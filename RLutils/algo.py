import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import torch
import numpy as np
import gymnasium as gym
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from torch.distributions import Categorical, kl_divergence
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList
from omegaconf import DictConfig, OmegaConf

from .reward_strategies import (
    InternalRewardStrategy,
    CuriousRewardStrategy,
    GoalSelectionStrategy,
    RandomGoalStrategy,
)
from .spatial_strategies import create_spatial_representation_strategy
from .other import synthesize
from .analysis import mutual_info_policy
from RLutils.goal_video_recorder import GoalMarkedVideoRecorder

logger = logging.getLogger(__name__)


# ============================================================================
# Experience Collection
# ============================================================================

@dataclass
class StepData:
    """Data from a single environment step."""
    obs: Dict
    action: torch.Tensor
    reward: float
    value: torch.Tensor
    SR: torch.Tensor
    dist: Any
    log_prob: torch.Tensor
    loc: Tuple[int, int]
    mask: float
    reward_past: Optional[float] = None  # reward credited to the *previous* step


class ExperienceBuffer:
    """Buffer for storing and managing experience data."""
    
    def __init__(
            self,
            num_frames: int,
            SR_size: int,
            device: torch.device,
            action_space=None,
    ):
        self.num_frames = num_frames
        self.device = device
        
        # Initialize buffers
        self.obss = [None] * num_frames
        self.locs = [None] * num_frames
        self.masks = torch.zeros(num_frames, device=device)
        self.continuous_actions = isinstance(action_space, gym.spaces.Box)
        if self.continuous_actions:
            self.actions = torch.zeros(
                (num_frames, *action_space.shape), device=device, dtype=torch.float32
            )
        else:
            self.actions = torch.zeros(num_frames, device=device, dtype=torch.int)
        self.values = torch.zeros(num_frames, device=device)
        self.SRs = torch.zeros((num_frames, SR_size), device=device)
        self.rewards = torch.zeros(num_frames, device=device)
        self.advantages = torch.zeros(num_frames, device=device)
        self.log_probs = torch.zeros(num_frames, device=device)
        self.all_rewards = {'extrinsic': self.rewards}
        
        # The lists below are only relevant if pRNN is being trained
        self.done_indices = [0]
        self.last_observations = []
        self.last_actions = []
        # In past-SR PPO, the first stored SR after an environment reset is a
        # zero placeholder.  Keep the actual SR from the preceding terminal
        # action so intrinsic rewards can retain the within-episode transition.
        self.past_SR_terminal_states = {}
    
    def store_step(self, idx: int, step_data: StepData):
        """Store data from a single step."""
        self.obss[idx] = step_data.obs
        self.locs[idx] = step_data.loc
        self.SRs[idx] = step_data.SR
        self.masks[idx] = step_data.mask
        self.actions[idx] = step_data.action
        self.values[idx] = step_data.value
        self.rewards[idx] = step_data.reward
        self.log_probs[idx] = step_data.log_prob
        if step_data.reward_past is not None and idx > 0:
            self.rewards[idx - 1] += step_data.reward_past

    def store_rewards(self, name: str, rewards: torch.Tensor):
        """Store additional reward signals."""
        self.all_rewards[name] = rewards
    
    def add_traj_end(
            self,
            idx: int,
            obs: Dict,
            act: np.ndarray,
            past_SR_terminal_state: torch.Tensor | None = None,
        ):
        """Mark trajectory end for pRNN training."""
        self.done_indices.append(idx + 1)
        self.last_observations.append(obs)
        self.last_actions.append(act)
        if past_SR_terminal_state is not None:
            self.past_SR_terminal_states[idx] = past_SR_terminal_state.detach().clone()

    def reset_trajectories(self):
        """Reset trajectory tracking."""
        self.done_indices = [0]
        self.last_observations = []
        self.last_actions = []
        self.past_SR_terminal_states = {}
    
    def compute_advantages(
            self,
            discount: float,
            gae_lambda: float,
            next_value: torch.Tensor,
            final_mask: float
        ):
        """Compute advantages using GAE."""
        for i in reversed(range(self.num_frames)):
            next_mask = self.masks[i + 1] if i < self.num_frames - 1 else final_mask
            next_val = self.values[i + 1] if i < self.num_frames - 1 else next_value
            next_adv = self.advantages[i + 1] if i < self.num_frames - 1 else 0
            
            # Augmented reward
            reward_term = sum(rew[i] for rew in self.all_rewards.values())
            
            delta = reward_term + discount * next_val * next_mask - self.values[i]
            self.advantages[i] = delta + discount * gae_lambda * next_adv * next_mask
    
    def to_dict_list(self, preprocess_fn, device: torch.device) -> DictList:
        """Convert buffer to DictList format for training."""
        exps = DictList()
        exps.obs = preprocess_fn(self.obss, device=device)
        exps.SR = self.SRs
        exps.action = self.actions
        exps.value = self.values
        exps.reward = self.rewards
        exps.advantage = self.advantages
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs
        
        return exps


# ============================================================================
# Logging and Metrics
# ============================================================================

class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self, env, loc_mask: List[bool]):
        self.env = env
        self.loc_mask = loc_mask
        self.continuous = bool(getattr(env, "continuous", False))
        
        # Episode metrics
        self.episode_return = 0
        self.episode_num_frames = 0
        self.done_counter = 0
        
        # History
        self.returns = []
        self.num_frames_list = []
        
        # Location tracking
        # NOTE: specific for Minigrid
        self.loc_visits = None if self.continuous else np.zeros([env.width, env.height])
        self.loc_history = None if self.continuous else [np.zeros(np.sum(loc_mask))] * 5
    
    def update_step(self, reward: float, past_reward: Optional[float] = None):
        """Update metrics for a single step."""
        self.episode_return += reward
        if past_reward is not None:
            self.episode_return += past_reward
        self.episode_num_frames += 1
    
    def update_location_visit(self, loc: Tuple[int, int]):
        """Track location visit."""
        if self.continuous:
            return
        self.loc_visits[loc] += 1
    
    def episode_done(self):
        """Mark episode as done and record metrics."""
        self.done_counter += 1
        self.returns.append(self.episode_return)
        self.num_frames_list.append(self.episode_num_frames)
        
        # Reset episode metrics
        self.episode_return = 0
        self.episode_num_frames = 0
    
    def compute_location_entropy(self) -> Tuple[float, float]:
        """Compute location entropy metrics."""
        if self.continuous:
            # Grid-cell occupancy/entropy is a MiniGrid diagnostic.  Do not
            # silently quantise continuous Miniworld positions here.
            return float("nan"), float("nan")
        visits_filtered = self.loc_visits.flatten('F')[self.loc_mask]
        loc_entropy = entropy(visits_filtered, base=2)
        
        # Update history
        self.loc_history.pop(0)
        self.loc_history.append(visits_filtered)
        loc_entropy_5 = entropy(np.sum(self.loc_history, axis=0), base=2)
        
        # Reset visits
        self.loc_visits = np.zeros([self.env.width, self.env.height])
        
        return loc_entropy, loc_entropy_5
    
    def get_episode_logs(self) -> List:
        """Get and clear episode logs."""
        logs_return = self.returns.copy()
        logs_frames = self.num_frames_list.copy()
        
        self.returns.clear()
        self.num_frames_list.clear()
        
        return logs_return, logs_frames


# ============================================================================
# Main PPO Algorithm
# ============================================================================

class PredictivePPOAlgo:
    """
    Modular PPO algorithm with spatial representations and reward augmentation.
    
    Refactored for improved:
    - Separation of concerns
    - Testability
    - Extensibility
    - Readability
    """
    
    def __init__(
        self,
        env,
        acmodel: torch.nn.Module,
        predictiveNet: Any,
        ppo_config: DictConfig,
        spatial_config: DictConfig,
        reward_config: DictConfig,
        device: Optional[torch.device] = None,
        preprocess_obss=None,
        joint_probabilities: bool = True,
    ):
        """
        Initialize PPO algorithm with modular configuration.
        
        Args:
            env: Environment instance
            acmodel: Actor-critic model
            ppo_config: PPO hyperparameters
            spatial_config: Spatial representation configuration
            reward_config: Reward augmentation configuration
            training_config: Additional training settings
            device: Torch device
            preprocess_obss: Observation preprocessing function
        """
        logger.info("Initializing PredictivePPOAlgo")
        
        # Store core components
        self.env = env
        self.acmodel = acmodel
        self.predictiveNet = predictiveNet
        self.device = device or torch.device('cpu')
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        
        # Store configurations
        self.config = ppo_config
        self.spatial_config = spatial_config
        self.reward_config = reward_config
        self.continuous_actions = isinstance(self.env.action_space, gym.spaces.Box)
        self.log_joint_probabilities = bool(joint_probabilities) and not self.continuous_actions
        
        # Validate configurations
        self._validate_config()
        
        # Initialize components
        self._setup_environment()
        self._setup_models()
        self._setup_spatial_representation()
        self._setup_reward_strategies()
        self._setup_experience_buffer()
        self._setup_metrics()
        self._setup_optimizer()
        
        # Policy prior
        self._setup_policy_prior()
        
        # Training state
        self.batch_num = 0
        
        # Log storage
        self._logs_collect = {}
        self._logs_update = {}
        
        logger.info("PredictivePPOAlgo initialization complete")
    
    def _validate_config(self):
        """Validate configuration consistency."""
        assert (self.acmodel.recurrent or self.config.recurrence == 1), \
            "Non-recurrent model requires recurrence=1"
        assert (self.config.num_frames % self.config.recurrence == 0), \
            "num_frames must be divisible by recurrence"
        assert (self.config.batch_size % self.config.recurrence == 0), \
            "batch_size must be divisible by recurrence"
        assert (self.spatial_config.past_SR ^ ('Next' in str(self.env.encodeAction))), \
            "pastSR configuration mismatch with environment"
        if self.spatial_config.predictive_net is not None:
            assert (self.config.num_frames % self.spatial_config.predictive_net.seqdur == 0), \
                "num_frames must be divisible by pRNN sequence duration"
    
    def _setup_environment(self):
        """Setup environment-related attributes."""
        if getattr(self.env, "continuous", False):
            self.loc_mask = []
            self.obs = self.env.reset()
            self.loc = self._get_agent_pos()
            self.mask = 1
            return
        # Get location mask
        # NOTE: specific for Minigrid
        if hasattr(self.env, 'loc_mask'):
            self.loc_mask = self.env.loc_mask
        elif 'Shell' in str(type(self.env)):
            self.loc_mask = [x is None or x.can_overlap() for x in self.env.env.grid.grid]
        else:
            self.loc_mask = [x is None or x.can_overlap() for x in self.env.grid.grid]
        
        # Initialize environment state
        self.obs = self.env.reset()
        self.loc = self._get_agent_pos()
        self.mask = 1
    
    def _setup_models(self):
        """Setup and configure models."""
        self.acmodel.to(self.device)
        self.acmodel.train()
    
    def _setup_spatial_representation(self):
        """Initialize spatial representation strategy."""
        self.SR_strategy = create_spatial_representation_strategy(
            self.spatial_config,
            self.env,
            self.predictiveNet,
            self.device
        )
        self.SR = self.SR_strategy.initialize_SR(obs=self.obs)
        logger.debug(f"Spatial representation size: {self.SR.shape}")
    
    def _setup_reward_strategies(self):
        """Initialize reward augmentation strategies."""

        self.all_rewards = []
        self.rewards = torch.tensor([], device=self.device)
        self.all_rewards.append(self.rewards)
        
        # Internal rewards
        if self.reward_config.internal_enabled:
            
            SR_size = self.SR_strategy.get_SR_size(self.SR)
            self.internal_strategy = InternalRewardStrategy(
                k_int=self.reward_config.internal_coef,
                SR_size=SR_size,
                device=self.device,
                num_frames=self.config.num_frames,
                mask_internal=self.spatial_config.mask_internal,
                mask_indices=self.spatial_config.mask_indices
            )
            logger.info("Internal reward strategy enabled")
        else:
            self.internal_strategy = None
            self.internal_rewards = None
        
        # Curious rewards
        if self.reward_config.curious_enabled:
            assert self.predictiveNet is not None, \
                "Curious requires predictive network"
            self.curious_strategy = CuriousRewardStrategy(
                predictive_net=self.predictiveNet,
                k_curious=self.reward_config.curious_coef,
                device=self.device,
            )
            logger.info("Curious reward strategy enabled")
        else:
            self.curious_strategy = None
            self.curious_rewards = None
    
    def _setup_experience_buffer(self):
        """Initialize experience buffer."""
        SR_size = self.SR_strategy.get_SR_size(self.SR)
        self.experience_buffer = ExperienceBuffer(
            num_frames=self.config.num_frames,
            SR_size=SR_size,
            device=self.device,
            action_space=self.env.action_space,
        )
    
    def _setup_metrics(self):
        """Initialize metrics tracker."""
        self.metrics = MetricsTracker(self.env, self.loc_mask)
    
    def _setup_policy_prior(self):
        """Setup policy prior tensor if configured."""
        prior = getattr(self.config, 'policy_prior', None)
        kl_coef = getattr(self.config, 'prior_kl_coef', 0.0)
        
        if self.continuous_actions and prior is not None and kl_coef > 0.0:
            raise ValueError(
                "Categorical policy_prior is not defined for continuous actions. "
                "Set ppo.prior_kl_coef=0 for Miniworld PPO."
            )
        if prior is not None and kl_coef > 0.0:
            # Accept list/ListConfig/tensor
            prior_list = OmegaConf.to_container(prior) if hasattr(prior, '_metadata') else list(prior)
            prior_tensor = torch.tensor(prior_list, dtype=torch.float32, device=self.device)
            prior_tensor = prior_tensor / prior_tensor.sum()  # normalise defensively
            self.policy_prior = prior_tensor
            logger.info(f"Policy prior enabled: {prior_list}, kl_coef={kl_coef}")
        else:
            self.policy_prior = None
        
        self.prior_kl_coef = kl_coef

    def _setup_optimizer(self):
        """Initialize optimizer."""
        self.optimizer = torch.optim.Adam(
            self.acmodel.parameters(), 
            lr=self.config.lr, 
            eps=self.config.adam_eps
        )
    
    def _get_agent_pos(self) -> Tuple[int, int]:
        """Get current agent position from environment."""
        if hasattr(self.env, 'get_agent_pos'):
            return self.env.get_agent_pos()
        else:
            return self.env.agent_pos

    def _get_hd(self):
        """Return the current continuous HD when the Shell exposes one."""
        if hasattr(self.env, "get_agent_dir"):
            return self.env.get_agent_dir()
        return None

    def _state_for_prnn(self, hd=None):
        """Provide the HD belonging to the observation sent to pRNN.

        Miniworld Shells expand this one HD only as an encoding detail. They
        must not receive a past/current HD pair: for past-SR networks the
        selected observation is ``o_t`` and gets ``HD_t``; for an offset-action
        current-SR network it is ``o_(t+1)`` and gets ``HD_(t+1)`` alongside
        the action's already-available previous speed.
        """
        if hd is None:
            hd = self._get_hd()
        if hd is None:
            return None
        return {"agent_dir": np.float32(hd)}

    def _hd_for_spatial_representation(self, past_hd, current_hd):
        """Choose HD from the same timestep as the strategy's observation."""
        return past_hd if self.spatial_config.past_SR else current_hd

    def _environment_action(self, action: torch.Tensor):
        """Convert a one-policy-batch action into Gymnasium's scalar/vector form."""
        action_cpu = action.detach().cpu()
        if self.continuous_actions:
            return action_cpu.squeeze(0).numpy()
        return int(action_cpu.item())

    @staticmethod
    def _policy_log_prob(dist, action: torch.Tensor) -> torch.Tensor:
        """Return one joint action log probability per batch item."""
        log_prob = dist.log_prob(action)
        return log_prob.sum(dim=-1) if log_prob.ndim > 1 else log_prob
    
    def _select_action(self) -> Tuple[torch.Tensor, Any, torch.Tensor, np.ndarray]:
        """
        Select action using current policy.
        
        Returns:
            action, distribution, value, deterministic_action
        """
        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        
        with torch.no_grad():
            dist, value = self.acmodel(preprocessed_obs, SR=self.SR)
        
        action = dist.sample()
        det_action = self._environment_action(action)
        
        return action, dist, value, det_action
    
    def _collect_single_step(self, idx: int) -> StepData:
        """
        Collect a single step of experience.
        
        Args:
            idx: Current step index
        
        Returns:
            StepData for this step
        """
        # Select action
        action, dist, value, det_action = self._select_action()
        past_hd = self._get_hd()
        
        # Execute action
        new_obs, reward, terminated, truncated, _ = self.env.step(det_action)
        
        # Handle exploration mode
        if self.reward_config.exploration:
            reward, terminated, truncated = 0, False, False
        
        done = terminated or truncated
        done = done or (self.reward_config.exploration and 
                (idx + 1) % self.spatial_config.predictive_net.seqdur == 0)
        
        new_loc = self._get_agent_pos()
        current_hd = self._get_hd()
        
        # Compute spatial representation
        SR_new = self.SR_strategy.compute_SR(
            action=det_action,
            past_obs=self.obs,
            new_obs=new_obs,
            state=self._state_for_prnn(
                self._hd_for_spatial_representation(past_hd, current_hd)
            ),
            )
        
        # Create step data
        step_data = StepData(
            obs=self.obs,
            action=action.squeeze(0),
            reward=reward,
            value=value,
            SR=self.SR,
            dist=dist,
            log_prob=self._policy_log_prob(dist, action).squeeze(0),
            loc=self.loc,
            mask=self.mask,
        )
        
        # Update state
        self.obs = new_obs
        self.loc = new_loc
        self.SR = SR_new
        self.mask = 1 - done
        
        return step_data, done
    
    def _handle_episode_end(self, idx: int):
        """Handle end of episode cleanup and logging."""

        _, _, _, det_action = self._select_action()

        # Update internal reference if applicable
        if self.internal_strategy and self.experience_buffer.rewards[idx] > 1e-5:
            SR = self.SR_strategy.last_SR(
                SR=self.SR,
                det_action=det_action,
                obs=self.obs,
                state=self._state_for_prnn(),
            )
            self.internal_strategy.update_reference(SR)
        
        # Record episode metrics
        self.metrics.episode_done()
        terminal_state = None
        if self.spatial_config.past_SR:
            # ``self.SR`` is the state produced by the terminal action.  It is
            # replaced by a zero placeholder for the next episode below.
            terminal_state = self.SR.squeeze(0)
        self.experience_buffer.add_traj_end(
            idx, self.obs, det_action, past_SR_terminal_state=terminal_state
        )
        
        # Reset environment and SR
        if self.spatial_config.predictive_net:
            self.predictiveNet.reset_state(device=self.device)
        
        self.SR = self.SR_strategy.initialize_SR(obs=self.obs)
        self.obs = self.env.reset()
        
        logger.debug(f"Episode ended at step {idx}")
    
    def _compute_augmented_rewards(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute all augmented rewards.
        
        Returns:
            internal_rewards, curious_rewards
        """
        # Internal rewards
        if self.internal_strategy:
            if self.spatial_config.past_SR:
                # Need additional SR computation for last state
                _, _, _, det_action = self._select_action()
                SR_last = self.SR_strategy.last_SR(
                    SR=self.SR,
                    det_action=det_action,
                    obs=self.obs,
                    state=self._state_for_prnn(),
                )
                SRs_all = torch.cat((self.experience_buffer.SRs[1:], SR_last), dim=0)
                SRs_all = self._substitute_past_SR_terminal_states(SRs_all)
                episode_end_indices = self.experience_buffer.past_SR_terminal_states.keys()
            else:
                SRs_all = self.experience_buffer.SRs
                episode_end_indices = None
            internal_rewards = self.internal_strategy.compute_rewards(
                SRs=SRs_all, episode_end_indices=episode_end_indices
            )
            
            self.experience_buffer.store_rewards('internal', internal_rewards)
        
        # Curious rewards
        if self.curious_strategy:
            actions_np = self.experience_buffer.actions.cpu().numpy()
            obss_all = self.experience_buffer.obss + [self.obs]
            curious_rewards = self.curious_strategy.compute_rewards(
                obss=obss_all,
                actions=actions_np,
                num_frames=self.config.num_frames,
                done_indices=self.experience_buffer.done_indices,
                last_observations=self.experience_buffer.last_observations,
                last_actions=self.experience_buffer.last_actions
            )
            self.experience_buffer.store_rewards('curious', curious_rewards)

    def _substitute_past_SR_terminal_states(self, SRs: torch.Tensor) -> torch.Tensor:
        """Replace shifted reset placeholders with terminal pRNN states.

        ``SRs[t]`` is the pRNN state reached by the action at rollout index
        ``t``.  For a completed episode, the state at that index would
        otherwise be the zero SR stored at the next episode's first action.
        """
        if not self.experience_buffer.past_SR_terminal_states:
            return SRs

        aligned_SRs = SRs.clone()
        for index, terminal_state in self.experience_buffer.past_SR_terminal_states.items():
            aligned_SRs[index] = terminal_state.to(
                device=aligned_SRs.device, dtype=aligned_SRs.dtype
            )
        return aligned_SRs
    
    def _compute_joint_probabilities(self) -> np.ndarray:
        """Compute joint probability distribution over states and actions."""
        joint_probs = np.zeros(
            (getattr(self.env, "numHDs"),
             self.env.width,
             self.env.height,
             getattr(self.acmodel, "act_dim")),
            dtype=np.float32
        )
        
        # Recompute distributions for all states
        with torch.no_grad():
            for i in range(self.config.num_frames):
                dist, _ = self.acmodel(
                    self.preprocess_obss([self.experience_buffer.obss[i]], device=self.device),
                    SR=self.experience_buffer.SRs[i:i+1]
                )
                
                # Get HD and location
                try:
                    hd = self.experience_buffer.obss[i]["direction"]
                except KeyError:
                    hd = self.experience_buffer.obss[i]["HD"]
                
                x, y = self.experience_buffer.locs[i]
                act_probs = dist.probs.detach().cpu().numpy().squeeze()
                joint_probs[hd, x, y, :] += act_probs
        
        return joint_probs
    
    def collect_experiences(self) -> DictList:
        """
        Collect rollouts and compute advantages.
        Stores logs in self._logs_collect.
        
        Returns:
            experiences: DictList containing all experience data
        """
        logger.debug("Starting experience collection")
        self.experience_buffer.past_SR_terminal_states = {}
        
        # Collect experiences
        any_done = False
        for i in range(self.config.num_frames):
            # Collect single step
            step_data, done = self._collect_single_step(i)
            
            # Store in buffer
            self.experience_buffer.store_step(i, step_data)
            
            # Update metrics
            self.metrics.update_step(step_data.reward, step_data.reward_past)
            self.metrics.update_location_visit(step_data.loc)
            
            # Handle episode end
            if done:
                any_done = True
                self._handle_episode_end(i)
        
        # If no episode ended, count the last frame as done
        if not any_done:
            self._handle_episode_end(self.config.num_frames - 1)
        
        # Compute augmented rewards
        self._compute_augmented_rewards()
        
        # Compute advantages
        _, _, next_value, _ = self._select_action()
        
        self.experience_buffer.compute_advantages(
            discount=self.config.discount,
            gae_lambda=self.config.gae_lambda,
            next_value=next_value,
            final_mask=self.mask
        )
        
        # Convert to DictList
        exps = self.experience_buffer.to_dict_list(self.preprocess_obss, self.device)
        
        # Reset predictive network state
        if self.spatial_config.predictive_net:
            self.predictiveNet.reset_state(device=self.device)
        
        # Compute metrics
        loc_entropy, loc_entropy_5 = self.metrics.compute_location_entropy()
        returns, num_frames = self.metrics.get_episode_logs()
        joint_probs = self._compute_joint_probabilities() if self.log_joint_probabilities else None
        
        # Store logs
        self._logs_collect = {
            "return_per_episode": returns,
            "num_frames_per_episode": num_frames,
            "num_frames": self.config.num_frames,
            "num_episodes": self.metrics.done_counter,
            "values": self.experience_buffer.values.tolist(),
            "advantages": self.experience_buffer.advantages.tolist(),
            "loc_entropy": loc_entropy,
            "loc_entropy_5": loc_entropy_5,
        }
        if joint_probs is not None:
            self._logs_collect["joint_dist"] = joint_probs
        
        if self.internal_strategy:
            self._logs_collect["internal_rewards"] = \
            self.experience_buffer.all_rewards['internal'].tolist()

        if self.curious_strategy:
            self._logs_collect["curious_rewards"] = \
            self.experience_buffer.all_rewards['curious'].tolist()
        
        logger.debug(f"Collected {self.config.num_frames} frames, "
                    f"{self.metrics.done_counter} pRNN trajectories")
        
        return exps
    
    def _compute_ppo_loss(self, sb) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss for a sub-batch.
        
        Args:
            sb: Sub-batch of experiences
        
        Returns:
            loss, metrics dictionary
        """
        # Forward pass
        dist, value = self.acmodel(sb.obs, SR=sb.SR)
        
        # Policy loss (PPO clip objective)
        policy_entropy = dist.entropy().mean()
        ratio = torch.exp(self._policy_log_prob(dist, sb.action) - sb.log_prob)
        surr1 = ratio * sb.advantage
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 
                           1.0 + self.config.clip_eps) * sb.advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (clipped)
        value_clipped = sb.value + torch.clamp(
            value - sb.value, -self.config.clip_eps, self.config.clip_eps
        )
        surr1 = (value - sb.returnn).pow(2)
        surr2 = (value_clipped - sb.returnn).pow(2)
        value_loss = torch.max(surr1, surr2).mean()
        
        # KL divergence from policy prior
        if self.policy_prior is not None:
            prior_expanded = self.policy_prior.unsqueeze(0).expand(dist.probs.shape[0], -1)
            prior_dist = Categorical(probs=prior_expanded)
            prior_kl = kl_divergence(dist, prior_dist).mean()
        else:
            prior_kl = torch.tensor(0.0, device=self.device)
        
        # Total loss
        loss = (policy_loss - 
                self.config.entropy_coef * policy_entropy + 
                self.config.value_loss_coef * value_loss +
                self.prior_kl_coef * prior_kl)
        
        # Metrics
        metrics = {
            'entropy': policy_entropy.item() / torch.log(torch.tensor(2.0)),  # nats to bits
            'value': value.mean().item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'prior_kl': prior_kl.item()
        }
        
        return loss, metrics
    
    def _get_batches_starting_indexes(self) -> List[np.ndarray]:
        """
        Generate batch starting indexes for training.
        
        Returns:
            List of index arrays for each batch
        """
        indexes = np.arange(0, self.config.num_frames, self.config.recurrence)
        indexes = np.random.permutation(indexes)
        
        # Shift indexes by recurrence//2 alternately for diversity
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.config.recurrence) % self.config.num_frames != 0]
            indexes += self.config.recurrence // 2
        self.batch_num += 1
        
        num_indexes = self.config.batch_size // self.config.recurrence
        batches = [indexes[i:i + num_indexes] 
                  for i in range(0, len(indexes), num_indexes)]
        
        return batches
    
    def _train_predictive_network(self, exps: DictList):
        """Train the pRNN on collected experiences."""
        if not self.spatial_config.train:
            return
        
        logger.debug("Training pRNN")
        
        pN = self.predictiveNet
        pN.pRNN.to(self.device)
        
        # Extract episode info before batching
        done_indices = self.experience_buffer.done_indices
        last_observations = self.experience_buffer.last_observations
        
        for idx in range(1, len(done_indices)):
            start_episode = done_indices[idx - 1]
            end_episode = done_indices[idx]
            last_obs = last_observations[idx - 1]
            
            # Extract episode data
            images_tensor = exps.obs.image[start_episode:end_episode]
            hd_tensor = exps.obs.direction[start_episode:end_episode]
            
            obs_for_pN = [
                {'image': images_tensor[i].cpu().numpy(), 
                 'direction': hd_tensor[i].item()}
                for i in range(len(images_tensor))
            ]
            act_for_pN = exps.action[start_episode:end_episode].cpu().numpy()
            
            # Convert and train
            obs, act = pN.env_shell.env2pred(obs_for_pN + [last_obs], act_for_pN)
            obs = obs.to(self.device)
            act = act.to(self.device)
            
            pN.trainStep(obs, act)
            pN.numTrainingEpochs += 1
        
        self.experience_buffer.reset_trajectories()
    
    def update_parameters(self, exps: DictList, update_params: bool = True) -> None:
        """
        Update model parameters using collected experiences.
        Stores logs in self._logs_update.
        
        Args:
            exps: Experience dictionary
            update_params: Whether to actually update parameters (for debugging)
        """
        logger.debug("Updating parameters")
        
        # Training loop
        all_metrics = {
            'entropy': [],
            'value': [],
            'policy_loss': [],
            'value_loss': [],
            'prior_kl': [],
            'grad_norm': []
        }
        
        for epoch in range(self.config.epochs):
            for batch_inds in self._get_batches_starting_indexes():
                batch_loss = 0
                batch_metrics= {k: 0 for k in all_metrics.keys()}
                
                for i in range(self.config.recurrence):
                    # Get sub-batch
                    sb = exps[batch_inds + i]
                    
                    # Compute loss
                    loss, metrics = self._compute_ppo_loss(sb)
                    batch_loss += loss
                    
                    for key in metrics.keys():
                        batch_metrics[key] += metrics[key]
                
                # Average over recurrence steps
                for key in batch_metrics.keys():
                    batch_metrics[key] /= self.config.recurrence
                
                # Update parameters
                if update_params:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    
                    # Compute gradient norm
                    grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2 
                        for p in self.acmodel.parameters()
                    ) ** 0.5
                    batch_metrics['grad_norm'] = grad_norm
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.acmodel.parameters(), 
                        self.config.max_grad_norm
                    )
                    
                    self.optimizer.step()
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    all_metrics[key].append(value)
        
        # Train predictive network
        self._train_predictive_network(exps)
        
        # Store logs
        self._logs_update = {key: np.mean(values) for key, values in all_metrics.items()}
        
        logger.debug(f"Training complete - Policy loss: {self._logs_update.get('policy_loss', 0):.4f}, "
                    f"Value loss: {self._logs_update.get('value_loss', 0):.4f}")

    def process_logs(self) -> Dict:
        """
        Process raw logs stored in self._logs_collect and self._logs_update
        into final metrics ready for logging.
        
        Returns:
            Dictionary of processed metrics ready for wandb logging
        """
        logs_collect = self._logs_collect
        logs_update = self._logs_update
        processed = {}
        
        # Process episode-level metrics
        if "num_frames_per_episode" in logs_collect:
            num_frames_per_episode = synthesize(logs_collect["num_frames_per_episode"])
            for key, value in num_frames_per_episode.items():
                processed[f"steps_per_trial_{key}"] = value
        
        if "return_per_episode" in logs_collect:
            return_per_episode = synthesize(logs_collect["return_per_episode"], signs=True)
            for key, value in return_per_episode.items():
                processed[f"return_{key}"] = value
        
        # Process intrinsic/curious rewards
        if "internal_rewards" in logs_collect:
            int_rewards = synthesize(logs_collect["internal_rewards"], abs=True)
            for key, value in int_rewards.items():
                processed[f"int_reward_{key}"] = value
        
        if "curious_rewards" in logs_collect:
            cur_rewards = synthesize(logs_collect["curious_rewards"], abs=True)
            for key, value in cur_rewards.items():
                processed[f"cur_reward_{key}"] = value
        
        # Process values and advantages
        if "values" in logs_collect:
            values = synthesize(logs_collect["values"])
            for key, value in values.items():
                processed[f"values_{key}"] = value
        
        if "advantages" in logs_collect:
            advantages = synthesize(logs_collect["advantages"])
            for key, value in advantages.items():
                processed[f"advantages_{key}"] = value

        # Process initial goal distances (goal-conditioned variants)
        if "distances_success" in logs_collect and len(logs_collect["distances_success"]) > 0:
            distances_success = synthesize(logs_collect["distances_success"])
            for key, value in distances_success.items():
                processed[f"distance_success_{key}"] = value

        if "distances_fail" in logs_collect and len(logs_collect["distances_fail"]) > 0:
            distances_fail = synthesize(logs_collect["distances_fail"])
            for key, value in distances_fail.items():
                processed[f"distance_fail_{key}"] = value
        
        # Add scalar metrics from collect_experiences
        for key in ["num_episodes", "loc_entropy", "loc_entropy_5", "num_frames"]:
            if key in logs_collect:
                processed[key] = logs_collect[key]
        
        # Add metrics from update_parameters
        for key in ["entropy", "policy_loss", "value_loss", "prior_kl", "grad_norm"]:
            if key in logs_update:
                processed[key] = logs_update[key]
        
        # Compute mutual information if joint distribution is available
        if "joint_dist" in logs_collect:
            processed["MI_policy"] = mutual_info_policy(logs_collect["joint_dist"])
        
        # Add projection similarity if available (for theta experiments)
        if "proj_sim" in logs_update:
            processed["projection_similarity"] = logs_update["proj_sim"]
        
        return processed


    def randomAgent_collect_exp_and_update(self, agent):
        assert self.spatial_config.train, \
        "The only reason to have random actions in algo is to train the pRNN geinus..."
        pN = self.predictiveNet
        num_frames = self.config.num_frames
        pN.pRNN.to(self.device)
        seqdur = self.spatial_config.predictive_net.seqdur
        numtrials = math.ceil(num_frames / seqdur)
        locs = [None] * num_frames
        loc_visits = np.zeros([self.env.width, self.env.height])
        loc_history = [np.zeros(np.sum(self.loc_mask))] * 5

        log_curr_seqdurs = []
        for bb in range(numtrials):
            curr_seqdur = min(
                    seqdur,
                    num_frames - (bb)*seqdur
                )
            log_curr_seqdurs.append(curr_seqdur)
            #The above is needed if seqdur is not a perfect divisor of num frames.
            # It implies that the last trial might have < seqdur steps

            obs,act,state,_ = pN.collectObservationSequence(self.env,
                                                                 agent, curr_seqdur)
            
            #Train
            obs, act = obs.to(self.device), act.to(self.device)
            _,_,_ = pN.trainStep(obs, act)
            pN.numTrainingEpochs += 1

            #Collect location info
            locs_array = state['agent_pos'][:-1,:]
            loc_list_current = [tuple(thisloc) for thisloc in locs_array]

            startidx = bb*seqdur
            endidx = min(num_frames, (bb+1)*seqdur)
            locs[startidx:endidx] = loc_list_current

        for loc in locs:
            loc_visits[loc] += 1
        loc_visits = loc_visits.flatten('F')[self.loc_mask]
        loc_entropy = entropy(loc_visits, base=2)

        loc_history.pop(0)
        loc_history.append(loc_visits)
        loc_entropy_5 = entropy(np.sum(loc_history, axis=0), base=2)

        policy_entropy = entropy(agent.default_action_probability, base=2)

        return {"num_frames": num_frames,
                "num_frames_per_episode": log_curr_seqdurs,
                "num_episodes": numtrials,
                "entropy": policy_entropy,
                "loc_entropy": loc_entropy,
                "loc_entropy_5": loc_entropy_5}


# ============================================================================
# Goal-Conditioned PPO
# ============================================================================

class GoalConditionedPPOAlgo(PredictivePPOAlgo):
    """
    Goal-conditioned extension of PredictivePPOAlgo.

    At the start of every ``collect_experiences`` call a goal SR is drawn from
    *goal_pool* via *goal_strategy* (defaults to :class:`RandomGoalStrategy`).
    The same goal is re-drawn after each episode ends within the rollout.

    The goal SR is:
    * Set as the reference of the ``InternalRewardStrategy`` so that
      distance-based internal rewards track progress toward the goal.
    * Concatenated to the current SR and passed to ``ACModelSR`` as its ``SR``
      argument.  The model must therefore be constructed with
      ``SR_size = raw_SR_size + goal_SR_size``.

    A terminal bonus reward is issued when the agent's new SR is closer than
    *goal_threshold* (cosine distance) to the goal:
    * ``past_SR=False`` -> +1 added to the **current** step's reward.
    * ``past_SR=True``  -> +1 added to the **previous** step's reward.
    """

    def __init__(
        self,
        env,
        acmodel: torch.nn.Module,
        predictiveNet: Any,
        ppo_config: DictConfig,
        spatial_config: DictConfig,
        reward_config: DictConfig,
        goal_pool: Dict,
        goal_threshold: float,
        goal_strategy: Optional[GoalSelectionStrategy] = None,
        check_location: bool = False,
        exclude_loactions: Optional[np.ndarray] = None,
        video_log_freq: Optional[int] = None,
        video_folder: Optional[str] = None,
        video_ext: str = '',
        device: Optional[torch.device] = None,
        preprocess_obss=None,
    ):
        """
        Args:
            goal_pool:       Tensor of candidate goal SRs  [n_goals, SR_dim].
            goal_threshold:  Cosine-distance threshold for reaching a goal.
            goal_strategy:   How to pick a goal from the pool.
                             Defaults to :class:`RandomGoalStrategy`.
            exclude_loactions: Optional 2xN numpy array of excluded goal
                               coordinates where each column is [x, y].
            (other args):    Forwarded to :class:`PredictivePPOAlgo`.
        """
        # Store goal pool early so _setup_experience_buffer can use it.
        # device is resolved inside super().__init__; store raw tensor for now.
        self._goal_pool_raw = goal_pool['h']
        self._goal_locs = goal_pool['state']['agent_pos']
        self._video_ext = video_ext

        # Optionally filter out excluded goal locations.
        if exclude_loactions is not None:
            exclude_arr = np.asarray(exclude_loactions)
            if exclude_arr.ndim != 2 or exclude_arr.shape[0] != 2:
                raise ValueError(
                    "exclude_loactions must be a 2xN numpy array "
                    "with columns [x, y]."
                )

            excluded = {
                (int(exclude_arr[0, i]), int(exclude_arr[1, i]))
                for i in range(exclude_arr.shape[1])
            }
            goal_locs_arr = np.asarray(self._goal_locs)

            keep_mask_np = np.array(
                [tuple(map(int, loc)) not in excluded for loc in goal_locs_arr],
                dtype=bool,
            )

            if not keep_mask_np.any():
                raise ValueError(
                    "All goal locations were excluded. "
                    "Provide a less restrictive exclude_loactions set."
                )

            self._goal_locs = goal_locs_arr[keep_mask_np]

            if isinstance(self._goal_pool_raw, torch.Tensor):
                keep_mask = torch.as_tensor(keep_mask_np[:-1], device=self._goal_pool_raw.device)
                self._goal_pool_raw = self._goal_pool_raw[:, keep_mask]
            else:
                self._goal_pool_raw = np.asarray(self._goal_pool_raw)[:, keep_mask_np[:-1]]

        self.goal_threshold = goal_threshold
        self.goal_strategy = goal_strategy or RandomGoalStrategy()
        self.check_location = check_location
        self.video_log_freq = int(video_log_freq) if video_log_freq is not None else 0
        self.video_folder = video_folder
        self._video_recorder = None
        self._episode_counter = 0

        if self.video_log_freq < 0:
            raise ValueError("video_log_freq must be >= 0")
        if self.video_log_freq > 0:
            if not self.video_folder:
                raise ValueError(
                    "video_folder must be provided when video_log_freq is enabled"
                )
            os.makedirs(self.video_folder, exist_ok=True)

        super().__init__(
            env, acmodel, predictiveNet,
            ppo_config, spatial_config, reward_config,
            device=device, preprocess_obss=preprocess_obss,
        )

        # Move pool to the resolved device and re-init experience buffer at the
        # correct (doubled) SR size.
        self.goal_pool = self._goal_pool_raw.to(self.device)

        raw_SR_size = self.SR_strategy.get_SR_size(self.SR)
        self.experience_buffer = ExperienceBuffer(
            num_frames=self.config.num_frames,
            SR_size=raw_SR_size * 2,
            device=self.device,
            action_space=self.env.action_space,
        )

        # InternalRewardStrategy is required for goal-reaching checks.
        assert self.internal_strategy is not None or self.check_location, (
            "To check goal-reaching, either internal rewards must be enabled or check_location=True must be set."
        )

        # Select and apply the first goal.
        self._select_goal()
        logger.info("GoalConditionedPPOAlgo initialised")

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def _select_goal(self) -> torch.Tensor:
        """Draw a new goal from the pool using the configured strategy."""
        self.goal, idx = self.goal_strategy.select_goal(self.goal_pool)
        self.goal_loc = self._goal_locs[idx]
        if self.internal_strategy:
            self.internal_strategy.set_reference(self.goal)

    def _should_record_current_episode(self) -> bool:
        """Whether the current episode should be recorded."""
        return self.video_log_freq > 0 and self._episode_counter % self.video_log_freq == 0

    def _ensure_episode_video_recorder(self) -> None:
        """Start a recorder for the current episode when logging is enabled."""
        if self._video_recorder is not None:
            return
        if not self._should_record_current_episode():
            return

        base_path = os.path.join(
            self.video_folder,
            f"goal_episode_{self._episode_counter:07d}_{self._video_ext}",
        )
        self._video_recorder = GoalMarkedVideoRecorder(
            env=self.env.env,
            base_path=base_path,
            goal_location=self.goal_loc,
            enabled=True,
            disable_logger=True,
        )
        self._video_recorder.capture_frame()

    def _stop_episode_video_recorder(self) -> None:
        """Close and clear the currently active episode recorder."""
        if self._video_recorder is None:
            return
        try:
            self._video_recorder.close()
        finally:
            self._video_recorder = None

    def _set_new_goal(self) -> None:
        """Select a new goal from the pool and update the internal strategy."""
        self._select_goal()
        logger.debug("New goal selected")

    def _compute_goal_start_distance(self) -> float:
        """Compute Euclidean distance between current location and goal location."""
        current_loc = np.asarray(self._get_agent_pos(), dtype=np.float32)
        goal_loc = np.asarray(self.goal_loc, dtype=np.float32)
        return float(np.linalg.norm(current_loc - goal_loc))

    def _check_goal(self, loc) -> Tuple[float, bool]:
        if (loc == self.goal_loc).all():
            return 1.0, True
        else:
            return 0.0, False

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _select_action(self) -> Tuple[torch.Tensor, Any, torch.Tensor, np.ndarray]:
        """Select action using goal-conditioned SR (current SR ‖ goal)."""
        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        goal_conditioned_SR = torch.cat([self.SR, self.goal], dim=1)

        with torch.no_grad():
            dist, value = self.acmodel(preprocessed_obs, SR=goal_conditioned_SR)

        action = dist.sample()
        det_action = self._environment_action(action)
        return action, dist, value, det_action

    def _collect_single_step(self, idx: int) -> Tuple[StepData, bool]:
        """
        Collect one step.  Checks for goal reaching and issues a terminal
        bonus reward when the cosine distance between the new SR and the goal
        drops below *goal_threshold*.
        """
        self._ensure_episode_video_recorder()

        action, dist, value, det_action = self._select_action()
        past_hd = self._get_hd()

        new_obs, reward, terminated, truncated, _ = self.env.step(det_action)

        if self._video_recorder is not None:
            self._video_recorder.capture_frame()

        if self.reward_config.exploration:
            reward, terminated, truncated = 0, False, False

        done = terminated or truncated
        done = done or (
            self.reward_config.exploration
            and (idx + 1) % self.spatial_config.predictive_net.seqdur == 0
        )

        new_loc = self._get_agent_pos()
        current_hd = self._get_hd()

        SR_new = self.SR_strategy.compute_SR(
            action=det_action,
            past_obs=self.obs,
            new_obs=new_obs,
            state=self._state_for_prnn(
                self._hd_for_spatial_representation(past_hd, current_hd)
            ),
        )

        # --- Goal-reaching check -----------------------------------------
        if self.check_location:
            reward_new, goal_reached = self._check_goal(new_loc)
            reward_past = None  # not used when check_location is True
        else:
            reward_new, reward_past, goal_reached = self.internal_strategy.check_goal(
                SR_new=SR_new,
                goal_threshold=self.goal_threshold,
                past_SR=self.spatial_config.past_SR,
            )
        reward += reward_new
        done = done or goal_reached
        if goal_reached:
            self._episode_goal_reached = True

        # Store goal-conditioned SR so training uses the same representation
        # that was used for action selection.
        goal_conditioned_SR = torch.cat([self.SR, self.goal], dim=1)

        step_data = StepData(
            obs=self.obs,
            action=action.squeeze(0),
            reward=reward,
            value=value,
            SR=goal_conditioned_SR,
            dist=dist,
            log_prob=self._policy_log_prob(dist, action).squeeze(0),
            loc=self.loc,
            mask=self.mask,
            reward_past=reward_past,
        )

        # Update state
        self.obs = new_obs
        self.loc = new_loc
        self.SR = SR_new          # always store *raw* SR; goal is kept separately
        self.mask = 1 - done

        return step_data, done

    def _handle_episode_end(self, idx: int) -> None:
        """Handle episode end AND select a fresh goal for the next episode."""
        # Log the starting distance of the episode that just ended.
        if self._current_episode_start_distance is not None:
            if self._episode_goal_reached:
                self._distances_success.append(self._current_episode_start_distance)
            else:
                self._distances_fail.append(self._current_episode_start_distance)

        self._stop_episode_video_recorder()

        super()._handle_episode_end(idx)
        self._set_new_goal()

        # super()._handle_episode_end resets the env, so refresh current location.
        self.loc = self._get_agent_pos()
        self._current_episode_start_distance = self._compute_goal_start_distance()
        self._episode_goal_reached = False
        self._episode_counter += 1

    def collect_experiences(self) -> DictList:
        """
        Collect rollouts.

        Selects a fresh goal at the very start of the collection window so
        that the first episode always has a well-defined goal even when the
        collector is called for the first time.
        """
        self._distances_success = []
        self._distances_fail = []
        self._episode_goal_reached = False
        self._stop_episode_video_recorder()

        self._set_new_goal()
        self._current_episode_start_distance = self._compute_goal_start_distance()

        try:
            exps = super().collect_experiences()
        finally:
            self._stop_episode_video_recorder()

        self._logs_collect["distances_success"] = self._distances_success
        self._logs_collect["distances_fail"] = self._distances_fail

        return exps

    def _compute_augmented_rewards(self) -> None:
        """
        Compute augmented rewards.

        The experience buffer stores goal-conditioned SRs
        (shape ``[num_frames, raw_SR_size + goal_SR_size]``).
        The internal strategy operates on *raw* SRs only, so we strip the
        goal suffix before calling ``compute_rewards``.
        """
        # --- Internal (proximity) rewards --------------------------------
        if self.internal_strategy:
            raw_sr_size = self.SR.shape[1]          # self.SR is always the raw SR

            if self.spatial_config.past_SR:
                _, _, _, det_action = self._select_action()
                SR_last = self.SR_strategy.last_SR(
                    SR=self.SR,
                    det_action=det_action,
                    obs=self.obs,
                )
                raw_SRs = torch.cat(
                    (self.experience_buffer.SRs[1:, :raw_sr_size], SR_last), dim=0
                )
                raw_SRs = self._substitute_past_SR_terminal_states(raw_SRs)
                episode_end_indices = self.experience_buffer.past_SR_terminal_states.keys()
            else:
                raw_SRs = self.experience_buffer.SRs[:, :raw_sr_size]
                episode_end_indices = None

            internal_rewards = self.internal_strategy.compute_rewards(
                SRs=raw_SRs, episode_end_indices=episode_end_indices
            )
            self.experience_buffer.store_rewards("internal", internal_rewards)

        # --- Curious rewards (unchanged from base class) -----------------
        if self.curious_strategy:
            actions_np = self.experience_buffer.actions.cpu().numpy()
            obss_all = self.experience_buffer.obss + [self.obs]
            curious_rewards = self.curious_strategy.compute_rewards(
                obss=obss_all,
                actions=actions_np,
                num_frames=self.config.num_frames,
                done_indices=self.experience_buffer.done_indices,
                last_observations=self.experience_buffer.last_observations,
                last_actions=self.experience_buffer.last_actions,
            )
            self.experience_buffer.store_rewards("curious", curious_rewards)

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import torch
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList
from omegaconf import DictConfig

from .reward_strategies import InternalRewardStrategy, CuriousRewardStrategy
from .spatial_strategies import create_spatial_representation_strategy
from .other import synthesize
from .analysis import mutual_info_policy

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


class ExperienceBuffer:
    """Buffer for storing and managing experience data."""
    
    def __init__(self, num_frames: int, SR_size: int, device: torch.device):
        self.num_frames = num_frames
        self.device = device
        
        # Initialize buffers
        self.obss = [None] * num_frames
        self.locs = [None] * num_frames
        self.masks = torch.zeros(num_frames, device=device)
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
        # if step_data.reward_past is not None:
        #     self.rewards[idx-1] += step_data.reward_past

    def store_rewards(self, name: str, rewards: torch.Tensor):
        """Store additional reward signals."""
        self.all_rewards[name] = rewards
    
    def add_traj_end(self, idx: int, obs: Dict, act: np.ndarray):
        """Mark trajectory end for pRNN training."""
        self.done_indices.append(idx + 1)
        self.last_observations.append(obs)
        self.last_actions.append(act)

    def reset_trajectories(self):
        """Reset trajectory tracking."""
        self.done_indices = [0]
        self.last_observations = []
        self.last_actions = []
    
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
        
        # Episode metrics
        self.episode_return = 0
        self.episode_num_frames = 0
        self.done_counter = 0
        
        # History
        self.returns = []
        self.num_frames_list = []
        
        # Location tracking
        # NOTE: specific for Minigrid
        self.loc_visits = np.zeros([env.width, env.height])
        self.loc_history = [np.zeros(np.sum(loc_mask))] * 5
    
    def update_step(self, reward: float):
        """Update metrics for a single step."""
        self.episode_return += reward
        self.episode_num_frames += 1
    
    def update_location_visit(self, loc: Tuple[int, int]):
        """Track location visit."""
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
        preprocess_obss=None
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
        assert (self.config.num_frames % self.spatial_config.predictive_net.seqdur == 0), \
            "num_frames must be divisible by pRNN sequence duration"
    
    def _setup_environment(self):
        """Setup environment-related attributes."""
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
                num_frames=self.config.num_frames
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
            device=self.device
        )
    
    def _setup_metrics(self):
        """Initialize metrics tracker."""
        self.metrics = MetricsTracker(self.env, self.loc_mask)
    
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
        det_action = action.cpu().numpy()
        
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
        
        # Execute action
        new_obs, reward, terminated, truncated, _ = self.env.step(det_action)
        
        # Handle exploration mode
        if self.reward_config.exploration:
            reward, terminated, truncated = 0, False, False
        
        done = terminated or truncated
        done = done or (self.reward_config.exploration and 
                (idx + 1) % self.spatial_config.predictive_net.seqdur == 0)
        
        new_loc = self._get_agent_pos()
        
        # Compute spatial representation
        SR_new = self.SR_strategy.compute_SR(
            action=det_action,
            past_obs=self.obs,
            new_obs=new_obs
            )

        # reward_new, reward_past, done = self.SR_strategy.check_goal(SR_new, SR_ref)
        # reward += reward_new
        
        # Create step data
        step_data = StepData(
            obs=self.obs,
            action=action,
            reward=reward,
            value=value,
            SR=self.SR,
            dist=dist,
            log_prob=dist.log_prob(action),
            loc=self.loc,
            mask=self.mask,
            # reward_past=reward_past
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
                obs=self.obs
            )
            self.internal_strategy.update_reference(SR)
        
        # Record episode metrics
        self.metrics.episode_done()
        self.experience_buffer.add_traj_end(idx, self.obs, det_action)
        
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
                    obs=self.obs
                )
                SRs_all = torch.cat((self.experience_buffer.SRs[1:], SR_last), dim=0)
            else:
                SRs_all = self.experience_buffer.SRs
            internal_rewards = self.internal_strategy.compute_rewards(SRs=SRs_all)
            
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
        
        # Collect experiences
        for i in range(self.config.num_frames):
            # Collect single step
            step_data, done = self._collect_single_step(i)
            
            # Store in buffer
            self.experience_buffer.store_step(i, step_data)
            
            # Update metrics
            self.metrics.update_step(step_data.reward)
            self.metrics.update_location_visit(step_data.loc)
            
            # Handle episode end
            if done:
                self._handle_episode_end(i)
        
        # Compute augmented rewards
        self._compute_augmented_rewards()
        
        # Compute advantages
        preprocessed_obs = self.preprocess_obss([self.obs], device=self.device)
        with torch.no_grad():
            _, next_value = self.acmodel(preprocessed_obs, SR=self.SR)
        
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
        joint_probs = self._compute_joint_probabilities()
        
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
            "joint_dist": joint_probs
        }
        
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
        ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
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
        
        # Total loss
        loss = (policy_loss - 
                self.config.entropy_coef * policy_entropy + 
                self.config.value_loss_coef * value_loss)
        
        # Metrics
        metrics = {
            'entropy': policy_entropy.item() / torch.log(torch.tensor(2.0)),  # nats to bits
            'value': value.mean().item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
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
        
        # Add scalar metrics from collect_experiences
        for key in ["num_episodes", "loc_entropy", "loc_entropy_5", "num_frames"]:
            if key in logs_collect:
                processed[key] = logs_collect[key]
        
        # Add metrics from update_parameters
        for key in ["entropy", "policy_loss", "value_loss", "grad_norm"]:
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

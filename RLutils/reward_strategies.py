"""
Reward augmentation strategies for RL algorithms.

This module provides various reward augmentation strategies that can be used
to enhance learning through internal rewards and curiosity-driven exploration.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import torch
import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


# ============================================================================
# Reward Strategy Pattern
# ============================================================================

class RewardStrategy(ABC):
    """Abstract base class for reward augmentation strategies."""
    
    @abstractmethod
    def compute_rewards(self, **kwargs) -> torch.Tensor:
        """Compute augmented rewards."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset strategy state."""
        pass


# ============================================================================
# Goal Selection Strategies
# ============================================================================

class GoalSelectionStrategy(ABC):
    """
    Abstract base class for selecting a goal SR from a pool.

    Subclass this to implement alternative goal-selection schemes
    (e.g. curriculum, distance-based, or adversarial selection).
    """

    @abstractmethod
    def select_goal(self, goal_pool: torch.Tensor) -> torch.Tensor:
        """
        Select a single goal SR from the pool.

        Args:
            goal_pool: Tensor of candidate goal SRs [n_goals, SR_dim].

        Returns:
            Selected goal SR [1, SR_dim].
        """
        pass


class RandomGoalStrategy(GoalSelectionStrategy):
    """Select a goal uniformly at random from the pool."""

    def select_goal(self, goal_pool: torch.Tensor) -> torch.Tensor:
        idx = torch.randint(goal_pool.shape[1], (1,)).item()
        return goal_pool[:, idx], idx


class InternalRewardStrategy(RewardStrategy):
    """Compute internal rewards based on spatial representation similarity."""
    
    def __init__(
            self,
            k_int: float,
            SR_size: int,
            device: torch.device,
            num_frames: int
        ):
        self.k_int = k_int
        self.device = device
        self.ref = torch.zeros((1, SR_size), device=device)
        self.nrefs = 0
        self.rewards = torch.zeros(num_frames, device=device)
    
    def compute_rewards(self, SRs: torch.Tensor) -> torch.Tensor:
        """
        Compute internal rewards based on cosine distance to reference SR.
        
        Args:
            SRs: Spatial representations [num_frames, SR_dim]
            pastSR: Whether using past SR mode
        
        Returns:
            Internal rewards [num_frames]
        """
        if not any(self.ref[0]):
            return self.rewards
        
        SRs_cpu = SRs.cpu()
        ref_cpu = self.ref.squeeze().cpu()
        
        # Compute errors for all timesteps
        errors = torch.tensor(
            [cosine(SR, ref_cpu) for SR in SRs_cpu], 
            device=self.device
        )
        errors = torch.cat((errors[0][None], errors), dim=0)
        
        # Internal reward is decrease in error
        internal_rewards = errors[:-1] - errors[1:]
        return self.k_int * internal_rewards
    
    def update_reference(self, SR: torch.Tensor):
        """Update reference SR using running average."""
        self.nrefs += 1
        self.ref = self.ref + (SR - self.ref) / self.nrefs
    
    def set_reference(self, SR: torch.Tensor):
        """Set reference SR directly."""
        self.nrefs = 1
        self.ref = SR

    def check_goal(
        self,
        SR_new: torch.Tensor,
        goal_threshold: float,
        past_SR: bool,
    ) -> Tuple[float, float, bool]:
        """
        Check whether *SR_new* is within *goal_threshold* cosine distance of
        the current reference (goal) SR.

        Args:
            SR_new:          New spatial representation  [1, SR_dim]  (or [SR_dim]).
            goal_threshold:  Cosine-distance threshold; smaller == more similar.
            past_SR:         Reward routing flag.
                             True  -> reward goes to the *previous* step  (reward_past=1).
                             False -> reward goes to the *current*  step  (reward_new=1).

        Returns:
            reward_new:   Scalar reward to add to the *current* step.
            reward_past:  Scalar reward to add to the *previous* step.
            goal_reached: Whether the goal was reached.
        """
        if not any(self.ref[0]):
            return 0.0, 0.0, False

        dist = cosine(
            SR_new.detach().cpu().squeeze().numpy(),
            self.ref.detach().cpu().squeeze().numpy(),
        )
        goal_reached = bool(dist < goal_threshold)

        if goal_reached:
            if past_SR:
                return 0.0, 1.0, True
            else:
                return 1.0, 0.0, True
        return 0.0, 0.0, False

    def reset(self):
        """Reset strategy state."""
        self.ref = torch.zeros_like(self.ref)
        self.nrefs = 0
        self.rewards = torch.zeros_like(self.rewards)


class CuriousRewardStrategy(RewardStrategy):
    """Compute curious rewards based on prediction error."""
    
    def __init__(self, predictive_net, k_curious: float,
                 device: torch.device):
        self.pN = predictive_net
        self.k_curious = k_curious
        self.device = device
    
    def compute_rewards(self, obss: List[Dict], actions: np.ndarray, num_frames: int,
                        done_indices: List[int], last_observations: List[Dict],
                        last_actions: List[np.ndarray]) -> torch.Tensor:
        """
        Compute curious rewards based on prediction error.
        
        Args:
            obss: List of observations
            actions: Actions taken [num_frames]
        
        Returns:
            Curious rewards [num_frames]
        """
        with torch.no_grad():
            MSEs = torch.zeros(num_frames, device=self.device)

            for idx in range(1, len(done_indices)):
                start_episode, end_episode = done_indices[idx-1], done_indices[idx]
                last_obs = last_observations[idx-1]
                last_act = last_actions[idx-1]
                acts_now = actions[start_episode:end_episode]
                obs_now = obss[start_episode:end_episode] + [last_obs]
                obs_formatted, act_formatted = self.pN.env_shell.env2pred(obs_now, acts_now)
                obs_formatted, act_formatted = obs_formatted.to(self.device), act_formatted.to(self.device)
                obs_pred, obs_next, _ = self.pN.predict(obs_formatted, act_formatted)
                obs_pred, obs_next = obs_pred.squeeze(0), obs_next.squeeze(0)
                MSEs[start_episode:end_episode] = ((obs_pred - obs_next) ** 2).mean(dim=1)
            
        return self.k_curious * MSEs
    
    def reset(self):
        """Reset strategy state."""
        # Curious strategy is stateless, but include for consistency
        pass


class NextCuriousRewardStrategy(CuriousRewardStrategy):
    """Compute curious rewards based on prediction error from next state."""

    def compute_rewards(self, obss: List[Dict], actions: np.ndarray, num_frames: int,
                        done_indices: List[int], last_observations: List[Dict],
                        last_actions: List[np.ndarray]) -> torch.Tensor:
        """
        Compute curious rewards based on prediction error.
        
        Args:
            obss: List of observations
            actions: Actions taken [num_frames]
        
        Returns:
            Curious rewards [num_frames]
        """
        with torch.no_grad():
            MSEs = torch.zeros(num_frames, device=self.device)

            for idx in range(1, len(done_indices)):
                start_episode, end_episode = done_indices[idx-1], done_indices[idx]
                last_obs = last_observations[idx-1]
                last_act = last_actions[idx-1]
                acts_now = np.concatenate([actions[start_episode:end_episode], last_act])
                # Adding two last_obs is a hack, because prednet expects one more obs than acts,
                # it shouldn't affect same-step prediction
                obs_now = obss[start_episode:end_episode] + [last_obs] + [last_obs]
                obs_formatted, act_formatted = self.pN.env_shell.env2pred(obs_now, acts_now)
                obs_formatted, act_formatted = obs_formatted.to(self.device), act_formatted.to(self.device)
                obs_pred, obs_next, _ = self.pN.predict(obs_formatted, act_formatted)
                # Remove the first prediction because we get reward for the first action, which corresponds to the second observation
                obs_pred, obs_next = obs_pred.squeeze(0)[1:], obs_next.squeeze(0)[1:]
                MSEs[start_episode:end_episode] = ((obs_pred - obs_next) ** 2).mean(dim=1)
            
        return self.k_curious * MSEs

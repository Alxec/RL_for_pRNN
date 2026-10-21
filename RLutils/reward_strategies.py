"""
Reward augmentation strategies for RL algorithms.

This module provides various reward augmentation strategies that can be used
to enhance learning through internal rewards and curiosity-driven exploration.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from scipy.spatial.distance import cosine

from prnn.utils.Architectures import pRNN_th

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


class InternalRewardStrategy(RewardStrategy):
    """Compute internal rewards based on spatial representation similarity."""
    
    def __init__(
            self,
            k_int: float,
            SR_size: int,
            device: torch.device,
            num_frames: int,
            mask_internal: bool = False,
            mask_indices: List[int] = None,
        ):
        self.k_int = k_int
        self.device = device
        self.ref = torch.zeros((1, SR_size), device=device)
        self.nrefs = 0
        self.rewards = torch.zeros(num_frames, device=device)
        self.mask_internal = mask_internal
        self.mask_indices = mask_indices
        assert not (mask_internal and mask_indices is None), "Must provide mask_indices if mask_internal is True"
    
    def compute_rewards(
            self,
            SRs: torch.Tensor,
            next_SRs: torch.Tensor,
            references: torch.Tensor | None = None,
        ) -> torch.Tensor:
        """
        Compute internal rewards based on cosine distance to a reference SR.

        Reward ``t`` is ``E(SRs[t]) - E(next_SRs[t])`` with ``E`` the cosine
        distance to the reference.  Pairs never cross an episode boundary, so
        no transition to a reset state is rewarded.

        Args:
            SRs: Spatial representations at the start of each action
                [num_frames, SR_dim].
            next_SRs: Representations reached by those actions, same shape.
            references: Optional per-action references, same shape, for
                goals that change between episodes; otherwise the stored
                reference applies to every action.  An all-zero reference
                (no goal yet, or no goal in the episode) gives a zero reward.

        Returns:
            One reward per action [num_frames].
        """
        if references is None:
            references = self.ref.reshape(1, -1).expand_as(SRs)
        if not (SRs.shape == next_SRs.shape == references.shape):
            raise ValueError(
                "SRs, next_SRs and references must have one shape; received "
                f"{tuple(SRs.shape)}, {tuple(next_SRs.shape)} and {tuple(references.shape)}."
            )
        SRs_cpu, next_cpu, refs_cpu = (
            x.detach().cpu().clone() for x in (SRs, next_SRs, references)
        )
        has_reference = refs_cpu.abs().sum(dim=1) > 0
        if self.mask_internal:
            for x in (SRs_cpu, next_cpu, refs_cpu):
                x[:, self.mask_indices] = 0
        rewards = torch.zeros(SRs.shape[0])
        for t in torch.nonzero(has_reference).flatten().tolist():
            rewards[t] = cosine(SRs_cpu[t], refs_cpu[t]) - cosine(next_cpu[t], refs_cpu[t])
        return self.k_int * rewards.to(self.device)

    def update_reference(self, SR: torch.Tensor):
        """Update reference SR using running average."""
        self.nrefs += 1
        self.ref = self.ref + (SR - self.ref) / self.nrefs
    
    def set_reference(self, SR: torch.Tensor):
        """Set reference SR directly."""
        self.nrefs = 1
        self.ref = SR

    def reset(self):
        """Reset strategy state."""
        self.ref = torch.zeros_like(self.ref)
        self.nrefs = 0
        self.rewards = torch.zeros_like(self.rewards)


def prnn_inputs(predictive_net, obs: List, act: np.ndarray, state: Optional[Dict] = None):
    """Convert one stored trajectory into pRNN ``(obs, act)`` tensors.

    MiniGrid Shells read HD from the observations.  Miniworld Shells need the
    trajectory's continuous HDs, one per observation, as ``state``.
    """
    if state is None:
        return predictive_net.env_shell.env2pred(obs, act)
    return predictive_net.env_shell.env2pred(obs, act, state=state)


def _append_action(actions: np.ndarray, action) -> np.ndarray:
    """Append one scalar (MiniGrid) or vector (Miniworld) action."""
    action = np.asarray(action, dtype=actions.dtype).reshape((1,) + actions.shape[1:])
    return np.concatenate([actions, action])


def is_rollout_prnn(pRNN) -> bool:
    """Rollout pRNNs predict ``k`` further steps from every timestep."""
    return isinstance(pRNN, pRNN_th)


def min_prnn_trajectory_actions(predictive_net) -> int:
    """Fewest actions a trajectory needs for one pRNN prediction.

    A rollout pRNN predicts ``k`` steps ahead and so needs more than ``k``
    actions; other pRNNs predict from trajectories of any length.
    """
    pRNN = predictive_net.pRNN
    return pRNN.k + 1 if is_rollout_prnn(pRNN) else 1


def prediction_target_indices(pRNN, rows: int, cols: int) -> np.ndarray:
    """Observation index targeted by each output of ``PredictiveNet.predict``.

    Predictions are laid out as (rows, cols).  Ordinary pRNNs have one row,
    and output ``c`` targets ``o_(c + predOffset)``.  Rollout pRNNs have one
    row per rollout step (see ``pRNN_th.restructure_inputs``): row 0 at
    column ``c`` is predicted with ``o_c`` as input (stream A) and row
    ``r >= 1`` is step ``r`` of the rollout started at ``c``, predicted from
    actions alone (stream B); both target ``o_(c + r + predOffset)``.
    """
    return np.add.outer(np.arange(rows), np.arange(cols)) + pRNN.predOffset


def prediction_action_indices(pRNN, rows: int, cols: int) -> np.ndarray:
    """Index of the action input at the pRNN step that made each prediction.

    Same layout as :func:`prediction_target_indices`: the step producing
    prediction ``(r, c)`` receives ``a_(c + r - actOffset)``.
    """
    return np.add.outer(np.arange(rows), np.arange(cols)) - pRNN.actOffset


def taken_action_predictions(pRNN, rows: int, cols: int) -> np.ndarray:
    """Which predictions follow the actions that were actually taken.

    Only those predictions have a ground-truth outcome to be surprised by.
    Rollouts of ``rollout_action="full"`` replay the taken actions, so all of
    them do.  Rollouts imagining other action sequences would need a mask of
    the rollout steps that still match the taken sequence.
    """
    if is_rollout_prnn(pRNN) and pRNN.actionTheta is not True:
        raise NotImplementedError(
            "Curious rewards need rollouts that replay the taken actions "
            "(rollout_action='full'); this pRNN imagines other action sequences."
        )
    return np.ones((rows, cols), dtype=bool)


class CuriousRewardStrategy(RewardStrategy):
    """Reward each action with the pRNN's prediction errors.

    ``timing`` selects which predictions belong to ``a_t``:

    - ``causal``: those whose target is ``o_(t+1)``, the observation ``a_t``
      leads to;
    - ``actor_aligned``: those made at the pRNN steps that received ``a_t``
      as input, i.e. the steps producing the SR state of ``a_t``.

    They differ only for pRNNs that pair ``a_t`` with ``o_t`` and target
    ``o_t`` (``actOffset=0``, ``predOffset=0``: past-SR RL).  Action-offset
    pRNNs (current-SR RL) and ``NextStep`` give the same alignment in both.
    A rollout pRNN makes up to ``k + 1`` predictions per action: one with the
    observation as input (stream A) and those of the ``k`` preceding
    timesteps' rollouts (stream B).  ``predictions="rollout"`` averages
    stream B only.

    ``center`` subtracts, and ``normalize_std`` divides by, the mean and
    standard deviation of the previous rollout's raw errors (the first rollout
    uses its own), before ``k_curious`` scales the reward.  Statistics cover
    only actions that received a prediction error.
    """

    PREDICTIONS = ("all", "rollout")
    TIMINGS = ("actor_aligned", "causal")

    def __init__(self, predictive_net, k_curious: float,
                 device: torch.device, predictions: str = "all",
                 timing: str = "actor_aligned", center: bool = False,
                 normalize_std: bool = False):
        if timing not in self.TIMINGS:
            raise ValueError(
                f"rewards.curious_reward_timing must be one of {self.TIMINGS}, got {timing!r}."
            )
        if predictions not in self.PREDICTIONS:
            raise ValueError(
                f"rewards.curious_predictions must be one of {self.PREDICTIONS}, "
                f"got {predictions!r}."
            )
        if predictions == "rollout" and not is_rollout_prnn(predictive_net.pRNN):
            raise ValueError(
                "rewards.curious_predictions=rollout needs a rollout pRNN."
            )
        self.pN = predictive_net
        self.k_curious = k_curious
        self.device = device
        self.predictions = predictions
        self.timing = timing
        self.center = bool(center)
        self.normalize_std = bool(normalize_std)
        self.previous_stats = None  # (mean, std) of the last rollout's raw errors
        self.raw_rewards = None  # this rollout's unscaled errors, for logging

    def compute_rewards(self, obss: List[Dict], actions: np.ndarray, num_frames: int,
                        done_indices: List[int], last_observations: List[Dict],
                        last_actions: List[np.ndarray],
                        episode_hds: Optional[List[np.ndarray]] = None) -> torch.Tensor:
        """
        Compute curious rewards for every trajectory of a rollout.

        Args:
            obss: Observations, one per action
            actions: Actions taken [num_frames]
            done_indices: Trajectory boundaries
            last_observations: Observation after each trajectory's last action
            last_actions: Action selected at that observation (not taken)
            episode_hds: Per-trajectory HDs, one per observation (Miniworld only)

        Returns:
            Curious rewards [num_frames]
        """
        errors = torch.zeros(num_frames, device=self.device)
        rewarded = torch.zeros(num_frames, dtype=torch.bool, device=self.device)
        min_actions = min_prnn_trajectory_actions(self.pN)
        with torch.no_grad():
            for idx in range(1, len(done_indices)):
                start, end = done_indices[idx-1], done_indices[idx]
                # Extend by the selected next action and a placeholder
                # observation, so that pRNNs targeting o_t at step t
                # (predOffset=0) also predict the final outcome o_T.  Neither
                # is rewarded; the placeholder is never a used target.
                acts = _append_action(actions[start:end], last_actions[idx-1])
                obs = obss[start:end] + [last_observations[idx-1]] * 2
                if len(acts) < min_actions:
                    continue  # too short for a rollout pRNN; no reward
                state = None
                if episode_hds is not None:
                    hds = episode_hds[idx-1]
                    state = {"agent_dir": np.append(hds, hds[-1])}
                obs_pN, act_pN = prnn_inputs(self.pN, obs, acts, state)
                obs_pred, obs_target, _ = self.pN.predict(
                    obs_pN.to(self.device), act_pN.to(self.device)
                )
                errors[start:end], rewarded[start:end] = self._outcome_errors(
                    obs_pred, obs_target, end - start
                )

        self.raw_rewards = errors
        return self.k_curious * self._scale(errors, rewarded)

    def _scale(self, errors: torch.Tensor, rewarded: torch.Tensor) -> torch.Tensor:
        """Center and/or normalize with the previous rollout's statistics."""
        if not rewarded.any():
            return errors
        current = (errors[rewarded].mean(), errors[rewarded].std(unbiased=False))
        mean, std = self.previous_stats if self.previous_stats is not None else current
        self.previous_stats = current
        scaled = errors
        if self.center:
            scaled = scaled - mean
        if self.normalize_std:
            scaled = scaled / std.clamp(min=1e-8)
        return torch.where(rewarded, scaled, torch.zeros_like(scaled))

    def _outcome_errors(self, obs_pred: torch.Tensor, obs_target: torch.Tensor,
                        num_actions: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean error of the predictions of each action, and which actions have any."""
        errors = ((obs_pred - obs_target) ** 2).flatten(start_dim=2).mean(dim=2)
        rows, cols = errors.shape
        pRNN = self.pN.pRNN
        if self.timing == "causal":
            action = prediction_target_indices(pRNN, rows, cols) - 1  # a_t leads to o_(t+1)
        else:
            action = prediction_action_indices(pRNN, rows, cols)
        used = taken_action_predictions(pRNN, rows, cols)
        if self.predictions == "rollout":
            used[0] = False
        used &= (action >= 0) & (action < num_actions)

        action = torch.as_tensor(action[used], device=errors.device)
        errors = errors[torch.as_tensor(used, device=errors.device)]
        totals = torch.zeros(num_actions, device=errors.device).index_add_(0, action, errors)
        counts = torch.zeros(num_actions, device=errors.device).index_add_(
            0, action, torch.ones_like(errors)
        )
        return totals / counts.clamp(min=1), counts > 0

    def reset(self):
        """Reset strategy state."""
        # Curious strategy is stateless, but include for consistency
        pass

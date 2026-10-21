"""Goal pools and goal-reaching thresholds for goal-conditioned PPO.

A goal is a state given to the actor, optionally with the grid location it
represents.  For pRNN goal-conditioned PPO the state is a pRNN state (also the
internal-reward reference), from one of four pools:

- ``sampled``: states recorded at single visits before training (the
  original pool);
- ``location_mean``: the same recording averaged per location;
- ``location_ema``: a running average per location, updated online from the
  states the agent visits, so it follows a pRNN that trains during RL;
- ``spontaneous``: the end state of noise-driven spontaneous pRNN activity
  with adaptation, started from the agent's current waking state.  Longer
  runs drift further, so the number of steps sets the goal distance.

The visual location-goal baseline uses :class:`LocationGoalPool`, whose state
is the location itself, encoded as ``onehot(x) || onehot(y)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

GOAL_POOL_TYPES = ("sampled", "location_mean", "location_ema", "spontaneous")


@dataclass
class Goal:
    """A goal state for the actor and, when known, the location it stands for."""

    state: torch.Tensor
    location: Optional[np.ndarray] = None


def location_key(location) -> tuple[int, int]:
    """Hashable integer grid location."""
    return tuple(int(value) for value in np.asarray(location).reshape(-1)[:2])


def excluded_locations(exclude: Optional[np.ndarray]) -> frozenset:
    """Convert a 2xN ``[x; y]`` array of excluded goal locations to a set."""
    if exclude is None:
        return frozenset()
    exclude = np.asarray(exclude)
    if exclude.ndim != 2 or exclude.shape[0] != 2:
        raise ValueError("Excluded goal locations must be a 2xN numpy array with columns [x, y].")
    return frozenset(location_key(exclude[:, i]) for i in range(exclude.shape[1]))


class GoalPool(ABC):
    """Source of goals for goal-conditioned PPO."""

    #: Whether goals carry a location (needed for location-based goal checks).
    has_locations = True

    @abstractmethod
    def sample(
        self,
        excluded: frozenset = frozenset(),
        waking_state: Optional[Callable[[], torch.Tensor]] = None,
    ) -> Optional[Goal]:
        """Draw a goal, or ``None`` when no goal is available yet."""

    def observe(self, state: torch.Tensor, location) -> None:
        """Record a visited state and its location (online pools only)."""

    def stats(self) -> Dict[str, float]:
        """Scalars worth logging."""
        return {}


class FixedGoalPool(GoalPool):
    """A fixed set of goals, one per entry of ``locations``, sampled uniformly."""

    def __init__(self, locations: np.ndarray):
        self.locations = np.asarray(locations)
        self._allowed: Dict[frozenset, np.ndarray] = {}

    @abstractmethod
    def goal_state(self, idx: int) -> torch.Tensor:
        """The actor's goal input for goal ``idx``."""

    def allowed(self, excluded: frozenset) -> np.ndarray:
        """Indices of goals outside ``excluded``; at least one must remain."""
        if excluded not in self._allowed:
            keep = np.array([location_key(loc) not in excluded for loc in self.locations])
            if not keep.any():
                raise ValueError(
                    "All goal locations were excluded. Provide a less restrictive exclusion."
                )
            self._allowed[excluded] = np.flatnonzero(keep)
        return self._allowed[excluded]

    def sample(self, excluded=frozenset(), waking_state=None) -> Goal:
        candidates = self.allowed(excluded)
        idx = candidates[torch.randint(len(candidates), (1,)).item()]
        return Goal(self.goal_state(idx), self.locations[idx])

    def stats(self) -> Dict[str, float]:
        return {"goal_pool_size": float(len(self.locations))}


class StateGoalPool(FixedGoalPool):
    """Fixed pRNN states with their locations.

    ``states`` has shape ``(rows, n_goals, units)``; ``rows`` is the pRNN
    activity's first dimension and is kept so that a goal has the same shape
    as before.
    """

    def __init__(self, states: torch.Tensor, locations: np.ndarray):
        if states.ndim != 3 or states.shape[1] != len(locations):
            raise ValueError(
                "Goal states must have shape (rows, n_goals, units) with one location per "
                f"goal; received {tuple(states.shape)} for {len(locations)} locations."
            )
        super().__init__(locations)
        self.states = states

    @classmethod
    def from_recording(cls, activity: torch.Tensor, locations: np.ndarray, average: bool = False):
        """Goals from recorded activity: every visit, or one mean per location."""
        locations = np.asarray(locations)
        if not average:
            return cls(activity, locations)
        keys = [location_key(location) for location in locations]
        unique = sorted(set(keys))
        index = {key: i for i, key in enumerate(unique)}
        groups = torch.as_tensor([index[key] for key in keys], device=activity.device)
        totals = torch.zeros(
            (activity.shape[0], len(unique), activity.shape[2]),
            dtype=activity.dtype, device=activity.device,
        ).index_add_(1, groups, activity)
        counts = torch.bincount(groups, minlength=len(unique)).to(activity.dtype)
        return cls(totals / counts[None, :, None], np.asarray(unique))

    def to(self, device) -> "StateGoalPool":
        self.states = self.states.to(device)
        return self

    def goal_state(self, idx: int) -> torch.Tensor:
        return self.states[:, idx]


def encode_location(location, width: int, height: int, device=None) -> torch.Tensor:
    """A batch-one ``onehot(x) || onehot(y)`` encoding of a grid location.

    It keeps the factorisation of a grid location while making each
    coordinate explicit to an otherwise visual actor-critic.
    """
    x, y = location_key(location)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Goal {(x, y)} is outside the {width}x{height} grid.")
    goal = torch.zeros((1, width + height), dtype=torch.float32, device=device)
    goal[0, x] = 1.0
    goal[0, width + y] = 1.0
    return goal


class LocationGoalPool(FixedGoalPool):
    """Integer grid locations given to the actor as ``onehot(x) || onehot(y)``.

    The goal input of the visual location-goal baseline, which has no pRNN.
    """

    def __init__(self, locations: np.ndarray, width: int, height: int, device=None):
        locations = np.asarray(locations)
        if locations.ndim != 2 or locations.shape[1] != 2 or len(locations) == 0:
            raise ValueError(
                "Location goal pool must have shape (num_goals, 2) with at least one goal."
            )
        rounded = np.rint(locations).astype(int)
        if not np.allclose(locations, rounded):
            raise ValueError("Location goal pools must contain integer grid coordinates.")
        in_bounds = (
            (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
        )
        if not in_bounds.all():
            raise ValueError("Location goal pool contains coordinates outside the environment grid.")
        super().__init__(rounded)
        self.width, self.height = int(width), int(height)
        self.device = device

    @property
    def goal_size(self) -> int:
        return self.width + self.height

    def goal_state(self, idx: int) -> torch.Tensor:
        return encode_location(self.locations[idx], self.width, self.height, self.device)


class LocationEMAGoalPool(GoalPool):
    """Per-location running average of visited pRNN states.

    The first ``1 / rate`` visits are averaged equally; later visits update an
    exponential moving average with weight ``rate``, so a reference follows a
    changing pRNN over roughly its last ``1 / rate`` visits.  Only locations
    visited at least ``min_visits`` times are offered as goals.
    """

    def __init__(self, rate: float, min_visits: int = 1):
        if not 0 < rate <= 1:
            raise ValueError("The goal-pool EMA rate must be in (0, 1].")
        self.rate = float(rate)
        self.min_visits = int(min_visits)
        self.references: Dict[tuple, torch.Tensor] = {}
        self.visits: Dict[tuple, int] = {}

    def observe(self, state: torch.Tensor, location) -> None:
        key = location_key(location)
        visits = self.visits.get(key, 0) + 1
        self.visits[key] = visits
        state = state.detach()
        if key not in self.references:
            self.references[key] = state.clone()
        else:
            reference = self.references[key]
            reference += max(self.rate, 1 / visits) * (state - reference)

    def eligible(self, excluded: frozenset = frozenset()) -> list:
        return [
            key for key, visits in self.visits.items()
            if visits >= self.min_visits and key not in excluded
        ]

    def sample(self, excluded=frozenset(), waking_state=None) -> Optional[Goal]:
        keys = self.eligible(excluded)
        if not keys:
            return None
        key = keys[torch.randint(len(keys), (1,)).item()]
        return Goal(self.references[key].clone(), np.asarray(key))

    def stats(self) -> Dict[str, float]:
        return {"goal_pool_size": float(len(self.eligible()))}


class SpontaneousGoalPool(GoalPool):
    """Goals from noise-driven spontaneous pRNN activity with adaptation.

    Starting from the agent's waking state, the recurrent cell runs without
    input for a random number of steps in ``[min_steps, max_steps]``, driven by
    Gaussian noise.  An adaptation current, as in pRNN's adapting cells, is
    subtracted from the cell's noise input::

        a_(t+1) = a_t * (1 - 1 / tau) + (gain / tau) * h_t

    where ``h_t`` is the state a step starts from, exactly as in pRNN's
    ``AdaptingRNNCell``: recently active units are suppressed and activity
    keeps moving instead of settling.  It is applied at sampling time only, so
    any trained pRNN can be used.  The final hidden state is the goal.

    The waking state is passed in by the caller, so a later variant can start
    from, or be driven by, other inputs (e.g. imagined action sequences).
    """

    has_locations = False

    def __init__(
        self,
        predictive_net,
        min_steps: int,
        max_steps: int,
        noise_mean: float = 0.0,
        noise_std: float = 0.03,
        adaptation_gain: float = 0.3,
        adaptation_tau: float = 8.0,
    ):
        if not 1 <= int(min_steps) <= int(max_steps):
            raise ValueError("Spontaneous goal steps need 1 <= min_steps <= max_steps.")
        if adaptation_tau <= 0:
            raise ValueError("adaptation_tau must be positive.")
        self.pN = predictive_net
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.noise_mean = float(noise_mean)
        self.noise_std = float(noise_std)
        self.adaptation_gain = float(adaptation_gain)
        self.adaptation_tau = float(adaptation_tau)
        self.last_steps = None

    def sample(self, excluded=frozenset(), waking_state=None) -> Goal:
        if excluded:
            raise ValueError("Spontaneous goals have no locations to exclude.")
        if waking_state is None:
            raise ValueError("Spontaneous goals start from the agent's waking state.")
        self.last_steps = int(torch.randint(self.min_steps, self.max_steps + 1, (1,)).item())
        return Goal(self.run(waking_state(), self.last_steps), None)

    def run(self, start: torch.Tensor, steps: int) -> torch.Tensor:
        """Spontaneous activity for ``steps`` steps from ``start`` [1, units]."""
        cell = self.pN.pRNN.rnn.cell
        device = cell.weight_hh.device
        hidden = start.reshape(1, -1).to(device)
        if hidden.shape[1] != cell.hidden_size:
            raise ValueError(
                f"The waking state has {hidden.shape[1]} units but the pRNN cell has "
                f"{cell.hidden_size}."
            )
        silence = torch.zeros((1, cell.input_size), device=device)
        adaptation = torch.zeros_like(hidden)
        # thetaRNNLayer passes (h, a) with zero cell adaptation at the start
        state = (hidden, 0)
        with torch.no_grad():
            for _ in range(steps):
                noise = self.noise_mean + self.noise_std * torch.randn_like(hidden)
                new_hidden, state = cell(silence, noise - adaptation, state)
                adaptation = (
                    adaptation * (1 - 1 / self.adaptation_tau)
                    + (self.adaptation_gain / self.adaptation_tau) * hidden
                )
                hidden = new_hidden
        return hidden[:, : self.pN.hidden_size]

    def stats(self) -> Dict[str, float]:
        return {} if self.last_steps is None else {"goal_spontaneous_steps": float(self.last_steps)}


class AdaptiveGoalThreshold:
    """Online estimate of a quantile of goal distances.

    ``value <- value + rate * (quantile - 1[distance < value])`` converges to
    the ``quantile``-th quantile of the distances it is updated with, so the
    goal region rescales itself as the pRNN's representation changes.
    Goal-conditioned PPO updates it with the distance to the goal at every
    step of a goal episode.
    """

    def __init__(self, initial: float, quantile: float, rate: float):
        if not 0 < quantile < 1:
            raise ValueError("The goal-threshold quantile must be in (0, 1).")
        if rate <= 0:
            raise ValueError("The goal-threshold rate must be positive.")
        self.value = float(initial)
        self.quantile = float(quantile)
        self.rate = float(rate)

    def update(self, distance: float) -> None:
        self.value = max(0.0, self.value + self.rate * (self.quantile - float(distance < self.value)))


def make_goal_pool(
    pool_type: str,
    *,
    recording: Optional[Dict] = None,
    predictive_net=None,
    ema_rate: float = 0.2,
    min_visits: int = 1,
    spontaneous: Optional[Dict] = None,
) -> GoalPool:
    """Build one of :data:`GOAL_POOL_TYPES`.

    ``recording`` is the aligned ``(activity, locations)`` pair of a random-walk
    recording, needed by ``sampled`` and ``location_mean``.
    """
    if pool_type in ("sampled", "location_mean"):
        if recording is None:
            raise ValueError(f"The {pool_type!r} goal pool needs a recorded trajectory.")
        activity, locations = recording
        return StateGoalPool.from_recording(activity, locations, average=pool_type == "location_mean")
    if pool_type == "location_ema":
        return LocationEMAGoalPool(ema_rate, min_visits)
    if pool_type == "spontaneous":
        if predictive_net is None:
            raise ValueError("Spontaneous goals need a pRNN.")
        return SpontaneousGoalPool(predictive_net, **dict(spontaneous or {}))
    raise ValueError(f"Unknown goal pool {pool_type!r}; choose one of {GOAL_POOL_TYPES}.")

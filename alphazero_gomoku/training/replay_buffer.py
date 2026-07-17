"""Bounded, persistable AlphaZero self-play replay buffer."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ReplaySample:
    state: np.ndarray
    policy: np.ndarray
    outcome: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._samples: deque[ReplaySample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._samples)

    def append(self, sample: ReplaySample) -> None:
        state = np.ascontiguousarray(sample.state, dtype=np.float32)
        policy = np.ascontiguousarray(sample.policy, dtype=np.float32)
        if state.ndim != 3 or state.shape[0] != 4:
            raise ValueError("state must have shape (4, height, width)")
        if policy.ndim != 1 or policy.size != state.shape[1] * state.shape[2]:
            raise ValueError("policy size must match the board area")
        if not np.isfinite(state).all() or not np.isfinite(policy).all():
            raise ValueError("state and policy must contain finite values")
        if not np.isclose(policy.sum(), 1.0, atol=1e-5):
            raise ValueError("policy probabilities must sum to one")
        if sample.outcome not in (-1.0, 0.0, 1.0):
            raise ValueError("outcome must be -1, 0, or 1")
        self._samples.append(ReplaySample(state, policy, float(sample.outcome)))

    def extend(self, samples: Iterable[ReplaySample]) -> None:
        for sample in samples:
            self.append(sample)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self):
            raise ValueError("batch_size exceeds the number of stored samples")
        indices = rng.choice(len(self), size=batch_size, replace=False)
        selected = [self._samples[int(index)] for index in indices]
        return (
            np.stack([sample.state for sample in selected]),
            np.stack([sample.policy for sample in selected]),
            np.asarray([sample.outcome for sample in selected], dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not self._samples:
            raise ValueError("cannot persist an empty replay buffer")
        np.savez_compressed(
            destination,
            states=np.stack([sample.state for sample in self._samples]),
            policies=np.stack([sample.policy for sample in self._samples]),
            outcomes=np.asarray([sample.outcome for sample in self._samples], dtype=np.float32),
            capacity=np.asarray(self.capacity, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer:
        with np.load(path, allow_pickle=False) as data:
            buffer = cls(capacity=int(data["capacity"]))
            for state, policy, outcome in zip(
                data["states"], data["policies"], data["outcomes"], strict=True
            ):
                buffer.append(ReplaySample(state, policy, float(outcome)))
        return buffer

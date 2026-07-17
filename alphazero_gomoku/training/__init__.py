"""Reproducible training infrastructure for AlphaZero Gomoku."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import ExperimentConfig, load_experiment_config
from .metrics import JsonlMetricsWriter, read_metrics
from .replay_buffer import ReplayBuffer, ReplaySample
from .reproducibility import seed_everything

if TYPE_CHECKING:
    from .checkpoint import CheckpointManager, ResumeBundle, TrainingState

_CHECKPOINT_EXPORTS = {"CheckpointManager", "ResumeBundle", "TrainingState"}

__all__ = [
    "CheckpointManager",
    "ExperimentConfig",
    "JsonlMetricsWriter",
    "ReplayBuffer",
    "ReplaySample",
    "ResumeBundle",
    "TrainingState",
    "load_experiment_config",
    "read_metrics",
    "seed_everything",
]


def __getattr__(name: str) -> Any:
    """Load PyTorch-backed checkpoint helpers only when they are requested."""
    if name in _CHECKPOINT_EXPORTS:
        from . import checkpoint

        return getattr(checkpoint, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

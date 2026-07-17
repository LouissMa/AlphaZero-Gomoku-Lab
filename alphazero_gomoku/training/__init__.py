"""Reproducible training infrastructure for AlphaZero Gomoku."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .config import ExperimentConfig, load_experiment_config
from .metrics import JsonlMetricsWriter, read_metrics
from .replay_buffer import ReplayBuffer, ReplaySample
from .reproducibility import seed_everything

if TYPE_CHECKING:
    from .benchmark import BenchmarkReport
    from .checkpoint import CheckpointManager, ResumeBundle, TrainingState
    from .inference import BatchedPolicyEvaluator, InferenceStats
    from .self_play import SelfPlayBatch, SelfPlayProfile, SelfPlayResult
    from .trainer import AlphaZeroTrainer

_LAZY_EXPORTS = {
    "AlphaZeroTrainer": ".trainer",
    "BatchedPolicyEvaluator": ".inference",
    "BenchmarkReport": ".benchmark",
    "CheckpointManager": ".checkpoint",
    "InferenceStats": ".inference",
    "ResumeBundle": ".checkpoint",
    "SelfPlayBatch": ".self_play",
    "SelfPlayProfile": ".self_play",
    "SelfPlayResult": ".self_play",
    "TrainingState": ".checkpoint",
}

__all__ = [
    "AlphaZeroTrainer",
    "BatchedPolicyEvaluator",
    "BenchmarkReport",
    "CheckpointManager",
    "ExperimentConfig",
    "InferenceStats",
    "JsonlMetricsWriter",
    "ReplayBuffer",
    "ReplaySample",
    "ResumeBundle",
    "SelfPlayBatch",
    "SelfPlayProfile",
    "SelfPlayResult",
    "TrainingState",
    "load_experiment_config",
    "read_metrics",
    "seed_everything",
]


def __getattr__(name: str) -> Any:
    """Load PyTorch-backed training components only when requested."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        return getattr(import_module(module_name, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

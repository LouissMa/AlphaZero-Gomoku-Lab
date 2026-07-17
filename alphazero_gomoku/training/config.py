"""Typed TOML configuration for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class BoardConfig:
    width: int = 6
    height: int = 6
    n_in_row: int = 4

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.n_in_row) <= 0:
            raise ValueError("board dimensions and n_in_row must be positive")
        if self.n_in_row > min(self.width, self.height):
            raise ValueError("n_in_row cannot exceed the smaller board dimension")


@dataclass(frozen=True, slots=True)
class NetworkSettings:
    channels: int = 64
    residual_blocks: int = 4
    policy_channels: int = 2
    value_channels: int = 1
    value_hidden_size: int = 128
    device: str = "auto"
    amp: bool = True
    compile_model: bool = False
    compile_backend: str | None = None

    def __post_init__(self) -> None:
        numeric = (
            self.channels,
            self.residual_blocks,
            self.policy_channels,
            self.value_channels,
            self.value_hidden_size,
        )
        if any(value <= 0 for value in numeric):
            raise ValueError("network dimensions must be positive")


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    games_per_iteration: int = 1
    simulations_per_move: int = 400
    c_puct: float = 5.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.games_per_iteration <= 0 or self.simulations_per_move <= 0:
            raise ValueError("self-play counts must be positive")
        if self.c_puct <= 0 or self.temperature <= 0:
            raise ValueError("c_puct and temperature must be positive")


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs_per_iteration: int = 5
    replay_capacity: int = 10_000
    gradient_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if min(self.batch_size, self.epochs_per_iteration, self.replay_capacity) <= 0:
            raise ValueError("optimization counts must be positive")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str = "gomoku-baseline"
    seed: int = 2026
    deterministic: bool = False
    iterations: int = 1_500
    checkpoint_every: int = 50
    output_dir: str = "runs"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("run name cannot be empty")
        if self.seed < 0 or self.iterations <= 0 or self.checkpoint_every <= 0:
            raise ValueError("seed must be non-negative and run counts positive")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    board: BoardConfig = field(default_factory=BoardConfig)
    network: NetworkSettings = field(default_factory=NetworkSettings)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        known_sections = {"board", "network", "self_play", "optimization", "run"}
        unknown = set(values) - known_sections
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        return cls(
            board=BoardConfig(**values.get("board", {})),
            network=NetworkSettings(**values.get("network", {})),
            self_play=SelfPlayConfig(**values.get("self_play", {})),
            optimization=OptimizationConfig(**values.get("optimization", {})),
            run=RunConfig(**values.get("run", {})),
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment TOML file."""
    config_path = Path(path)
    with config_path.open("rb") as file:
        values = tomllib.load(file)
    return ExperimentConfig.from_dict(values)

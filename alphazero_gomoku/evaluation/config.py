"""Typed configuration for reproducible evaluation tournaments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from alphazero_gomoku.training.config import BoardConfig

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    board: BoardConfig = field(default_factory=BoardConfig)
    games_per_opponent: int = 20
    simulations_per_move: int = 100
    pure_mcts_playouts: int = 100
    c_puct: float = 5.0
    confidence: float = 0.95
    promotion_score: float = 0.55
    promotion_lower_bound: float = 0.50
    seed: int = 2026
    device: str = "auto"
    baselines: tuple[str, ...] = ("random", "heuristic", "pure_mcts")

    def __post_init__(self) -> None:
        if self.games_per_opponent <= 0 or self.games_per_opponent % 2:
            raise ValueError("games_per_opponent must be a positive even number")
        if self.simulations_per_move <= 0 or self.pure_mcts_playouts <= 0:
            raise ValueError("MCTS simulation counts must be positive")
        if self.c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if not 0 <= self.promotion_lower_bound <= self.promotion_score <= 1:
            raise ValueError("promotion bounds must satisfy 0 <= lower <= score <= 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        allowed = {"random", "heuristic", "pure_mcts"}
        unknown = set(self.baselines) - allowed
        if unknown:
            raise ValueError(f"unknown arena baselines: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_arena_config(path: str | Path) -> ArenaConfig:
    with Path(path).open("rb") as file:
        values = tomllib.load(file)
    unknown = set(values) - {"board", "arena"}
    if unknown:
        raise ValueError(f"unknown arena configuration sections: {sorted(unknown)}")
    arena = dict(values.get("arena", {}))
    if "baselines" in arena:
        arena["baselines"] = tuple(arena["baselines"])
    return ArenaConfig(
        board=BoardConfig(**values.get("board", {})),
        **arena,
    )

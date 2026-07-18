"""Equal-budget PUCT versus Gumbel AlphaZero comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from alphazero_gomoku.evaluation.arena import MatchReport, run_match
from alphazero_gomoku.evaluation.config import ArenaConfig
from alphazero_gomoku.evaluation.neural import resolve_model_path
from alphazero_gomoku.game import Board
from alphazero_gomoku.mcts_alphaZero import MCTSPlayer
from alphazero_gomoku.policy_value_net_pytorch import PolicyValueNet
from alphazero_gomoku.training.config import BoardConfig

from .player import GumbelMCTSPlayer

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

COMPARISON_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class SearchComparisonConfig:
    board: BoardConfig = field(default_factory=BoardConfig)
    games: int = 20
    simulations_per_move: int = 64
    max_considered_actions: int = 16
    c_puct: float = 5.0
    gumbel_scale: float = 1.0
    q_value_scale: float = 0.1
    q_visit_offset: float = 50.0
    confidence: float = 0.95
    seed: int = 2026
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.games <= 0 or self.games % 2:
            raise ValueError("games must be a positive even number")
        if self.simulations_per_move <= 0 or self.max_considered_actions <= 0:
            raise ValueError("search counts must be positive")
        if self.c_puct <= 0 or self.q_value_scale <= 0:
            raise ValueError("search scales must be positive")
        if self.gumbel_scale < 0 or self.q_visit_offset < 0:
            raise ValueError("Gumbel scale and Q visit offset must be non-negative")
        if not 0 < self.confidence < 1 or self.seed < 0:
            raise ValueError("confidence and seed are invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_search_comparison_config(path: str | Path) -> SearchComparisonConfig:
    with Path(path).open("rb") as file:
        values = tomllib.load(file)
    unknown = set(values) - {"board", "comparison"}
    if unknown:
        raise ValueError(f"unknown comparison configuration sections: {sorted(unknown)}")
    return SearchComparisonConfig(
        board=BoardConfig(**values.get("board", {})),
        **values.get("comparison", {}),
    )


@dataclass(frozen=True, slots=True)
class PuctFactory:
    network: PolicyValueNet
    config: SearchComparisonConfig
    name: str = "puct"

    def create(self, seed: int) -> MCTSPlayer:
        return MCTSPlayer(
            self.network.policy_value_fn,
            c_puct=self.config.c_puct,
            n_playout=self.config.simulations_per_move,
            is_selfplay=0,
            rng=np.random.default_rng(seed),
        )


@dataclass(frozen=True, slots=True)
class GumbelFactory:
    network: PolicyValueNet
    config: SearchComparisonConfig
    name: str = "gumbel"

    def create(self, seed: int) -> GumbelMCTSPlayer:
        return GumbelMCTSPlayer(
            self.network.policy_value_fn,
            simulations=self.config.simulations_per_move,
            max_considered_actions=self.config.max_considered_actions,
            gumbel_scale=self.config.gumbel_scale,
            q_value_scale=self.config.q_value_scale,
            q_visit_offset=self.config.q_visit_offset,
            is_selfplay=False,
            rng=np.random.default_rng(seed),
        )


@dataclass(frozen=True, slots=True)
class RootProfile:
    simulations_per_move: int
    elapsed_seconds: float
    selected_action: int
    policy_entropy: float


@dataclass(frozen=True, slots=True)
class SearchComparisonReport:
    format_version: int
    timestamp: str
    model: str
    config: dict[str, Any]
    match: MatchReport
    profiles: dict[str, RootProfile]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _profile_root(
    factory: PuctFactory | GumbelFactory, config: SearchComparisonConfig
) -> RootProfile:
    board = Board(
        width=config.board.width,
        height=config.board.height,
        n_in_row=config.board.n_in_row,
    )
    board.init_board()
    player = factory.create(config.seed)
    player.set_player_ind(board.players[0])
    started = perf_counter()
    action, policy = player.get_action(board, return_prob=True)
    elapsed = perf_counter() - started
    positive = policy[policy > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return RootProfile(
        simulations_per_move=config.simulations_per_move,
        elapsed_seconds=elapsed,
        selected_action=int(action),
        policy_entropy=entropy,
    )


def compare_search_algorithms(
    model_path: str | Path,
    config: SearchComparisonConfig,
) -> SearchComparisonReport:
    source = resolve_model_path(model_path)
    network = PolicyValueNet.from_checkpoint(
        source,
        device=config.device,
        load_optimizer=False,
        amp=False,
    )
    if (network.config.board_width, network.config.board_height) != (
        config.board.width,
        config.board.height,
    ):
        raise ValueError("checkpoint and comparison board dimensions do not match")
    gumbel = GumbelFactory(network, config)
    puct = PuctFactory(network, config)
    arena_config = ArenaConfig(
        board=config.board,
        games_per_opponent=config.games,
        simulations_per_move=config.simulations_per_move,
        pure_mcts_playouts=1,
        c_puct=config.c_puct,
        confidence=config.confidence,
        seed=config.seed,
        device=config.device,
        baselines=(),
    )
    match = run_match(gumbel, puct, arena_config, np.random.default_rng(config.seed))
    return SearchComparisonReport(
        format_version=COMPARISON_FORMAT_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=str(source),
        config=config.to_dict(),
        match=match,
        profiles={
            "gumbel": _profile_root(gumbel, config),
            "puct": _profile_root(puct, config),
        },
    )


def write_search_comparison_report(
    report: SearchComparisonReport,
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination

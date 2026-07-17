"""Parallel self-play data generation with centralized batched inference."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from alphazero_gomoku.game import Board, Game
from alphazero_gomoku.mcts_alphaZero import MCTSPlayer
from alphazero_gomoku.policy_value_net_pytorch import PolicyValueNet

from .config import BoardConfig, SelfPlayConfig
from .inference import BatchedPolicyEvaluator, InferenceStats
from .replay_buffer import ReplaySample

PolicyValueFunction = Callable[[Any], tuple[Iterable[tuple[int, float]], float]]


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    winner: int
    samples: tuple[ReplaySample, ...]
    tree_reuse_count: int
    tree_reset_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SelfPlayProfile:
    games: int
    positions: int
    elapsed_seconds: float
    simulations: int
    tree_reuse_count: int
    tree_reset_count: int
    inference: InferenceStats

    @property
    def positions_per_second(self) -> float:
        return self.positions / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def simulations_per_second(self) -> float:
        return self.simulations / self.elapsed_seconds if self.elapsed_seconds else 0.0


@dataclass(frozen=True, slots=True)
class SelfPlayBatch:
    results: tuple[SelfPlayResult, ...]
    profile: SelfPlayProfile


def generate_self_play_game(
    policy_value_fn: PolicyValueFunction,
    board_config: BoardConfig,
    self_play_config: SelfPlayConfig,
    rng: np.random.Generator,
) -> SelfPlayResult:
    """Play one actor-local game while reusing its MCTS search tree."""
    started = perf_counter()
    board = Board(
        width=board_config.width,
        height=board_config.height,
        n_in_row=board_config.n_in_row,
    )
    game = Game(board)
    player = MCTSPlayer(
        policy_value_fn,
        c_puct=self_play_config.c_puct,
        n_playout=self_play_config.simulations_per_move,
        is_selfplay=1,
        rng=rng,
    )
    winner, records = game.start_self_play(
        player,
        temp=self_play_config.temperature,
    )
    samples = tuple(
        ReplaySample(state=state, policy=policy, outcome=float(outcome))
        for state, policy, outcome in records
    )
    if not samples:
        raise RuntimeError("self-play produced no training samples")
    return SelfPlayResult(
        winner=int(winner),
        samples=samples,
        tree_reuse_count=player.mcts.root_reuse_count,
        tree_reset_count=player.mcts.root_reset_count,
        elapsed_seconds=perf_counter() - started,
    )


def generate_self_play_games(
    network: PolicyValueNet,
    board_config: BoardConfig,
    self_play_config: SelfPlayConfig,
    rng: np.random.Generator,
) -> SelfPlayBatch:
    """Run actor-local MCTS games in parallel with one batched evaluator."""
    game_count = self_play_config.games_per_iteration
    actor_count = min(self_play_config.parallel_games, game_count)
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=game_count, dtype=np.uint64)
    started = perf_counter()

    with BatchedPolicyEvaluator(
        network,
        batch_size=self_play_config.inference_batch_size,
        max_wait_ms=self_play_config.inference_wait_ms,
    ) as evaluator:

        def play(seed: np.uint64) -> SelfPlayResult:
            return generate_self_play_game(
                evaluator.policy_value_fn,
                board_config,
                self_play_config,
                np.random.default_rng(seed),
            )

        with ThreadPoolExecutor(
            max_workers=actor_count,
            thread_name_prefix="alphazero-self-play",
        ) as actors:
            results = tuple(actors.map(play, seeds))
        inference = evaluator.snapshot()

    elapsed = perf_counter() - started
    positions = sum(len(result.samples) for result in results)
    profile = SelfPlayProfile(
        games=game_count,
        positions=positions,
        elapsed_seconds=elapsed,
        simulations=inference.requests,
        tree_reuse_count=sum(result.tree_reuse_count for result in results),
        tree_reset_count=sum(result.tree_reset_count for result in results),
        inference=inference,
    )
    return SelfPlayBatch(results=results, profile=profile)

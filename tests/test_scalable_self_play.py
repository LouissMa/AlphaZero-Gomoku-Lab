"""Tests for batched inference, parallel actors, and performance reports."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphazero_gomoku.game import Board  # noqa: E402
from alphazero_gomoku.policy_value_net_pytorch import (  # noqa: E402
    NetworkConfig,
    PolicyValueNet,
)
from alphazero_gomoku.training.benchmark import (  # noqa: E402
    run_self_play_benchmark,
    write_benchmark_report,
)
from alphazero_gomoku.training.config import (  # noqa: E402
    BoardConfig,
    ExperimentConfig,
    NetworkSettings,
    OptimizationConfig,
    RunConfig,
    SelfPlayConfig,
)
from alphazero_gomoku.training.inference import BatchedPolicyEvaluator  # noqa: E402
from alphazero_gomoku.training.self_play import generate_self_play_games  # noqa: E402


@pytest.fixture
def network() -> PolicyValueNet:
    config = NetworkConfig(
        board_width=3,
        board_height=3,
        channels=4,
        residual_blocks=1,
        policy_channels=1,
        value_channels=1,
        value_hidden_size=8,
    )
    return PolicyValueNet(3, 3, config=config, device="cpu", amp=False)


def make_board(first_move: int | None = None) -> Board:
    board = Board(width=3, height=3, n_in_row=3)
    board.init_board()
    if first_move is not None:
        board.do_move(first_move)
    return board


def parallel_config(*, games: int = 2) -> SelfPlayConfig:
    return SelfPlayConfig(
        games_per_iteration=games,
        simulations_per_move=1,
        c_puct=1.5,
        temperature=1.0,
        parallel_games=2,
        inference_batch_size=4,
        inference_wait_ms=100.0,
    )


def test_batched_evaluator_combines_concurrent_requests(network: PolicyValueNet) -> None:
    boards = [make_board(move) for move in range(4)]

    with BatchedPolicyEvaluator(network, batch_size=4, max_wait_ms=100.0) as evaluator:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(evaluator.policy_value_fn, boards))
        stats = evaluator.snapshot()

    assert stats.requests == 4
    assert stats.batches == 1
    assert stats.max_batch_size == 4
    assert stats.mean_batch_size == pytest.approx(4.0)
    for board, (priors, value) in zip(boards, results, strict=True):
        assert {move for move, _ in priors} == set(board.availables)
        assert sum(probability for _, probability in priors) == pytest.approx(1.0)
        assert -1.0 <= value <= 1.0


def test_parallel_self_play_is_reproducible_and_reuses_trees(
    network: PolicyValueNet,
) -> None:
    board_config = BoardConfig(width=3, height=3, n_in_row=3)
    config = parallel_config()

    first = generate_self_play_games(
        network,
        board_config,
        config,
        np.random.default_rng(29),
    )
    second = generate_self_play_games(
        network,
        board_config,
        config,
        np.random.default_rng(29),
    )

    assert first.profile.games == 2
    assert first.profile.inference.max_batch_size >= 2
    assert first.profile.inference.requests == first.profile.simulations
    assert first.profile.tree_reuse_count >= first.profile.positions - first.profile.games
    assert first.profile.positions_per_second > 0
    assert first.profile.simulations_per_second > 0
    assert [result.winner for result in first.results] == [
        result.winner for result in second.results
    ]
    for left, right in zip(first.results, second.results, strict=True):
        assert len(left.samples) == len(right.samples)
        for left_sample, right_sample in zip(left.samples, right.samples, strict=True):
            np.testing.assert_array_equal(left_sample.state, right_sample.state)
            np.testing.assert_array_equal(left_sample.policy, right_sample.policy)
            assert left_sample.outcome == right_sample.outcome


def test_benchmark_writes_versioned_hardware_report(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        board=BoardConfig(width=3, height=3, n_in_row=3),
        network=NetworkSettings(
            channels=4,
            residual_blocks=1,
            policy_channels=1,
            value_channels=1,
            value_hidden_size=8,
            device="cpu",
            amp=False,
        ),
        self_play=parallel_config(),
        optimization=OptimizationConfig(batch_size=2, replay_capacity=32),
        run=RunConfig(
            name="benchmark",
            seed=31,
            deterministic=True,
            iterations=1,
            checkpoint_every=1,
            output_dir=str(tmp_path),
        ),
    )

    report = run_self_play_benchmark(config)
    destination = write_benchmark_report(report, tmp_path / "benchmark.json")

    assert report.format_version == 1
    assert report.device == "cpu"
    assert report.hardware["pytorch"] == torch.__version__
    assert report.workload["games"] == 2
    assert report.performance["simulations_per_second"] > 0
    assert destination.is_file()
    assert 0 < report.performance["inference_batch_utilization"] <= 1

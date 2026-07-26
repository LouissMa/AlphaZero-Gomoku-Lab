from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphazero_gomoku.training.config import BoardConfig, SelfPlayConfig  # noqa: E402
from alphazero_gomoku.training.self_play import generate_self_play_game  # noqa: E402


def uniform_policy(board):
    probability = 1.0 / len(board.availables)
    return [(move, probability) for move in board.availables], 0.0


def gumbel_config() -> SelfPlayConfig:
    return SelfPlayConfig(
        games_per_iteration=1,
        simulations_per_move=4,
        search_algorithm="gumbel",
        max_considered_actions=3,
        gumbel_scale=1.0,
        q_value_scale=0.1,
        q_visit_offset=10.0,
        parallel_games=1,
        inference_batch_size=1,
    )


def test_self_play_config_preserves_puct_default() -> None:
    assert SelfPlayConfig().search_algorithm == "puct"


def test_self_play_config_rejects_unknown_search_algorithm() -> None:
    with pytest.raises(ValueError, match="search_algorithm"):
        SelfPlayConfig(search_algorithm="uct")


def test_gumbel_self_play_produces_normalized_replay_targets() -> None:
    result = generate_self_play_game(
        uniform_policy,
        BoardConfig(width=3, height=3, n_in_row=3),
        gumbel_config(),
        np.random.default_rng(9),
    )

    assert result.samples
    assert result.search_algorithm == "gumbel"
    assert result.simulations > 0
    for sample in result.samples:
        assert sample.policy.shape == (9,)
        assert sample.policy.sum() == pytest.approx(1.0)
        assert np.isfinite(sample.policy).all()


def test_gumbel_self_play_is_seed_reproducible() -> None:
    board = BoardConfig(width=3, height=3, n_in_row=3)
    first = generate_self_play_game(
        uniform_policy, board, gumbel_config(), np.random.default_rng(4)
    )
    second = generate_self_play_game(
        uniform_policy, board, gumbel_config(), np.random.default_rng(4)
    )

    assert first.winner == second.winner
    assert len(first.samples) == len(second.samples)
    for left, right in zip(first.samples, second.samples, strict=True):
        np.testing.assert_array_equal(left.state, right.state)
        np.testing.assert_allclose(left.policy, right.policy)

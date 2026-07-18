from __future__ import annotations

import numpy as np
import pytest

from alphazero_gomoku.game import Board
from alphazero_gomoku.gumbel.player import GumbelMCTSPlayer
from alphazero_gomoku.gumbel.search import GumbelSearch


def uniform_policy(board: Board):
    probability = 1.0 / len(board.availables)
    return [(move, probability) for move in board.availables], 0.0


def board_with_opening() -> Board:
    board = Board(width=3, height=3, n_in_row=3)
    board.init_board()
    board.do_move(4)
    return board


def test_gumbel_search_is_legal_reproducible_and_budgeted() -> None:
    first = GumbelSearch(
        uniform_policy,
        simulations=8,
        max_considered_actions=4,
        rng=np.random.default_rng(7),
    ).search(board_with_opening())
    second = GumbelSearch(
        uniform_policy,
        simulations=8,
        max_considered_actions=4,
        rng=np.random.default_rng(7),
    ).search(board_with_opening())

    assert first.action in board_with_opening().availables
    assert first.action == second.action
    np.testing.assert_allclose(first.policy, second.policy)
    assert first.simulations_used == 8
    assert len(first.considered_actions) == 4
    assert sum(first.root_visits.values()) == 7


def test_gumbel_search_returns_full_normalized_completed_q_policy() -> None:
    result = GumbelSearch(
        uniform_policy,
        simulations=4,
        max_considered_actions=2,
        rng=np.random.default_rng(11),
    ).search(board_with_opening())

    assert result.policy.shape == (9,)
    assert result.policy.sum() == pytest.approx(1.0)
    assert result.policy[4] == 0.0
    assert np.isfinite(result.policy).all()


def test_gumbel_player_reuses_selected_subtree() -> None:
    board = board_with_opening()
    player = GumbelMCTSPlayer(
        uniform_policy,
        simulations=6,
        max_considered_actions=3,
        is_selfplay=True,
        rng=np.random.default_rng(3),
    )

    move, policy = player.get_action(board, return_prob=True)
    board.do_move(move)
    player.get_action(board)

    assert policy.shape == (9,)
    assert player.search_engine.root_reuse_count >= 1


def test_gumbel_search_rejects_terminal_board() -> None:
    board = Board(width=1, height=1, n_in_row=1)
    board.init_board()
    board.do_move(0)

    with pytest.raises(ValueError, match="legal actions"):
        GumbelSearch(uniform_policy, simulations=2).search(board)

"""Regression tests for the framework-independent game engine."""

import unittest

import numpy as np

from alphazero_gomoku.game import Board


class BoardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board(width=6, height=6, n_in_row=4)
        self.board.init_board()

    def play(self, moves: list[int]) -> None:
        for move in moves:
            self.board.do_move(move)

    def test_coordinate_round_trip(self) -> None:
        for move in range(36):
            self.assertEqual(
                self.board.location_to_move(self.board.move_to_location(move)),
                move,
            )

    def test_horizontal_win(self) -> None:
        self.play([0, 6, 1, 7, 2, 8, 3])
        self.assertEqual(self.board.game_end(), (True, 1))

    def test_vertical_win(self) -> None:
        self.play([0, 1, 6, 2, 12, 3, 18])
        self.assertEqual(self.board.game_end(), (True, 1))

    def test_both_diagonal_wins(self) -> None:
        self.play([0, 1, 7, 2, 14, 3, 21])
        self.assertEqual(self.board.game_end(), (True, 1))

        self.board.init_board()
        self.play([3, 0, 8, 1, 13, 2, 18])
        self.assertEqual(self.board.game_end(), (True, 1))

    def test_state_tensor_contract(self) -> None:
        self.board.do_move(35)
        state = self.board.current_state()

        self.assertEqual(state.shape, (4, 6, 6))
        self.assertEqual(state.dtype.kind, "f")
        self.assertEqual(np.count_nonzero(state[1]), 1)
        self.assertEqual(np.count_nonzero(state[2]), 1)

    def test_invalid_board_size_is_rejected(self) -> None:
        board = Board(width=3, height=6, n_in_row=4)
        with self.assertRaises(ValueError):
            board.init_board()


if __name__ == "__main__":
    unittest.main()

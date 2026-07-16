"""Regression tests for rectangular board state encoding."""

import unittest

import numpy as np

from alphazero_gomoku.game import Board


class RectangularBoardTest(unittest.TestCase):
    def test_state_layout_is_channels_height_width(self) -> None:
        board = Board(width=5, height=6, n_in_row=4)
        board.init_board()
        board.do_move(29)

        state = board.current_state()

        self.assertEqual(state.shape, (4, 6, 5))
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(state[1, 0, 4], 1.0)
        self.assertEqual(state[2, 0, 4], 1.0)


if __name__ == "__main__":
    unittest.main()

"""Framework-independent arena players and baseline factories."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from alphazero_gomoku.mcts_pure import MCTSPlayer as LegacyPureMCTSPlayer


class ArenaPlayer(Protocol):
    def set_player_ind(self, player: int) -> None: ...
    def get_action(self, board: object) -> int: ...


class PlayerFactory(Protocol):
    name: str

    def create(self, seed: int) -> ArenaPlayer: ...


class RandomPlayer:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.player = 0

    def set_player_ind(self, player: int) -> None:
        self.player = player

    def get_action(self, board: object) -> int:
        return int(self.rng.choice(board.availables))


class HeuristicPlayer(RandomPlayer):
    """Win immediately, block immediate losses, then prefer central moves."""

    def _winning_moves(self, board: object, player: int) -> list[int]:
        winning: list[int] = []
        for move in board.availables:
            candidate = copy.deepcopy(board)
            candidate.current_player = player
            candidate.do_move(move)
            won, winner = candidate.has_a_winner()
            if won and winner == player:
                winning.append(int(move))
        return winning

    def get_action(self, board: object) -> int:
        winning = self._winning_moves(board, board.current_player)
        if winning:
            return int(self.rng.choice(winning))
        opponent = (
            board.players[0] if board.current_player == board.players[1] else board.players[1]
        )
        blocks = self._winning_moves(board, opponent)
        if blocks:
            return int(self.rng.choice(blocks))
        center_row = (board.height - 1) / 2
        center_column = (board.width - 1) / 2
        distances = {
            int(move): abs(move // board.width - center_row)
            + abs(move % board.width - center_column)
            for move in board.availables
        }
        minimum = min(distances.values())
        preferred = [move for move, distance in distances.items() if distance == minimum]
        return int(self.rng.choice(preferred))


class PureMCTSPlayer:
    def __init__(self, seed: int, *, c_puct: float, playouts: int) -> None:
        np.random.seed(seed % (2**32))
        self.delegate = LegacyPureMCTSPlayer(c_puct=c_puct, n_playout=playouts)

    def set_player_ind(self, player: int) -> None:
        self.delegate.set_player_ind(player)

    def get_action(self, board: object) -> int:
        return int(self.delegate.get_action(board))


@dataclass(frozen=True, slots=True)
class RandomFactory:
    name: str = "random"

    def create(self, seed: int) -> RandomPlayer:
        return RandomPlayer(seed)


@dataclass(frozen=True, slots=True)
class HeuristicFactory:
    name: str = "heuristic"

    def create(self, seed: int) -> HeuristicPlayer:
        return HeuristicPlayer(seed)


@dataclass(frozen=True, slots=True)
class PureMCTSFactory:
    c_puct: float
    playouts: int
    name: str = "pure_mcts"

    def create(self, seed: int) -> PureMCTSPlayer:
        return PureMCTSPlayer(seed, c_puct=self.c_puct, playouts=self.playouts)

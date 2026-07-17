"""Self-play data generation for the reproducible training pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from alphazero_gomoku.game import Board, Game
from alphazero_gomoku.mcts_alphaZero import MCTSPlayer
from alphazero_gomoku.policy_value_net_pytorch import PolicyValueNet

from .config import BoardConfig, SelfPlayConfig
from .replay_buffer import ReplaySample


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    winner: int
    samples: tuple[ReplaySample, ...]


def generate_self_play_game(
    network: PolicyValueNet,
    board_config: BoardConfig,
    self_play_config: SelfPlayConfig,
) -> SelfPlayResult:
    """Play one neural-guided game and return immutable training samples."""
    board = Board(
        width=board_config.width,
        height=board_config.height,
        n_in_row=board_config.n_in_row,
    )
    game = Game(board)
    player = MCTSPlayer(
        network.policy_value_fn,
        c_puct=self_play_config.c_puct,
        n_playout=self_play_config.simulations_per_move,
        is_selfplay=1,
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
    return SelfPlayResult(winner=int(winner), samples=samples)

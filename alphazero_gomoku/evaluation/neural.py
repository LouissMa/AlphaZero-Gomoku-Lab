"""Modern policy-value network adapter for the evaluation arena."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alphazero_gomoku.mcts_alphaZero import MCTSPlayer
from alphazero_gomoku.policy_value_net_pytorch import PolicyValueNet


def resolve_model_path(path: str | Path) -> Path:
    source = Path(path)
    if source.is_dir():
        source = source / "model.pt"
    if not source.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {source}")
    return source


@dataclass(slots=True)
class NeuralMCTSFactory:
    model_path: Path
    simulations_per_move: int
    c_puct: float
    device: str = "auto"
    name: str = "candidate"
    network: PolicyValueNet = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = resolve_model_path(self.model_path)
        self.network = PolicyValueNet.from_checkpoint(
            self.model_path,
            device=self.device,
            load_optimizer=False,
            amp=False,
        )

    def validate_board(self, *, width: int, height: int) -> None:
        model_width = self.network.config.board_width
        model_height = self.network.config.board_height
        if (model_width, model_height) != (width, height):
            raise ValueError(
                f"model board is {model_width}x{model_height}, but arena board is {width}x{height}"
            )

    def create(self, seed: int) -> MCTSPlayer:
        return MCTSPlayer(
            self.network.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.simulations_per_move,
            is_selfplay=0,
            rng=np.random.default_rng(seed),
        )

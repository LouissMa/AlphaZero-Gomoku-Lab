from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from alphazero_gomoku.evaluation.neural import NeuralMCTSFactory  # noqa: E402
from alphazero_gomoku.policy_value_net_pytorch import NetworkConfig, PolicyValueNet  # noqa: E402


def test_neural_factory_loads_checkpoint_directory(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    network = PolicyValueNet(
        3,
        3,
        device="cpu",
        config=NetworkConfig(
            board_width=3,
            board_height=3,
            channels=8,
            residual_blocks=1,
            value_hidden_size=16,
        ),
        amp=False,
    )
    network.save_model(checkpoint / "model.pt")

    factory = NeuralMCTSFactory(
        checkpoint,
        simulations_per_move=1,
        c_puct=5.0,
        device="cpu",
    )

    player = factory.create(seed=7)
    assert factory.model_path == checkpoint / "model.pt"
    assert player is not None


def test_neural_factory_rejects_mismatched_arena_board(tmp_path) -> None:
    model = tmp_path / "model.pt"
    network = PolicyValueNet(
        3,
        3,
        device="cpu",
        config=NetworkConfig(
            board_width=3,
            board_height=3,
            channels=8,
            residual_blocks=1,
            value_hidden_size=16,
        ),
        amp=False,
    )
    network.save_model(model)
    factory = NeuralMCTSFactory(model, simulations_per_move=1, c_puct=5.0, device="cpu")

    with pytest.raises(ValueError, match="3x3.*4x4"):
        factory.validate_board(width=4, height=4)

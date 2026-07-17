"""End-to-end tests for complete training snapshots."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphazero_gomoku.policy_value_net_pytorch import (  # noqa: E402
    NetworkConfig,
    PolicyValueNet,
)
from alphazero_gomoku.training.checkpoint import (  # noqa: E402
    CheckpointManager,
    TrainingState,
)
from alphazero_gomoku.training.config import (  # noqa: E402
    BoardConfig,
    ExperimentConfig,
    NetworkSettings,
    OptimizationConfig,
)
from alphazero_gomoku.training.replay_buffer import (  # noqa: E402
    ReplayBuffer,
    ReplaySample,
)
from alphazero_gomoku.training.reproducibility import seed_everything  # noqa: E402


def test_complete_checkpoint_round_trip(tmp_path) -> None:
    config = ExperimentConfig(
        board=BoardConfig(width=4, height=4, n_in_row=4),
        network=NetworkSettings(channels=8, residual_blocks=1, value_hidden_size=16),
        optimization=OptimizationConfig(batch_size=2, replay_capacity=8),
    )
    network_config = NetworkConfig(
        board_width=4,
        board_height=4,
        channels=8,
        residual_blocks=1,
        value_hidden_size=16,
    )
    network = PolicyValueNet(4, 4, config=network_config, device="cpu", amp=False)
    replay = ReplayBuffer(capacity=8)
    for move, outcome in [(0, -1.0), (1, 0.0), (2, 1.0)]:
        state = np.zeros((4, 4, 4), dtype=np.float32)
        policy = np.zeros(16, dtype=np.float32)
        policy[move] = 1.0
        replay.append(ReplaySample(state, policy, outcome))
    random_context = seed_everything(19)
    state = TrainingState(iteration=7, total_games=4, total_positions=31)
    manager = CheckpointManager(tmp_path / "checkpoints")

    checkpoint = manager.save(network, replay, state, config, random_context.numpy)
    expected_random_value = random_context.numpy.random()
    restored = manager.load(checkpoint, device="cpu")

    assert manager.latest() == checkpoint
    assert restored.config == config
    assert restored.state == state
    assert len(restored.replay) == 3
    assert restored.network.config == network_config
    assert restored.random_generator.random() == expected_random_value

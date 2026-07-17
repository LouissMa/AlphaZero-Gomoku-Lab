"""Tests for the modern PyTorch policy-value backend."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphazero_gomoku.game import Board, Game  # noqa: E402
from alphazero_gomoku.mcts_alphaZero import MCTSPlayer  # noqa: E402
from alphazero_gomoku.policy_value_net_pytorch import (  # noqa: E402
    NetworkConfig,
    PolicyValueModel,
    PolicyValueNet,
)


@pytest.fixture
def config() -> NetworkConfig:
    return NetworkConfig(
        board_width=4,
        board_height=4,
        channels=8,
        residual_blocks=1,
        policy_channels=2,
        value_channels=1,
        value_hidden_size=16,
    )


@pytest.fixture
def network(config: NetworkConfig) -> PolicyValueNet:
    return PolicyValueNet(4, 4, config=config, device="cpu", amp=False)


def test_model_output_contract(config: NetworkConfig) -> None:
    model = PolicyValueModel(config).eval()
    states = torch.zeros((3, 4, 4, 4), dtype=torch.float32)

    with torch.inference_mode():
        log_policy, values = model(states)

    assert log_policy.shape == (3, 16)
    assert values.shape == (3, 1)
    assert torch.allclose(log_policy.exp().sum(dim=1), torch.ones(3), atol=1e-6)
    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)


def test_policy_value_fn_masks_and_normalizes_legal_actions(network: PolicyValueNet) -> None:
    board = Board(width=4, height=4, n_in_row=4)
    board.init_board()
    board.do_move(0)
    board.do_move(5)

    action_priors, value = network.policy_value_fn(board)
    action_priors = list(action_priors)

    assert {action for action, _ in action_priors} == set(board.availables)
    assert math.isclose(sum(probability for _, probability in action_priors), 1.0, abs_tol=1e-6)
    assert -1.0 <= value <= 1.0


def test_policy_value_fn_handles_terminal_full_board(network: PolicyValueNet) -> None:
    board = Board(width=4, height=4, n_in_row=4)
    board.init_board()
    for move in range(16):
        board.do_move(move)

    action_priors, value = network.policy_value_fn(board)

    assert list(action_priors) == []
    assert -1.0 <= value <= 1.0


def test_training_step_updates_parameters(network: PolicyValueNet) -> None:
    rng = np.random.default_rng(7)
    states = rng.random((4, 4, 4, 4), dtype=np.float32)
    target_policy = np.full((4, 16), 1.0 / 16, dtype=np.float32)
    winners = np.asarray([1.0, -1.0, 0.0, 1.0], dtype=np.float32)
    before = [parameter.detach().clone() for parameter in network.model.parameters()]

    metrics = network.train_batch(states, target_policy, winners, learning_rate=1e-3)

    assert all(
        math.isfinite(value)
        for value in (
            metrics.loss,
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
            metrics.gradient_norm,
        )
    )
    assert metrics.loss == pytest.approx(metrics.policy_loss + metrics.value_loss)
    assert network.training_step == 1
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, network.model.parameters(), strict=True)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_amp_training_step(config: NetworkConfig) -> None:
    network = PolicyValueNet(4, 4, config=config, device="cuda", amp=True)
    states = np.zeros((2, 4, 4, 4), dtype=np.float32)
    target_policy = np.full((2, 16), 1.0 / 16, dtype=np.float32)
    winners = np.asarray([1.0, -1.0], dtype=np.float32)

    metrics = network.train_batch(states, target_policy, winners)

    assert network.amp_enabled
    assert network.device.type == "cuda"
    assert math.isfinite(metrics.loss)
    assert network.training_step == 1


def test_checkpoint_round_trip(
    tmp_path: Path,
    network: PolicyValueNet,
) -> None:
    states = np.zeros((2, 4, 4, 4), dtype=np.float32)
    expected_policy, expected_values = network.policy_value(states)
    checkpoint = tmp_path / "policy.pt"

    network.save_model(checkpoint, metadata={"purpose": "test"})
    restored = PolicyValueNet.from_checkpoint(checkpoint, device="cpu", amp=False)
    actual_policy, actual_values = restored.policy_value(states)

    np.testing.assert_allclose(actual_policy, expected_policy, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual_values, expected_values, rtol=1e-6, atol=1e-7)
    assert restored.config == network.config


def test_torch_compile_with_eager_backend(config: NetworkConfig) -> None:
    network = PolicyValueNet(
        4,
        4,
        config=config,
        device="cpu",
        amp=False,
        compile_model=True,
        compile_backend="eager",
    )

    probabilities, _ = network.policy_value(np.zeros((1, 4, 4, 4), dtype=np.float32))

    assert network.compile_enabled
    assert probabilities.sum() == pytest.approx(1.0)


def test_compile_failure_falls_back_to_eager(network: PolicyValueNet) -> None:
    def broken_compiled_model(_: torch.Tensor) -> None:
        raise RuntimeError("compiler unavailable")

    network.compile_enabled = True
    network._forward_model = broken_compiled_model

    with pytest.warns(RuntimeWarning, match="falling back to eager mode"):
        probabilities, _ = network.policy_value(np.zeros((1, 4, 4, 4), dtype=np.float32))

    assert not network.compile_enabled
    assert probabilities.sum() == pytest.approx(1.0)


def test_network_integrates_with_mcts(network: PolicyValueNet) -> None:
    board = Board(width=4, height=4, n_in_row=4)
    board.init_board()
    player = MCTSPlayer(network.policy_value_fn, c_puct=5, n_playout=4)

    move = player.get_action(board)

    assert move in board.availables


def test_network_completes_self_play_game(network: PolicyValueNet) -> None:
    board = Board(width=4, height=4, n_in_row=4)
    game = Game(board)
    player = MCTSPlayer(
        network.policy_value_fn,
        c_puct=5,
        n_playout=2,
        is_selfplay=1,
    )

    winner, play_data = game.start_self_play(player, temp=1.0)
    records = list(play_data)

    assert winner in {-1, 1, 2}
    assert 7 <= len(records) <= 16
    assert all(state.shape == (4, 4, 4) for state, _, _ in records)

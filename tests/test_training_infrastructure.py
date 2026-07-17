"""Tests for reproducible training infrastructure."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from alphazero_gomoku.training.config import ExperimentConfig, load_experiment_config
from alphazero_gomoku.training.metrics import JsonlMetricsWriter, read_metrics
from alphazero_gomoku.training.replay_buffer import ReplayBuffer, ReplaySample
from alphazero_gomoku.training.reproducibility import seed_everything


def make_sample(move: int, outcome: float = 1.0) -> ReplaySample:
    state = np.zeros((4, 4, 4), dtype=np.float32)
    state[0, move // 4, move % 4] = 1.0
    policy = np.zeros(16, dtype=np.float32)
    policy[move] = 1.0
    return ReplaySample(state, policy, outcome)


def test_loads_typed_toml_config() -> None:
    config = load_experiment_config("configs/train_6x6.toml")

    assert config.board.width == 6
    assert config.board.n_in_row == 4
    assert config.network.channels == 64
    assert config.optimization.batch_size == 512
    assert config.to_dict()["run"]["seed"] == 2026


def test_config_rejects_unknown_sections() -> None:
    with pytest.raises(ValueError, match="unknown configuration sections"):
        ExperimentConfig.from_dict({"surprise": {}})


def test_replay_buffer_is_bounded_and_samples_deterministically() -> None:
    replay = ReplayBuffer(capacity=3)
    replay.extend(make_sample(move, outcome=float(move % 3 - 1)) for move in range(4))

    first = replay.sample(2, np.random.default_rng(11))
    second = replay.sample(2, np.random.default_rng(11))

    assert len(replay) == 3
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_replay_buffer_round_trip(tmp_path: Path) -> None:
    replay = ReplayBuffer(capacity=8)
    replay.extend([make_sample(1, -1.0), make_sample(2, 0.0), make_sample(3, 1.0)])
    path = tmp_path / "replay.npz"

    replay.save(path)
    restored = ReplayBuffer.load(path)
    states, policies, outcomes = restored.sample(3, np.random.default_rng(4))

    assert restored.capacity == 8
    assert states.shape == (3, 4, 4, 4)
    assert policies.shape == (3, 16)
    assert sorted(outcomes.tolist()) == [-1.0, 0.0, 1.0]


def test_replay_buffer_validates_policy() -> None:
    sample = make_sample(0)
    invalid = ReplaySample(sample.state, np.zeros(16, dtype=np.float32), sample.outcome)

    with pytest.raises(ValueError, match="sum to one"):
        ReplayBuffer(capacity=2).append(invalid)


def test_seed_everything_reproduces_python_and_numpy() -> None:
    first_context = seed_everything(42)
    first = (random.random(), np.random.random(), first_context.numpy.random())
    second_context = seed_everything(42)
    second = (random.random(), np.random.random(), second_context.numpy.random())

    assert first == second


def test_jsonl_metrics_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    writer = JsonlMetricsWriter(path)

    writer.write(1, {"loss": 2.5, "games": 3})
    writer.write(2, {"loss": 1.5, "games": 5})
    records = list(read_metrics(path))

    assert [record["step"] for record in records] == [1, 2]
    assert records[1]["loss"] == 1.5
    assert "timestamp" in records[0]
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)

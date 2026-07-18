"""Integration tests for the end-to-end reproducible training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphazero_gomoku.cli import _build_parser  # noqa: E402
from alphazero_gomoku.training.config import (  # noqa: E402
    BoardConfig,
    ExperimentConfig,
    NetworkSettings,
    OptimizationConfig,
    RunConfig,
    SelfPlayConfig,
)
from alphazero_gomoku.training.trainer import AlphaZeroTrainer  # noqa: E402


def tiny_config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        board=BoardConfig(width=3, height=3, n_in_row=3),
        network=NetworkSettings(
            channels=4,
            residual_blocks=1,
            policy_channels=1,
            value_channels=1,
            value_hidden_size=8,
            device="cpu",
            amp=False,
        ),
        self_play=SelfPlayConfig(
            games_per_iteration=1,
            simulations_per_move=1,
            c_puct=1.5,
            temperature=1.0,
        ),
        optimization=OptimizationConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=2,
            epochs_per_iteration=1,
            replay_capacity=64,
            gradient_clip_norm=5.0,
        ),
        run=RunConfig(
            name="smoke",
            seed=17,
            deterministic=True,
            iterations=2,
            checkpoint_every=1,
            output_dir=str(output_dir),
        ),
    )


def test_training_pipeline_runs_and_resumes(tmp_path: Path) -> None:
    trainer = AlphaZeroTrainer.create(tiny_config(tmp_path))

    first_state = trainer.run(target_iteration=1)
    first_checkpoint = trainer.checkpoints.latest()

    assert first_state.iteration == 1
    assert first_state.total_games == 1
    assert first_state.total_positions >= 5
    assert trainer.network.training_step == 1
    assert first_checkpoint.name == "step_000001"
    manifest = json.loads((first_checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    with np.load(first_checkpoint / "replay.npz", allow_pickle=False) as replay_data:
        assert int(replay_data["format_version"]) == 1

    resumed = AlphaZeroTrainer.resume(first_checkpoint, device="cpu")
    final_state = resumed.run(target_iteration=2)
    with pytest.raises(FileExistsError, match="resume its latest checkpoint"):
        AlphaZeroTrainer.create(tiny_config(tmp_path))

    assert final_state.iteration == 2
    assert final_state.total_games == 2
    assert final_state.total_positions > first_state.total_positions
    assert resumed.network.training_step == 2
    assert resumed.checkpoints.latest().name == "step_000002"

    records = [
        json.loads(line)
        for line in (resumed.run_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2]
    assert all(record["optimizer_updates"] == 1 for record in records)
    assert all("checkpoint" in record for record in records)


def test_train_cli_requires_exactly_one_source() -> None:
    parser = _build_parser()

    config_args = parser.parse_args(["train", "--config", "experiment.toml"])
    resume_args = parser.parse_args(["train", "--resume", "checkpoints/step_000001"])
    benchmark_args = parser.parse_args(
        [
            "benchmark",
            "--config",
            "experiment.toml",
            "--games",
            "4",
        ]
    )

    assert config_args.config == Path("experiment.toml")
    assert config_args.resume is None
    assert resume_args.resume == Path("checkpoints/step_000001")
    assert resume_args.config is None
    assert benchmark_args.config == Path("experiment.toml")
    assert benchmark_args.games == 4

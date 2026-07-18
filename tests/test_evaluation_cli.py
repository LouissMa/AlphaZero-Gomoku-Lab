from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from alphazero_gomoku.cli import main  # noqa: E402
from alphazero_gomoku.policy_value_net_pytorch import NetworkConfig, PolicyValueNet  # noqa: E402


def test_arena_cli_evaluates_reports_and_promotes(tmp_path, capsys) -> None:
    checkpoint = tmp_path / "checkpoint"
    model = checkpoint / "model.pt"
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
    config = tmp_path / "arena.toml"
    config.write_text(
        """[board]
width = 3
height = 3
n_in_row = 3

[arena]
games_per_opponent = 2
simulations_per_move = 1
pure_mcts_playouts = 1
baselines = ["random"]
promotion_score = 0.0
promotion_lower_bound = 0.0
seed = 7
device = "cpu"
""",
        encoding="utf-8",
    )
    report = tmp_path / "reports" / "arena.json"
    promoted = tmp_path / "models" / "best.pt"

    exit_code = main(
        [
            "arena",
            "--candidate",
            str(checkpoint),
            "--incumbent",
            str(model),
            "--config",
            str(config),
            "--output",
            str(report),
            "--promote-to",
            str(promoted),
        ]
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert [match["opponent"] for match in payload["matches"]] == [
        "random",
        "incumbent",
    ]
    assert payload["promotion"]["promoted"] is True
    assert promoted.read_bytes() == model.read_bytes()
    assert "Arena complete" in capsys.readouterr().out


def test_arena_cli_rejects_promotion_without_incumbent(tmp_path) -> None:
    with pytest.raises(SystemExit, match="--promote-to requires --incumbent"):
        main(
            [
                "arena",
                "--candidate",
                str(tmp_path / "candidate.pt"),
                "--config",
                str(tmp_path / "arena.toml"),
                "--promote-to",
                str(tmp_path / "best.pt"),
            ]
        )

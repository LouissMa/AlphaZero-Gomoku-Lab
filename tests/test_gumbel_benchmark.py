from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from alphazero_gomoku.cli import main  # noqa: E402
from alphazero_gomoku.gumbel.benchmark import (  # noqa: E402
    load_search_comparison_config,
)
from alphazero_gomoku.policy_value_net_pytorch import (  # noqa: E402
    NetworkConfig,
    PolicyValueNet,
)


def write_checkpoint_and_config(tmp_path):
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
    config = tmp_path / "compare.toml"
    config.write_text(
        """[board]
width = 3
height = 3
n_in_row = 3

[comparison]
games = 2
simulations_per_move = 2
max_considered_actions = 2
c_puct = 1.5
gumbel_scale = 1.0
q_value_scale = 0.1
q_visit_offset = 10.0
confidence = 0.95
seed = 13
device = "cpu"
""",
        encoding="utf-8",
    )
    return model, config


def test_comparison_config_requires_even_games(tmp_path) -> None:
    _, path = write_checkpoint_and_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace("games = 2", "games = 3")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="positive even"):
        load_search_comparison_config(path)


def test_compare_search_cli_writes_equal_budget_report(tmp_path, capsys) -> None:
    model, config = write_checkpoint_and_config(tmp_path)
    destination = tmp_path / "reports" / "comparison.json"

    exit_code = main(
        [
            "compare-search",
            "--model",
            str(model),
            "--config",
            str(config),
            "--output",
            str(destination),
        ]
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["format_version"] == 1
    assert payload["match"]["summary"]["games"] == 2
    assert [game["first_player"] for game in payload["match"]["games"]] == [
        "gumbel",
        "puct",
    ]
    assert payload["profiles"]["gumbel"]["simulations_per_move"] == 2
    assert payload["profiles"]["puct"]["simulations_per_move"] == 2
    assert "Search comparison complete" in capsys.readouterr().out

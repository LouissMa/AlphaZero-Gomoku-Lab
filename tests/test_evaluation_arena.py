from __future__ import annotations

import json

import numpy as np
import pytest

from alphazero_gomoku.evaluation.arena import run_match, write_tournament_report
from alphazero_gomoku.evaluation.config import ArenaConfig, load_arena_config
from alphazero_gomoku.evaluation.players import HeuristicFactory, RandomFactory
from alphazero_gomoku.evaluation.promotion import decide_promotion, promote_model
from alphazero_gomoku.evaluation.stats import ScoreSummary
from alphazero_gomoku.training.config import BoardConfig


def tiny_arena_config(**overrides: object) -> ArenaConfig:
    values = {
        "board": BoardConfig(width=3, height=3, n_in_row=3),
        "games_per_opponent": 4,
        "simulations_per_move": 1,
        "pure_mcts_playouts": 1,
        "baselines": ("random", "heuristic"),
        "seed": 17,
    }
    values.update(overrides)
    return ArenaConfig(**values)


def test_arena_config_requires_even_games() -> None:
    with pytest.raises(ValueError, match="positive even"):
        tiny_arena_config(games_per_opponent=3)


def test_load_arena_config_reads_board_and_baselines(tmp_path) -> None:
    path = tmp_path / "arena.toml"
    path.write_text(
        "[board]\nwidth = 3\nheight = 3\nn_in_row = 3\n\n[arena]\n"
        'games_per_opponent = 2\nbaselines = ["random"]\n',
        encoding="utf-8",
    )

    config = load_arena_config(path)

    assert config.board.width == 3
    assert config.games_per_opponent == 2
    assert config.baselines == ("random",)


def test_score_summary_treats_draw_as_half_point() -> None:
    summary = ScoreSummary.from_results(["win", "draw", "loss", "draw"], confidence=0.95)

    assert summary.score == 0.5
    assert summary.elo_difference == pytest.approx(0.0)
    assert summary.confidence_low < summary.score < summary.confidence_high


def test_match_alternates_first_player_and_is_seed_reproducible() -> None:
    config = tiny_arena_config()
    candidate = HeuristicFactory(name="candidate")
    opponent = RandomFactory()

    first = run_match(candidate, opponent, config, np.random.default_rng(config.seed))
    second = run_match(candidate, opponent, config, np.random.default_rng(config.seed))

    assert [game.first_player for game in first.games] == [
        "candidate",
        "random",
        "candidate",
        "random",
    ]
    assert [(game.seed, game.result, game.moves) for game in first.games] == [
        (game.seed, game.result, game.moves) for game in second.games
    ]


def test_report_is_machine_readable_json(tmp_path) -> None:
    config = tiny_arena_config(games_per_opponent=2)
    match = run_match(
        HeuristicFactory(name="candidate"),
        RandomFactory(),
        config,
        np.random.default_rng(config.seed),
    )
    from alphazero_gomoku.evaluation.arena import TournamentReport

    report = TournamentReport.create("candidate", config, (match,))
    destination = write_tournament_report(report, tmp_path / "nested" / "arena.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["format_version"] == 1
    assert payload["matches"][0]["opponent"] == "random"
    assert payload["overall"]["games"] == 2


def test_promotion_is_confidence_gated_and_atomic(tmp_path) -> None:
    summary = ScoreSummary.from_results(["win"] * 20, confidence=0.95)
    decision = decide_promotion(summary, required_score=0.55, required_lower_bound=0.5)
    candidate = tmp_path / "candidate.pt"
    destination = tmp_path / "models" / "best.pt"
    candidate.write_bytes(b"new-model")

    promoted = promote_model(candidate, destination, decision)

    assert decision.promoted is True
    assert promoted == destination
    assert destination.read_bytes() == b"new-model"
    assert not (destination.parent / ".best.pt.tmp").exists()


def test_promotion_does_not_copy_when_confidence_is_insufficient(tmp_path) -> None:
    summary = ScoreSummary.from_results(["win", "loss"], confidence=0.95)
    decision = decide_promotion(summary, required_score=0.5, required_lower_bound=0.5)

    assert decision.promoted is False
    assert promote_model(tmp_path / "missing.pt", tmp_path / "best.pt", decision) is None

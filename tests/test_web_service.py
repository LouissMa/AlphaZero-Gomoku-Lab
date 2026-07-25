"""Behavior tests for the framework-independent web game service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from alphazero_gomoku.web.catalog import (
    ModelCatalog,
    ModelDescriptor,
    ModelNotFoundError,
)
from alphazero_gomoku.web.service import (
    GameConflictError,
    GameNotFoundError,
    GameService,
    InvalidMoveError,
    browser_to_move,
    move_to_browser,
)


@dataclass(frozen=True)
class FixedRuntime:
    descriptor: ModelDescriptor

    def policy_value_fn(self, board: object) -> tuple[list[tuple[int, float]], float]:
        legal = list(board.availables)
        weights = np.asarray([0.5**index for index in range(len(legal))], dtype=np.float64)
        weights /= weights.sum()
        return list(zip(legal, weights.tolist(), strict=True)), 0.25


class FixedCatalog:
    def __init__(self) -> None:
        self.descriptor = ModelDescriptor(
            id="fixed-4x4",
            name="Fixed test policy",
            width=4,
            height=4,
            n_in_row=3,
            kind="heuristic",
        )
        self.runtime = FixedRuntime(self.descriptor)

    def list_models(self) -> tuple[ModelDescriptor, ...]:
        return (self.descriptor,)

    def load(self, model_id: str) -> FixedRuntime:
        if model_id != self.descriptor.id:
            raise ModelNotFoundError(model_id)
        return self.runtime


@pytest.fixture
def service() -> GameService:
    return GameService(FixedCatalog(), seed=7)


def test_catalog_exposes_only_allowlisted_models(tmp_path: Path) -> None:
    catalog = ModelCatalog(tmp_path)

    models = catalog.list_models()

    assert [(model.id, model.width, model.height, model.kind) for model in models] == [
        ("heuristic-6x6", 6, 6, "heuristic"),
        ("alphazero-6x6", 6, 6, "numpy"),
        ("alphazero-8x8", 8, 8, "numpy"),
    ]
    with pytest.raises(ModelNotFoundError, match="unknown model"):
        catalog.load("../../private")


@pytest.mark.parametrize(
    ("row", "column", "move"),
    [(0, 0, 25), (0, 4, 29), (5, 0, 0), (5, 4, 4)],
)
def test_browser_coordinates_flip_the_legacy_row(
    row: int,
    column: int,
    move: int,
) -> None:
    assert browser_to_move(row, column, width=5, height=6) == move
    assert move_to_browser(move, width=5, height=6) == (row, column)


def test_create_black_game_has_normalized_analysis(service: GameService) -> None:
    state = service.create_game(
        model_id="fixed-4x4",
        human_color="black",
        search="puct",
        simulations=8,
    )

    assert state["status"] == "playing"
    assert state["current_color"] == "black"
    assert state["human_color"] == "black"
    assert state["moves"] == []
    assert state["analysis"]["color"] == "black"
    assert state["analysis"]["value"] == pytest.approx(0.25)
    assert sum(state["analysis"]["policy"]) == pytest.approx(1.0)


def test_white_game_starts_with_one_ai_move(service: GameService) -> None:
    state = service.create_game(
        model_id="fixed-4x4",
        human_color="white",
        search="gumbel",
        simulations=8,
    )

    assert state["current_color"] == "white"
    assert state["human_color"] == "white"
    assert state["move_count"] == 1
    assert state["moves"][0]["color"] == "black"
    assert state["analysis"]["latency_ms"] >= 0


def test_human_move_is_followed_by_ai_response(service: GameService) -> None:
    initial = service.create_game(
        model_id="fixed-4x4",
        human_color="black",
        search="puct",
        simulations=8,
    )

    state = service.play_move(initial["id"], row=0, column=0)

    assert [move["color"] for move in state["moves"]] == ["black", "white"]
    assert state["current_color"] == "black"
    assert state["last_move"] == state["moves"][-1]
    assert state["can_undo"] is True


def test_invalid_move_does_not_mutate_game(service: GameService) -> None:
    initial = service.create_game(
        model_id="fixed-4x4",
        human_color="black",
        search="puct",
        simulations=8,
    )
    after_turn = service.play_move(initial["id"], row=0, column=0)

    with pytest.raises(InvalidMoveError, match="occupied"):
        service.play_move(initial["id"], row=0, column=0)
    with pytest.raises(InvalidMoveError, match="outside"):
        service.play_move(initial["id"], row=8, column=0)

    assert service.get_game(initial["id"])["moves"] == after_turn["moves"]


def test_human_win_has_no_post_game_analysis(service: GameService) -> None:
    state = service.create_game(
        model_id="fixed-4x4",
        human_color="black",
        search="puct",
        simulations=8,
    )
    state = service.play_move(state["id"], row=0, column=0)
    state = service.play_move(state["id"], row=0, column=1)
    state = service.play_move(state["id"], row=0, column=2)

    assert state["status"] == "black_won"
    assert state["winner"] == "black"
    assert state["move_count"] == 5
    assert state["analysis"] is None
    with pytest.raises(GameConflictError, match="finished"):
        service.play_move(state["id"], row=1, column=1)


def test_undo_rewinds_one_complete_human_turn(service: GameService) -> None:
    initial = service.create_game(
        model_id="fixed-4x4",
        human_color="black",
        search="gumbel",
        simulations=8,
    )
    service.play_move(initial["id"], row=0, column=0)

    state = service.undo(initial["id"])

    assert state["moves"] == []
    assert state["current_color"] == "black"
    assert state["can_undo"] is False
    with pytest.raises(GameConflictError, match="nothing to undo"):
        service.undo(initial["id"])


def test_replay_is_complete_and_returned_as_fresh_data(service: GameService) -> None:
    initial = service.create_game(
        model_id="fixed-4x4",
        human_color="black",
        search="puct",
        simulations=8,
    )
    current = service.play_move(initial["id"], row=0, column=0)

    replay = service.replay(initial["id"])

    assert len(replay) == 3
    assert replay[0]["moves"] == []
    assert replay[-1]["moves"] == current["moves"]
    replay[-1]["moves"].clear()
    assert len(service.replay(initial["id"])[-1]["moves"]) == 2


def test_unknown_session_is_rejected(service: GameService) -> None:
    with pytest.raises(GameNotFoundError, match="unknown game"):
        service.get_game("missing")


@pytest.mark.parametrize("simulations", [0, 7, 9, 128])
def test_simulation_budget_is_whitelisted(service: GameService, simulations: int) -> None:
    with pytest.raises(ValueError, match="simulations"):
        service.create_game(
            model_id="fixed-4x4",
            human_color="black",
            search="puct",
            simulations=simulations,
        )

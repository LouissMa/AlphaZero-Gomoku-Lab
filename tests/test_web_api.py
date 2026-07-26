"""HTTP contract tests for the interactive web application."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from alphazero_gomoku.cli import main
from alphazero_gomoku.web.api import create_app
from alphazero_gomoku.web.catalog import ModelCatalog
from alphazero_gomoku.web.service import GameService


class ASGIClient:
    def __init__(self, application: object) -> None:
        self.application = application

    def request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.application)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path: str) -> httpx.Response:
        return self.request("GET", path)

    def post(self, path: str, **kwargs: object) -> httpx.Response:
        return self.request("POST", path, **kwargs)


@pytest.fixture
def client(tmp_path: Path) -> ASGIClient:
    service = GameService(ModelCatalog(tmp_path), seed=11)
    return ASGIClient(create_app(service))


def create_game(client: ASGIClient) -> dict[str, object]:
    response = client.post(
        "/api/games",
        json={
            "model_id": "heuristic-6x6",
            "human_color": "black",
            "search": "puct",
            "simulations": 8,
        },
    )
    assert response.status_code == 201
    return response.json()


def test_health_and_models_are_discoverable(client: ASGIClient) -> None:
    health = client.get("/api/health")
    models = client.get("/api/models")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": "alphazero-gomoku-web",
        "version": 1,
    }
    assert models.status_code == 200
    assert [model["id"] for model in models.json()] == [
        "heuristic-6x6",
        "alphazero-6x6",
        "alphazero-8x8",
    ]


def test_game_move_undo_and_replay_round_trip(client: ASGIClient) -> None:
    initial = create_game(client)

    fetched = client.get(f"/api/games/{initial['id']}")
    moved = client.post(f"/api/games/{initial['id']}/moves", json={"row": 0, "column": 0})
    replay = client.get(f"/api/games/{initial['id']}/replay")
    undone = client.post(f"/api/games/{initial['id']}/undo")

    assert fetched.status_code == 200
    assert moved.status_code == 200
    assert moved.json()["move_count"] == 2
    assert len(replay.json()) == 3
    assert undone.status_code == 200
    assert undone.json()["move_count"] == 0


def test_invalid_requests_have_stable_status_codes(client: ASGIClient) -> None:
    missing = client.get("/api/games/not-a-session")
    invalid_settings = client.post(
        "/api/games",
        json={
            "model_id": "heuristic-6x6",
            "human_color": "black",
            "search": "puct",
            "simulations": 12,
        },
    )
    initial = create_game(client)
    outside = client.post(
        f"/api/games/{initial['id']}/moves",
        json={"row": 99, "column": 0},
    )
    no_undo = client.post(f"/api/games/{initial['id']}/undo")

    assert missing.status_code == 404
    assert missing.json()["detail"].startswith("unknown game")
    assert invalid_settings.status_code == 422
    assert outside.status_code == 422
    assert no_undo.status_code == 409


def test_unknown_model_is_not_a_path_escape(client: ASGIClient) -> None:
    response = client.post(
        "/api/games",
        json={
            "model_id": "../../private",
            "human_color": "black",
            "search": "gumbel",
            "simulations": 8,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "unknown model: ../../private"


def test_index_is_delivered_as_html(client: ASGIClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "AlphaZero" in response.text


def test_serve_help_documents_runtime_controls(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["serve", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--host" in output
    assert "--port" in output
    assert "--reload" in output
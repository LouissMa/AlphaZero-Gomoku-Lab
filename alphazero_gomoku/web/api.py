"""FastAPI boundary for browser play and packaged static assets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import WEB_API_VERSION
from .catalog import ModelCatalog, ModelNotFoundError
from .service import (
    GameConflictError,
    GameNotFoundError,
    GameService,
    InvalidMoveError,
)

STATIC_ROOT = Path(__file__).resolve().parent / "static"


class CreateGameRequest(BaseModel):
    model_id: str
    human_color: Literal["black", "white"] = "black"
    search: Literal["puct", "gumbel"] = "gumbel"
    simulations: Literal[8, 16, 32, 64] = 32


class MoveRequest(BaseModel):
    row: int
    column: int


def _application_root() -> Path:
    working = Path.cwd()
    if (working / "models").is_dir():
        return working
    return Path(__file__).resolve().parents[2]


def create_app(service: GameService | None = None) -> FastAPI:
    game_service = service or GameService(ModelCatalog(_application_root()))
    application = FastAPI(
        title="AlphaZero Gomoku Lab",
        version=str(WEB_API_VERSION),
        description="Interactive policy-value inference and Gomoku play.",
    )

    @application.exception_handler(ModelNotFoundError)
    @application.exception_handler(GameNotFoundError)
    async def not_found_handler(_request: object, error: Exception) -> None:
        raise HTTPException(status_code=404, detail=str(error))

    @application.exception_handler(InvalidMoveError)
    async def invalid_move_handler(_request: object, error: InvalidMoveError) -> None:
        raise HTTPException(status_code=422, detail=str(error))

    @application.exception_handler(GameConflictError)
    async def conflict_handler(_request: object, error: GameConflictError) -> None:
        raise HTTPException(status_code=409, detail=str(error))

    @application.get("/api/health")
    def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "service": "alphazero-gomoku-web",
            "version": WEB_API_VERSION,
        }

    @application.get("/api/models")
    def models() -> list[dict[str, str | int]]:
        return game_service.list_models()

    @application.post("/api/games", status_code=201)
    def create_game(request: CreateGameRequest) -> dict[str, object]:
        return game_service.create_game(**request.model_dump())

    @application.get("/api/games/{game_id}")
    def get_game(game_id: str) -> dict[str, object]:
        return game_service.get_game(game_id)

    @application.post("/api/games/{game_id}/moves")
    def play_move(game_id: str, request: MoveRequest) -> dict[str, object]:
        return game_service.play_move(game_id, **request.model_dump())

    @application.post("/api/games/{game_id}/undo")
    def undo(game_id: str) -> dict[str, object]:
        return game_service.undo(game_id)

    @application.get("/api/games/{game_id}/replay")
    def replay(game_id: str) -> list[dict[str, object]]:
        return game_service.replay(game_id)

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    application.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
    return application


app = create_app()
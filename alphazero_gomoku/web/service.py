"""Framework-independent game sessions for the interactive web application."""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from alphazero_gomoku.game import Board
from alphazero_gomoku.gumbel.player import GumbelMCTSPlayer
from alphazero_gomoku.mcts_alphaZero import MCTSPlayer

from .catalog import ModelCatalog, ModelDescriptor, PolicyRuntime

Color = Literal["black", "white"]
SearchAlgorithm = Literal["puct", "gumbel"]
ALLOWED_SIMULATIONS = frozenset({8, 16, 32, 64})
PLAYER_TO_COLOR: dict[int, Color] = {1: "black", 2: "white"}
COLOR_TO_PLAYER: dict[Color, int] = {"black": 1, "white": 2}


class Catalog(Protocol):
    def list_models(self) -> tuple[ModelDescriptor, ...]: ...
    def load(self, model_id: str) -> PolicyRuntime: ...


class GameNotFoundError(LookupError):
    def __init__(self, game_id: str) -> None:
        super().__init__(f"unknown game: {game_id}")


class InvalidMoveError(ValueError):
    """Raised when browser coordinates do not identify a legal move."""


class GameConflictError(RuntimeError):
    """Raised when a valid request conflicts with the current game state."""


def browser_to_move(row: int, column: int, *, width: int, height: int) -> int:
    if not (0 <= row < height and 0 <= column < width):
        raise InvalidMoveError(f"point ({row}, {column}) is outside the board")
    return (height - 1 - row) * width + column


def move_to_browser(move: int, *, width: int, height: int) -> tuple[int, int]:
    if not 0 <= move < width * height:
        raise InvalidMoveError(f"move {move} is outside the board")
    legacy_row, column = divmod(move, width)
    return height - 1 - legacy_row, column


@dataclass(slots=True)
class GameSession:
    id: str
    descriptor: ModelDescriptor
    runtime: PolicyRuntime
    human_color: Color
    search: SearchAlgorithm
    simulations: int
    board: Board
    seed: int
    moves: list[dict[str, int | str]] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def human_player(self) -> int:
        return COLOR_TO_PLAYER[self.human_color]

    @property
    def ai_player(self) -> int:
        return 2 if self.human_player == 1 else 1


class GameService:
    """Own ephemeral games and expose JSON-ready state transitions."""

    def __init__(
        self,
        catalog: Catalog | None = None,
        *,
        seed: int = 2026,
    ) -> None:
        self.catalog = catalog if catalog is not None else ModelCatalog(".")
        self._rng = np.random.default_rng(seed)
        self._sessions: dict[str, GameSession] = {}
        self._lock = threading.RLock()

    def list_models(self) -> list[dict[str, str | int]]:
        return [descriptor.to_dict() for descriptor in self.catalog.list_models()]

    def create_game(
        self,
        *,
        model_id: str,
        human_color: Color,
        search: SearchAlgorithm,
        simulations: int,
    ) -> dict[str, object]:
        if human_color not in COLOR_TO_PLAYER:
            raise ValueError("human_color must be black or white")
        if search not in {"puct", "gumbel"}:
            raise ValueError("search must be puct or gumbel")
        if simulations not in ALLOWED_SIMULATIONS:
            raise ValueError(f"simulations must be one of {sorted(ALLOWED_SIMULATIONS)}")
        with self._lock:
            runtime = self.catalog.load(model_id)
            descriptor = runtime.descriptor
            board = Board(
                width=descriptor.width,
                height=descriptor.height,
                n_in_row=descriptor.n_in_row,
            )
            board.init_board(start_player=0)
            game_id = str(uuid.uuid4())
            session = GameSession(
                id=game_id,
                descriptor=descriptor,
                runtime=runtime,
                human_color=human_color,
                search=search,
                simulations=simulations,
                board=board,
                seed=int(self._rng.integers(0, 2**32, dtype=np.uint32)),
            )
            self._sessions[game_id] = session
            if session.ai_player == board.current_player:
                self._play_ai(session)
            return self._serialize(session)

    def _require(self, game_id: str) -> GameSession:
        try:
            return self._sessions[game_id]
        except KeyError as error:
            raise GameNotFoundError(game_id) from error

    def get_game(self, game_id: str) -> dict[str, object]:
        with self._lock:
            return self._serialize(self._require(game_id))

    def play_move(self, game_id: str, *, row: int, column: int) -> dict[str, object]:
        with self._lock:
            session = self._require(game_id)
            ended, _ = session.board.game_end()
            if ended:
                raise GameConflictError("game is already finished")
            if session.board.current_player != session.human_player:
                raise GameConflictError("it is not the human turn")
            move = browser_to_move(
                row,
                column,
                width=session.board.width,
                height=session.board.height,
            )
            if move not in session.board.availables:
                raise InvalidMoveError(f"point ({row}, {column}) is occupied")
            self._apply_move(session, move)
            ended, _ = session.board.game_end()
            if not ended:
                self._play_ai(session)
            return self._serialize(session)

    def _apply_move(self, session: GameSession, move: int) -> None:
        player = int(session.board.current_player)
        row, column = move_to_browser(
            move,
            width=session.board.width,
            height=session.board.height,
        )
        session.board.do_move(move)
        session.moves.append(
            {
                "move": int(move),
                "row": row,
                "column": column,
                "color": PLAYER_TO_COLOR[player],
                "ply": len(session.moves) + 1,
            }
        )

    def _play_ai(self, session: GameSession) -> None:
        started = time.perf_counter()
        generator = np.random.default_rng(session.seed + len(session.moves))
        if session.search == "gumbel":
            player = GumbelMCTSPlayer(
                session.runtime.policy_value_fn,
                simulations=session.simulations,
                max_considered_actions=min(16, len(session.board.availables)),
                rng=generator,
            )
        else:
            player = MCTSPlayer(
                session.runtime.policy_value_fn,
                c_puct=5.0,
                n_playout=session.simulations,
                is_selfplay=0,
                rng=generator,
            )
        move = player.get_action(session.board)
        if move is None:
            raise RuntimeError("search did not return a legal move")
        self._apply_move(session, int(move))
        session.latency_ms = (time.perf_counter() - started) * 1000

    def undo(self, game_id: str) -> dict[str, object]:
        with self._lock:
            session = self._require(game_id)
            if not session.moves:
                raise GameConflictError("there is nothing to undo")
            if len(session.moves) == 1 and session.moves[0]["color"] != session.human_color:
                raise GameConflictError("there is nothing to undo")
            remove_count = 2 if session.moves[-1]["color"] != session.human_color else 1
            session.moves = session.moves[:-remove_count]
            self._rebuild_board(session)
            session.latency_ms = 0.0
            return self._serialize(session)

    def _rebuild_board(self, session: GameSession) -> None:
        moves = [int(record["move"]) for record in session.moves]
        session.board.init_board(start_player=0)
        for move in moves:
            session.board.do_move(move)

    def replay(self, game_id: str) -> list[dict[str, object]]:
        with self._lock:
            session = self._require(game_id)
            board = Board(
                width=session.board.width,
                height=session.board.height,
                n_in_row=session.board.n_in_row,
            )
            board.init_board()
            frames = [self._serialize_frame(board, [])]
            replay_moves: list[dict[str, int | str]] = []
            for record in session.moves:
                board.do_move(int(record["move"]))
                replay_moves.append(copy.deepcopy(record))
                frames.append(self._serialize_frame(board, replay_moves))
            return frames

    @staticmethod
    def _status(board: Board) -> tuple[str, Color | None]:
        ended, winner = board.game_end()
        if not ended:
            return "playing", None
        if winner == -1:
            return "draw", None
        color = PLAYER_TO_COLOR[int(winner)]
        return f"{color}_won", color

    def _analysis(self, session: GameSession) -> dict[str, object] | None:
        status, _ = self._status(session.board)
        if status != "playing":
            return None
        priors, value = session.runtime.policy_value_fn(session.board)
        policy = np.zeros(session.board.width * session.board.height, dtype=np.float64)
        for move, probability in priors:
            if int(move) in session.board.availables:
                policy[int(move)] = max(0.0, float(probability))
        total = float(policy.sum())
        if total <= 0:
            policy[session.board.availables] = 1.0 / len(session.board.availables)
        else:
            policy /= total
        browser_policy = np.zeros_like(policy)
        for move, probability in enumerate(policy):
            row, column = move_to_browser(
                move,
                width=session.board.width,
                height=session.board.height,
            )
            browser_policy[row * session.board.width + column] = probability
        return {
            "color": PLAYER_TO_COLOR[int(session.board.current_player)],
            "value": float(np.clip(value, -1.0, 1.0)),
            "policy": browser_policy.tolist(),
            "latency_ms": round(session.latency_ms, 2),
        }

    def _serialize_frame(
        self,
        board: Board,
        moves: list[dict[str, int | str]],
    ) -> dict[str, object]:
        status, winner = self._status(board)
        return {
            "status": status,
            "winner": winner,
            "current_color": PLAYER_TO_COLOR[int(board.current_player)],
            "moves": copy.deepcopy(moves),
            "last_move": copy.deepcopy(moves[-1]) if moves else None,
            "move_count": len(moves),
        }

    def _serialize(self, session: GameSession) -> dict[str, object]:
        frame = self._serialize_frame(session.board, session.moves)
        can_undo = bool(session.moves) and not (
            len(session.moves) == 1 and session.moves[0]["color"] != session.human_color
        )
        return {
            "id": session.id,
            "width": session.board.width,
            "height": session.board.height,
            "n_in_row": session.board.n_in_row,
            **frame,
            "human_color": session.human_color,
            "ai_color": PLAYER_TO_COLOR[session.ai_player],
            "model": session.descriptor.to_dict(),
            "search": session.search,
            "simulations": session.simulations,
            "can_undo": can_undo,
            "analysis": self._analysis(session),
        }

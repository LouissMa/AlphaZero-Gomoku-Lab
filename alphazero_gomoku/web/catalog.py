"""Allowlisted model discovery and lazy policy runtime loading."""

from __future__ import annotations

import copy
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from alphazero_gomoku.policy_value_net_numpy import PolicyValueNetNumpy


class ModelNotFoundError(LookupError):
    """Raised when a client requests a model outside the public catalog."""

    def __init__(self, model_id: str) -> None:
        super().__init__(f"unknown model: {model_id}")


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    id: str
    name: str
    width: int
    height: int
    n_in_row: int
    kind: str
    relative_path: str | None = None

    def to_dict(self) -> dict[str, str | int]:
        public = asdict(self)
        public.pop("relative_path")
        return public


class PolicyRuntime(Protocol):
    descriptor: ModelDescriptor

    def policy_value_fn(self, board: object) -> tuple[object, float]: ...


@dataclass(slots=True)
class NumpyPolicyRuntime:
    descriptor: ModelDescriptor
    network: PolicyValueNetNumpy

    def policy_value_fn(self, board: object) -> tuple[object, float]:
        return self.network.policy_value_fn(board)


@dataclass(frozen=True, slots=True)
class HeuristicPolicyRuntime:
    descriptor: ModelDescriptor

    @staticmethod
    def _winning_moves(board: object, player: int) -> set[int]:
        winning: set[int] = set()
        for move in board.availables:
            candidate = copy.deepcopy(board)
            candidate.current_player = player
            candidate.do_move(move)
            won, winner = candidate.has_a_winner()
            if won and winner == player:
                winning.add(int(move))
        return winning

    def policy_value_fn(self, board: object) -> tuple[list[tuple[int, float]], float]:
        legal = [int(move) for move in board.availables]
        if not legal:
            return [], 0.0
        current = int(board.current_player)
        opponent = board.players[0] if current == board.players[1] else board.players[1]
        winning = self._winning_moves(board, current)
        blocking = self._winning_moves(board, opponent)
        center_row = (board.height - 1) / 2
        center_column = (board.width - 1) / 2
        weights = np.asarray(
            [
                100.0
                if move in winning
                else 25.0
                if move in blocking
                else 1.0
                / (
                    1.0
                    + abs(move // board.width - center_row)
                    + abs(move % board.width - center_column)
                )
                for move in legal
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        value = 0.8 if winning else 0.2 if blocking else 0.0
        return list(zip(legal, weights.tolist(), strict=True)), value


class ModelCatalog:
    """Fixed public model catalog rooted at one repository/application directory."""

    _MODELS = (
        ModelDescriptor(
            id="heuristic-6x6",
            name="Tactical heuristic · 6×6",
            width=6,
            height=6,
            n_in_row=4,
            kind="heuristic",
        ),
        ModelDescriptor(
            id="alphazero-6x6",
            name="AlphaZero policy · 6×6",
            width=6,
            height=6,
            n_in_row=4,
            kind="numpy",
            relative_path="models/best_policy_6_6_4.model",
        ),
        ModelDescriptor(
            id="alphazero-8x8",
            name="AlphaZero policy · 8×8",
            width=8,
            height=8,
            n_in_row=5,
            kind="numpy",
            relative_path="models/best_policy_8_8_5.model",
        ),
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._by_id = {model.id: model for model in self._MODELS}
        self._cache: dict[str, PolicyRuntime] = {}

    def list_models(self) -> tuple[ModelDescriptor, ...]:
        return self._MODELS

    def load(self, model_id: str) -> PolicyRuntime:
        try:
            descriptor = self._by_id[model_id]
        except KeyError as error:
            raise ModelNotFoundError(model_id) from error
        cached = self._cache.get(model_id)
        if cached is not None:
            return cached
        if descriptor.kind == "heuristic":
            runtime: PolicyRuntime = HeuristicPolicyRuntime(descriptor)
        else:
            model_path = (self.root / str(descriptor.relative_path)).resolve()
            if self.root not in model_path.parents or not model_path.is_file():
                raise FileNotFoundError(f"bundled model is missing: {descriptor.relative_path}")
            data = model_path.read_bytes()
            try:
                parameters = pickle.loads(data)
            except UnicodeDecodeError:
                parameters = pickle.loads(data, encoding="bytes")
            runtime = NumpyPolicyRuntime(
                descriptor,
                PolicyValueNetNumpy(descriptor.width, descriptor.height, parameters),
            )
        self._cache[model_id] = runtime
        return runtime

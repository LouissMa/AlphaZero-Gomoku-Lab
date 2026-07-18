"""Alternating-color, reproducible evaluation arena."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from alphazero_gomoku.game import Board

from .config import ArenaConfig
from .players import PlayerFactory
from .stats import ScoreSummary

ARENA_REPORT_VERSION = 1


@dataclass(frozen=True, slots=True)
class GameRecord:
    index: int
    seed: int
    candidate_player: int
    first_player: str
    winner: str
    result: str
    moves: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class MatchReport:
    opponent: str
    summary: ScoreSummary
    games: tuple[GameRecord, ...]


@dataclass(frozen=True, slots=True)
class TournamentReport:
    format_version: int
    timestamp: str
    candidate: str
    config: dict[str, object]
    matches: tuple[MatchReport, ...]
    overall: ScoreSummary
    promotion: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        candidate: str,
        config: ArenaConfig,
        matches: tuple[MatchReport, ...],
    ) -> TournamentReport:
        all_results = [game.result for match in matches for game in match.games]
        return cls(
            format_version=ARENA_REPORT_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            candidate=candidate,
            config=config.to_dict(),
            matches=matches,
            overall=ScoreSummary.from_results(
                all_results,
                confidence=config.confidence,
            ),
        )


def play_game(
    candidate_factory: PlayerFactory,
    opponent_factory: PlayerFactory,
    config: ArenaConfig,
    *,
    index: int,
    seed: int,
) -> GameRecord:
    started = perf_counter()
    candidate_first = index % 2 == 0
    first_factory = candidate_factory if candidate_first else opponent_factory
    second_factory = opponent_factory if candidate_first else candidate_factory
    first = first_factory.create(seed)
    second = second_factory.create(seed ^ 0x9E3779B97F4A7C15)
    board = Board(
        width=config.board.width,
        height=config.board.height,
        n_in_row=config.board.n_in_row,
    )
    board.init_board(start_player=0)
    first.set_player_ind(board.players[0])
    second.set_player_ind(board.players[1])
    players = {board.players[0]: first, board.players[1]: second}
    candidate_player = board.players[0] if candidate_first else board.players[1]
    moves = 0

    while True:
        current = board.get_current_player()
        move = int(players[current].get_action(board))
        if move not in board.availables:
            raise ValueError(f"{current} selected illegal move {move}")
        board.do_move(move)
        moves += 1
        ended, winner = board.game_end()
        if ended:
            break

    if winner == -1:
        result = "draw"
        winner_name = "draw"
    elif winner == candidate_player:
        result = "win"
        winner_name = candidate_factory.name
    else:
        result = "loss"
        winner_name = opponent_factory.name
    return GameRecord(
        index=index,
        seed=seed,
        candidate_player=candidate_player,
        first_player=first_factory.name,
        winner=winner_name,
        result=result,
        moves=moves,
        elapsed_seconds=perf_counter() - started,
    )


def run_match(
    candidate_factory: PlayerFactory,
    opponent_factory: PlayerFactory,
    config: ArenaConfig,
    rng: np.random.Generator,
) -> MatchReport:
    seeds = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=config.games_per_opponent,
        dtype=np.uint64,
    )
    games = tuple(
        play_game(
            candidate_factory,
            opponent_factory,
            config,
            index=index,
            seed=int(seed),
        )
        for index, seed in enumerate(seeds)
    )
    summary = ScoreSummary.from_results(
        [game.result for game in games],
        confidence=config.confidence,
    )
    return MatchReport(opponent=opponent_factory.name, summary=summary, games=games)


def run_tournament(
    candidate_factory: PlayerFactory,
    opponent_factories: list[PlayerFactory],
    config: ArenaConfig,
) -> TournamentReport:
    if not opponent_factories:
        raise ValueError("at least one opponent is required")
    rng = np.random.default_rng(config.seed)
    matches = tuple(
        run_match(candidate_factory, opponent, config, rng) for opponent in opponent_factories
    )
    return TournamentReport.create(candidate_factory.name, config, matches)


def write_tournament_report(report: TournamentReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination

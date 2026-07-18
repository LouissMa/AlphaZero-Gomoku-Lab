"""Statistical summaries for head-to-head arena results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    confidence: float
    confidence_low: float
    confidence_high: float
    elo_difference: float

    @classmethod
    def from_results(
        cls,
        results: list[str] | tuple[str, ...],
        *,
        confidence: float,
    ) -> ScoreSummary:
        if not results:
            raise ValueError("at least one arena result is required")
        allowed = {"win", "draw", "loss"}
        unknown = set(results) - allowed
        if unknown:
            raise ValueError(f"unknown results: {sorted(unknown)}")
        wins = results.count("win")
        draws = results.count("draw")
        losses = results.count("loss")
        games = len(results)
        score = (wins + 0.5 * draws) / games
        low, high = wilson_interval(wins + 0.5 * draws, games, confidence)
        elo = elo_difference(score, games)
        return cls(
            games=games,
            wins=wins,
            draws=draws,
            losses=losses,
            score=score,
            confidence=confidence,
            confidence_low=low,
            confidence_high=high,
            elo_difference=elo,
        )


def wilson_interval(successes: float, games: int, confidence: float) -> tuple[float, float]:
    if games <= 0:
        raise ValueError("games must be positive")
    if not 0 <= successes <= games:
        raise ValueError("successes must be within the game count")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    probability = successes / games
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denominator = 1 + z * z / games
    center = (probability + z * z / (2 * games)) / denominator
    margin = (
        z
        * math.sqrt(probability * (1 - probability) / games + z * z / (4 * games * games))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def elo_difference(score: float, games: int) -> float:
    if games <= 0:
        raise ValueError("games must be positive")
    clipped = min(max(score, 0.5 / games), 1 - 0.5 / games)
    return 400 * math.log10(clipped / (1 - clipped))

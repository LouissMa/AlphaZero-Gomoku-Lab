"""Confidence-gated automatic model promotion."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .stats import ScoreSummary


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    reason: str
    score: float
    confidence_low: float
    required_score: float
    required_lower_bound: float


def decide_promotion(
    summary: ScoreSummary,
    *,
    required_score: float,
    required_lower_bound: float,
) -> PromotionDecision:
    promoted = summary.score >= required_score and summary.confidence_low >= required_lower_bound
    if promoted:
        reason = "candidate passed score and confidence thresholds"
    elif summary.score < required_score:
        reason = "candidate score is below the promotion threshold"
    else:
        reason = "confidence lower bound is below the promotion threshold"
    return PromotionDecision(
        promoted=promoted,
        reason=reason,
        score=summary.score,
        confidence_low=summary.confidence_low,
        required_score=required_score,
        required_lower_bound=required_lower_bound,
    )


def promote_model(
    candidate_model: str | Path,
    destination: str | Path,
    decision: PromotionDecision,
) -> Path | None:
    if not decision.promoted:
        return None
    source = Path(candidate_model)
    target = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(f"candidate model does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    os.replace(temporary, target)
    return target

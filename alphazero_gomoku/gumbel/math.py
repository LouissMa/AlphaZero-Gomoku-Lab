"""Numerical primitives used by Gumbel AlphaZero."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GumbelTopK:
    actions: np.ndarray
    logits: np.ndarray
    gumbels: np.ndarray


@dataclass(frozen=True, slots=True)
class CompletedQValues:
    raw: np.ndarray
    transformed: np.ndarray
    completion_value: float


def sequential_halving_schedule(
    max_considered_actions: int,
    simulations: int,
) -> tuple[int, ...]:
    if max_considered_actions <= 0 or simulations <= 0:
        raise ValueError("action count and simulations must be positive")
    if max_considered_actions == 1:
        return tuple(range(simulations))
    rounds = int(math.ceil(math.log2(max_considered_actions)))
    schedule: list[int] = []
    visits = [0] * max_considered_actions
    considered = max_considered_actions
    while len(schedule) < simulations:
        extra = max(1, simulations // (rounds * considered))
        for _ in range(extra):
            schedule.extend(visits[:considered])
        for index in range(considered):
            visits[index] += 1
        considered = max(2, considered // 2)
    return tuple(schedule[:simulations])


def sample_gumbel_top_k(
    actions: np.ndarray,
    prior_probs: np.ndarray,
    max_considered_actions: int,
    rng: np.random.Generator,
    *,
    gumbel_scale: float = 1.0,
) -> GumbelTopK:
    actions = np.asarray(actions, dtype=np.int64)
    prior_probs = np.asarray(prior_probs, dtype=np.float64)
    if actions.size == 0:
        raise ValueError("at least one action is required")
    if actions.shape != prior_probs.shape:
        raise ValueError("actions and prior probabilities must have the same shape")
    if max_considered_actions <= 0:
        raise ValueError("max_considered_actions must be positive")
    if gumbel_scale < 0:
        raise ValueError("gumbel_scale must be non-negative")
    if np.any(prior_probs < 0) or float(prior_probs.sum()) <= 0:
        raise ValueError("prior probabilities must have positive mass")
    probabilities = prior_probs / prior_probs.sum()
    logits = np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))
    gumbels = gumbel_scale * rng.gumbel(size=actions.size)
    count = min(max_considered_actions, actions.size)
    indices = np.argsort(logits + gumbels)[-count:][::-1]
    return GumbelTopK(
        actions=actions[indices],
        logits=logits[indices],
        gumbels=gumbels[indices],
    )


def completed_qvalues(
    q_values: np.ndarray,
    visit_counts: np.ndarray,
    raw_value: float,
    prior_probs: np.ndarray,
    *,
    value_scale: float = 0.1,
    visit_offset: float = 50.0,
    use_mixed_value: bool = True,
    epsilon: float = 1e-8,
) -> CompletedQValues:
    q_values = np.asarray(q_values, dtype=np.float64)
    visits = np.asarray(visit_counts, dtype=np.int64)
    priors = np.asarray(prior_probs, dtype=np.float64)
    if q_values.shape != visits.shape or q_values.shape != priors.shape:
        raise ValueError("Q values, visits, and priors must have the same shape")
    if q_values.size == 0:
        raise ValueError("at least one Q value is required")
    if np.any(visits < 0):
        raise ValueError("visit counts must be non-negative")
    if value_scale <= 0 or visit_offset < 0:
        raise ValueError("value scale must be positive and visit offset non-negative")

    visited = visits > 0
    total_visits = int(visits.sum())
    completion = float(raw_value)
    if use_mixed_value and visited.any():
        visited_mass = float(priors[visited].sum())
        if visited_mass > 0:
            weighted_q = float(np.sum(priors[visited] * q_values[visited]) / visited_mass)
            completion = (float(raw_value) + total_visits * weighted_q) / (total_visits + 1)
    raw = np.where(visited, q_values, completion)
    minimum = float(raw.min())
    maximum = float(raw.max())
    normalized = (raw - minimum) / max(maximum - minimum, epsilon)
    transformed = (visit_offset + int(visits.max())) * value_scale * normalized
    return CompletedQValues(
        raw=raw,
        transformed=transformed,
        completion_value=completion,
    )


def improved_policy(
    prior_probs: np.ndarray,
    transformed_qvalues: np.ndarray,
    legal_mask: np.ndarray | None = None,
) -> np.ndarray:
    priors = np.asarray(prior_probs, dtype=np.float64)
    qvalues = np.asarray(transformed_qvalues, dtype=np.float64)
    if priors.shape != qvalues.shape or priors.size == 0:
        raise ValueError("priors and Q values must have the same non-empty shape")
    legal = np.ones(priors.shape, dtype=bool) if legal_mask is None else np.asarray(legal_mask)
    if legal.shape != priors.shape or not legal.any():
        raise ValueError("legal mask must match priors and contain a legal action")
    if np.any(priors < 0) or float(priors[legal].sum()) <= 0:
        raise ValueError("legal prior probabilities must have positive mass")
    logits = np.full(priors.shape, -np.inf, dtype=np.float64)
    logits[legal] = np.log(np.maximum(priors[legal], np.finfo(np.float64).tiny)) + qvalues[legal]
    maximum = float(np.max(logits[legal]))
    weights = np.zeros(priors.shape, dtype=np.float64)
    weights[legal] = np.exp(logits[legal] - maximum)
    return weights / weights.sum()


def deterministic_action_selection(
    prior_probs: np.ndarray,
    transformed_qvalues: np.ndarray,
    visit_counts: np.ndarray,
) -> int:
    """Choose visits whose frequencies converge to the improved policy."""
    visits = np.asarray(visit_counts, dtype=np.int64)
    if np.any(visits < 0):
        raise ValueError("visit counts must be non-negative")
    policy = improved_policy(prior_probs, transformed_qvalues)
    if policy.shape != visits.shape:
        raise ValueError("visit counts must match the policy shape")
    scores = policy - visits / (1 + visits.sum())
    return int(np.argmax(scores))

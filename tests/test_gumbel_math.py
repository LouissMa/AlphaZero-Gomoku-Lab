from __future__ import annotations

import numpy as np
import pytest

from alphazero_gomoku.gumbel.math import (
    completed_qvalues,
    deterministic_action_selection,
    improved_policy,
    sample_gumbel_top_k,
    sequential_halving_schedule,
)


def test_sequential_halving_schedule_spends_exact_budget() -> None:
    schedule = sequential_halving_schedule(max_considered_actions=8, simulations=24)

    assert len(schedule) == 24
    assert schedule[:8] == (0,) * 8
    assert max(schedule) >= 2


def test_sequential_halving_schedule_handles_one_action() -> None:
    assert sequential_halving_schedule(1, 4) == (0, 1, 2, 3)


def test_sample_gumbel_top_k_is_unique_and_reproducible() -> None:
    actions = np.array([2, 4, 6, 8])
    priors = np.array([0.1, 0.2, 0.3, 0.4])

    first = sample_gumbel_top_k(actions, priors, 3, np.random.default_rng(7))
    second = sample_gumbel_top_k(actions, priors, 3, np.random.default_rng(7))

    assert first.actions.shape == (3,)
    assert len(set(first.actions.tolist())) == 3
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_allclose(first.gumbels, second.gumbels)


def test_completed_qvalues_use_mixed_value_for_unvisited_actions() -> None:
    completed = completed_qvalues(
        q_values=np.array([0.8, -0.2, 0.0]),
        visit_counts=np.array([2, 1, 0]),
        raw_value=0.1,
        prior_probs=np.array([0.5, 0.25, 0.25]),
        value_scale=0.1,
        visit_offset=50.0,
        use_mixed_value=True,
    )

    expected_mixed = (0.1 + 3 * ((2 / 3) * 0.8 + (1 / 3) * -0.2)) / 4
    assert completed.raw[2] == pytest.approx(expected_mixed)
    assert completed.transformed.min() == pytest.approx(0.0)
    assert completed.transformed.max() == pytest.approx(5.2)


def test_improved_policy_is_finite_normalized_and_masks_illegal_actions() -> None:
    target = improved_policy(
        prior_probs=np.array([0.2, 0.3, 0.5, 0.0]),
        transformed_qvalues=np.array([0.0, 1.0, -1.0, 100.0]),
        legal_mask=np.array([True, True, True, False]),
    )

    assert np.isfinite(target).all()
    assert target.sum() == pytest.approx(1.0)
    assert target[3] == 0.0
    assert target[1] > target[0] > target[2]


def test_deterministic_action_selection_balances_improved_policy_and_visits() -> None:
    action = deterministic_action_selection(
        prior_probs=np.array([0.7, 0.3]),
        transformed_qvalues=np.array([0.0, 0.0]),
        visit_counts=np.array([2, 0]),
    )

    assert action == 1


@pytest.mark.parametrize(
    ("actions", "priors", "message"),
    [
        (np.array([], dtype=int), np.array([], dtype=float), "at least one"),
        (np.array([1, 2]), np.array([1.0]), "same shape"),
        (np.array([1, 2]), np.array([0.0, 0.0]), "positive mass"),
    ],
)
def test_sample_gumbel_top_k_validates_inputs(actions, priors, message) -> None:
    with pytest.raises(ValueError, match=message):
        sample_gumbel_top_k(actions, priors, 1, np.random.default_rng(1))

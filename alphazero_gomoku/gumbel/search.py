"""Gumbel AlphaZero tree search with Sequential Halving at the root."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .math import (
    completed_qvalues,
    deterministic_action_selection,
    improved_policy,
    sample_gumbel_top_k,
)

PolicyValueFunction = Callable[[Any], tuple[Iterable[tuple[int, float]], float]]


@dataclass(slots=True)
class SearchNode:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    raw_value: float = 0.0
    children: dict[int, SearchNode] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def update(self, value: float) -> None:
        self.visits += 1
        self.value_sum += value


@dataclass(frozen=True, slots=True)
class SearchResult:
    action: int
    policy: np.ndarray
    simulations_used: int
    considered_actions: tuple[int, ...]
    root_visits: dict[int, int]


class GumbelSearch:
    def __init__(
        self,
        policy_value_fn: PolicyValueFunction,
        *,
        simulations: int,
        max_considered_actions: int = 16,
        gumbel_scale: float = 1.0,
        q_value_scale: float = 0.1,
        q_visit_offset: float = 50.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if simulations <= 0 or max_considered_actions <= 0:
            raise ValueError("simulations and max_considered_actions must be positive")
        if gumbel_scale < 0 or q_value_scale <= 0 or q_visit_offset < 0:
            raise ValueError("Gumbel and Q transformation settings are invalid")
        self.policy_value_fn = policy_value_fn
        self.simulations = simulations
        self.max_considered_actions = max_considered_actions
        self.gumbel_scale = gumbel_scale
        self.q_value_scale = q_value_scale
        self.q_visit_offset = q_visit_offset
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root = SearchNode(prior=1.0)
        self.root_reuse_count = 0
        self.root_reset_count = 0

    @staticmethod
    def _terminal_value(board: Any, winner: int) -> float:
        if winner == -1:
            return 0.0
        return 1.0 if winner == board.get_current_player() else -1.0

    def _evaluate_and_expand(self, node: SearchNode, board: Any) -> float:
        action_priors, value = self.policy_value_fn(board)
        pairs = [(int(action), float(probability)) for action, probability in action_priors]
        legal = set(board.availables)
        pairs = [(action, probability) for action, probability in pairs if action in legal]
        if not pairs:
            raise ValueError("policy did not return legal actions")
        probabilities = np.asarray([max(0.0, pair[1]) for pair in pairs], dtype=np.float64)
        if probabilities.sum() <= 0:
            raise ValueError("policy probabilities must have positive mass")
        probabilities /= probabilities.sum()
        node.raw_value = float(value)
        for (action, _), probability in zip(pairs, probabilities, strict=True):
            node.children.setdefault(action, SearchNode(prior=float(probability)))
        return node.raw_value

    def _completed_for(self, node: SearchNode) -> tuple[list[int], np.ndarray, np.ndarray]:
        actions = list(node.children)
        children = [node.children[action] for action in actions]
        priors = np.asarray([child.prior for child in children], dtype=np.float64)
        values = np.asarray([child.q_value for child in children], dtype=np.float64)
        visits = np.asarray([child.visits for child in children], dtype=np.int64)
        completed = completed_qvalues(
            values,
            visits,
            node.raw_value,
            priors,
            value_scale=self.q_value_scale,
            visit_offset=self.q_visit_offset,
        )
        return actions, priors, completed.transformed

    def _select_interior_action(self, node: SearchNode) -> int:
        actions, priors, transformed = self._completed_for(node)
        visits = np.asarray([node.children[action].visits for action in actions])
        index = deterministic_action_selection(priors, transformed, visits)
        return int(actions[index])

    def _simulate(self, board: Any, root_action: int) -> None:
        state = copy.deepcopy(board)
        node = self.root
        action = root_action
        path: list[SearchNode] = []
        while True:
            child = node.children[action]
            state.do_move(action)
            path.append(child)
            ended, winner = state.game_end()
            if ended:
                leaf_value = self._terminal_value(state, winner)
                break
            node = child
            if not node.children:
                leaf_value = self._evaluate_and_expand(node, state)
                break
            action = self._select_interior_action(node)

        value = -float(leaf_value)
        for visited_node in reversed(path):
            visited_node.update(value)
            value = -value

    def _root_scores(
        self,
        considered_actions: np.ndarray,
        logits: np.ndarray,
        gumbels: np.ndarray,
    ) -> np.ndarray:
        actions, _, transformed = self._completed_for(self.root)
        transformed_by_action = dict(zip(actions, transformed, strict=True))
        return (
            logits
            + gumbels
            + np.asarray([transformed_by_action[int(action)] for action in considered_actions])
        )

    def search(self, board: Any) -> SearchResult:
        if not board.availables:
            raise ValueError("search requires legal actions")
        simulations_used = 0
        if not self.root.children:
            self._evaluate_and_expand(self.root, board)
            simulations_used = 1

        legal_actions = np.asarray(list(self.root.children), dtype=np.int64)
        legal_priors = np.asarray(
            [self.root.children[int(action)].prior for action in legal_actions],
            dtype=np.float64,
        )
        top_k = sample_gumbel_top_k(
            legal_actions,
            legal_priors,
            self.max_considered_actions,
            self.rng,
            gumbel_scale=self.gumbel_scale,
        )
        active = np.arange(top_k.actions.size)
        remaining = self.simulations - simulations_used
        while remaining > 0 and active.size > 1:
            rounds_left = max(1, int(math.ceil(math.log2(active.size))))
            visits_each = max(1, remaining // (rounds_left * active.size))
            for index in active:
                for _ in range(visits_each):
                    if remaining <= 0:
                        break
                    self._simulate(board, int(top_k.actions[index]))
                    simulations_used += 1
                    remaining -= 1
            scores = self._root_scores(top_k.actions, top_k.logits, top_k.gumbels)
            survivor_count = max(1, active.size // 2)
            active = active[np.argsort(scores[active])[-survivor_count:]]
        while remaining > 0:
            scores = self._root_scores(top_k.actions, top_k.logits, top_k.gumbels)
            winner_index = int(active[np.argmax(scores[active])])
            self._simulate(board, int(top_k.actions[winner_index]))
            simulations_used += 1
            remaining -= 1

        scores = self._root_scores(top_k.actions, top_k.logits, top_k.gumbels)
        action = int(top_k.actions[int(active[np.argmax(scores[active])])])
        actions, priors, transformed = self._completed_for(self.root)
        legal_target = improved_policy(priors, transformed)
        policy = np.zeros(board.width * board.height, dtype=np.float64)
        policy[actions] = legal_target
        root_visits = {candidate: self.root.children[candidate].visits for candidate in actions}
        return SearchResult(
            action=action,
            policy=policy,
            simulations_used=simulations_used,
            considered_actions=tuple(int(item) for item in top_k.actions),
            root_visits=root_visits,
        )

    def update_with_move(self, move: int) -> None:
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root_reuse_count += 1
        else:
            self.root = SearchNode(prior=1.0)
            self.root_reset_count += 1

    def reset(self) -> None:
        self.update_with_move(-1)

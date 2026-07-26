"""Player adapter for Gumbel AlphaZero search."""

from __future__ import annotations

from typing import Any

import numpy as np

from .search import GumbelSearch, PolicyValueFunction


class GumbelMCTSPlayer:
    def __init__(
        self,
        policy_value_fn: PolicyValueFunction,
        *,
        simulations: int,
        max_considered_actions: int = 16,
        gumbel_scale: float = 1.0,
        q_value_scale: float = 0.1,
        q_visit_offset: float = 50.0,
        is_selfplay: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.is_selfplay = is_selfplay
        self.player = 0
        self.search_engine = GumbelSearch(
            policy_value_fn,
            simulations=simulations,
            max_considered_actions=max_considered_actions,
            gumbel_scale=gumbel_scale,
            q_value_scale=q_value_scale,
            q_visit_offset=q_visit_offset,
            rng=rng,
        )

    def set_player_ind(self, player: int) -> None:
        self.player = player

    def reset_player(self) -> None:
        self.search_engine.reset()

    def get_action(
        self,
        board: Any,
        temp: float = 1e-3,
        return_prob: bool = False,
    ) -> int | tuple[int, np.ndarray]:
        del temp
        result = self.search_engine.search(board)
        if self.is_selfplay:
            self.search_engine.update_with_move(result.action)
        else:
            self.search_engine.reset()
        if return_prob:
            return result.action, result.policy
        return result.action

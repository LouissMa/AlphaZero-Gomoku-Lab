"""End-to-end reproducible AlphaZero training orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from alphazero_gomoku.policy_value_net_pytorch import NetworkConfig, PolicyValueNet

from .checkpoint import CheckpointManager, TrainingState
from .config import ExperimentConfig
from .metrics import JsonlMetricsWriter
from .replay_buffer import ReplayBuffer
from .reproducibility import seed_everything
from .self_play import generate_self_play_games

IterationCallback = Callable[[TrainingState, dict[str, Any]], None]


class AlphaZeroTrainer:
    """Coordinate self-play, optimization, metrics, and resumable snapshots."""

    def __init__(
        self,
        config: ExperimentConfig,
        network: PolicyValueNet,
        replay: ReplayBuffer,
        state: TrainingState,
        random_generator: np.random.Generator,
        *,
        run_root: str | Path | None = None,
    ) -> None:
        self.config = config
        self.network = network
        self.replay = replay
        self.state = state
        self.random_generator = random_generator
        self.run_root = (
            Path(run_root)
            if run_root is not None
            else Path(config.run.output_dir) / config.run.name
        )
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.checkpoints = CheckpointManager(self.run_root / "checkpoints")
        self.metrics = JsonlMetricsWriter(self.run_root / "metrics.jsonl")

    @classmethod
    def create(cls, config: ExperimentConfig) -> AlphaZeroTrainer:
        """Create a new experiment from a validated configuration."""
        run_root = Path(config.run.output_dir) / config.run.name
        if (run_root / "metrics.jsonl").exists() or (run_root / "checkpoints" / "latest").exists():
            raise FileExistsError(
                f"experiment already exists at {run_root}; resume its latest checkpoint"
            )

        random_context = seed_everything(
            config.run.seed,
            deterministic=config.run.deterministic,
        )
        architecture = NetworkConfig(
            board_width=config.board.width,
            board_height=config.board.height,
            channels=config.network.channels,
            residual_blocks=config.network.residual_blocks,
            policy_channels=config.network.policy_channels,
            value_channels=config.network.value_channels,
            value_hidden_size=config.network.value_hidden_size,
        )
        network = PolicyValueNet(
            config.board.width,
            config.board.height,
            config=architecture,
            device=config.network.device,
            learning_rate=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
            gradient_clip_norm=config.optimization.gradient_clip_norm,
            amp=config.network.amp,
            compile_model=config.network.compile_model,
            compile_backend=config.network.compile_backend,
        )
        return cls(
            config=config,
            network=network,
            replay=ReplayBuffer(config.optimization.replay_capacity),
            state=TrainingState(),
            random_generator=random_context.numpy,
            run_root=run_root,
        )

    @classmethod
    def resume(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "auto",
    ) -> AlphaZeroTrainer:
        """Restore a complete experiment snapshot and continue in its run directory."""
        checkpoint_path = Path(checkpoint)
        manager = CheckpointManager(checkpoint_path.parent)
        bundle = manager.load(checkpoint_path, device=device)
        return cls(
            config=bundle.config,
            network=bundle.network,
            replay=bundle.replay,
            state=bundle.state,
            random_generator=bundle.random_generator,
            run_root=checkpoint_path.parent.parent,
        )

    def _collect_self_play(self) -> tuple[int, int, dict[int, int], dict[str, Any]]:
        positions = 0
        winners: dict[int, int] = {}
        batch = generate_self_play_games(
            self.network,
            self.config.board,
            self.config.self_play,
            self.random_generator,
        )
        for result in batch.results:
            self.replay.extend(result.samples)
            positions += len(result.samples)
            winners[result.winner] = winners.get(result.winner, 0) + 1

        profile = batch.profile
        performance: dict[str, Any] = {
            "self_play_seconds": profile.elapsed_seconds,
            "positions_per_second": profile.positions_per_second,
            "simulations": profile.simulations,
            "simulations_per_second": profile.simulations_per_second,
            "tree_reuse_count": profile.tree_reuse_count,
            "tree_reset_count": profile.tree_reset_count,
            "inference_requests": profile.inference.requests,
            "inference_batches": profile.inference.batches,
            "mean_inference_batch_size": profile.inference.mean_batch_size,
            "inference_batch_utilization": profile.inference.mean_batch_size
            / self.config.self_play.inference_batch_size,
            "max_inference_batch_size": profile.inference.max_batch_size,
            "inference_seconds": profile.inference.inference_seconds,
            "parallel_games": min(
                self.config.self_play.parallel_games,
                self.config.self_play.games_per_iteration,
            ),
        }
        return profile.games, positions, winners, performance

    def _optimize(self) -> list[dict[str, float]]:
        if len(self.replay) < self.config.optimization.batch_size:
            return []
        updates: list[dict[str, float]] = []
        learning_rate = self.config.optimization.learning_rate * self.state.learning_rate_multiplier
        for _ in range(self.config.optimization.epochs_per_iteration):
            states, policies, outcomes = self.replay.sample(
                self.config.optimization.batch_size,
                self.random_generator,
            )
            metrics = self.network.train_batch(
                states,
                policies,
                outcomes,
                learning_rate=learning_rate,
            )
            updates.append(asdict(metrics))
        return updates

    @staticmethod
    def _mean_metrics(updates: list[dict[str, float]]) -> dict[str, float]:
        if not updates:
            return {}
        return {name: fmean(update[name] for update in updates) for name in updates[0]}

    def run(
        self,
        *,
        target_iteration: int | None = None,
        on_iteration: IterationCallback | None = None,
    ) -> TrainingState:
        """Train until an absolute iteration target and return the final state."""
        target = self.config.run.iterations if target_iteration is None else target_iteration
        if target <= 0:
            raise ValueError("target iteration must be positive")
        if target < self.state.iteration:
            raise ValueError(
                f"target iteration {target} precedes checkpoint iteration {self.state.iteration}"
            )
        if target == self.state.iteration:
            return self.state

        for iteration in range(self.state.iteration + 1, target + 1):
            games, positions, winners, performance = self._collect_self_play()
            updates = self._optimize()
            self.state = replace(
                self.state,
                iteration=iteration,
                total_games=self.state.total_games + games,
                total_positions=self.state.total_positions + positions,
            )
            record: dict[str, Any] = {
                "games": games,
                "positions": positions,
                "replay_size": len(self.replay),
                "optimizer_updates": len(updates),
                "training_step": self.network.training_step,
                **performance,
                "winners": {str(key): value for key, value in sorted(winners.items())},
                **self._mean_metrics(updates),
            }
            should_checkpoint = (
                iteration % self.config.run.checkpoint_every == 0 or iteration == target
            )
            if should_checkpoint:
                checkpoint = self.checkpoints.save(
                    self.network,
                    self.replay,
                    self.state,
                    self.config,
                    self.random_generator,
                )
                record["checkpoint"] = str(checkpoint)

            self.metrics.write(iteration, record)

            if on_iteration is not None:
                on_iteration(self.state, record)

        return self.state

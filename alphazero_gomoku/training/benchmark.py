"""Reproducible self-play profiling and benchmark reports."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from alphazero_gomoku.policy_value_net_pytorch import NetworkConfig, PolicyValueNet

from .config import ExperimentConfig
from .reproducibility import seed_everything
from .self_play import generate_self_play_games

BENCHMARK_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    format_version: int
    timestamp: str
    device: str
    hardware: dict[str, Any]
    workload: dict[str, Any]
    performance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _create_network(config: ExperimentConfig) -> PolicyValueNet:
    architecture = NetworkConfig(
        board_width=config.board.width,
        board_height=config.board.height,
        channels=config.network.channels,
        residual_blocks=config.network.residual_blocks,
        policy_channels=config.network.policy_channels,
        value_channels=config.network.value_channels,
        value_hidden_size=config.network.value_hidden_size,
    )
    return PolicyValueNet(
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


def run_self_play_benchmark(
    config: ExperimentConfig,
    *,
    games: int | None = None,
) -> BenchmarkReport:
    """Measure a deterministic self-play workload without writing training state."""
    if games is not None:
        if games <= 0:
            raise ValueError("games must be positive")
        config = replace(
            config,
            self_play=replace(config.self_play, games_per_iteration=games),
        )

    random_context = seed_everything(
        config.run.seed,
        deterministic=config.run.deterministic,
    )
    network = _create_network(config)
    if network.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(network.device)

    batch = generate_self_play_games(
        network,
        config.board,
        config.self_play,
        random_context.numpy,
    )
    profile = batch.profile
    cuda_name = (
        torch.cuda.get_device_name(network.device) if network.device.type == "cuda" else None
    )
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(network.device) / (1024 * 1024)
        if network.device.type == "cuda"
        else 0.0
    )
    return BenchmarkReport(
        format_version=BENCHMARK_FORMAT_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        device=str(network.device),
        hardware={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "cuda_device": cuda_name,
            "cuda_peak_memory_mb": peak_memory_mb,
        },
        workload={
            "board": asdict(config.board),
            "games": profile.games,
            "parallel_games": min(
                config.self_play.parallel_games,
                config.self_play.games_per_iteration,
            ),
            "simulations_per_move": config.self_play.simulations_per_move,
            "inference_batch_size": config.self_play.inference_batch_size,
            "positions": profile.positions,
            "simulations": profile.simulations,
        },
        performance={
            "elapsed_seconds": profile.elapsed_seconds,
            "positions_per_second": profile.positions_per_second,
            "simulations_per_second": profile.simulations_per_second,
            "inference_seconds": profile.inference.inference_seconds,
            "inference_batches": profile.inference.batches,
            "mean_inference_batch_size": profile.inference.mean_batch_size,
            "inference_batch_utilization": profile.inference.mean_batch_size
            / config.self_play.inference_batch_size,
            "max_inference_batch_size": profile.inference.max_batch_size,
            "tree_reuse_count": profile.tree_reuse_count,
            "tree_reset_count": profile.tree_reset_count,
        },
    )


def write_benchmark_report(report: BenchmarkReport, path: str | Path) -> Path:
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

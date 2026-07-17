"""Command-line entry points for AlphaZero Gomoku."""

from __future__ import annotations

import argparse
import importlib.util
import platform
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "0.1.0"


def _default_model_path() -> Path:
    return Path.cwd() / "models" / "best_policy_8_8_5.model"


def _doctor(_: argparse.Namespace) -> int:
    model_path = _default_model_path()
    print(f"AlphaZero Gomoku {VERSION}")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print(f"Default model: {'found' if model_path.is_file() else 'missing'} ({model_path})")
    print(f"Pygame: {'available' if importlib.util.find_spec('pygame') else 'not installed'}")
    return 0


def _report_training_iteration(state: Any, metrics: dict[str, Any]) -> None:
    loss = metrics.get("loss")
    loss_text = "warming-up" if loss is None else f"loss={loss:.4f}"
    print(
        f"iteration={state.iteration} games={state.total_games} "
        f"positions={state.total_positions} replay={metrics['replay_size']} {loss_text}"
    )


def _train(args: argparse.Namespace) -> int:
    try:
        from alphazero_gomoku.training.config import load_experiment_config
        from alphazero_gomoku.training.trainer import AlphaZeroTrainer
    except ImportError as error:
        if error.name == "torch":
            raise SystemExit(
                'PyTorch is required for training. Install it with: pip install -e ".[train]"'
            ) from error
        raise

    if args.resume is not None:
        trainer = AlphaZeroTrainer.resume(
            args.resume,
            device=args.device or "auto",
        )
    else:
        config = load_experiment_config(args.config)
        if args.device is not None:
            config = replace(
                config,
                network=replace(config.network, device=args.device),
            )
        trainer = AlphaZeroTrainer.create(config)

    state = trainer.run(
        target_iteration=args.iterations,
        on_iteration=_report_training_iteration,
    )
    checkpoint = trainer.checkpoints.latest()
    print(f"Training complete at iteration {state.iteration}. Checkpoint: {checkpoint}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    try:
        from alphazero_gomoku.training.benchmark import (
            run_self_play_benchmark,
            write_benchmark_report,
        )
        from alphazero_gomoku.training.config import load_experiment_config
    except ImportError as error:
        if error.name == "torch":
            raise SystemExit(
                'PyTorch is required for benchmarking. Install it with: pip install -e ".[train]"'
            ) from error
        raise

    config = load_experiment_config(args.config)
    if args.device is not None:
        config = replace(
            config,
            network=replace(config.network, device=args.device),
        )
    report = run_self_play_benchmark(config, games=args.games)
    destination = write_benchmark_report(report, args.output)
    performance = report.performance
    print(
        f"Benchmark complete: {performance['simulations_per_second']:.2f} simulations/s, "
        f"mean batch={performance['mean_inference_batch_size']:.2f}. Report: {destination}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gomoku",
        description="Modern AlphaZero Gomoku development toolkit.",
    )
    parser.add_argument("--version", action="version", version=VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="Check the local runtime and bundled model availability.",
    )
    doctor.set_defaults(handler=_doctor)

    train = subparsers.add_parser(
        "train",
        help="Run or resume the reproducible AlphaZero training pipeline.",
    )
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="TOML experiment configuration.")
    source.add_argument("--resume", type=Path, help="Checkpoint directory to resume.")
    train.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Absolute iteration target; defaults to the configured target.",
    )
    train.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="Override the configured device.",
    )
    train.set_defaults(handler=_train)
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Profile parallel self-play and batched neural inference.",
    )
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--games", type=int, default=None)
    benchmark.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/self_play.json"),
    )
    benchmark.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
    )
    benchmark.set_defaults(handler=_benchmark)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the project command-line interface."""
    args = _build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

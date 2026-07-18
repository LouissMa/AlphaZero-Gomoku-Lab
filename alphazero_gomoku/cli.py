"""Command-line entry points for AlphaZero Gomoku."""

from __future__ import annotations

import argparse
import importlib.util
import platform
from collections.abc import Sequence
from dataclasses import asdict, replace
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


def _arena(args: argparse.Namespace) -> int:
    if args.promote_to is not None and args.incumbent is None:
        raise SystemExit("--promote-to requires --incumbent")
    try:
        from alphazero_gomoku.evaluation.arena import (
            run_tournament,
            write_tournament_report,
        )
        from alphazero_gomoku.evaluation.config import load_arena_config
        from alphazero_gomoku.evaluation.neural import (
            NeuralMCTSFactory,
            resolve_model_path,
        )
        from alphazero_gomoku.evaluation.players import (
            HeuristicFactory,
            PureMCTSFactory,
            RandomFactory,
        )
        from alphazero_gomoku.evaluation.promotion import (
            decide_promotion,
            promote_model,
        )
    except ImportError as error:
        if error.name == "torch":
            raise SystemExit(
                'PyTorch is required for evaluation. Install it with: pip install -e ".[train]"'
            ) from error
        raise

    config = load_arena_config(args.config)
    if args.device is not None:
        config = replace(config, device=args.device)
    candidate = NeuralMCTSFactory(
        args.candidate,
        simulations_per_move=config.simulations_per_move,
        c_puct=config.c_puct,
        device=config.device,
    )
    candidate.validate_board(
        width=config.board.width,
        height=config.board.height,
    )
    factories = {
        "random": lambda: RandomFactory(),
        "heuristic": lambda: HeuristicFactory(),
        "pure_mcts": lambda: PureMCTSFactory(
            c_puct=config.c_puct,
            playouts=config.pure_mcts_playouts,
        ),
    }
    opponents = [factories[name]() for name in config.baselines]
    if args.incumbent is not None:
        incumbent_factory = NeuralMCTSFactory(
            args.incumbent,
            simulations_per_move=config.simulations_per_move,
            c_puct=config.c_puct,
            device=config.device,
            name="incumbent",
        )
        incumbent_factory.validate_board(
            width=config.board.width,
            height=config.board.height,
        )
        opponents.append(incumbent_factory)

    report = run_tournament(candidate, opponents, config)
    if args.incumbent is not None:
        incumbent = next(match for match in report.matches if match.opponent == "incumbent")
        decision = decide_promotion(
            incumbent.summary,
            required_score=config.promotion_score,
            required_lower_bound=config.promotion_lower_bound,
        )
        report = replace(report, promotion=asdict(decision))
        if args.promote_to is not None:
            promote_model(resolve_model_path(args.candidate), args.promote_to, decision)
    destination = write_tournament_report(report, args.output)
    print(
        f"Arena complete: score={report.overall.score:.3f}, "
        f"Elo={report.overall.elo_difference:+.1f}, "
        f"{config.confidence:.0%} CI=[{report.overall.confidence_low:.3f}, "
        f"{report.overall.confidence_high:.3f}]. Report: {destination}"
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
    arena = subparsers.add_parser(
        "arena",
        help="Evaluate a checkpoint against baselines and an incumbent model.",
    )
    arena.add_argument("--candidate", type=Path, required=True)
    arena.add_argument("--config", type=Path, required=True)
    arena.add_argument("--incumbent", type=Path)
    arena.add_argument("--promote-to", type=Path)
    arena.add_argument(
        "--output",
        type=Path,
        default=Path("reports/arena.json"),
    )
    arena.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=None,
    )
    arena.set_defaults(handler=_arena)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the project command-line interface."""
    args = _build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

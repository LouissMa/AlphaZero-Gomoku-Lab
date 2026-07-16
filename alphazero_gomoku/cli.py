"""Command-line entry points for AlphaZero Gomoku."""

from __future__ import annotations

import argparse
import importlib.util
import platform
from collections.abc import Sequence
from pathlib import Path

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the project command-line interface."""
    args = _build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

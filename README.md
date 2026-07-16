# AlphaZero Gomoku Lab

A modern, reproducible AlphaZero self-play research and engineering lab for
Gomoku. The project is being upgraded milestone by milestone while preserving
the bundled NumPy inference path and pretrained models.

## Current capabilities

- Configurable Gomoku board and win conditions.
- AlphaZero-style neural-guided Monte Carlo Tree Search.
- Pure MCTS baseline player.
- Terminal and Pygame human-versus-AI interfaces.
- NumPy inference with bundled 6x6/4-in-a-row and 8x8/5-in-a-row models.
- Framework-independent game-engine regression tests.
- Python 3.10-3.13 continuous integration.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m pip install -e ".[gui]"
python -m alphazero_gomoku doctor
python gui_play.py
```

For terminal play:

```bash
python human_play.py
```

## Development

Install the development dependencies and run the checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check alphazero_gomoku/cli.py alphazero_gomoku/__main__.py tests
```

The modernization work is organized into independently testable milestones.
See the [roadmap](docs/ROADMAP.md) and [contribution guide](CONTRIBUTING.md).

## Roadmap

1. Modern engineering baseline.
2. Modern PyTorch residual policy-value network.
3. Reproducible training pipeline.
4. Batched inference and parallel self-play.
5. Elo evaluation arena.
6. Gumbel AlphaZero.
7. Interactive web application.
8. Containerized open-source release and benchmark report.

## Project origin

This repository modernizes the educational
[AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) implementation
by Junxiao Song. The original README is preserved in
[`docs/ORIGINAL_README.md`](docs/ORIGINAL_README.md), and the original MIT license
is retained.

## License

MIT

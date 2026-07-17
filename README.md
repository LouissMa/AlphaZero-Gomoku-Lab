# AlphaZero Gomoku Lab

[![CI](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/ci.yml)
[![PyTorch backend](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/pytorch.yml/badge.svg)](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/pytorch.yml)

A modern, reproducible AlphaZero self-play research and engineering lab for
Gomoku. The project is upgraded through independently testable milestones while
preserving the bundled NumPy inference path and pretrained models.

## Current capabilities

- Configurable Gomoku board and win conditions.
- AlphaZero-style neural-guided Monte Carlo Tree Search.
- Modern PyTorch residual policy-value network.
- AdamW training, gradient clipping, CUDA AMP, and optional `torch.compile`.
- Versioned, portable checkpoints containing complete architecture metadata.
- Pure MCTS baseline player.
- Terminal and Pygame human-versus-AI interfaces.
- NumPy inference with bundled 6x6/4-in-a-row and 8x8/5-in-a-row models.
- Python 3.10-3.13 continuous integration and dedicated PyTorch validation.

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

## Modern PyTorch backend

Install the training dependencies:

```bash
python -m pip install -e ".[train]"
```

Create a configurable network:

```python
from alphazero_gomoku.policy_value_net_pytorch import NetworkConfig, PolicyValueNet

config = NetworkConfig(
    board_width=6,
    board_height=6,
    channels=64,
    residual_blocks=4,
)
network = PolicyValueNet(6, 6, config=config, device="auto")
```

See the [PyTorch backend guide](docs/PYTORCH_BACKEND.md) for training and
checkpoint examples.

## Development

```bash
python -m pip install -e ".[dev,train]"
python -m pytest
ruff check alphazero_gomoku/cli.py alphazero_gomoku/policy_value_net_pytorch.py tests
```

See the [roadmap](docs/ROADMAP.md) and [contribution guide](CONTRIBUTING.md).

## Roadmap

- [x] Modern engineering baseline.
- [x] Modern PyTorch residual policy-value network.
- [ ] Reproducible training pipeline.
- [ ] Batched inference and parallel self-play.
- [ ] Elo evaluation arena.
- [ ] Gumbel AlphaZero.
- [ ] Interactive web application.
- [ ] Containerized open-source release and benchmark report.

## Project origin

This repository modernizes the educational
[AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) implementation
by Junxiao Song. The original README is preserved in
[`docs/ORIGINAL_README.md`](docs/ORIGINAL_README.md), and the original MIT license
is retained.

## License

MIT

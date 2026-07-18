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
- Reproducible self-play training with persistent replay, JSONL metrics, and
  interruption-safe resume.
- Centralized batched neural inference across parallel self-play actors.
- MCTS tree reuse, throughput profiling, hardware metadata, and versioned
  benchmark reports.
- Reproducible evaluation tournaments with alternating first player, Wilson
  confidence intervals, Elo estimates, and confidence-gated model promotion.
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

## Reproducible training

Run a small end-to-end smoke experiment:

```bash
python -m alphazero_gomoku train --config configs/train_smoke.toml
```

Start the full experiment or resume an exact snapshot:

```bash
gomoku train --config configs/train_6x6.toml
gomoku train --resume runs/gomoku-6x6-baseline/checkpoints/step_000050
```

See the [training guide](docs/TRAINING.md) for outputs and reproducibility details.

## Parallel self-play benchmark

Profile the complete scalable search path without changing training state:

```bash
gomoku benchmark --config configs/train_smoke.toml --games 2 --device cpu
```

The report includes simulations per second, inference batch utilization, tree
reuse, runtime versions, CPU threads, and CUDA memory when available. See the
[scalable self-play guide](docs/SCALABLE_SELF_PLAY.md) and committed
[smoke CPU report](benchmarks/smoke_cpu.json).

## Evaluation arena

Evaluate a checkpoint against the configured baseline suite:

```bash
gomoku arena --candidate runs/gomoku-smoke/checkpoints/step_000001 \
  --config configs/arena_smoke.toml --output reports/arena_smoke.json
```

Add `--incumbent models/best.pt --promote-to models/best.pt` to run a
head-to-head promotion gate. See the [evaluation arena guide](docs/EVALUATION_ARENA.md)
for statistical interpretation and the full workflow.

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
- [x] Reproducible training pipeline.
- [x] Batched inference and parallel self-play.
- [x] Elo evaluation arena.
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

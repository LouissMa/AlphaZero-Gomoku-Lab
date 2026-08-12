# AlphaZero Gomoku Lab

[English](README.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/ci.yml)
[![PyTorch backend](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/pytorch.yml/badge.svg)](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/pytorch.yml)
[![Container](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/container.yml/badge.svg)](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/actions/workflows/container.yml)
[![Release](https://img.shields.io/badge/release-1.0.0-7c3aed)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-0f766e.svg)](LICENSE)

A modern, reproducible AlphaZero self-play research and engineering lab for
Gomoku. The project is upgraded through independently testable milestones while
preserving the bundled NumPy inference path and pretrained models.

The repository is both a research workbench and an engineering portfolio:
experiments are reproducible, search improvements are compared at matched
budgets, release evidence is machine-readable, and the complete browser demo
runs locally without a cloud service.

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
- Gumbel AlphaZero with Gumbel-Top-k root sampling, Sequential Halving,
  Completed-Q policy targets, and equal-budget PUCT comparisons.
- Interactive Canvas web application with real model selection, policy heatmaps,
  value estimates, undo, replay, and PUCT/Gumbel controls.
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

For the interactive browser application:

```bash
python -m pip install -e ".[web]"
gomoku serve
```

Or run the same non-root service in Docker:

```bash
docker build -t alphazero-gomoku-lab:1.0.0 .
docker run --rm -p 8000:8000 alphazero-gomoku-lab:1.0.0
```

Open `http://127.0.0.1:8000` and use `/api/health` for a deployment health
check.

## Research workflow

```text
typed experiment config -> seeded self-play -> persistent replay
                        -> optimization/checkpoint -> evaluation arena
                        -> confidence-gated promotion -> web inference
```

The training snapshot stores network and optimizer state, replay data, random
number generator states, configuration, and version metadata. The arena then
alternates colors and reports confidence intervals instead of promoting a model
from a single headline win rate.

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

## Gumbel AlphaZero

Train with the low-simulation policy-improvement search and compare it fairly
against PUCT using one checkpoint:

```bash
gomoku train --config configs/train_gumbel_smoke.toml --iterations 1
gomoku compare-search \
  --model runs/gomoku-gumbel-smoke/checkpoints/step_000001 \
  --config configs/compare_search_smoke.toml
```

See the [Gumbel AlphaZero guide](docs/GUMBEL_ALPHAZERO.md) for the formulas,
configuration, benchmark schema, and implementation scope.

## Evidence and model transparency

| Artifact | Purpose |
| --- | --- |
| [Bundled model card](models/README.md) | File inventory, SHA-256 hashes, intended use, provenance gaps, and limitations |
| [Scalable self-play smoke report](benchmarks/smoke_cpu.json) | Batched inference, parallel actors, tree reuse, and hardware metadata |
| [Gumbel vs PUCT smoke report](benchmarks/gumbel_vs_puct_smoke.json) | Equal simulation budget and deterministic comparison schema |
| [Arena smoke report](reports/arena_smoke.json) | Alternating colors, baselines, confidence intervals, and Elo schema |

These deliberately small reports prove execution paths and reproducibility
metadata. They do not establish general playing strength or cross-hardware
performance. See the [benchmark interpretation guide](benchmarks/README.md).

## Interactive web application

Launch the portfolio-ready browser experience:

```bash
python -m pip install -e ".[web]"
gomoku serve
```

Play against bundled AlphaZero models with PUCT or Gumbel search, inspect the
policy heatmap and value estimate, undo turns, and replay the complete game. A
non-root Docker image is included. See the [web application guide](docs/WEB_APP.md).

## Development

```bash
python -m pip install -e ".[dev,train,web]"
python -m pytest
ruff check alphazero_gomoku/cli.py alphazero_gomoku/policy_value_net_pytorch.py alphazero_gomoku/training alphazero_gomoku/evaluation alphazero_gomoku/gumbel alphazero_gomoku/web tests
```

See the [roadmap](docs/ROADMAP.md) and [contribution guide](CONTRIBUTING.md).

Before proposing a release, run the executable repository audit:

```bash
gomoku release-check
```

## Roadmap

- [x] Modern engineering baseline.
- [x] Modern PyTorch residual policy-value network.
- [x] Reproducible training pipeline.
- [x] Batched inference and parallel self-play.
- [x] Elo evaluation arena.
- [x] Gumbel AlphaZero.
- [x] Interactive web application.
- [x] Containerized open-source release and benchmark report.

## Limitations

- The bundled NumPy weights predate the reproducible training pipeline; exact
  seeds, replay data, hardware, and controlled strength measurements are not
  available.
- Committed benchmarks are smoke workloads, not leaderboard results.
- Training a competitive network remains compute-intensive and is not performed
  by the default test suite.
- The engine demonstrates freestyle Gomoku and does not implement every
  tournament rule variant.

## Community and releases

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
- Use the structured GitHub issue forms for reproducible bugs and scoped ideas.
- Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).
- See [SUPPORT.md](SUPPORT.md), the [changelog](CHANGELOG.md), and the
  [release runbook](docs/RELEASING.md) for lifecycle details.

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff). GitHub can
render it through **Cite this repository** after the release lands on `main`.

## Project origin

This repository modernizes the educational
[AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) implementation
by Junxiao Song. The original README is preserved in
[`docs/ORIGINAL_README.md`](docs/ORIGINAL_README.md), and the original MIT license
is retained.

## License

MIT

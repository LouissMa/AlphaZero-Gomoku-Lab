# Reproducible training

The phase-three pipeline connects neural-guided self-play, replay storage,
optimizer updates, metrics, and complete resumable checkpoints.

## Install

```bash
python -m pip install -e ".[train]"
```

For development and tests:

```bash
python -m pip install -e ".[dev,train]"
```

## Start an experiment

```bash
gomoku train --config configs/train_6x6.toml
```

The same command works without installing the console script:

```bash
python -m alphazero_gomoku train --config configs/train_6x6.toml
```

Use a smaller absolute iteration target for a quick validation:

```bash
gomoku train --config configs/train_6x6.toml --iterations 2 --device cpu
```

Outputs are written to `<output_dir>/<run.name>/`:

```text
runs/gomoku-6x6-baseline/
??? metrics.jsonl
??? checkpoints/
    ??? latest
    ??? step_000050/
        ??? manifest.json
        ??? model.pt
        ??? replay.npz
```

Each checkpoint contains the validated experiment configuration, model and
optimizer states, replay buffer, training counters, architecture metadata, and
Python/NumPy/PyTorch random-number states.

## Resume exactly

Pass the checkpoint directory, not the `latest` pointer file:

```bash
gomoku train \
  --resume runs/gomoku-6x6-baseline/checkpoints/step_000050 \
  --iterations 100
```

`--iterations` is an absolute target. In this example, a checkpoint at
iteration 50 continues through iteration 100. Omitting it uses the target saved
in the original TOML configuration.

A device can be changed while restoring a portable checkpoint:

```bash
gomoku train --resume <checkpoint-directory> --device cuda
```

## Reproducibility contract

When `run.deterministic = true`, the pipeline seeds Python, legacy NumPy,
the dedicated NumPy generator, PyTorch CPU, and all available CUDA devices.
Checkpoint restore reinstates those random states before the next self-play
game. Hardware, PyTorch, and CUDA differences can still affect floating-point
results, so record the environment alongside benchmark results.

The append-only `metrics.jsonl` file records self-play games and positions,
replay size, optimizer updates, training step, winner counts, losses, entropy,
gradient norm, and checkpoint paths.

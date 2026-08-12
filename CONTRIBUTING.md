# Contributing to AlphaZero Gomoku Lab

Thank you for helping make the project more reproducible, understandable, and
useful. By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

AlphaZero Gomoku Lab supports Python 3.10 and newer. Create an isolated
environment and install the development dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[dev,train,web]"
```

The optional Pygame interface can be added with:

```bash
python -m pip install -e ".[dev,gui,train,web]"
```

## Checks

Run the same core checks used by continuous integration:

```bash
python -m pytest
ruff check alphazero_gomoku/cli.py alphazero_gomoku/release.py \
  alphazero_gomoku/policy_value_net_pytorch.py alphazero_gomoku/training \
  alphazero_gomoku/evaluation alphazero_gomoku/gumbel alphazero_gomoku/web tests
python -m alphazero_gomoku doctor
python -m alphazero_gomoku release-check
```

New game rules and search behavior should include deterministic regression tests.
Performance changes should include before-and-after benchmark results and hardware
details.

## Pull requests

Keep each pull request focused and explain the observable behavior it changes.
Add a failing regression test before fixing a bug, and include deterministic
tests for new rules, search behavior, serialization, or release contracts.
Update the English and Chinese entry points together when user-facing commands
or capabilities change.

Use semantic commit subjects such as `Add batched inference metrics` or
`Fix checkpoint RNG restoration`. Never commit API tokens, private training
data, virtual environments, generated build directories, or large unreviewed
checkpoints.

## Research evidence

- Performance changes include matched before/after JSON reports and complete
  hardware/workload metadata.
- Search comparisons use the same model, seed policy, board, and simulation
  budget for every method.
- Strength claims include enough games, alternating colors, confidence
  intervals, and the exact configuration.
- New bundled weights require a model-card entry with intended use, limitations,
  provenance, license, board rules, and SHA-256 hash.
- Changed report schemas increment their format version and update the benchmark
  guide.

## Release metadata

Changes that affect installation, commands, compatibility, or public behavior
must update `CHANGELOG.md`. Before requesting a release, run `gomoku
release-check`, build both package formats, and follow `docs/RELEASING.md`.

## Modernization policy

The existing NumPy inference implementation and bundled models remain supported
alongside the modern PyTorch backend. Historical framework code is archived for
reference; avoid introducing new features into the Theano, TensorFlow 1.x,
legacy Keras, or PyTorch 0.x implementations.

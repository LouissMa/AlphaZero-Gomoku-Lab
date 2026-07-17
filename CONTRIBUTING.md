# Contributing

## Development setup

AlphaZero Gomoku Lab supports Python 3.10 and newer. Create an isolated
environment and install the development dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[dev,train]"
```

The optional Pygame interface can be installed with:

```bash
python -m pip install -e ".[dev,gui,train]"
```

## Checks

Run the same core checks used by continuous integration:

```bash
python -m pytest
ruff check alphazero_gomoku/cli.py alphazero_gomoku/policy_value_net_pytorch.py tests
python -m alphazero_gomoku doctor
```

New game rules and search behavior should include deterministic regression tests.
Performance changes should include before-and-after benchmark results and hardware
details.

## Modernization policy

The existing NumPy inference implementation and bundled models remain supported
alongside the modern PyTorch backend. Historical framework code is archived for
reference; avoid introducing new features into the Theano, TensorFlow 1.x,
legacy Keras, or PyTorch 0.x implementations.

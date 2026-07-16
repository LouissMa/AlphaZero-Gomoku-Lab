# Contributing

## Development setup

AlphaZero Gomoku Lab supports Python 3.10 and newer. Create an isolated
environment and install the development dependencies:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

The optional Pygame interface can be installed with:

```bash
python -m pip install -e ".[dev,gui]"
```

## Checks

Run the same core checks used by continuous integration:

```bash
python -m pytest
ruff check alphazero_gomoku/cli.py alphazero_gomoku/__main__.py tests
python -m alphazero_gomoku doctor
```

New game rules and search behavior should include deterministic regression tests.
Performance changes should include before-and-after benchmark results and hardware
details.

## Modernization policy

The existing NumPy inference implementation and bundled models remain supported
while the legacy training backends are replaced. Avoid introducing new features
into the Theano, TensorFlow 1.x, or legacy Keras implementations.

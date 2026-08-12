# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-12

### Added

- Modern Python packaging, CLI diagnostics, Python 3.10-3.13 CI, and deterministic
  game-engine regression tests.
- Configurable PyTorch residual policy-value network with portable checkpoints,
  mixed precision, gradient clipping, and optional compilation.
- Reproducible self-play training with typed TOML configuration, persistent
  replay, RNG snapshots, JSONL metrics, and exact resume.
- Parallel self-play actors, centralized batched inference, tree reuse, and
  versioned performance reports.
- Evaluation arena with alternating colors, deterministic baselines, Wilson
  confidence intervals, Elo estimates, and atomic model promotion.
- Gumbel AlphaZero root search with Sequential Halving, completed-Q targets,
  deterministic sampling, and equal-budget PUCT comparison.
- Interactive FastAPI and Canvas application with bundled model selection,
  policy heatmaps, value estimates, undo, replay, PUCT/Gumbel controls, and a
  non-root Docker image.
- Bilingual project entry points, bundled-model card, benchmark evidence index,
  community policies, executable release audit, container CI, and tag-driven
  GitHub/GHCR release automation.

### Changed

- Archived historical framework backends while preserving NumPy inference and
  the original bundled models.
- Reframed the upstream educational codebase as a reproducible AlphaZero
  research and engineering lab.

### Security

- Containerized web serving runs as an unprivileged user and exposes a health
  check for deployment monitoring.

[1.0.0]: https://github.com/LouissMa/AlphaZero-Gomoku-Lab/releases/tag/v1.0.0

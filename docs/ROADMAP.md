# AlphaZero Gomoku Lab roadmap

The project will evolve through independently testable milestones. Each milestone
must leave the main branch runnable and include measurable acceptance criteria.

## 1. Modern engineering baseline

**Status: complete.**

- Package metadata, dependency groups, CLI, lint configuration, and game-engine tests.
- Preserve the existing NumPy inference path and bundled models.
- Document legacy training backends until their replacement is complete.

## 2. Modern PyTorch network

**Status: complete.**

- A configurable residual policy-value network on a supported PyTorch release.
- Device-independent checkpoints, mixed precision, and optional compilation.
- Compatibility tests for policy shapes, legal-action masks, and training steps.

## 3. Reproducible training pipeline

**Status: complete.**

- Typed configuration files, deterministic seeds, resumable checkpoints, and metrics.
- Replay-buffer persistence and explicit model/data version metadata.
- End-to-end self-play, optimization, CLI execution, and resume smoke tests.

## 4. Scalable self-play and MCTS

- Batched neural inference, parallel actors, search-tree reuse, and profiling.
- Publish simulations-per-second and hardware utilization benchmarks.

## 5. Evaluation arena

- Alternating colors, confidence intervals, Elo ratings, and automatic model promotion.
- Reproducible tournaments against random, heuristic, pure-MCTS, and neural players.

## 6. Gumbel AlphaZero

- Sequential-halving root search and completed-Q-value policy targets.
- Fair comparisons with PUCT at equal simulation budgets.

## 7. Interactive web application

- Browser play, model selection, policy heatmaps, value estimates, and game replay.
- A small inference API with containerized local deployment.

## 8. Open-source release

- CI, Docker images, model cards, benchmark reports, bilingual documentation, and releases.

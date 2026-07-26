# Gumbel AlphaZero Design

## Objective

Add a research-grade Gumbel AlphaZero search path to the existing Gomoku lab
without replacing the PUCT implementation. The new path must produce improved
Completed-Q policy targets, support reproducible self-play, and provide a fair
equal-simulation comparison against PUCT.

## Chosen approach

Use a native NumPy/Python implementation that shares the existing `Board`,
policy-value callback, replay samples, batched evaluator, and arena framework.
This keeps dependencies unchanged and makes the algorithm inspectable. A JAX
`mctx` dependency was rejected because it would duplicate the model runtime;
modifying the legacy PUCT class in place was rejected because it would weaken
the A/B comparison and increase regression risk.

## Algorithm

At every root, convert legal-action priors to logits and sample independent
Gumbel noise. Consider at most `max_considered_actions` actions using top-k of
`logit + gumbel`. Allocate the fixed simulation budget with Sequential Halving:
give surviving actions balanced visits, rank them by
`gumbel + logit + sigma(completed_q)`, and halve until one remains.

Interior nodes use deterministic policy improvement scores derived from the
prior logits, visit counts, and transformed Completed-Q values. Missing Q values
are filled with a mixed value combining the raw network value and prior-weighted
visited-action values. Completed values are min-max normalized and scaled by
`(max_visit + q_visit_offset) * q_value_scale`.

The training target is the normalized improved policy
`softmax(log_prior + transformed_completed_q)`, restricted to legal actions.
It remains a full board-sized probability vector so the existing replay buffer
and cross-entropy loss require no schema change.

## Components

- `alphazero_gomoku/gumbel/math.py`: Gumbel sampling, stable softmax,
  Sequential Halving schedules, mixed values, Q completion and policy targets.
- `alphazero_gomoku/gumbel/search.py`: tree nodes, simulations, root halving,
  deterministic interior selection, reuse counters and search diagnostics.
- `alphazero_gomoku/gumbel/player.py`: existing `Game.start_self_play` adapter.
- `alphazero_gomoku/gumbel/benchmark.py`: alternating-color equal-budget match,
  latency/search statistics and versioned JSON report.
- Existing training config/self-play: select `puct` or `gumbel` and carry Gumbel
  hyperparameters without changing default PUCT behavior.
- CLI: `gomoku compare-search` loads one checkpoint and produces an auditable
  PUCT-vs-Gumbel report.

## Correctness and failure handling

Configuration rejects non-positive budgets, invalid action caps/scales, and
unknown algorithms. Search masks illegal actions, rejects empty legal sets, and
always spends no more than the configured simulation budget. Model/board shape
validation occurs before benchmarking.

## Verification

Unit tests cover exact schedules, top-k uniqueness, Q completion, normalized
policy targets, deterministic seeds and budget accounting. Integration tests
cover Gumbel self-play samples, PUCT backward compatibility, checkpoint config
round trips, CLI JSON output, alternating colors, and equal budgets. A tiny 3x3
smoke benchmark is committed; CI runs all tests on Python 3.10-3.13 plus the
PyTorch workflow.

## Sources

- Danihelka et al., “Policy improvement by planning with Gumbel,” ICLR 2022.
- Google DeepMind `mctx`, especially `policies.py`, `qtransforms.py`, and
  `seq_halving.py`.

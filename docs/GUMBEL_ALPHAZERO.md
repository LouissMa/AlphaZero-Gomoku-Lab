# Gumbel AlphaZero

Phase 6 adds a separate Gumbel AlphaZero search path while preserving PUCT as
the default. It follows the policy-improvement design in Danihelka et al.,
[Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO),
and cross-checks the numerical rules against Google DeepMind's official
[`mctx`](https://github.com/google-deepmind/mctx) implementation.

## What changes

At a root node, the search samples actions without replacement using the
Gumbel-Top-k trick. Sequential Halving allocates the fixed budget across these
actions and repeatedly removes the lower-scoring half. The score combines the
sampled Gumbel value, the network prior logit, and transformed Completed-Q.

For unvisited actions, Completed-Q substitutes a mixed state value derived from
the raw value prediction and prior-weighted values of visited actions. Values
are normalized and scaled before policy improvement. Interior action selection
uses the paper's deterministic rule:

```text
argmax(improved_policy - visit_count / (1 + total_visits))
```

This makes visit frequencies approach the improved policy. Replay targets are
`softmax(log_prior + transformed_completed_q)`, not raw root visit counts.

## Train with Gumbel search

Run the small end-to-end configuration:

```bash
gomoku train --config configs/train_gumbel_smoke.toml --iterations 1
```

To switch an existing experiment configuration, set these `[self_play]` keys:

```toml
search_algorithm = "gumbel"
max_considered_actions = 16
gumbel_scale = 1.0
q_value_scale = 0.1
q_visit_offset = 50.0
```

`search_algorithm = "puct"` remains the default. Checkpoints store all search
settings, so resume reconstructs the same self-play algorithm.

## Equal-budget comparison

Use one checkpoint for both players and give them the same simulation count:

```bash
gomoku compare-search \
  --model runs/gomoku-gumbel-smoke/checkpoints/step_000001 \
  --config configs/compare_search_smoke.toml \
  --output benchmarks/gumbel_vs_puct_smoke.json
```

The match alternates first player, uses deterministic per-game seeds, and
records Gumbel's W/D/L score, confidence interval, Elo difference, root-search
latency, and policy entropy. The two `simulations_per_move` values in the report
must be identical.

The smoke configuration validates the pipeline only. Use more games and a
larger simulation budget before drawing strength conclusions.

## Scope

This is Gumbel AlphaZero for a known deterministic game model. It is not Gumbel
MuZero: no learned dynamics or reward model is introduced. The implementation
is native NumPy/Python so it can reuse the project's PyTorch model, batched
evaluator, board engine, replay format, and checkpoint system without adding a
JAX runtime dependency.

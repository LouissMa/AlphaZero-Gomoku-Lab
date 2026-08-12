# Benchmark evidence

AlphaZero Gomoku Lab commits machine-readable smoke reports so that important
research and engineering paths remain inspectable. Each JSON document has a
versioned schema, workload configuration, seed, timestamp, and relevant runtime
metadata.

| Evidence | What it exercises | Scope |
| --- | --- | --- |
| [`benchmarks/smoke_cpu.json`](smoke_cpu.json) | Parallel actors, centralized batched inference, tree reuse, and profiler output | Two 3x3 CPU games with two simulations per move |
| [`benchmarks/gumbel_vs_puct_smoke.json`](gumbel_vs_puct_smoke.json) | Equal-budget Gumbel AlphaZero and PUCT comparison | Two 3x3 games with four simulations per move |
| [`reports/arena_smoke.json`](../reports/arena_smoke.json) | Alternating colors, deterministic opponents, Wilson intervals, Elo estimates, and promotion-report schema | Six 3x3 games across random, heuristic, and pure-MCTS baselines |

## Reproduce the paths

```bash
gomoku benchmark --config configs/train_smoke.toml --games 2 --device cpu
gomoku compare-search \
  --model runs/gomoku-gumbel-smoke/checkpoints/step_000001 \
  --config configs/compare_search_smoke.toml
gomoku arena \
  --candidate runs/gomoku-smoke/checkpoints/step_000001 \
  --config configs/arena_smoke.toml \
  --output reports/arena_smoke.json
```

## Interpretation limits

These are deliberately tiny smoke workloads. They prove that the complete code
paths and report schemas execute; they do not establish broad playing strength,
statistical superiority, or production throughput. Do not compare simulations
per second across different hardware, software versions, board sizes, model
architectures, actor counts, batch sizes, or simulation budgets. Use repeated,
matched configurations and report confidence intervals for research claims.

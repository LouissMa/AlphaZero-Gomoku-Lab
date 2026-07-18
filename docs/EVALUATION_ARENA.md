# Evaluation arena

The phase-5 arena provides reproducible, alternating-first-player matches for
comparing a candidate checkpoint with fixed baselines and the current best
neural checkpoint. Every run emits a versioned JSON report containing each
game, aggregate W/D/L score, a Wilson confidence interval, and Elo difference.

## Run a smoke evaluation

First create the tiny checkpoint with the training smoke configuration:

```bash
gomoku train --config configs/train_smoke.toml --iterations 1
```

Then evaluate it against random, tactical heuristic, and pure-MCTS players:

```bash
gomoku arena \
  --candidate runs/gomoku-smoke/checkpoints/step_000001 \
  --config configs/arena_smoke.toml \
  --output reports/arena_smoke.json
```

Use `configs/arena_6x6.toml` for the main 6x6 experiment. The checkpoint's
stored board dimensions must match the arena board.

## Compare and promote a model

Pass the current best checkpoint as the incumbent. Promotion is performed only
when the candidate's head-to-head score and Wilson lower confidence bound both
meet the configured thresholds:

```bash
gomoku arena \
  --candidate runs/candidate/checkpoints/step_000100 \
  --incumbent models/best.pt \
  --config configs/arena_6x6.toml \
  --promote-to models/best.pt \
  --output reports/candidate_vs_best.json
```

The replacement is atomic: a passing candidate is copied to a temporary file
beside the destination and then moved into place. A rejected candidate never
changes the incumbent. The report still records the decision and thresholds.

## Reproducibility and interpretation

- Each opponent receives an even number of games, with the candidate moving
  first in exactly half.
- The configured seed generates deterministic per-game RNG streams.
- Draws count as half a point in score, confidence intervals, and Elo.
- Elo is descriptive for the completed match set; the confidence interval is
  the promotion safety gate.
- The JSON `format_version` allows downstream dashboards to validate schema
  compatibility.

For a meaningful promotion decision, increase `games_per_opponent` beyond the
smoke setting. Two games validate plumbing but cannot provide strong evidence.

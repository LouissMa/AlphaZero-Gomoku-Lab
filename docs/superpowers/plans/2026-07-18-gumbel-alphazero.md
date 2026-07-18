# Gumbel AlphaZero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible Gumbel AlphaZero self-play and an equal-budget PUCT comparison workflow.

**Architecture:** A separate NumPy Gumbel search package adapts to the current policy-value callback and `Game` API. Training selects the search algorithm from typed TOML configuration; a benchmark module and CLI compare both implementations with one checkpoint.

**Tech Stack:** Python 3.10+, NumPy, existing PyTorch policy-value network, pytest, Ruff.

## Global Constraints

- Preserve `puct` as the default and keep legacy tests passing.
- No new runtime dependencies.
- All randomness comes from injected `numpy.random.Generator` instances.
- Both algorithms receive the same per-move simulation budget in comparisons.
- Reports are JSON, versioned, finite-valued, and reproducible apart from timing/timestamp fields.

---

### Task 1: Mathematical primitives

**Files:**
- Create: `alphazero_gomoku/gumbel/math.py`
- Test: `tests/test_gumbel_math.py`

**Interfaces:**
- Produces: `sequential_halving_schedule`, `sample_gumbel_top_k`,
  `completed_qvalues`, `improved_policy`.

- [ ] Write tests asserting schedule length/budget, unique top-k actions, raw and
  mixed-value completion, finite normalized policy, and validation failures.
- [ ] Run `python -m pytest tests/test_gumbel_math.py -q` and confirm import
  failure before implementation.
- [ ] Implement stable, vector-shaped NumPy functions with explicit validation.
- [ ] Rerun the test file and Ruff until green.

### Task 2: Gumbel search and player

**Files:**
- Create: `alphazero_gomoku/gumbel/search.py`
- Create: `alphazero_gomoku/gumbel/player.py`
- Create: `alphazero_gomoku/gumbel/__init__.py`
- Test: `tests/test_gumbel_search.py`

**Interfaces:**
- Consumes: a callable returning legal `(action, probability)` pairs and value.
- Produces: `GumbelSearch.search(board) -> SearchResult` and
  `GumbelMCTSPlayer.get_action(board, temp, return_prob)`.

- [ ] Write tests for deterministic results, legal actions, exact budget upper
  bound, root action coverage, policy shape/sum, and tree reuse.
- [ ] Run the tests and confirm missing-module/API failures.
- [ ] Implement tree expansion/backpropagation, root Sequential Halving,
  deterministic interior selection, Completed-Q targets and player adapter.
- [ ] Rerun search and math tests, then refactor while green.

### Task 3: Training integration

**Files:**
- Modify: `alphazero_gomoku/training/config.py`
- Modify: `alphazero_gomoku/training/self_play.py`
- Modify: `configs/train_smoke.toml`
- Create: `configs/train_gumbel_smoke.toml`
- Test: `tests/test_gumbel_training.py`

**Interfaces:**
- Adds `search_algorithm`, `max_considered_actions`, `gumbel_scale`,
  `q_value_scale`, and `q_visit_offset` to `SelfPlayConfig`.

- [ ] Write tests for configuration validation, PUCT default compatibility,
  deterministic Gumbel self-play and valid replay policy targets.
- [ ] Confirm failures before implementation.
- [ ] Add typed configuration and select the player factory in self-play.
- [ ] Run Gumbel and existing training tests until green.

### Task 4: Equal-budget benchmark and CLI

**Files:**
- Create: `alphazero_gomoku/gumbel/benchmark.py`
- Modify: `alphazero_gomoku/cli.py`
- Create: `configs/compare_search_smoke.toml`
- Test: `tests/test_gumbel_benchmark.py`

**Interfaces:**
- Produces: `compare_search_algorithms`, `write_search_comparison_report`, and
  `gomoku compare-search --model ... --config ... --output ...`.

- [ ] Write tests for alternating first player, equal budgets, JSON schema,
  deterministic outcomes and end-to-end CLI execution.
- [ ] Confirm missing API/command failures.
- [ ] Implement checkpoint-backed factories, match runner, metrics and CLI.
- [ ] Run focused tests and a real 3x3 comparison command.

### Task 5: Documentation, CI and release evidence

**Files:**
- Create: `docs/GUMBEL_ALPHAZERO.md`
- Create: `benchmarks/gumbel_vs_puct_smoke.json`
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/pytorch.yml`

- [ ] Document formulas, limitations, configuration and commands with links to
  the ICLR paper and official DeepMind implementation.
- [ ] Commit the real smoke comparison report and mark phase 6 complete.
- [ ] Run full pytest, Ruff, compileall, CLI smoke and diff checks.
- [ ] Commit, push `codex/gumbel-alphazero`, create a stacked Draft PR against
  `codex/evaluation-arena`, and wait for every GitHub check to pass.

# Scalable self-play and batched MCTS inference

Phase four runs multiple actor-local MCTS games concurrently while keeping all
PyTorch model access in one centralized batching worker.

## Architecture

Each actor owns its board, MCTS tree, and NumPy random generator. A policy
callback sends leaf states and legal-action masks to `BatchedPolicyEvaluator`.
The evaluator waits for a small configurable window, combines pending requests,
performs one `PolicyValueNet.policy_value` call, masks illegal moves, and wakes
the actors with their individual policy/value results.

This design provides:

- neural inference batches across independent games;
- one model-access thread, avoiding unsafe concurrent optimizer/model calls;
- search-tree reuse within every game;
- deterministic actor seeds allocated from the checkpointed experiment RNG;
- bounded latency when the batch is not full;
- ordered results independent of actor completion order.

## Configuration

The `[self_play]` section controls scaling:

```toml
games_per_iteration = 8
parallel_games = 4
simulations_per_move = 400
inference_batch_size = 16
inference_wait_ms = 2.0
```

`parallel_games` is capped by `games_per_iteration`. Larger inference waits
usually improve batch utilization but add latency, especially on CPU. GPU runs
normally benefit from more actors and larger inference batches.

## Profiling

Every training iteration records:

- self-play elapsed time and positions per second;
- simulations and simulations per second;
- inference requests, batches, elapsed time, mean/max batch size, and batch utilization;
- MCTS root reuse and reset counts;
- configured parallel actor count.

These fields are appended to the same `metrics.jsonl` stream as optimizer
losses and checkpoint paths.

## Benchmark command

Run a workload without modifying training state:

```bash
gomoku benchmark \
  --config configs/train_smoke.toml \
  --games 2 \
  --device cpu \
  --output benchmarks/smoke_cpu.json
```

The versioned JSON report contains workload dimensions, throughput, batching,
tree reuse, Python/PyTorch/platform metadata, CPU thread count, CUDA device
name, and peak CUDA memory when applicable.

The repository includes a measured smoke baseline in
[`benchmarks/smoke_cpu.json`](../benchmarks/smoke_cpu.json). It is a functional
regression baseline from one machine, not a cross-hardware performance claim.

## Tuning guidance

- Increase `parallel_games` until the accelerator is fed consistently.
- Increase `inference_batch_size` when mean batch size reaches the current limit.
- Keep `inference_wait_ms` small for low actor counts and interactive workloads.
- Compare throughput only with identical board, network, simulations, and game counts.
- Use the evaluation arena in phase five to measure playing strength separately
  from systems throughput.

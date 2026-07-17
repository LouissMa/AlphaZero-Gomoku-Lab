# Benchmarks

Benchmark reports in this directory are produced by the public
`gomoku benchmark` command. Reports use a versioned JSON schema and record both
the workload and hardware/software environment needed to interpret throughput.

`smoke_cpu.json` validates the complete path on a tiny 3x3 workload: parallel
actors, centralized batched inference, MCTS tree reuse, and profiler output.
Run the same command on another machine to create a comparable local report;
do not compare simulations per second across different workload settings.

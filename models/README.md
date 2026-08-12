# Bundled model card

This directory contains legacy NumPy policy-value weights preserved from the
original AlphaZero Gomoku project. They make the terminal, Pygame, and web demos
usable without training a new network first.

## Intended use

- Educational demonstrations of neural-guided Monte Carlo Tree Search.
- Local human-versus-AI play on the matching board configuration.
- Compatibility and regression checks for the preserved NumPy inference path.

These files are not modern PyTorch checkpoints and cannot be resumed by the
reproducible training pipeline. New training runs produce portable checkpoint
directories with architecture, optimizer, replay, RNG, and data-version
metadata; see [`docs/TRAINING.md`](../docs/TRAINING.md).

## Inventory and integrity

| File | Board | Win length | Runtime | SHA-256 |
| --- | ---: | ---: | --- | --- |
| `best_policy_6_6_4.model` | 6x6 | 4 | NumPy | `00443477e3ad971406482a3c1517ca950c270e6e3d28221021a69d020dfdb176` |
| `best_policy_6_6_4.model2` | 6x6 | 4 | NumPy | `250c0a9ec1df26a7d4dfc6e5df56eed2d5ebc79694926ad7e5d74a283a96ce61` |
| `best_policy_8_8_5.model` | 8x8 | 5 | NumPy | `c089392ec7aa62d34c4cc9c710603d84263388d35901e5cac81ebb3eb3fa996d` |
| `best_policy_8_8_5.model2` | 8x8 | 5 | NumPy | `260441024f211c005692200da772dc63217967cb2dadfa7cc46b85a9c02a82db` |

Verify a file on PowerShell with:

```powershell
Get-FileHash -Algorithm SHA256 models\best_policy_8_8_5.model
```

## Provenance and limitations

The files originate from the upstream educational repository retained in this
project's history. Exact training code revisions, random seeds, replay data,
hardware, training duration, licenses for any external training data, and
statistically controlled strength evaluations were not recorded with the
artifacts. This project therefore makes no claim about their Elo, optimality,
fairness, or suitability for competitive ranking.

The models play only the board and win-length combination encoded by their
filename. Gomoku rule variants differ; these demonstrations use this project's
freestyle board engine and should not be treated as an authority on tournament
rules. Outputs are research/demo results, not safety-critical decisions.

## Evaluation

Committed reports in [`benchmarks/`](../benchmarks/) and [`reports/`](../reports/)
exercise the modern infrastructure on tiny smoke workloads. They do not measure
the playing strength of these bundled legacy weights. Evaluate a modern
checkpoint with the reproducible arena before making comparative claims.

## License

The repository, including these preserved artifacts, is distributed under the
project's [MIT license](../LICENSE) and retains attribution to the original
AlphaZero Gomoku project.

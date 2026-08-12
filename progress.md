# AlphaZero Gomoku Lab modernization progress

## Completed milestones

1. Modern Python engineering baseline and preserved NumPy inference.
2. Configurable PyTorch residual policy-value network.
3. Reproducible, resumable self-play training pipeline.
4. Parallel actors, batched inference, tree reuse, and profiling.
5. Statistical evaluation arena and confidence-gated promotion.
6. Gumbel AlphaZero and equal-budget PUCT comparison.
7. Interactive FastAPI and Canvas application with containerized serving.
8. Auditable open-source `1.0.0` release candidate.

## Phase 8 implementation

- Added equivalent English and Simplified Chinese landing pages with verified
  commands, evidence links, limitations, community paths, and citation guidance.
- Added a collection-level model card covering all four bundled legacy models,
  exact SHA-256 hashes, intended use, provenance gaps, and evaluation limits.
- Expanded the benchmark index to cover scalable self-play, Gumbel-versus-PUCT,
  and arena smoke reports without overstating the tiny workloads.
- Added changelog, citation, security, support, conduct, issue, pull-request, and
  generated-release-notes metadata.
- Added a standard-library repository audit and `gomoku release-check`, driven
  by red-green tests for missing files, version drift, unlisted evidence,
  language navigation, and tag mismatch.
- Added pull-request Docker builds and a tag-driven release workflow that
  validates packages, publishes `linux/amd64` and `linux/arm64` images to GHCR,
  generates checksums, and creates the GitHub Release with least-privilege jobs.
- Added an operator runbook that keeps manual dispatch validation-only and
  reserves `v1.0.0` publication for a verified commit on `main`.

## Release boundary

The repository is prepared for `1.0.0`, but the tag and GitHub Release are not
created from this feature branch. After the final pull request merges and all
checks pass on `main`, follow `docs/RELEASING.md` to publish `v1.0.0`.

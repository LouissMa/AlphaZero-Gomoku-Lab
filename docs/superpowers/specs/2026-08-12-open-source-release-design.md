# Open-source release design

## Context

AlphaZero Gomoku Lab has completed seven independently testable milestones. The
repository already contains a modern Python package, reproducible training,
scalable self-play, an evaluation arena, Gumbel AlphaZero, committed benchmark
evidence, and a containerized web application. GitHub pull request #11 merged
the web application into the feature chain, while `main` currently ends at the
Gumbel AlphaZero milestone. The release phase must therefore both finish the
open-source surface and provide a clean final path back to `main`.

## Goals

- Present the project professionally in English and Simplified Chinese.
- Document every bundled model and committed benchmark without inventing
  provenance or performance claims that the repository cannot prove.
- Provide normal open-source contribution, support, security, citation, and
  change-history files.
- Make release readiness executable and testable rather than a manual promise.
- Validate the container on pull requests and publish versioned multi-platform
  images and Python artifacts from signed-off version tags.
- Prepare version `1.0.0` as the first modernized public release.
- Mark roadmap milestone 8 complete only after all acceptance checks pass.

## Non-goals

- No new game-playing, training, or search algorithm features.
- No claims that bundled legacy models were produced by the modern training
  pipeline; their original training metadata is unavailable.
- No automatic publication to PyPI, because the project does not yet have a
  configured trusted publisher or an explicitly authorized package namespace.
- No release tag or GitHub Release before the final pull request is merged to
  `main`. Publishing from an unmerged branch would make the release history
  misleading.
- No large benchmark reruns or new model training in this phase.

## Considered approaches

### A. Documentation-only release

Add bilingual documentation, model cards, and community files but leave
publishing manual. This is low risk, but it does not demonstrate production
release engineering and allows documentation or version drift.

### B. Fully automated release from any branch

Build and publish images and GitHub Releases from a manually dispatched
workflow. This is convenient, but a mistaken dispatch can publish an unreviewed
commit and create tags that do not correspond to `main`.

### C. Verified tag-driven release candidate (selected)

Add executable release-readiness checks, pull-request container builds, and a
tag-triggered release workflow. Tags matching `v*.*.*` build source and wheel
artifacts, validate their contents, publish a multi-platform image to GitHub
Container Registry, generate checksums, and create the GitHub Release. The
workflow verifies that the tag version equals the package version. Actual
publication happens only after this branch is reviewed and merged to `main`.

This option has the strongest portfolio value while keeping irreversible
publication behind an explicit tag.

## Release surface

### Project entry points

`README.md` remains the canonical English landing page and links to
`README.zh-CN.md`. Both pages describe the same capabilities, installation
paths, web demo, reproducible research workflow, evidence, limitations, and
project origin. Examples must use commands and paths that exist in the current
tree.

### Model and benchmark documentation

`models/README.md` acts as the collection-level model card. It records board
shape, win length, runtime backend, intended use, known limitations, file
inventory, integrity hashes, and the fact that exact training provenance and
evaluation metrics for legacy weights are unavailable. It must distinguish the
bundled NumPy models from modern portable PyTorch checkpoints.

`benchmarks/README.md` becomes the evidence index for scalable self-play,
Gumbel-versus-PUCT, and the arena smoke report. It explains that smoke reports
validate code paths rather than establish playing strength or cross-hardware
performance.

### Community and lifecycle files

The repository adds:

- `CHANGELOG.md` in Keep a Changelog form with a `1.0.0` release section;
- `SECURITY.md`, `SUPPORT.md`, `CODE_OF_CONDUCT.md`, and `CITATION.cff`;
- structured bug and feature issue forms, issue-template configuration, and a
  pull-request template;
- GitHub generated-release-notes categories.

`CONTRIBUTING.md` is updated so local commands match CI and explain when model,
benchmark, and release metadata must change.

## Release-readiness contract

A small standard-library module, `alphazero_gomoku/release.py`, exposes a
deterministic repository audit. It accepts a repository root and returns a list
of human-readable violations. The audit checks:

- package version is valid semantic version `1.0.0`;
- all required release, bilingual, model-card, benchmark, container, and
  workflow files exist;
- English and Chinese READMEs link to one another;
- every tracked bundled `*.model` and `*.model2` file appears in the model card;
- every tracked benchmark/report JSON evidence file appears in the benchmark
  index;
- roadmap milestone 8 is marked complete;
- changelog and citation metadata contain the package version.

The CLI gains `gomoku release-check`, prints each violation to standard error,
and exits non-zero on failure. Unit tests drive the audit with temporary
repository fixtures, while a repository integration test checks the actual
tree. CI runs the command on every push and pull request.

## Automation architecture

### Pull-request container validation

`.github/workflows/container.yml` uses Buildx to build the Linux image without
pushing it. This proves that the Docker context, package metadata, static web
assets, models, non-root user, and command remain compatible.

### Tag-driven publication

`.github/workflows/release.yml` runs only for tags matching `v*.*.*` and manual
dispatch in validation-only mode. It:

1. checks out the exact ref and installs build tooling;
2. verifies tag/package version equality and runs `gomoku release-check`;
3. runs the full non-PyTorch test suite and builds wheel/source archives;
4. installs the wheel into a clean environment and smoke-tests the CLI;
5. generates SHA-256 checksums and uploads artifacts;
6. builds and pushes `linux/amd64` and `linux/arm64` images to GHCR when the
   event is a tag;
7. creates a GitHub Release with generated notes and attached artifacts.

The workflow uses least-privilege `contents: write` and `packages: write`
permissions only in the publishing job. It does not require repository secrets.

## Error handling and safety

- The readiness command reports all detected problems in one run.
- Missing files, malformed metadata, unlisted evidence, and version mismatch
  fail before artifact publication.
- Manual workflow dispatch builds and validates but never pushes an image or
  creates a release.
- The Docker job does not publish on pull requests.
- Model cards state unknowns explicitly; no inferred training facts are
  presented as evidence.

## Testing and acceptance criteria

The phase is implementation-complete when:

1. New release-audit tests demonstrate a red-green cycle and cover missing
   files, version mismatch, unlisted models/evidence, and a valid tree.
2. `python -m pytest` passes on the complete repository.
3. Ruff passes for all maintained Python packages and tests.
4. `python -m build` creates both wheel and source distributions, and the wheel
   installs and passes `python -m alphazero_gomoku doctor` in an isolated
   environment.
5. `gomoku release-check` reports a ready repository.
6. Workflow YAML parses and the GitHub pull request checks pass.
7. Documentation cross-references and model hashes are verified against the
   actual repository files.

The implementation is committed and pushed on `codex/open-source-release`, and
a pull request targets `main`. Creating tag `v1.0.0` and the corresponding
GitHub Release remains the final, explicit post-merge action.

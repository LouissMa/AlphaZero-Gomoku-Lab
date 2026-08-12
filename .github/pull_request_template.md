## What changed

Describe the user-visible or developer-visible outcome and why it is needed.

## Evidence

List the exact tests, benchmarks, screenshots, or report JSON used to validate
the change. For a bug, explain the failing regression test and root cause.

## Checklist

- [ ] The change is focused and contains no secrets, private data, or generated build output.
- [ ] New behavior or bug fixes include deterministic tests that failed before the implementation.
- [ ] `python -m pytest` passes.
- [ ] Ruff passes for the modern maintained modules and tests listed in `CONTRIBUTING.md`.
- [ ] User-facing commands and both README languages are synchronized.
- [ ] Model changes update the model card, provenance, limitations, and SHA-256.
- [ ] Performance or strength claims include matched, machine-readable benchmark evidence.
- [ ] Public behavior is recorded in `CHANGELOG.md`.
- [ ] `python -m alphazero_gomoku release-check` passes.

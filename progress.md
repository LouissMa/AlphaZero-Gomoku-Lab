Original prompt: 接下来进行下一个阶段，完整地完成

## Phase 7 progress

- 2026-07-25: Selected FastAPI + native Canvas and committed the reviewed design.
- Baseline before changes: 61 tests passed.
- Added an allowlisted model catalog, top-left coordinate boundary, thread-safe game sessions, PUCT/Gumbel play, direct policy/value analysis, undo, and immutable replay frames.
- Added typed FastAPI endpoints, `gomoku serve`, packaged static assets, and stable 404/409/422 errors.
- Built the Machine Kifu responsive Canvas interface with model/search controls, heatmap, value telemetry, replay, fullscreen, and accessibility hooks.
- Added a non-root Docker image, healthcheck, Web documentation, README showcase, roadmap status, and Python 3.10-3.13 CI coverage.
- Web verification: 23 focused tests passed; real browser human move + AI response succeeded; heatmap, undo, replay slider, fullscreen, and 390x844 responsive layout succeeded; browser console reported 0 errors and 0 warnings.
- Environment note: Codex image-view helper remained unavailable after a C-drive exhaustion event. Playwright screenshots, visible headed Chromium, DOM geometry, semantic snapshots, and text state were used for visual cross-checking.

## Final verification

- Full suite: 84 passed.
- CI-scoped Ruff: all checks passed.
- Python compileall, `doctor`, `serve --help`, JavaScript syntax, and wheel build passed.
- Built wheel contains `index.html`, `styles.css`, and `app.js`.
- Local health endpoint and real headed-Chromium game flow passed with zero console errors/warnings.
- Docker CLI is not installed on this machine, so the Dockerfile could not be built locally.

## Current TODO

- Publish the phase branch and wait for GitHub Actions.
- Phase 8 remains: open-source release, model cards, bilingual docs, versioned release artifacts.
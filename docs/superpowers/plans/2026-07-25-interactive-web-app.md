# Interactive Web Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a containerized browser application that plays real games against bundled AlphaZero engines and visualizes policy, value, and replay data.

**Architecture:** A synchronous, thread-safe Python domain service owns ephemeral game sessions and adapts existing policy-value/search engines. FastAPI exposes typed JSON endpoints and packaged static assets. A framework-free Canvas client renders and controls the complete product.

**Tech Stack:** Python 3.10+, NumPy, existing PUCT/Gumbel engines, FastAPI, Uvicorn, HTTPX, HTML5 Canvas, CSS, vanilla JavaScript, Pytest, Playwright, Docker.

## Global Constraints

- Keep FastAPI, Uvicorn, and HTTPX in the optional `web` dependency group.
- Keep the base installation and existing CLI commands usable without web dependencies.
- Accept only allowlisted model identifiers; never accept a client-supplied filesystem path.
- Browser coordinates use top-left origin; the legacy board remains bottom-origin internally.
- Search simulations must be exactly one of 8, 16, 32, or 64.
- The entry HTML file is `index.html`.
- Expose `window.render_game_to_text`, `window.advanceTime(ms)`, and `F` fullscreen.
- Include a subtle clickable `Created By Deerflow` link to `https://deerflow.tech`.

---

### Task 1: Model catalog and game session domain

**Files:**
- Create: `alphazero_gomoku/web/__init__.py`
- Create: `alphazero_gomoku/web/catalog.py`
- Create: `alphazero_gomoku/web/service.py`
- Test: `tests/test_web_service.py`

**Interfaces:**
- Produces: `ModelCatalog(root: Path)`, `ModelDescriptor`, `GameService(catalog, seed=...)`.
- Produces: `create_game`, `get_game`, `play_move`, `undo`, and `replay`.
- Consumes: `Board`, `MCTSPlayer`, `GumbelMCTSPlayer`, `PolicyValueNetNumpy`, and `HeuristicPlayer`.

- [ ] **Step 1: Write catalog and coordinate tests**

Write failing tests that assert the three fixed descriptors, reject unknown
identifiers, convert top-left `(row, column)` to the correct legacy move, and
serialize engine moves back to top-left coordinates.

- [ ] **Step 2: Verify the tests fail for missing web modules**

Run: `python -m pytest tests/test_web_service.py -q -p no:cacheprovider`

Expected: collection fails because `alphazero_gomoku.web` does not exist.

- [ ] **Step 3: Implement the allowlisted catalog and conversion boundary**

Implement immutable descriptors for `heuristic-6x6`, `alphazero-6x6`, and
`alphazero-8x8`; lazy-load bundled pickle parameters; add explicit coordinate
helpers that flip rows exactly once.

- [ ] **Step 4: Write failing session behavior tests**

Test game creation, AI opening for a white human, legal human+AI turns, occupied
and out-of-range rejection, terminal win serialization, and policy normalization.
Use a deterministic policy runtime in the test catalog so tests exercise the real
service without slow neural inference.

- [ ] **Step 5: Implement the minimal thread-safe session service**

Create immutable response dictionaries from mutable boards; use snapshots for
undo/replay; choose PUCT or Gumbel with equal simulation parameters; provide
direct-policy analysis and stable domain exceptions.

- [ ] **Step 6: Write and pass undo/replay tests**

Assert undo removes a complete human turn, replay begins empty, replay ends at
the current board, and replay frames cannot be mutated through returned data.

- [ ] **Step 7: Run the domain tests**

Run: `python -m pytest tests/test_web_service.py -q -p no:cacheprovider`

Expected: all web service tests pass.

### Task 2: FastAPI inference and static delivery

**Files:**
- Create: `alphazero_gomoku/web/api.py`
- Create: `tests/test_web_api.py`
- Modify: `pyproject.toml`
- Modify: `alphazero_gomoku/cli.py`

**Interfaces:**
- Consumes: `GameService`.
- Produces: `create_app(service: GameService | None = None) -> FastAPI`.
- Produces: `gomoku serve --host --port --reload`.

- [ ] **Step 1: Add optional dependencies and write failing API tests**

Add `web = ["fastapi>=0.116", "uvicorn>=0.35", "httpx>=0.28"]`. Test health,
models, create/get game, valid and invalid moves, undo, replay, unknown sessions,
and `/` serving HTML.

- [ ] **Step 2: Verify API tests fail for missing application**

Run: `python -m pytest tests/test_web_api.py -q -p no:cacheprovider`

Expected: import failure for `alphazero_gomoku.web.api`.

- [ ] **Step 3: Implement request models, routes, and error mapping**

Use Pydantic literals for colors/search and a literal simulation whitelist.
Return 404 for unknown resources, 409 for game-state conflicts, and 422 for
coordinate or configuration validation.

- [ ] **Step 4: Add lazy CLI server command**

`_serve` imports Uvicorn only when selected. Parser options are `--host`,
`--port`, and `--reload`; `--reload` passes the application import string while
normal startup passes an application instance.

- [ ] **Step 5: Run API and CLI tests**

Run: `python -m pytest tests/test_web_api.py tests/test_game.py -q -p no:cacheprovider`

Expected: all selected tests pass.

### Task 3: Canvas product interface

**Files:**
- Create: `alphazero_gomoku/web/static/index.html`
- Create: `alphazero_gomoku/web/static/styles.css`
- Create: `alphazero_gomoku/web/static/app.js`

**Interfaces:**
- Consumes: all `/api` routes from Task 2.
- Produces: responsive browser play, analysis heatmap, undo/restart, replay,
  fullscreen, and automation hooks.

- [ ] **Step 1: Build semantic entry markup**

Create the setup controls, Canvas, live status, analysis readout, action buttons,
replay slider, accessible error region, and Deerflow attribution in `index.html`.

- [ ] **Step 2: Implement the visual system**

Use CSS custom properties for rice paper, ink, mineral blue, and vermilion.
Implement asymmetric desktop layout, mobile stacking, keyboard focus, reduced
motion, loading state, and a restrained staggered entrance.

- [ ] **Step 3: Implement state and API orchestration**

Fetch models, create a default game, map Canvas clicks to top-left intersections,
disable input during requests, update state atomically, surface errors, and
support restart/undo/heatmap/replay controls.

- [ ] **Step 4: Implement high-DPI Canvas rendering**

Draw paper/wood texture, grid, star points, heat circles, stones, last-move mark,
hover preview, and replay frames. Recompute dimensions on resize/fullscreen.

- [ ] **Step 5: Add deterministic browser hooks**

Expose `render_game_to_text` with visible interaction state and coordinate rules.
Expose `advanceTime(ms)` to advance the pulse animation and redraw. Bind `F` to
fullscreen and redraw after `fullscreenchange`.

### Task 4: Container, documentation, and CI

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `docs/WEB_APP.md`
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/pytorch.yml`
- Modify: `progress.md`

**Interfaces:**
- Produces: local `pip install -e ".[web]"` and Docker startup instructions.
- Produces: CI coverage for domain/API tests and Ruff coverage for web modules.

- [ ] **Step 1: Add production container**

Use `python:3.12-slim`, install the `web` extra without editable mode, create a
non-root user, expose 8000, add `/api/health` healthcheck, and run `gomoku serve`.

- [ ] **Step 2: Document operation and architecture**

Document local startup, Docker commands, controls, API endpoints, model/search
semantics, coordinate conventions, and troubleshooting in `docs/WEB_APP.md`.
Add a concise README showcase and mark roadmap phase 7 complete.

- [ ] **Step 3: Extend CI**

Install `.[dev,web]` in the standard matrix, run the whole suite, lint web code,
and include web files in PyTorch workflow path filters without duplicating tests.

- [ ] **Step 4: Update progress tracking**

Record completed features, verification results, known trade-offs, and phase 8
as the only remaining roadmap work.

### Task 5: End-to-end and visual verification

**Files:**
- Create: `.tmp/web-actions-*.json` only as ignored local test artifacts.
- Create: `.tmp/web-screenshots/*.png` only as ignored local test artifacts.

**Interfaces:**
- Consumes: packaged application and the standard web-game Playwright client.
- Produces: verified desktop/mobile screenshots, console log, and text state.

- [ ] **Step 1: Run full static verification**

Run the complete Pytest suite, Ruff, `compileall`, `gomoku doctor`,
`gomoku serve --help`, package-data inspection, and `git diff --check`.

- [ ] **Step 2: Start the real server**

Run `python -m alphazero_gomoku serve --host 127.0.0.1 --port 8765` and confirm
health, model listing, and index delivery.

- [ ] **Step 3: Exercise the game with the standard Playwright client**

Use short click bursts with pauses to start a game, play a stone, wait for the AI,
toggle heatmap, undo, and move the replay slider. Inspect each returned
`render_game_to_text` state and browser console output.

- [ ] **Step 4: Inspect screenshots**

Open desktop and mobile gameplay screenshots. Confirm the board, controls, stones,
analysis, and responsive layout are visible with no clipping or contrast defects.
Fix and repeat until correct.

- [ ] **Step 5: Verify fullscreen and regressions**

Use `F`, confirm the Canvas resizes and text state stays consistent, exit with
Escape, then replay the primary game flow from a fresh session.

- [ ] **Step 6: Commit and publish**

Stage only phase 7 files, commit with `Add interactive AlphaZero web app`, push
`codex/interactive-web-app`, create a draft PR against
`codex/gumbel-alphazero`, and wait for all GitHub Actions checks to pass.

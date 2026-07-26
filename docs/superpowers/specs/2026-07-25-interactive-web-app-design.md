# Interactive Web Application Design

## Purpose

Phase 7 turns the repository from an algorithm laboratory into a portfolio-ready
product that a reviewer can run and understand without reading the training code.
The browser must play a real game against the repository's existing engines and
explain the model's current opinion through policy and value visualizations.

## Product scope

The application provides:

- human-versus-AI play on the bundled 6x6 and 8x8 boards;
- model selection between the bundled AlphaZero networks and a deterministic
  heuristic baseline;
- PUCT or Gumbel AlphaZero search with a bounded simulation selector;
- a policy heatmap and current-player value estimate;
- last-move, turn, result, latency, and move-count feedback;
- restart, undo, fullscreen, heatmap visibility, and complete game replay;
- a health endpoint, command-line server, Docker image, and local documentation.

Training, authentication, persistent multi-user storage, matchmaking, and remote
deployment are outside this phase. Sessions are intentionally ephemeral.

## Considered approaches

### Recommended: FastAPI plus a native Canvas client

FastAPI supplies a typed inference boundary and OpenAPI documentation while a
framework-free HTML/CSS/JavaScript client keeps the build small. A single Canvas
owns the board rendering and input mapping. This approach has the clearest path
from the current Python code to a runnable Docker image.

### React single-page application plus FastAPI

React would make a larger interface easier to scale, but introduces a Node build,
two dependency graphs, and generated assets for a one-screen product. It is not
justified by the phase requirements.

### Pure static browser simulation

A static-only demo is easy to host, but it cannot execute the repository's Python
models or demonstrate the PUCT and Gumbel implementations. It would be a visual
mock rather than a real AlphaZero product.

## Architecture

The optional `web` dependency group contains FastAPI, Uvicorn, and HTTPX for API
integration tests. `gomoku serve` imports these dependencies lazily so the base
game-engine installation remains lightweight.

`alphazero_gomoku.web.catalog` discovers a fixed allowlist of bundled models.
Legacy pickle parameters are loaded through the existing NumPy policy-value
network, so the default web experience does not require PyTorch. A heuristic
policy-value adapter offers an instant baseline and supports test isolation.

`alphazero_gomoku.web.service` owns in-memory `GameSession` instances. A session
contains the board, selected model, search algorithm, human side, move history,
and snapshots required for undo and replay. Public service methods are protected
by a lock because inference calls can arrive from multiple server threads.

`alphazero_gomoku.web.api` validates HTTP payloads and maps domain exceptions to
stable status codes. Static assets are served from the packaged `static`
directory. API handlers call the synchronous service through normal FastAPI
thread-pool execution; CPU-bound NumPy inference therefore does not block the
async event loop.

## API contract

### `GET /api/health`

Returns `{status: "ok", service: "alphazero-gomoku-web", version: 1}`.

### `GET /api/models`

Returns model descriptors with `id`, `name`, board dimensions, `n_in_row`, and
`kind`. Only repository-owned allowlisted paths can be selected.

### `POST /api/games`

Accepts:

```json
{
  "model_id": "alphazero-6x6",
  "human_color": "black",
  "search": "gumbel",
  "simulations": 32
}
```

`human_color` is `black` or `white`; `search` is `puct` or `gumbel`;
`simulations` is one of 8, 16, 32, or 64. If the human chooses white, the
response includes the AI's opening move.

### `GET /api/games/{game_id}`

Returns the current serializable state.

### `POST /api/games/{game_id}/moves`

Accepts `{row, column}` in a top-left browser coordinate system. It rejects an
occupied point, an out-of-range point, a finished game, or a request made while
it is not the human's turn. A valid request applies the human move and, unless
the game ends, the AI response in the same transaction.

### `POST /api/games/{game_id}/undo`

Rewinds one human turn: normally the last AI and human stones, or only the last
human stone when it ended the game. The returned board is always ready for the
human unless the human selected white and the initial AI opening is the only
move.

### `GET /api/games/{game_id}/replay`

Returns all immutable frames from the empty board through the current position.
Every frame includes stones, last move, active player, status, and winner.

## State representation

Board coordinates use top-left origin, rows increasing downward, and columns
increasing rightward. The legacy engine stores bottom-origin rows, so the service
is the only conversion boundary. Clients never depend on the engine's internal
move numbering.

The game response contains:

- `id`, `width`, `height`, `n_in_row`;
- `status` (`playing`, `black_won`, `white_won`, or `draw`);
- `current_color`, `human_color`, and `ai_color`;
- ordered `moves` with row, column, color, and ply;
- `last_move`, `move_count`, and `can_undo`;
- `analysis` with normalized row-major policy values, scalar value in `[-1, 1]`,
  analyzed color, and AI search latency in milliseconds;
- the chosen model, search algorithm, and simulation count.

Policy values for occupied points are zero. The legal values sum to one while a
game is active. Analysis is omitted after a terminal result.

## Search and inference behavior

Each AI turn creates a fresh search player from the selected runtime. This avoids
keeping mutable MCTS trees across HTTP requests and makes undo deterministic.
PUCT uses the existing `MCTSPlayer`; Gumbel uses `GumbelMCTSPlayer`. Both receive
the same configured simulation budget and seeded NumPy generator.

The heatmap uses the direct policy head on the returned position rather than
performing another search. This keeps responses responsive and makes the
visualization semantically clear: it is the network's prior for the side shown.
The displayed value is from that same side's perspective.

## Visual design

The interface uses an editorial "AI棋谱 / Machine Kifu" direction: warm rice-paper
tones, ink-black typography, mineral blue analysis marks, and a single vermilion
accent. The layout is deliberately asymmetric, with the square board as the
dominant object and a narrow instrument panel for controls and diagnostics.

The board is a responsive high-DPI Canvas with wood grain, crisp intersections,
star points, dimensional stones, a last-move marker, and translucent heat
contours. It supports pointer input, keyboard fullscreen (`F`), Escape exit,
reduced motion, mobile stacking, and visible focus states.

The page includes a subtle, clickable "Created By Deerflow" attribution as
required by the frontend design standard.

## Browser automation hooks

The client exposes:

- `window.render_game_to_text()`, returning concise JSON with coordinate rules,
  current state, legal interaction mode, visible moves, analysis, and replay;
- `window.advanceTime(ms)`, which advances deterministic UI animation time and
  redraws the Canvas;
- `F` fullscreen toggling with Canvas resizing.

These hooks are testing interfaces, not alternate game logic.

## Errors and safety

- Unknown models and sessions return 404.
- Invalid settings and moves return 422 or 409 with a user-readable `detail`.
- Model paths cannot be supplied by clients.
- Session identifiers use unguessable UUIDs.
- The client disables board input while a request is pending and preserves the
  last valid state if a request fails.
- Model loading is lazy and cached; a load failure is reported without crashing
  unrelated sessions.

## Verification

Completion requires:

- unit tests for coordinate conversion, catalog safety, state serialization,
  move validation, win/draw state, undo, and replay;
- API integration tests for health, models, game creation, legal/illegal moves,
  unknown sessions, and static index delivery;
- the existing full Python suite and Ruff;
- `python -m compileall`, CLI help, and Docker configuration validation;
- a real server smoke test;
- Playwright interaction bursts that create a game, place a stone, receive an AI
  response, toggle the heatmap, use undo, inspect replay, and toggle fullscreen;
- visual inspection of desktop and mobile gameplay screenshots;
- zero new browser console errors.


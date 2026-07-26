# Interactive web application

Phase 7 adds a real browser product on top of the repository's existing policy-value networks, PUCT, and Gumbel AlphaZero search.

## Run locally

```bash
python -m pip install -e ".[web]"
gomoku serve --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Use `gomoku serve --help` for host, port, and development reload options.

## Run with Docker

```bash
docker build -t alphazero-gomoku-lab .
docker run --rm -p 8000:8000 alphazero-gomoku-lab
```

The image runs as an unprivileged user and exposes a health check at `/api/health`.

## Product controls

- Select the tactical heuristic or a bundled 6x6/8x8 AlphaZero model.
- Choose black/white, Gumbel/PUCT, and an equal simulation budget.
- Click a board intersection to play; the AI response is returned in the same turn.
- Toggle the policy prior heatmap, undo one human turn, or inspect every replay frame.
- Press `F` to toggle board fullscreen and Escape to exit.

The heatmap is the direct normalized policy prior for the side to move. The value score is also from that side's perspective, not a calibrated win probability. Sessions are in memory and intentionally disappear when the server restarts.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Liveness and API version |
| GET | `/api/models` | Allowlisted model descriptors |
| POST | `/api/games` | Create a configured game |
| GET | `/api/games/{id}` | Read current state |
| POST | `/api/games/{id}/moves` | Apply a browser-coordinate move and AI response |
| POST | `/api/games/{id}/undo` | Rewind one human turn |
| GET | `/api/games/{id}/replay` | Return immutable replay frames |

Interactive OpenAPI documentation is available at `/docs`.

## Coordinates and safety

The browser API uses top-left origin with rows increasing downward. The service converts once at the legacy engine boundary. Clients select fixed model identifiers and cannot supply filesystem paths. Invalid state transitions return stable 404, 409, or 422 responses.

## Troubleshooting

- `Web dependencies are required`: install `.[web]`.
- A neural model fails to load: run from the repository root and confirm `models/` is present.
- Slow moves: lower the simulation budget to 8 or choose the heuristic model.
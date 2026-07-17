"""Append-only JSON Lines experiment metrics."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlMetricsWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, step: int, metrics: Mapping[str, Any]) -> None:
        if step < 0:
            raise ValueError("step must be non-negative")
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            **dict(metrics),
        }
        with self.path.open("a", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, allow_nan=False, sort_keys=True)
            file.write("\n")


def read_metrics(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)

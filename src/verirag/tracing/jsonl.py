from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path


class JsonlTracer:
    """Appends one JSON object per line to a file (the "JSON lines" format).

    One line per step keeps the file easy to read with any tool (`tail`, `jq`)
    and safe to append to: a crash can damage at most the last line.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, step: str, data: Mapping[str, object]) -> None:
        line = {
            "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "step": step,
            "data": dict(data),
        }
        # default=str: a value JSON cannot encode (a Path, an Enum) is stored as text
        # instead of crashing the run. A log must never break the thing it observes.
        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

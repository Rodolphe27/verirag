from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path


def load_env_file(path: Path, environ: MutableMapping[str, str]) -> None:
    """Read KEY=value lines from a .env file into `environ`.

    A variable that is already set in the real environment wins, so a value
    typed in the terminal overrides the file. Comments (#) and blank lines are
    skipped, and quotes around a value are removed. A missing file is fine.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        environ.setdefault(key.strip(), value.strip().strip("\"'"))

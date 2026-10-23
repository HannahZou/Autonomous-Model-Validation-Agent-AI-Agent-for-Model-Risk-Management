"""Append-only audit trail shared by the automated run and the human review."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> datetime:
    return datetime.now(timezone.utc)


class AuditLog:
    """Append-only JSONL trail of everything the system and the reviewer did."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **data: Any) -> None:
        record = {"ts": now().isoformat(), "event": kind, **data}
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

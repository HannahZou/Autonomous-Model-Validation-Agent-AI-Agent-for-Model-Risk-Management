from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mva.schemas import Severity, SR117Pillar


@dataclass
class Issue:
    """A check result before it is numbered and attached to an agent."""

    check: str
    title: str
    severity: Severity
    pillar: SR117Pillar
    description: str
    recommendation: str
    evidence: dict[str, Any] = field(default_factory=dict)
    affected: list[str] = field(default_factory=list)


def rnd(x: float, n: int = 4) -> float:
    """Round for evidence so reports stay readable and JSON-safe."""
    return float(round(float(x), n))

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


def fmt_p(p: float) -> str:
    """Readable p-value: never prints a misleading 'p = 0'."""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"

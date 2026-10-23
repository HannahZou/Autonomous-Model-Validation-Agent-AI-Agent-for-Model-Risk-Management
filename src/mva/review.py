"""Human-in-the-loop review of automated findings.

The agents propose findings; a human validator decides. Every finding must be
accepted, edited or rejected before the validation can be signed off, and every
decision is written to the run's audit trail. Rejections and severity changes
require a written rationale, and nothing can change after sign-off.

The Streamlit app is a thin layer over this module, so the rules here apply no
matter how the review is carried out.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from mva.audit import AuditLog, now
from mva.config import ReviewPolicy
from mva.schemas import Finding, Severity, ValidationRun

REVIEW_FILE = "review.json"


class ReviewError(Exception):
    """Raised when a review action breaks the review rules."""


class ReviewStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


class FindingReview(BaseModel):
    finding_id: str
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer: str | None = None
    comment: str | None = None
    reviewed_at: datetime | None = None
    severity_override: Severity | None = None
    title_override: str | None = None
    description_override: str | None = None
    recommendation_override: str | None = None


class OverallRating(str, Enum):
    SATISFACTORY = "Satisfactory"
    SATISFACTORY_WITH_FINDINGS = "Satisfactory with findings"
    UNSATISFACTORY = "Unsatisfactory"


RATING_MEANING = {
    OverallRating.SATISFACTORY: "The model is fit for its intended use.",
    OverallRating.SATISFACTORY_WITH_FINDINGS: (
        "The model may be used for its intended purpose; findings must be remediated by their due dates."
    ),
    OverallRating.UNSATISFACTORY: (
        "The model should not be relied on for its intended use until High-severity findings are remediated, "
        "or only under compensating controls approved by the model risk committee."
    ),
}


class SignOff(BaseModel):
    reviewer: str
    signed_at: datetime
    rating: OverallRating
    comment: str = ""


class ReviewState(BaseModel):
    run_id: str
    reviews: dict[str, FindingReview] = Field(default_factory=dict)
    sign_off: SignOff | None = None


class FinalFinding(BaseModel):
    """A finding as it will appear in the report, after the reviewer's decision."""

    finding: Finding
    review: FindingReview
    original_severity: Severity
    remediation_due: str | None = None  # ISO date


def _clean(text: str | None) -> str | None:
    text = (text or "").strip()
    return text or None


class ReviewSession:
    def __init__(self, run_dir: str | Path, policy: ReviewPolicy | None = None):
        self.run_dir = Path(run_dir)
        results = self.run_dir / "results.json"
        if not results.exists():
            raise ReviewError(f"No results.json in {self.run_dir}")
        self.run = ValidationRun.model_validate_json(results.read_text())
        self.policy = policy or ReviewPolicy()
        self.audit = AuditLog(self.run_dir / "audit.jsonl")
        path = self.run_dir / REVIEW_FILE
        if path.exists():
            self.state = ReviewState.model_validate_json(path.read_text())
        else:
            self.state = ReviewState(run_id=self.run.run_id)
        for f in self.run.findings:  # new findings start pending
            self.state.reviews.setdefault(f.finding_id, FindingReview(finding_id=f.finding_id))
        self._findings = {f.finding_id: f for f in self.run.findings}

    # -- queries ---------------------------------------------------------------
    @property
    def signed_off(self) -> bool:
        return self.state.sign_off is not None

    def review(self, finding_id: str) -> FindingReview:
        if finding_id not in self.state.reviews:
            raise ReviewError(f"Unknown finding {finding_id}")
        return self.state.reviews[finding_id]

    def pending(self) -> list[str]:
        return [fid for fid, r in self.state.reviews.items() if r.status is ReviewStatus.PENDING]

    def progress(self) -> dict[str, int]:
        counts = {s.value: 0 for s in ReviewStatus}
        for r in self.state.reviews.values():
            counts[r.status.value] += 1
        return counts

    def final_findings(self, include_rejected: bool = False) -> list[FinalFinding]:
        """Findings with the reviewer's edits applied, in report order (severity, then ID)."""
        base_date = (self.state.sign_off.signed_at if self.state.sign_off else now()).date()
        days = dict(self.policy.remediation_days)
        out = []
        for fid, f in self._findings.items():
            r = self.state.reviews[fid]
            if r.status is ReviewStatus.REJECTED and not include_rejected:
                continue
            updated = f.model_copy(update={
                k: v for k, v in {
                    "severity": r.severity_override,
                    "title": r.title_override,
                    "description": r.description_override,
                    "recommendation": r.recommendation_override,
                }.items() if v is not None
            })
            due = None
            if r.status in (ReviewStatus.ACCEPTED, ReviewStatus.EDITED) and updated.severity.value in days:
                due = (base_date + timedelta(days=days[updated.severity.value])).isoformat()
            out.append(FinalFinding(finding=updated, review=r, original_severity=f.severity, remediation_due=due))
        out.sort(key=lambda x: (-x.finding.severity.rank, x.finding.finding_id))
        return out

    def rating(self) -> OverallRating:
        sev = {x.finding.severity for x in self.final_findings() if x.review.status is not ReviewStatus.PENDING}
        if Severity.HIGH in sev:
            return OverallRating.UNSATISFACTORY
        if Severity.MEDIUM in sev:
            return OverallRating.SATISFACTORY_WITH_FINDINGS
        return OverallRating.SATISFACTORY

    # -- actions -------------------------------------------------------------------
    def _guard(self, finding_id: str, reviewer: str) -> FindingReview:
        if self.signed_off:
            raise ReviewError("The validation has been signed off; reviews can no longer change.")
        if not _clean(reviewer):
            raise ReviewError("A reviewer name is required.")
        return self.review(finding_id)

    def _record(self, r: FindingReview, action: str, reviewer: str, **extra) -> None:
        r.reviewer = reviewer.strip()
        r.reviewed_at = now()
        self.audit.event("review_decision", finding_id=r.finding_id, action=action, reviewer=r.reviewer,
                         comment=r.comment, **extra)
        self.save()

    def accept(self, finding_id: str, reviewer: str, comment: str | None = None) -> None:
        r = self._guard(finding_id, reviewer)
        r.status, r.comment = ReviewStatus.ACCEPTED, _clean(comment)
        r.severity_override = r.title_override = r.description_override = r.recommendation_override = None
        self._record(r, "accept", reviewer)

    def edit(self, finding_id: str, reviewer: str, comment: str, *, severity: Severity | str | None = None,
             title: str | None = None, description: str | None = None, recommendation: str | None = None) -> None:
        r = self._guard(finding_id, reviewer)
        if not _clean(comment):
            raise ReviewError("Editing a finding requires a rationale.")
        original = self._findings[finding_id]
        sev = Severity(severity) if severity is not None else None
        changes = {
            "severity_override": sev if sev is not None and sev != original.severity else None,
            "title_override": _clean(title) if _clean(title) != original.title else None,
            "description_override": _clean(description) if _clean(description) != original.description else None,
            "recommendation_override": (
                _clean(recommendation) if _clean(recommendation) != original.recommendation else None
            ),
        }
        if not any(v is not None for v in changes.values()):
            raise ReviewError("No change was made; use accept instead.")
        for k, v in changes.items():
            setattr(r, k, v)
        r.status, r.comment = ReviewStatus.EDITED, _clean(comment)
        self._record(
            r, "edit", reviewer,
            changed=[k.replace("_override", "") for k, v in changes.items() if v is not None],
            severity_before=original.severity.value,
            severity_after=(r.severity_override or original.severity).value,
        )

    def reject(self, finding_id: str, reviewer: str, comment: str) -> None:
        r = self._guard(finding_id, reviewer)
        if not _clean(comment):
            raise ReviewError("Rejecting a finding requires a rationale.")
        r.status, r.comment = ReviewStatus.REJECTED, _clean(comment)
        r.severity_override = r.title_override = r.description_override = r.recommendation_override = None
        self._record(r, "reject", reviewer)

    def reopen(self, finding_id: str, reviewer: str) -> None:
        r = self._guard(finding_id, reviewer)
        self.state.reviews[finding_id] = FindingReview(finding_id=finding_id)
        self.audit.event("review_decision", finding_id=finding_id, action="reopen", reviewer=reviewer.strip(),
                         previous_status=r.status.value)
        self.save()

    def sign_off(self, reviewer: str, comment: str = "") -> SignOff:
        if self.signed_off:
            raise ReviewError("Already signed off.")
        if not _clean(reviewer):
            raise ReviewError("A reviewer name is required.")
        if self.pending():
            raise ReviewError(f"{len(self.pending())} finding(s) still pending review: {', '.join(self.pending())}")
        failed = [r.agent for r in self.run.agent_reports if r.status != "completed"]
        if failed and not _clean(comment):
            raise ReviewError(f"Agent(s) failed ({', '.join(failed)}); explain in the sign-off comment.")
        self.state.sign_off = SignOff(reviewer=reviewer.strip(), signed_at=now(), rating=self.rating(),
                                      comment=(comment or "").strip())
        self.audit.event("signed_off", reviewer=self.state.sign_off.reviewer, rating=self.state.sign_off.rating.value,
                         progress=self.progress())
        self.save()
        return self.state.sign_off

    def save(self) -> None:
        path = self.run_dir / REVIEW_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.state.model_dump_json(indent=2))
        tmp.replace(path)

"""Build the sample validation reports in docs/sample_reports from the example packages.

The review decisions below are an illustrative demo review, recorded under the
name "Demo reviewer"; in real use every decision is made by a validator in the app.

Usage:  python scripts/make_sample_reports.py      (PDF needs LibreOffice)
"""

import shutil
import tempfile
from pathlib import Path

from mva.config import load_settings
from mva.orchestrator import Orchestrator
from mva.report import build_report, convert_to_pdf
from mva.review import ReviewSession

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "sample_reports"
REVIEWER = "Demo reviewer"

DEMO_DECISIONS = {
    "credit_pd": {
        "reject": {"DQ-001": "Conflicting IDs arise from the duplicate rows already raised in DQ-006; tracked there."},
        "edit": {"AS-003": ("Insignificance of utilisation is a symptom of the collinearity in AS-001.",
                            {"recommendation": "Resolve together with AS-001 (multicollinearity)."})},
    },
    "market_var": {
        "edit": {"DQ-006": ("Three missing prices out of ~1,000; forward-fill impact is immaterial.",
                            {"recommendation": "Document the forward-fill rule in the methodology."})},
    },
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        settings = load_settings(use_llm=False, runs_dir=Path(tmp))
        for name in ("credit_pd", "market_var"):
            _, run_dir = Orchestrator(settings).validate(ROOT / "examples" / name)
            s = ReviewSession(run_dir)
            plan = DEMO_DECISIONS.get(name, {})
            for fid, why in plan.get("reject", {}).items():
                s.reject(fid, REVIEWER, why)
            for fid, (why, changes) in plan.get("edit", {}).items():
                s.edit(fid, REVIEWER, why, **changes)
            for fid in s.pending():
                s.accept(fid, REVIEWER)
            s.sign_off(REVIEWER, "Illustrative review of a synthetic model package.")
            docx = build_report(run_dir, OUT / f"validation_report_{name}.docx")
            pdf = convert_to_pdf(docx)
            print("wrote", docx, pdf or "(no PDF: LibreOffice not found)")

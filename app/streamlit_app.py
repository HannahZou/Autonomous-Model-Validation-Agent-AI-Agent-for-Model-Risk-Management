"""Model validation review app (human in the loop).

    streamlit run app/streamlit_app.py

1. Run a validation on a model package (or open an earlier run).
2. Accept, edit or reject each proposed finding, with a rationale where required.
3. Sign off once nothing is pending, then download the validation report.

All rules (rationale required, no changes after sign-off, audit logging) live in
mva.review, so the app cannot bypass them.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from mva.config import load_settings
from mva.llm import OpenAIExecutiveSummaryWriter
from mva.orchestrator import Orchestrator
from mva.package import PackageError
from mva.report import build_report, convert_to_pdf
from mva.review import RATING_MEANING, ReviewError, ReviewSession, ReviewStatus
from mva.schemas import Severity
from mva.turnaround import turnaround

ROOT = Path(__file__).resolve().parents[1]
STATUS_ICON = {"pending": "⏳", "accepted": "✅", "edited": "✏️", "rejected": "❌"}
SEVERITY_ICON = {"High": "🔴", "Medium": "🟠", "Low": "⚪", "Info": "▫️"}

st.set_page_config(page_title="Model Validation Review", page_icon="🛡️", layout="wide")
settings = load_settings()


def list_packages() -> list[str]:
    examples = ROOT / "examples"
    return sorted(str(p.parent.relative_to(ROOT)) for p in examples.glob("*/model_package.yaml"))


def list_runs() -> list[Path]:
    runs_dir = settings.runs_dir if settings.runs_dir.is_absolute() else ROOT / settings.runs_dir
    if not runs_dir.exists():
        return []
    return sorted((p for p in runs_dir.iterdir() if (p / "results.json").exists()), reverse=True)


def act(fn, *args, **kwargs) -> None:
    """Run a review action and refresh, or show the rule it broke."""
    try:
        fn(*args, **kwargs)
    except ReviewError as exc:
        st.error(str(exc))
        return
    st.rerun()


# -- sidebar: run or open ------------------------------------------------------------
with st.sidebar:
    st.header("Run a validation")
    packages = list_packages()
    choice = st.selectbox("Model package", packages + ["Other folder…"])
    package = st.text_input("Package folder") if choice == "Other folder…" else choice
    use_llm = st.toggle("LLM narratives (OpenAI)", value=settings.llm_enabled, disabled=not settings.openai_api_key,
                        help="Set OPENAI_API_KEY in .env to enable.")
    if st.button("Run validation", type="primary", disabled=not package):
        with st.spinner("Running the five validation agents…"):
            try:
                pkg_path = Path(package) if Path(package).is_absolute() else ROOT / package
                runs_dir = settings.runs_dir if settings.runs_dir.is_absolute() else ROOT / settings.runs_dir
                _, out_dir = Orchestrator(replace(settings, use_llm=use_llm, runs_dir=runs_dir)).validate(pkg_path)
                st.session_state["run_dir"] = str(out_dir)
            except PackageError as exc:
                st.error(str(exc))

    st.header("Open a run")
    runs = list_runs()
    if runs:
        names = [p.name for p in runs]
        current = Path(st.session_state.get("run_dir", runs[0])).name
        idx = names.index(current) if current in names else 0
        picked = st.selectbox("Validation run", names, index=idx)
        st.session_state["run_dir"] = str(runs[names.index(picked)])
    else:
        st.caption("No runs yet.")

    st.header("Reviewer")
    reviewer = st.text_input("Your name", key="reviewer", placeholder="e.g. Hannah Zou")

if "run_dir" not in st.session_state:
    st.title("Model Validation Review")
    st.info("Run a validation or open an earlier run from the sidebar.")
    st.stop()

run_dir = Path(st.session_state["run_dir"])
session = ReviewSession(run_dir)
run = session.run
signed = session.state.sign_off

# -- header -------------------------------------------------------------------------------
st.title(run.model_name)
st.caption(f"{run.model_id} · {run.model_type} · run {run.run_id}")
progress = session.progress()
total = sum(progress.values())
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Overall rating", session.rating().value + ("" if signed else " *"))
c2.metric("Pending", progress["pending"])
c3.metric("Accepted", progress["accepted"])
c4.metric("Edited", progress["edited"])
c5.metric("Rejected", progress["rejected"])
st.progress((total - progress["pending"]) / total if total else 1.0,
            text="Signed off" if signed else f"{total - progress['pending']} of {total} findings reviewed")
if not signed:
    st.caption("* provisional until sign-off; computed from the findings reviewed so far.")

review_tab, signoff_tab, details_tab = st.tabs(["Review findings", "Sign-off & report", "Run details"])

# -- review ---------------------------------------------------------------------------------
with review_tab:
    finals = session.final_findings(include_rejected=True)
    f1, f2, f3 = st.columns(3)
    agents = f1.multiselect("Area", sorted({x.finding.agent for x in finals}))
    severities = f2.multiselect("Severity", [s.value for s in Severity if s is not Severity.INFO])
    statuses = f3.multiselect("Review status", [s.value for s in ReviewStatus],
                              default=[] if signed else ["pending"])
    shown = [
        x for x in finals
        if (not agents or x.finding.agent in agents)
        and (not severities or x.finding.severity.value in severities)
        and (not statuses or x.review.status.value in statuses)
    ]
    if not shown:
        st.success("Nothing to show with these filters." if finals else "The agents proposed no findings.")
    for x in shown:
        f, rv = x.finding, x.review
        label = (f"{STATUS_ICON[rv.status.value]} {f.finding_id} · {SEVERITY_ICON[f.severity.value]} "
                 f"{f.severity.value} · {f.title}")
        with st.expander(label, expanded=False):
            st.caption(f"{f.agent} · {f.pillar.value}")
            if x.original_severity != f.severity:
                st.caption(f"Severity changed by reviewer from {x.original_severity.value}.")
            st.markdown(f.description)
            if f.commentary:
                st.markdown(f"**AI commentary** · _generated, for the reviewer_  \n{f.commentary}")
            st.markdown(f"**Recommendation:** {f.recommendation}")
            with st.popover("Evidence"):
                st.json(f.evidence)
            if rv.status is not ReviewStatus.PENDING:
                st.info(f"{rv.status.value.capitalize()} by {rv.reviewer} on {rv.reviewed_at:%Y-%m-%d %H:%M} UTC"
                        + (f" — {rv.comment}" if rv.comment else ""))
            if signed:
                continue
            a, e, r = st.columns(3)
            with a.form(f"accept-{f.finding_id}"):
                note = st.text_input("Note (optional)", key=f"acc-note-{f.finding_id}")
                if st.form_submit_button("Accept"):
                    act(session.accept, f.finding_id, reviewer, note)
            with e.form(f"edit-{f.finding_id}"):
                levels = [s.value for s in Severity if s is not Severity.INFO]
                sev = st.selectbox("Severity", levels, index=levels.index(f.severity.value), key=f"sev-{f.finding_id}")
                title = st.text_input("Title", value=f.title, key=f"title-{f.finding_id}")
                desc = st.text_area("Description", value=f.description, key=f"desc-{f.finding_id}")
                rec = st.text_area("Recommendation", value=f.recommendation, key=f"rec-{f.finding_id}")
                why = st.text_area("Rationale (required)", key=f"why-{f.finding_id}")
                if st.form_submit_button("Save edit"):
                    act(session.edit, f.finding_id, reviewer, why, severity=sev, title=title,
                        description=desc, recommendation=rec)
            with r.form(f"reject-{f.finding_id}"):
                why_not = st.text_area("Why is this not a finding? (required)", key=f"rej-{f.finding_id}")
                if st.form_submit_button("Reject"):
                    act(session.reject, f.finding_id, reviewer, why_not)
            if rv.status is not ReviewStatus.PENDING and st.button("Reopen", key=f"reopen-{f.finding_id}"):
                act(session.reopen, f.finding_id, reviewer)

# -- sign-off and report ---------------------------------------------------------------------------
with signoff_tab:
    rating = session.rating()
    st.subheader(f"Overall rating: {rating.value}")
    st.write(RATING_MEANING[rating])
    if signed:
        st.success(f"Signed off by {signed.reviewer} on {signed.signed_at:%Y-%m-%d %H:%M} UTC.")
    else:
        pending = session.pending()
        if pending:
            st.warning(f"{len(pending)} finding(s) still pending: {', '.join(pending)}")
        comment = st.text_area("Sign-off comment", placeholder="Scope limitations, conditions of use, …")
        if st.button("Sign off validation", type="primary", disabled=bool(pending)):
            act(session.sign_off, reviewer, comment)

    st.divider()
    st.subheader("Validation report")
    st.caption("Word document; marked DRAFT until sign-off. PDF needs LibreOffice installed.")
    want_pdf = st.checkbox("Also produce PDF")
    if st.button("Generate report"):
        writer = (OpenAIExecutiveSummaryWriter(settings.openai_model, settings.openai_api_key)
                  if use_llm and settings.llm_enabled else None)
        with st.spinner("Building report…"):
            docx_path = build_report(run_dir, exec_writer=writer)
            st.session_state["report"] = str(docx_path)
            st.session_state["report_pdf"] = None
            if want_pdf:
                pdf = convert_to_pdf(docx_path)
                st.session_state["report_pdf"] = str(pdf) if pdf else None
                if pdf is None:
                    st.warning("LibreOffice not found; open the .docx and export it to PDF.")
    if st.session_state.get("report") and Path(st.session_state["report"]).exists():
        path = Path(st.session_state["report"])
        st.download_button("Download Word report", path.read_bytes(), file_name=path.name,
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        if st.session_state.get("report_pdf"):
            pdf = Path(st.session_state["report_pdf"])
            st.download_button("Download PDF", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf")

# -- run details --------------------------------------------------------------------------------------
with details_tab:
    st.subheader("Turnaround (from the audit trail)")
    ta = turnaround(run_dir)
    m1, m2, m3 = st.columns(3)
    m1.metric("Automated analysis", f"{ta.get('automated_seconds', 0):.1f} s")
    m2.metric("Active review", f"{ta['active_review_minutes']} min" if "active_review_minutes" in ta else "—")
    m3.metric("Start to sign-off", f"{ta['total_hours']} h" if "total_hours" in ta else "—")
    if ta.get("agent_seconds"):
        st.dataframe(pd.DataFrame({"Agent": list(ta["agent_seconds"]), "Seconds": list(ta["agent_seconds"].values())}),
                     hide_index=True)
    st.subheader("Agent outputs")
    for rep in run.agent_reports:
        with st.expander(f"{rep.agent} · {rep.status} · {len(rep.findings)} finding(s) · narrative: {rep.narrative_source}"):
            if rep.error:
                st.error(rep.error)
            st.write(rep.summary)
            st.json(rep.metrics, expanded=False)
    st.subheader("Audit trail")
    events = session.audit.events()
    st.dataframe(pd.DataFrame([{"time": e["ts"], "event": e["event"],
                                "detail": json.dumps({k: v for k, v in e.items() if k not in ("ts", "event")},
                                                     default=str)[:160]} for e in events]),
                 hide_index=True)

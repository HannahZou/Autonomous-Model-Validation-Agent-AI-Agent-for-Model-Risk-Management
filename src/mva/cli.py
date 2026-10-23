"""Command line entry point.

    mva validate examples/credit_pd            # run the five agents (OpenAI narratives if a key is set)
    mva validate examples/market_var --no-llm  # fully deterministic
    mva report runs/<run_id> [--pdf]           # Word report (DRAFT until signed off in the review app)
    mva turnaround runs/<run_id>               # timings from the audit trail
    mva benchmark examples/credit_pd examples/market_var --repeat 3
    mva make-examples                          # regenerate synthetic packages

Human review and sign-off happen in the Streamlit app:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import replace
from pathlib import Path

from mva.config import Settings, load_settings
from mva.orchestrator import Orchestrator
from mva.package import PackageError


def executive_writer(settings: Settings):
    if settings.llm_enabled:
        from mva.llm import OpenAIExecutiveSummaryWriter

        return OpenAIExecutiveSummaryWriter(model=settings.openai_model, api_key=settings.openai_api_key)
    return None


def _print_run(run, out_dir) -> None:
    c = run.severity_counts()
    print(f"{run.model_name} ({run.model_id})")
    for r in run.agent_reports:
        status = "" if r.status == "completed" else f"  [FAILED: {r.error}]"
        print(f"  {r.agent:<16} {len(r.findings):>2} findings  ({r.duration_seconds:.2f}s, narrative: {r.narrative_source}){status}")
        for f in r.findings:
            print(f"    {f.finding_id}  {f.severity.value:<6}  {f.title}")
    print(f"Total: {c['High']} High, {c['Medium']} Medium, {c['Low']} Low  in {run.duration_seconds:.2f}s")
    print(f"Saved to {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mva", description="Autonomous Model Validation Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a model package folder")
    v.add_argument("package", type=Path)
    v.add_argument("--no-llm", action="store_true", help="skip OpenAI; use template narratives")
    v.add_argument("--runs-dir", type=Path, default=None)

    r = sub.add_parser("report", help="build the Word (and optionally PDF) validation report for a run")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--pdf", action="store_true", help="also convert to PDF (needs LibreOffice)")
    r.add_argument("--no-llm", action="store_true", help="template executive summary")
    r.add_argument("--out", type=Path, default=None)

    t = sub.add_parser("turnaround", help="show timings for a run from its audit trail")
    t.add_argument("run_dir", type=Path)

    b = sub.add_parser("benchmark", help="time the automated pipeline on one or more packages")
    b.add_argument("packages", type=Path, nargs="+")
    b.add_argument("--repeat", type=int, default=3)
    b.add_argument("--runs-dir", type=Path, default=None)

    e = sub.add_parser("make-examples", help="write synthetic example packages")
    e.add_argument("--out", type=Path, default=Path("examples"))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "make-examples":
        from mva.synthetic import write_credit_package, write_market_package

        print("wrote", write_credit_package(args.out / "credit_pd"))
        print("wrote", write_market_package(args.out / "market_var"))
        return 0

    if args.command == "turnaround":
        from mva.turnaround import turnaround

        print(json.dumps(turnaround(args.run_dir), indent=2))
        return 0

    overrides = {}
    if getattr(args, "no_llm", False):
        overrides["use_llm"] = False
    if getattr(args, "runs_dir", None):
        overrides["runs_dir"] = args.runs_dir
    settings = load_settings(**overrides)

    if args.command == "report":
        from mva.report import build_report, convert_to_pdf
        from mva.review import ReviewError

        try:
            path = build_report(args.run_dir, args.out, exec_writer=executive_writer(settings))
        except ReviewError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Report: {path}")
        if args.pdf:
            pdf = convert_to_pdf(path)
            print(f"PDF: {pdf}" if pdf else "PDF skipped: LibreOffice not found (open the .docx and export to PDF)")
        return 0

    if args.command == "benchmark":
        # Deterministic narratives so the timing measures the analysis, not API latency
        settings = replace(settings, use_llm=False)
        results = {}
        for pkg in args.packages:
            times: dict[str, list[float]] = {}
            for _ in range(args.repeat):
                run, _ = Orchestrator(settings).validate(pkg)
                times.setdefault("total", []).append(run.duration_seconds)
                for rep in run.agent_reports:
                    times.setdefault(rep.agent, []).append(rep.duration_seconds)
            results[str(pkg)] = {k: round(statistics.median(v), 3) for k, v in times.items()}
            print(f"{pkg}: median of {args.repeat} runs")
            for k, val in results[str(pkg)].items():
                print(f"  {k:<26} {val:>8.3f} s")
        out = settings.runs_dir / "benchmark.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"repeat": args.repeat, "median_seconds": results}, indent=2))
        print(f"Saved to {out}")
        return 0

    try:
        run, out_dir = Orchestrator(settings).validate(args.package)
    except PackageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_run(run, out_dir)
    print(f"Next: review in the app (streamlit run app/streamlit_app.py) or draft a report: mva report {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

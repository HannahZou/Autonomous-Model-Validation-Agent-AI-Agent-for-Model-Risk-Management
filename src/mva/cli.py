"""Command line entry point.

    mva validate examples/credit_pd            # uses OpenAI if OPENAI_API_KEY is set
    mva validate examples/market_var --no-llm  # fully deterministic
    mva make-examples                          # regenerate synthetic packages
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mva.config import load_settings
from mva.orchestrator import Orchestrator
from mva.package import PackageError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mva", description="Autonomous Model Validation Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="validate a model package folder")
    v.add_argument("package", type=Path)
    v.add_argument("--no-llm", action="store_true", help="skip OpenAI; use template narratives")
    v.add_argument("--runs-dir", type=Path, default=None)

    e = sub.add_parser("make-examples", help="write synthetic example packages")
    e.add_argument("--out", type=Path, default=Path("examples"))

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "make-examples":
        from mva.synthetic import write_credit_package, write_market_package

        print("wrote", write_credit_package(args.out / "credit_pd"))
        print("wrote", write_market_package(args.out / "market_var"))
        return 0

    overrides = {}
    if args.no_llm:
        overrides["use_llm"] = False
    if args.runs_dir:
        overrides["runs_dir"] = args.runs_dir
    settings = load_settings(**overrides)

    try:
        run, out_dir = Orchestrator(settings).validate(args.package)
    except PackageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    c = run.severity_counts()
    print(f"{run.model_name} ({run.model_id})")
    for r in run.agent_reports:
        status = "" if r.status == "completed" else f"  [FAILED: {r.error}]"
        print(f"  {r.agent:<16} {len(r.findings):>2} findings  ({r.duration_seconds:.2f}s, narrative: {r.narrative_source}){status}")
        for f in r.findings:
            print(f"    {f.finding_id}  {f.severity.value:<6}  {f.title}")
    print(f"Total: {c['High']} High, {c['Medium']} Medium, {c['Low']} Low  in {run.duration_seconds:.2f}s")
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

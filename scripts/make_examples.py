"""Regenerate the example model packages in ./examples.

Usage:  python scripts/make_examples.py
"""

from pathlib import Path

from mva.synthetic import write_credit_package, write_market_package

ROOT = Path(__file__).resolve().parents[1] / "examples"

if __name__ == "__main__":
    print("wrote", write_credit_package(ROOT / "credit_pd"))
    print("wrote", write_market_package(ROOT / "market_var"))

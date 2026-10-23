"""Synthetic model packages for demos and tests (no real customer or market data)."""

from mva.synthetic.credit import make_credit_data, write_credit_package
from mva.synthetic.market import make_price_data, write_market_package

__all__ = ["make_credit_data", "write_credit_package", "make_price_data", "write_market_package"]

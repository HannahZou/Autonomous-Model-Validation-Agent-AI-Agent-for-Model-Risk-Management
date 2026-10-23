"""Validation report: a Word document (and PDF when LibreOffice is available)."""

from mva.report.builder import build_report, convert_to_pdf

__all__ = ["build_report", "convert_to_pdf"]

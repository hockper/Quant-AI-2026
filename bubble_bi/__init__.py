"""Bubble Bi — reading the stock market like a language.

The notebook at the repo root is the entry point. Everything it needs is
exposed here.
"""

from bubble_bi import verify
from bubble_bi.report import CheckFailed, report, run_tests
from bubble_bi.settings import check, device, summary

__all__ = [
    "check", "device", "summary",     # settings
    "verify",                         # per-section checks
    "report", "run_tests", "CheckFailed",
]

"""Bubble Bi — reading the stock market like a language.

The notebook at the repo root is the entry point. Everything it needs is here.
"""

from bubble_bi import data, diagnostics, models, plots, training, verify
from bubble_bi.training import train
from bubble_bi.report import CheckFailed, report, run_tests
from bubble_bi.settings import check, device, summary

__all__ = [
    "check", "device", "summary",     # the notebook's settings
    "data",                           # download / add_features / find_leaks
    "models",                         # the VQVAE, used as both TS and CS
    "verify",                         # the check that closes each section
    "report", "run_tests", "CheckFailed",
]

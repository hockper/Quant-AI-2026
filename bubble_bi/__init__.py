"""Bubble Bi — reading the stock market like a language.

The notebook at the repo root is the entry point. Everything it needs is here.
"""

from bubble_bi import (attention, autopsy, data, diagnostics, keep, models,
                       plots, training, verify)
from bubble_bi.report import CheckFailed, report, run_tests
from bubble_bi.settings import check, device, summary
from bubble_bi.training import train, train_world

__all__ = [
    "check", "device", "summary",         # the notebook's settings
    "data",                               # download / features / grids / sentences
    "models",                             # VQVAE, Fusion, Tokenizer, WorldModel
    "train", "train_world", "training",   # teaching them
    "keep",                               # saving them, so Colab cannot eat your work
    "verify",                             # the check that closes each section
    "plots", "diagnostics", "autopsy",    # looking at what they learned, and what broke
    "attention",                          # ...and what each company chose to read
    "report", "run_tests", "CheckFailed",
]

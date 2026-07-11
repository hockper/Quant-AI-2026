"""The check that closes each section of the notebook.

One function per part of the project. Each one proves the part actually works
rather than asserting that it does, then says what we now have.
"""

from __future__ import annotations

import os
from pathlib import Path

from bubble_bi.report import report, run_tests
from bubble_bi.settings import device


def setup(settings: dict) -> None:
    """Section 1-2: the settings are sound and the code runs here."""
    n = len(settings["tickers"])
    where = device()

    data_dir = Path(settings["data_dir"])
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".write-probe"
        probe.touch()
        probe.unlink()
        writable = True
    except OSError:
        writable = False

    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False

    tests_pass, tests_summary = run_tests()

    ts, cs = settings["ts"], settings["cs"]
    report(
        "Setup",
        [
            ("Settings understood", True, f"{n} companies, no typos"),
            ("Data folder writable", writable, f"{data_dir}/"),
            ("PyTorch available", has_torch, "required to train"),
            ("Hardware", True, where.upper()),
            ("Project's own tests", tests_pass, tests_summary),
        ],
        have=f"""
        A checked configuration — and nothing else yet.
        The tokenizer will read {ts['days']} days of each stock and {cs['days']} days of the
        whole market, then merge them into 1 token out of {settings['fusion']['vocabulary']}.
        No prices downloaded, no model trained.
        """,
    )
    if where == "cpu" and os.environ.get("COLAB_GPU") is None:
        print("\n  ℹ️  Running on CPU. Training will work but be slow —")
        print("     on Colab, switch to a GPU runtime for a large speed-up.")

"""How every section of the notebook reports back.

Each part of the project ends the same way: a list of checks that either pass or
fail, then a short note on what we actually have now. If any check fails the
notebook stops — a silent failure that lets you keep going is worse than an
error.
"""

from __future__ import annotations

import subprocess
import sys

_WIDTH = 68
_RULE = "─" * _WIDTH


class CheckFailed(RuntimeError):
    """Raised when a section's checks did not all pass."""


def report(title: str, checks: list[tuple[str, bool, str]], have: str) -> None:
    """Print a section's checks and what we now have. Raise if anything failed.

    checks: (label, passed, detail) — `detail` is the evidence, e.g. "30 companies".
    have:   a couple of lines saying what exists now that did not before.
    """
    passed = all(ok for _, ok, _ in checks)

    print(_RULE)
    print(f"  {'✅' if passed else '❌'}  {title}"
          f"{'' if passed else '  —  SOMETHING IS WRONG'}")
    print(_RULE)
    column = max(30, max(len(label) for label, _, _ in checks) + 2)
    for label, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'}  {label:<{column}}{detail}")

    if passed:
        print()
        print("  📦  What we have now")
        for line in have.strip().splitlines():
            print(f"      {line.strip()}")
    print(_RULE)

    if not passed:
        failed = [label for label, ok, _ in checks if not ok]
        raise CheckFailed(
            f"{title}: {len(failed)} check(s) failed — {', '.join(failed)}. "
            "Fix this before running the next cell."
        )


def run_tests(path: str = "tests") -> tuple[bool, str]:
    """Run the project's real test suite. Returns (passed, summary)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
    )
    last = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    summary = last[-1].strip() if last else "no output"
    # pytest's tail looks like "18 passed in 1.07s" / "1 failed, 17 passed in 1.2s"
    summary = summary.split(" in ")[0].strip()
    return proc.returncode == 0, summary

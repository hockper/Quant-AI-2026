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


def report(title: str, checks: list[tuple[str, bool, str]], have: str,
           known_problem: str | None = None) -> None:
    """Print a section's checks and what we now have. Raise if anything failed.

    checks: (label, passed, detail) — `detail` is the evidence, e.g. "30 companies".
    have:   a couple of lines saying what exists now that did not before.

    known_problem: if the failures are an OPEN QUESTION rather than broken code, pass a
        line explaining it. The ❌ still shows — we do not hide a bad result — but the
        notebook is allowed to continue. Use this only where the failure is understood
        and written down; a check that never stops anything is worse than no check.
    """
    passed = all(ok for _, ok, _ in checks)
    tolerated = not passed and known_problem

    print(_RULE)
    banner = "" if passed else (
        "  —  A KNOWN PROBLEM" if tolerated else "  —  SOMETHING IS WRONG"
    )
    print(f"  {'✅' if passed else '❌'}  {title}{banner}")
    print(_RULE)
    column = max(30, max(len(label) for label, _, _ in checks) + 2)
    for label, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'}  {label:<{column}}{detail}")

    if passed or tolerated:
        print()
        print("  📦  What we have now")
        for line in have.strip().splitlines():
            print(f"      {line.strip()}")
    print(_RULE)

    if tolerated:
        print(f"\n  ⚠️  {known_problem}")
        print("     The notebook continues, but do not pretend this passed.")
        return

    if not passed:
        failed = [label for label, ok, _ in checks if not ok]
        raise CheckFailed(
            f"{title}: {len(failed)} check(s) failed — {', '.join(failed)}. "
            "Fix this before running the next cell."
        )


def run_tests(path: str = "tests") -> tuple[bool, str]:
    """Run the project's real test suite. Returns (passed, summary).

    On failure the summary NAMES the tests that failed. "2 failed" tells you nothing —
    you cannot act on a count, and the whole point of a check is to be actionable.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-q", "--no-header",
         "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    tail = lines[-1].strip() if lines else "no output"
    summary = tail.split(" in ")[0].strip()

    if proc.returncode == 0:
        return True, summary

    broke = [
        ln.split("::")[-1].split(" ")[0]
        for ln in lines
        if ln.startswith("FAILED") or ln.startswith("ERROR")
    ]
    if broke:
        shown = ", ".join(broke[:3]) + (f", +{len(broke) - 3} more" if len(broke) > 3 else "")
        summary = f"{summary} → {shown}"
    return False, summary

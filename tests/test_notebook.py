"""The notebook IS the project, so its own cells get tested like anything else.

These do not run the notebook — training on a laptop is pointless and slow. They read it,
and check the things that go wrong silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import bubble_bi as bb
from bubble_bi.settings import DEFAULTS

NOTEBOOK = Path(__file__).resolve().parent.parent / "Bubble_Bi.ipynb"


def _cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text())["cells"]


def _settings_cell() -> str:
    """The SETTINGS cell — the one the reader actually edits."""
    for cell in _cells():
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.lstrip().startswith("SETTINGS = dict("):
            return source
    raise AssertionError("the notebook has no SETTINGS cell any more")


def _typed_settings() -> dict:
    scope: dict = {}
    exec(_settings_cell(), scope)          # noqa: S102 — it is our own notebook
    return scope["SETTINGS"]


def test_the_notebook_can_turn_the_search_on():
    """⚠️ THE BUG THIS EXISTS FOR.

    The notebook told the reader, twice, to `SETTINGS["search"]["run"] = True` — and
    `SETTINGS` had no `search` block at all. It raised `KeyError` and the search was
    simply unreachable. Nothing caught it, because `check()` fills the default (`run =
    False`) behind the scenes, so the notebook ran perfectly and just never searched.

    A setting the notebook instructs you to change must EXIST in the cell you change it in.
    """
    typed = _typed_settings()

    assert "search" in typed, (
        "SETTINGS has no `search` block, but the notebook tells the reader to set "
        "SETTINGS['search']['run'] = True. That raises KeyError, and the search can "
        "never be turned on."
    )
    typed["search"]["run"] = True          # exactly what the notebook says to do
    assert bb.check(typed)["search"]["run"] is True


def test_the_search_ships_switched_off():
    """It is a one-time act of discovery: everyone after inherits `tuned.json` by cloning.
    Shipping it ON would burn 15 minutes of a GPU session for every reader, unasked."""
    assert _typed_settings()["search"]["run"] is False


def test_every_setting_the_notebook_types_is_one_the_project_accepts():
    """`check()` rejects unknown keys, so a typo in the notebook stops it dead on cell one.
    This has bitten us: the SETTINGS cell once still nested `commitment` under `loss` after
    it had moved, and the notebook could not run at all in its committed state."""
    bb.check(_typed_settings())            # must not raise


@pytest.mark.parametrize("block", ["ts", "cs", "fusion", "predictor", "loss", "search"])
def test_the_notebook_never_invents_a_setting_that_does_not_exist(block):
    typed = _typed_settings()
    unknown = set(typed.get(block, {})) - set(DEFAULTS[block])
    assert not unknown, f"SETTINGS[{block!r}] invents settings the project has never heard of: {sorted(unknown)}"


def test_the_notebook_is_committed_unrun():
    """Stored outputs would carry stale numbers — and, once, a pasted token. It is run on
    Colab, never here."""
    dirty = [i for i, c in enumerate(_cells())
             if c.get("outputs") or c.get("execution_count")]
    assert not dirty, f"cells {dirty} have stored output — clear them before committing"

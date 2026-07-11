import pytest

import bubble_bi as bb
from bubble_bi.settings import DEFAULTS


def test_minimal_settings_get_every_default_filled_in():
    s = bb.check({"tickers": ["AAPL"]})
    assert s["ts"]["days"] == 4
    assert s["cs"]["days"] == 5
    assert s["fusion"]["vocabulary"] == 512
    assert s["predictor"]["sentence_length"] == 64
    assert s["model_size"] == 128
    assert set(s) == set(DEFAULTS)


def test_a_partial_block_keeps_the_other_defaults_in_that_block():
    s = bb.check({"tickers": ["AAPL"], "ts": {"days": 10}})
    assert s["ts"]["days"] == 10            # what you set
    assert s["ts"]["vocabulary"] == 512     # what you left out
    assert s["cs"]["days"] == 5             # the other entry is untouched


def test_the_two_entries_are_independent():
    s = bb.check({
        "tickers": ["AAPL"],
        "ts": {"days": 4, "vocabulary": 256, "encoder_depth": 5},
        "cs": {"days": 9, "vocabulary": 64, "encoder_depth": 1},
    })
    assert (s["ts"]["days"], s["ts"]["vocabulary"], s["ts"]["encoder_depth"]) == (4, 256, 5)
    assert (s["cs"]["days"], s["cs"]["vocabulary"], s["cs"]["encoder_depth"]) == (9, 64, 1)


def test_tickers_are_cleaned_and_deduplicated():
    s = bb.check({"tickers": [" aapl ", "MSFT", "aapl", "", "  "]})
    assert s["tickers"] == ["AAPL", "MSFT"]


def test_empty_tickers_is_rejected():
    with pytest.raises(ValueError, match="list at least one company"):
        bb.check({"tickers": []})


def test_a_typo_in_a_setting_name_is_caught_not_silently_ignored():
    with pytest.raises(ValueError, match="Unknown setting"):
        bb.check({"tickers": ["AAPL"], "modelsize": 64})


def test_a_typo_inside_a_block_is_caught_too():
    with pytest.raises(ValueError, match=r"`ts` got unknown setting"):
        bb.check({"tickers": ["AAPL"], "ts": {"dayz": 4}})


@pytest.mark.parametrize("bad", [
    {"ts": {"days": 0}},
    {"cs": {"encoder_depth": -1}},
    {"predictor": {"sentence_length": 0}},
    {"model_size": 0},
    {"steps": 0},
])
def test_nonsense_sizes_are_rejected(bad):
    with pytest.raises(ValueError, match="at least 1"):
        bb.check({"tickers": ["AAPL"], **bad})


@pytest.mark.parametrize("block", ["ts", "cs", "fusion"])
def test_a_vocabulary_below_two_is_rejected(block):
    with pytest.raises(ValueError, match="at least 2"):
        bb.check({"tickers": ["AAPL"], block: {"vocabulary": 1}})


def test_booleans_are_not_accepted_as_sizes():
    # True == 1 in Python, so a bare isinstance(int) check would let this through.
    with pytest.raises(ValueError, match="whole number"):
        bb.check({"tickers": ["AAPL"], "ts": {"days": True}})


def test_summary_names_both_entries_and_the_merged_token():
    text = bb.summary(bb.check({"tickers": ["AAPL", "MSFT"]}))
    assert "2 companies" in text
    assert "TS" in text and "CS" in text
    assert "ONE token" in text


def test_device_reports_something_we_can_run_on():
    assert bb.device() in {"cpu", "gpu", "tpu"}

import inspect

import pytest

import bubble_bi as bb
from bubble_bi.models import VQVAE
from bubble_bi.settings import DEFAULTS


def test_minimal_settings_get_every_default_filled_in():
    s = bb.check({"tickers": ["AAPL"]})
    assert s["ts"]["days"] == 15
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


def test_no_setting_in_an_entry_block_is_decorative():
    """A setting no model reads is worse than no setting: it LIES.

    This is the test that would have caught the bug this whole spec exists for.
    `loss['commitment']` sat in SETTINGS for the entire project, was validated on every
    run, and was handed to nothing. We trained at commitment=1.0 while reading 0.25.
    """
    accepted = set(inspect.signature(VQVAE.__init__).parameters)
    for entry in ("ts", "cs"):
        unread = set(DEFAULTS[entry]) - accepted
        assert not unread, (
            f"SETTINGS[{entry!r}] contains settings VQVAE never reads: {sorted(unread)}. "
            "Either wire them up or delete them — a setting that does nothing is a lie."
        )


def test_the_codebook_knobs_actually_reach_the_codebook():
    model = VQVAE(companies=1, features=6, width=16,
                  **{**DEFAULTS["ts"], "commitment": 0.9, "diversity": 0.7, "decay": 0.5})
    assert model.codebook.commitment == 0.9
    assert model.codebook.diversity == 0.7
    assert model.codebook.decay == 0.5


def test_commitment_defaults_to_the_literature_value():
    """0.25, not 1.0. An over-strong commitment pins the encoder to the codebook and is a
    documented cause of the collapse we see in fusion."""
    assert DEFAULTS["ts"]["commitment"] == 0.25
    assert DEFAULTS["cs"]["commitment"] == 0.25
    assert VQVAE(companies=1, days=4, features=6, width=16).codebook.commitment == 0.25


def test_no_loss_weight_is_decorative():
    """Same disease as the entry blocks: the predictor's weights must reach the predictor."""
    from bubble_bi.models.world import WorldModel

    accepted = set(inspect.signature(WorldModel.__init__).parameters)
    unread = set(DEFAULTS["loss"]) - accepted
    assert not unread, (
        f"SETTINGS['loss'] contains weights WorldModel never reads: {sorted(unread)}."
    )


def test_the_optimiser_uses_the_weight_decay_setting():
    """It was hardcoded to 0.01 while STORM uses 0.05, and no setting existed at all."""
    import inspect as _inspect

    from bubble_bi import training

    source = _inspect.getsource(training)
    assert "weight_decay=0.01" not in source, (
        "training.py still hardcodes weight_decay=0.01 — it must come from settings."
    )
    assert DEFAULTS["weight_decay"] == 0.05


def test_no_fusion_setting_is_decorative():
    """The same trap as the entry blocks. This used to need two different checks,
    because `Tokenizer.__init__` took a whole `settings` dict rather than `**fusion` --
    `settings` is always in the signature, read or not, so a plain `inspect.signature`
    diff couldn't tell us anything. Now that `Tokenizer` takes the fusion block as
    explicit keyword arguments (the same convention as `VQVAE`), the honest
    signature-diff check used for ts/cs above works here too.

    Kept alongside it: build a Tokenizer with distinctive fusion values and check they
    actually reached the codebook/fusion modules, not just that they were accepted.
    """
    from bubble_bi.models import VQVAE
    from bubble_bi.models.world import Tokenizer

    ts = VQVAE(companies=1, days=4, features=6, width=16, heads=2)
    cs = VQVAE(companies=3, days=4, features=6, width=16, heads=2)
    tokenizer = Tokenizer(
        ts, cs, model_size=16, vocabulary=17, depth=3, attend_to="companies",
        commitment=0.81, diversity=0.62, decay=0.53,
    )

    assert tokenizer.codebook.words == 17
    assert tokenizer.codebook.commitment == 0.81
    assert tokenizer.codebook.diversity == 0.62
    assert tokenizer.codebook.decay == 0.53
    assert len(tokenizer.fusion.rounds) == 3
    assert tokenizer.attend_to == "companies"

    accepted = set(inspect.signature(Tokenizer.__init__).parameters)
    unread = set(DEFAULTS["fusion"]) - accepted
    assert not unread, (
        f"DEFAULTS['fusion'] contains settings Tokenizer never reads: {sorted(unread)}. "
        "Either wire them up or delete them -- a setting that does nothing is a lie."
    )

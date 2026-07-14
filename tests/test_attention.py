import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from bubble_bi.attention import _key_labels, neighbours, plot

TICKERS = ["AAPL", "MSFT", "JPM", "BAC", "NVDA"]


def _read(attention, how="companies", cs_days=5):
    attention = np.asarray(attention, dtype=float)
    keys = attention.shape[1]
    flat = 1.0 / keys
    sharpness = float(np.nanmean(np.abs(attention - flat)) / flat)
    return {
        "attention": attention,
        "tickers": TICKERS,
        "attend_to": how,
        "keys": keys,
        "flat": flat,
        "sharpness": sharpness,
        "choosing": sharpness > 0.15,
        "cs_days": cs_days,
    }


# ------------------------------------------------- choosing, or not choosing

def test_a_flat_map_is_reported_as_not_choosing():
    """The failure that never shows up in a loss curve. If every company reads the whole
    market equally, the cross-attention may as well not be there."""
    read = _read(np.full((5, 5), 0.2))            # perfectly uniform
    assert not read["choosing"]
    assert read["sharpness"] < 0.01


def test_a_map_with_real_preferences_is_reported_as_choosing():
    attention = np.full((5, 5), 0.05)
    for i in range(5):
        attention[i, (i + 1) % 5] = 0.80          # each company reads one other, hard
    read = _read(attention)
    assert read["choosing"]


def test_the_plot_says_out_loud_when_the_attention_is_flat():
    fig = plot(_read(np.full((5, 5), 0.2)))
    printed = " ".join(t.get_text() for t in fig.axes[0].texts)
    title = fig.axes[0].get_title(loc="left")
    assert "not choosing" in title
    assert "none of the work it was built for" in printed


def test_the_plot_says_so_when_it_IS_choosing():
    attention = np.full((5, 5), 0.05)
    attention[:, 0] = 0.80
    fig = plot(_read(attention))
    assert "it is choosing" in fig.axes[0].get_title(loc="left")


# --------------------------------------------------------- reading the map

def test_it_names_who_each_company_reads_most():
    attention = np.full((5, 5), 0.05)
    attention[2, 3] = 0.80                        # JPM reads BAC
    attention[4, 0] = 0.80                        # NVDA reads AAPL

    table = neighbours(_read(attention), top=1)
    assert "BAC" in table.loc["JPM", "reads most"]
    assert "AAPL" in table.loc["NVDA", "reads most"]


def test_a_company_reading_itself_is_reported_separately():
    # Self-attention is real and interesting, but it must not crowd out the answer to
    # "which OTHER companies does it read".
    attention = np.full((5, 5), 0.05)
    attention[0, 0] = 0.90                        # AAPL mostly reads itself
    attention[0, 1] = 0.30                        # ...then MSFT

    table = neighbours(_read(attention), top=1)
    assert "MSFT" in table.loc["AAPL", "reads most"]     # itself is excluded
    assert table.loc["AAPL", "reads itself"] == "90.0%"


def test_asking_who_it_reads_is_refused_when_the_keys_are_days():
    with pytest.raises(ValueError, match="attend_to"):
        neighbours(_read(np.full((5, 5), 0.2), how="days"))


def test_the_keys_are_labelled_with_the_tickers_when_they_are_companies():
    assert _key_labels(_read(np.full((5, 5), 0.2))) == TICKERS


def test_the_cells_map_folds_150_keys_into_two_readable_halves():
    rng = np.random.default_rng(0)
    attention = rng.random((5, 5 * 3))            # 5 companies x 3 days = 15 keys
    attention /= attention.sum(axis=1, keepdims=True)
    fig = plot(_read(attention, how="cells", cs_days=3))
    assert len(fig.axes) >= 2                     # one for companies, one for days


# ------------------------------------------- the bug that made the map a lie

def test_every_company_is_measured_not_just_the_first_few():
    """A real bug, caught by looking at the output -- and no longer possible to build.

    Under the old cached-latent design, sentences were served one COMPANY at a time,
    ordered by company, so sampling only the first few batches reached only the first
    few tickers. Under joint training a batch is one time WINDOW across ALL companies
    at once (see `WorldModel.forward`), so that ordering bug cannot happen any more —
    every batch structurally carries every company as its own axis.

    What replaces it: `gather()` must actually READ that axis correctly, rather than
    silently mislabelling it. This builds a tiny multi-company sentence loader (raw
    grids, the new batch shape) and checks every company comes back finite.

    ⚠️ Two things this test used to miss entirely, both now closed:

    1. `isfinite` alone can basically never fail any more -- the NaN-for-unmeasured-
       company path this test was ORIGINALLY written to catch is gone. It stayed green
       out of habit, not because it was still checking anything.
    2. The old fixture used `attend_to="companies"`, where `keys == companies` by
       construction -- so the attention map came back perfectly SQUARE. A `gather()`
       that silently swapped the company and key axes (mislabelling "who reads whom",
       the one thing this diagnostic exists to produce) would have passed unchanged:
       same shape, still finite.

    So this uses `attend_to="cells"` (`keys = companies * cs_days`, deliberately NOT
    equal to `companies`) and checks that every company's row of the map sums to 1
    across the KEY axis -- true of a real softmax output, and false of one whose axes
    have been swapped. With `companies != keys`, a literal axis transpose is not even
    shape-compatible any more: it fails loudly instead of quietly mislabelling.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    import bubble_bi as bb
    from bubble_bi.attention import gather

    companies, width, ts_days, cs_days, features, sentence = 5, 16, 2, 3, 6, 4

    class _Sentences(Dataset):
        """One item = one window of `sentence` days, every company at once — exactly
        how `bubble_bi.data.make_sentences` will serve them."""
        def __init__(self, n=8):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return {
                "ts_grid": torch.randn(sentence, companies, 1, ts_days, features),
                "cs_grid": torch.randn(sentence, companies, cs_days, features),
                "cs_present": torch.ones(sentence, companies, dtype=torch.bool),
                "candle": torch.randn(sentence, companies, 4),
            }

    ts = bb.models.VQVAE(companies=1, days=ts_days, features=features, vocabulary=16,
                         width=width, heads=2)
    cs = bb.models.VQVAE(companies=companies, days=cs_days, features=features,
                         vocabulary=16, width=width, heads=2)
    tokenizer = bb.models.Tokenizer(ts, cs, model_size=width, heads=2,
                                    attend_to="cells")
    world = bb.models.WorldModel(tokenizer, sentence=sentence, depth=1, heads=2)

    # `make_sentences()` returns a plain {period: DataLoader} dict, not nested under a
    # "loaders" key -- see the note on `gather()` about why that nesting used to exist
    # and why it was wrong to keep expecting it.
    book = {"test": DataLoader(_Sentences(), batch_size=3, shuffle=False)}
    settings = bb.check({"tickers": [f"T{i}" for i in range(companies)]})

    read = gather(world, book, None, settings)
    keys = companies * cs_days                       # attend_to="cells", deliberately != companies
    assert read["attention"].shape == (companies, keys)
    # THE assertion: every company must have been measured, not just the first few.
    assert np.isfinite(read["attention"]).all(), "a company was never reached"
    # Each row is an average of softmax shares, which sum to 1 across the keys THAT
    # company read. A company/key axis swap would still be finite, but would no longer
    # sum to 1 here (a random map is essentially never doubly-stochastic).
    assert np.allclose(read["attention"].sum(axis=1), 1.0, atol=1e-4), (
        "a company's row does not sum to 1 across the keys it read -- the company and "
        "key axes may have been swapped"
    )

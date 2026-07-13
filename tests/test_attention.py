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
    """A real bug, caught by looking at the output.

    The sentences are ordered BY COMPANY. Sampling the first N batches therefore only
    ever reaches the first few companies, and every company after them comes back blank
    — while the summary numbers quietly get computed on that biased sample. A diagnostic
    that measures the wrong thing is worse than no diagnostic.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    import bubble_bi as bb
    from bubble_bi.attention import gather
    from bubble_bi.settings import DEFAULTS

    companies, width, keys, days = 5, 16, 4, 6

    class _Ordered(Dataset):
        """Sentences laid out company by company — exactly like the real ones."""
        def __init__(self, per_company=20):
            self.who = np.repeat(np.arange(companies), per_company)

        def __len__(self):
            return len(self.who)

        def __getitem__(self, i):
            return {
                "z_ts": torch.randn(days, width),
                "market": torch.randn(days, keys, width),
                "candle": torch.randn(days, 4),
                "company": torch.tensor(int(self.who[i])),
                "last_day": torch.tensor(days - 1),
            }

    ts = bb.models.VQVAE(companies=1, days=4, features=6, vocabulary=16,
                         width=width, heads=2)
    cs = bb.models.VQVAE(companies=companies, days=keys, features=6, vocabulary=16,
                         width=width, heads=2)
    fusion = {**DEFAULTS["fusion"], "vocabulary": 16, "depth": 1}
    world = bb.models.WorldModel(
        bb.models.Tokenizer(ts, cs, model_size=width, heads=2, **fusion),
        sentence=days, depth=1, heads=2,
    )

    book = {"loaders": {"test": DataLoader(_Ordered(), batch_size=4, shuffle=False)}}
    settings = bb.check({"tickers": ["A", "B", "C", "D", "E"]})

    read = gather(world, book, None, settings)
    assert read["attention"].shape == (companies, keys)
    # THE assertion: the LAST company must have been measured too.
    assert np.isfinite(read["attention"]).all(), "a company was never reached"

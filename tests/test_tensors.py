import numpy as np
import pandas as pd
import pytest
import torch

import bubble_bi as bb
from bubble_bi.data.tensors import Scaler, make_tensors, split_days, to_arrays, tuning_loaders


@pytest.fixture
def table():
    """Three companies that look nothing like each other — as real ones do not.

    A penny stock, a blue chip and a mid-cap: prices spanning 25x, volumes spanning
    100x, and quite different volatilities. If the fixture made them interchangeable
    it could not catch the very thing this module exists to fix.
    """
    rng = np.random.default_rng(0)
    days = pd.date_range("2015-01-01", periods=700, freq="B", name="date")
    shape = {                       # ticker: (price level, daily vol, volume, range)
        "AAA": (20.0, 0.030, 5e7, 0.03),      # cheap, wild, heavily traded
        "BBB": (500.0, 0.008, 3e5, 0.005),    # expensive, sleepy, thin
        "CCC": (100.0, 0.015, 4e6, 0.012),    # middle of the road
    }
    frames = []
    for ticker, (level, vol, turnover, band) in shape.items():
        close = level * np.exp(np.cumsum(rng.normal(0, vol, len(days))))
        wobble = rng.uniform(0.4, 1.0, len(days))          # the daily range varies
        one = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, vol / 3, len(days))),
                "high": close * (1 + band * wobble),
                "low": close * (1 - band * wobble),
                "close": close,
                "volume": (turnover * rng.uniform(0.5, 1.5, len(days))),
            },
            index=days,
        )
        one["target"] = one["close"].shift(-1) / one["close"] - 1.0
        one["ticker"] = ticker
        frames.append(one.reset_index())
    raw = pd.concat(frames).set_index(["date", "ticker"]).sort_index()
    settings = bb.check({"tickers": ["AAA", "BBB", "CCC"]})
    return bb.data.add_features(raw, settings), settings


@pytest.fixture
def batches(table):
    """A real `Batches` built from the small synthetic table above."""
    data, settings = table
    return make_tensors(data, settings)


def test_the_table_becomes_a_dense_grid(table):
    data, settings = table
    a = to_arrays(data, settings)
    assert a.x.shape == (len(a.dates), 3, len(bb.data.names()))      # days × companies × features
    assert a.y.shape == (len(a.dates), 3)
    assert a.ok.shape == (len(a.dates), 3)
    assert a.ok.any()


# ---------------------------------------------------------- leak 1: shuffling time

def test_the_three_periods_are_in_time_order_and_never_overlap(table):
    data, settings = table
    a = to_arrays(data, settings)
    days = split_days(a, settings)

    assert days["learn"].max() < days["tune"].min()
    assert days["tune"].max() < days["test"].min()
    for period in ("learn", "tune", "test"):
        assert (np.diff(days[period]) > 0).all()   # strictly increasing: real time order


def test_no_days_are_shared_between_periods(table):
    data, settings = table
    days = split_days(to_arrays(data, settings), settings)
    everything = np.concatenate(list(days.values()))
    assert len(everything) == len(set(everything.tolist()))


# ------------------------------------------- leak 2: a target reaching over the border

def test_the_last_day_of_each_period_is_dropped(table):
    # Its target is TOMORROW's return -- and tomorrow belongs to the next period.
    data, settings = table
    a = to_arrays(data, settings)
    days = split_days(a, settings)
    total = len(a.dates)
    learn_end = int(total * settings["split"]["learn"])

    assert days["learn"].max() == learn_end - 2      # not learn_end - 1: the border day is gone
    assert days["tune"].min() == learn_end           # and the next period starts cleanly
    assert days["test"].max() == total - 2           # the very last day has no tomorrow


# -------------------------------------------- leak 3: scaling with the future

def test_the_scale_is_measured_on_the_learn_period_only(table):
    data, settings = table
    a = to_arrays(data, settings)
    days = split_days(a, settings)
    scaler = Scaler(a, days["learn"])

    # Sabotage the FUTURE only. If the scaler were looking at it, its numbers would move.
    tampered = to_arrays(data, settings)
    tampered.x[days["test"]] += 1000.0
    after = Scaler(tampered, days["learn"])

    assert np.allclose(scaler.middle, after.middle)
    assert np.allclose(scaler.spread, after.spread)


def test_the_later_periods_are_scaled_with_the_learn_numbers_not_their_own(table):
    # They must NOT be re-centred on themselves -- that would be quietly re-fitting on
    # data we are pretending not to have seen.
    data, settings = table
    batches = make_tensors(data, settings)
    a, days, scaler = batches.arrays, batches.days, batches.scaler
    scaled = scaler.apply(a.x)

    by_hand = (a.x[days["test"]] - scaler.middle[None]) / scaler.spread[None]
    usable = a.ok[days["test"]]
    assert np.allclose(scaled[days["test"]][usable], by_hand[usable].astype(np.float32),
                       atol=1e-5)


# ----------------------------------------------------------------- the grids

def test_ts_grids_fit_the_ts_model(table):
    data, settings = table
    batches = make_tensors(data, settings)
    model = bb.models.VQVAE(companies=1, features=len(bb.data.names()), width=settings["model_size"],
                            **settings["ts"])
    sample = next(iter(batches.ts["learn"]))
    assert sample["grid"].shape[1:] == (1, settings["ts"]["days"], len(bb.data.names()))
    out = model(sample)                                    # must simply work
    assert out["ids"].shape == (sample["grid"].shape[0],)


def test_cs_grids_fit_the_cs_model(table):
    data, settings = table
    batches = make_tensors(data, settings)
    model = bb.models.VQVAE(companies=3, features=len(bb.data.names()), width=settings["model_size"],
                            **settings["cs"])
    sample = next(iter(batches.cs["learn"]))
    assert sample["grid"].shape[1:] == (3, settings["cs"]["days"], len(bb.data.names()))
    assert sample["present"].shape[1:] == (3,)
    out = model(sample)
    assert out["ids"].shape == (sample["grid"].shape[0],)


def test_a_ts_grid_holds_that_companys_own_history_and_only_its_own(table):
    data, settings = table
    batches = make_tensors(data, settings)
    scaled = batches.scaler.apply(batches.arrays.x)
    window = settings["ts"]["days"]

    item = batches.ts["learn"].dataset[0]
    t, j = int(item["day"]), int(item["company"])
    expected = scaled[t - window + 1: t + 1, j]            # this company, these days
    assert np.allclose(item["grid"].numpy()[0], expected)


def test_a_grid_never_contains_a_day_after_its_own(table):
    # The window must END on its day, not straddle it.
    data, settings = table
    batches = make_tensors(data, settings)
    scaled = batches.scaler.apply(batches.arrays.x)
    window = settings["ts"]["days"]

    for i in (0, 5, 50):
        item = batches.ts["learn"].dataset[i]
        t, j = int(item["day"]), int(item["company"])
        got = item["grid"].numpy()[0]
        assert np.allclose(got[-1], scaled[t, j])          # last row IS the day itself
        future = scaled[t + 1: t + 1 + window, j]
        for row in got:
            assert not any(np.allclose(row, f) for f in future)


def test_only_companies_that_traded_are_marked_present(table):
    data, settings = table
    a = to_arrays(data, settings)
    a.ok[:, 1] = False                                     # company BBB never trades
    batches = make_tensors(data, settings)
    batches.arrays.ok[:, 1] = False
    from bubble_bi.data.tensors import CSGrids
    grids = CSGrids(batches.arrays, batches.scaler.apply(batches.arrays.x),
                    batches.days["learn"], settings["cs"]["days"])
    assert len(grids) > 0
    assert not bool(grids[0]["present"][1])                # and the model must ignore it


def test_every_period_has_samples(table):
    data, settings = table
    batches = make_tensors(data, settings)
    sizes = batches.sizes()
    assert (sizes["TS samples"] > 0).all()
    assert (sizes["CS samples"] > 0).all()
    assert sizes.loc["learn", "TS samples"] > sizes.loc["test", "TS samples"]


def test_learn_is_shuffled_but_test_keeps_its_order(table):
    data, settings = table
    batches = make_tensors(data, settings)
    assert batches.ts["learn"].sampler.__class__.__name__ == "RandomSampler"
    assert batches.ts["test"].sampler.__class__.__name__ == "SequentialSampler"


def test_a_feature_that_never_moves_is_left_alone_not_blown_up(table):
    # A near-constant feature has a standard deviation made of pure float noise.
    # Dividing by it would amplify that noise into +/-1 garbage and feed it to the
    # model as if it were signal.
    data, settings = table
    a = to_arrays(data, settings)
    days = split_days(a, settings)
    a.x[:, :, 3] = 7.0                                   # feature 3 never changes

    scaler = Scaler(a, days["learn"], a.names)
    assert scaler.constant[:, 3].all()                   # flat for every company
    assert (scaler.spread[:, 3] == 1.0).all()
    assert a.names[3] in scaler.flat_features

    scaled = scaler.apply(a.x)
    usable = scaled[a.ok]                                # warm-up rows are blank by design
    assert np.abs(usable[:, 3]).max() < 1e-6            # stays at zero, not +/-1
    assert np.isfinite(usable).all()


# ------------------------------------------ normalising each company against itself

def test_each_company_is_normalised_against_its_own_history(table):
    data, settings = table
    batches = make_tensors(data, settings)
    a, days, scaler = batches.arrays, batches.days, batches.scaler

    assert scaler.middle.shape == (3, len(bb.data.names()))        # one average PER COMPANY, per feature
    assert scaler.spread.shape == (3, len(bb.data.names()))

    scaled = scaler.apply(a.x)
    learn_ok = a.ok[days["learn"]]
    moves = ~scaler.constant
    for j in range(3):                            # every company, on its own
        mine = scaled[days["learn"]][:, j][learn_ok[:, j]]
        keep = moves[j]
        assert np.allclose(mine.mean(axis=0)[keep], 0, atol=1e-4)
        assert np.allclose(mine.std(axis=0)[keep], 1, atol=1e-3)


def test_a_companys_scale_is_not_affected_by_the_other_companies(table):
    # The whole point of "time only, not spatial": company AAA's numbers must not
    # move when company CCC is sabotaged.
    data, settings = table
    a = to_arrays(data, settings)
    days = split_days(a, settings)
    before = Scaler(a, days["learn"], a.names)

    tampered = to_arrays(data, settings)
    tampered.x[:, 2, :] *= 1000.0                 # company CCC goes berserk
    after = Scaler(tampered, days["learn"], a.names)

    assert np.allclose(before.middle[0], after.middle[0])    # AAA untouched
    assert np.allclose(before.spread[0], after.spread[0])
    assert not np.allclose(before.middle[2], after.middle[2])  # CCC did change


def test_normalising_erases_which_company_a_row_belongs_to(table):
    # Before: a company's average level identifies it, and the codebook would waste
    # words memorising "this is BBB". After: every company sits at its own zero.
    data, settings = table
    batches = make_tensors(data, settings)
    a, days = batches.arrays, batches.days
    scaled = batches.scaler.apply(a.x)

    def company_share(x, rows):
        """How much of a feature's variance is just WHICH company it is?"""
        x = np.where(a.ok[rows][:, :, None], x[rows], np.nan)
        per_company = np.nanmean(x, axis=0)                        # [N, F]
        return np.nanvar(per_company, axis=0) / np.nanvar(x, axis=(0, 1))

    learn = days["learn"]
    assert np.nanmax(company_share(a.x, learn)) > 0.10             # identity WAS there
    assert np.nanmax(company_share(scaled, learn)) < 1e-3          # and is gone

    # Over the WHOLE of history it does not vanish completely, and that is honest:
    # the scale may only be measured on the learn period, so a company that genuinely
    # drifts afterwards is allowed to sit off-centre. We are not permitted to peek at
    # the future to flatten it.
    everything = np.arange(len(a.dates))
    before = np.nanmax(company_share(a.x, everything))
    after = np.nanmax(company_share(scaled, everything))
    assert after < before / 3


def test_a_company_with_almost_no_history_is_refused_not_guessed(table):
    data, settings = table
    a = to_arrays(data, settings)
    days = split_days(a, settings)
    a.ok[:, 1] = False
    a.ok[:5, 1] = True                            # BBB has 5 usable days. Not enough.
    with pytest.raises(ValueError, match="Too little history"):
        Scaler(a, days["learn"], a.names)


def test_a_tiny_but_varying_feature_is_not_mistaken_for_a_flat_one(table):
    """The bug this guards against.

    Illiquidity for a mega-cap sits around 1e-6 while varying by 30% of itself --
    real, useful signal. An absolute "is it flat?" threshold threw it away and told
    the notebook it carried no information. Flatness must be judged relative to the
    feature's own size.
    """
    data, settings = table
    a = to_arrays(data, settings)
    days = split_days(a, settings)

    rng = np.random.default_rng(0)
    tiny = 1e-6 * (1 + 0.3 * rng.normal(size=a.x.shape[:2]))   # small, but ALIVE
    a.x[:, :, 5] = tiny.astype(np.float32)

    scaler = Scaler(a, days["learn"], a.names)
    assert not scaler.constant[:, 5].any()            # kept, not zeroed
    assert a.names[5] not in scaler.flat_features

    scaled = scaler.apply(a.x)[a.ok]
    assert scaled[:, 5].std() > 0.5                   # and it still varies once scaled


# ----------------------------------------------- tuning_loaders: search cannot see test

def test_the_search_cannot_reach_the_test_days(batches):
    """Not 'does not read test' -- CANNOT. The loader is never built.

    A search that could reach the test period would quietly tune on the answer, and the
    only defence that actually holds is not constructing the thing.
    """
    loaders = tuning_loaders(batches, "ts", days=4, batch=8)
    assert set(loaders) == {"learn", "tune"}
    assert "test" not in loaders


def test_tuning_loaders_rebuild_at_a_new_window_length(batches):
    short = tuning_loaders(batches, "ts", days=3, batch=8)
    long = tuning_loaders(batches, "ts", days=6, batch=8)
    assert next(iter(short["learn"]))["grid"].shape[2] == 3
    assert next(iter(long["learn"]))["grid"].shape[2] == 6


def test_tuning_loaders_serve_the_market_grid_for_cs(batches):
    loaders = tuning_loaders(batches, "cs", days=3, batch=4)
    grid = next(iter(loaders["learn"]))["grid"]
    assert grid.shape[1] == len(batches.arrays.tickers)   # every company, together
    assert grid.shape[2] == 3


def test_an_unknown_entry_is_rejected(batches):
    with pytest.raises(ValueError, match="'ts' or 'cs'"):
        tuning_loaders(batches, "fusion", days=3, batch=8)

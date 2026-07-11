import numpy as np
import pandas as pd
import pytest
import torch

import bubble_bi as bb
from bubble_bi.data.tensors import Scaler, make_tensors, split_days, to_arrays


@pytest.fixture
def table():
    """A small but realistic feature table: 3 companies, enough days to warm up."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2015-01-01", periods=700, freq="B", name="date")
    frames = []
    for ticker in ("AAA", "BBB", "CCC"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        one = pd.DataFrame(
            {
                "open": close * 0.999, "high": close * 1.01,
                "low": close * 0.99, "close": close,
                "volume": rng.integers(1e6, 5e6, len(days)).astype(float),
            },
            index=days,
        )
        one["target"] = one["close"].shift(-1) / one["close"] - 1.0
        one["ticker"] = ticker
        frames.append(one.reset_index())
    raw = pd.concat(frames).set_index(["date", "ticker"]).sort_index()
    settings = bb.check({"tickers": ["AAA", "BBB", "CCC"]})
    return bb.data.add_features(raw, settings), settings


def test_the_table_becomes_a_dense_grid(table):
    data, settings = table
    a = to_arrays(data, settings)
    assert a.x.shape == (len(a.dates), 3, 22)      # days × companies × features
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


def test_the_learn_period_is_centred_and_the_later_periods_are_left_alone(table):
    data, settings = table
    batches = make_tensors(data, settings)
    a, days = batches.arrays, batches.days
    scaled = batches.scaler.apply(a.x)

    learn = scaled[days["learn"]][a.ok[days["learn"]]]
    moves = ~batches.scaler.constant                       # a constant feature stays at 0
    assert np.allclose(learn.mean(axis=0), 0, atol=1e-4)   # centred, by construction
    assert np.allclose(learn.std(axis=0)[moves], 1, atol=1e-3)

    # The later periods get the SAME numbers applied to them -- they are not re-centred
    # on themselves, which would be re-fitting on data we are pretending not to have.
    test = scaled[days["test"]][a.ok[days["test"]]]
    by_hand = ((a.x[days["test"]][a.ok[days["test"]]] - batches.scaler.middle)
               / batches.scaler.spread)
    assert np.allclose(test, by_hand)


# ----------------------------------------------------------------- the grids

def test_ts_grids_fit_the_ts_model(table):
    data, settings = table
    batches = make_tensors(data, settings)
    model = bb.models.VQVAE(companies=1, features=22, width=settings["model_size"],
                            **settings["ts"])
    sample = next(iter(batches.ts["learn"]))
    assert sample["grid"].shape[1:] == (1, settings["ts"]["days"], 22)
    out = model(sample)                                    # must simply work
    assert out["ids"].shape == (sample["grid"].shape[0],)


def test_cs_grids_fit_the_cs_model(table):
    data, settings = table
    batches = make_tensors(data, settings)
    model = bb.models.VQVAE(companies=3, features=22, width=settings["model_size"],
                            **settings["cs"])
    sample = next(iter(batches.cs["learn"]))
    assert sample["grid"].shape[1:] == (3, settings["cs"]["days"], 22)
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
    assert scaler.constant[3]
    assert scaler.spread[3] == 1.0
    assert a.names[3] in scaler.flat_features

    scaled = scaler.apply(a.x)
    usable = scaled[a.ok]                                # warm-up rows are blank by design
    assert np.abs(usable[:, 3]).max() < 1e-6            # stays at zero, not +/-1
    assert np.isfinite(usable).all()

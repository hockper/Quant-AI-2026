from bubble_bi.data import FAMILIES, by_family, names
from bubble_bi.settings import DEFAULTS


def test_by_family_covers_every_feature_exactly_once():
    grouped = by_family(DEFAULTS)
    assert set(grouped) == set(FAMILIES)

    flat = [n for group in grouped.values() for n in group]
    assert sorted(flat) == sorted(names()), "a feature is in two families, or in none"


def test_by_family_knows_where_the_candle_and_the_volatility_are():
    grouped = by_family(DEFAULTS)
    assert grouped["candle"] == ["gap", "body", "upper_wick", "lower_wick"]
    assert "realized_vol" in grouped["volatility"]

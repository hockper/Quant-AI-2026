from bubble_bi.config import DataConfig
from bubble_bi.data.universe import load_universe


def test_load_universe_returns_configured_tickers():
    cfg = DataConfig(tickers=["AAPL", "MSFT", "AAPL"])
    assert load_universe(cfg) == ["AAPL", "MSFT"]  # dedup, order preserved


def test_load_universe_rejects_empty():
    import pytest
    with pytest.raises(ValueError):
        load_universe(DataConfig(tickers=[]))

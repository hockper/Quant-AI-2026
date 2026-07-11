"""Proving that nothing looks into the future.

This is the most important test in the project.

If a feature on Monday secretly uses Tuesday's price, the model will appear to
predict the market brilliantly — and will lose money the moment it meets a real
Tuesday it has not already seen. It is the classic way a financial model fools
the person who built it.

So we do not *assert* that our features are backward-looking. We prove it, on the
real data, every run:

    take the data → delete the future → recompute everything
    → check that not a single past value changed.

If a feature peeks ahead, deleting the future changes its past values, and this
catches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_leaks(prices: pd.DataFrame, settings: dict, cut: float = 0.6) -> list[str]:
    """Recompute the features with the future deleted. Return any that changed.

    An empty list means every feature is backward-looking. Anything in the list
    is looking at data it could not have had.
    """
    from bubble_bi.data.features import add_features

    dates = prices.index.get_level_values("date").unique().sort_values()
    cutoff = dates[int(len(dates) * cut)]

    full = add_features(prices, settings)
    past_only = add_features(prices[prices.index.get_level_values("date") <= cutoff], settings)

    full = full[full.index.get_level_values("date") <= cutoff].sort_index()
    past_only = past_only.sort_index()

    leaking = []
    for column in past_only.columns:
        if column == "target":
            continue                       # the target looks forward on purpose
        a = full[column].to_numpy(dtype=float)
        b = past_only[column].to_numpy(dtype=float)
        both_blank = np.isnan(a) & np.isnan(b)
        if not np.allclose(a[~both_blank], b[~both_blank], atol=1e-8, equal_nan=True):
            leaking.append(column)
    return leaking

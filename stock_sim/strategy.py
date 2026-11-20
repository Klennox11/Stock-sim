"""Moving-average crossover strategy.

Rule (long-only):
  * If the FAST average is above the SLOW average at today's close -> we want to own the stock.
  * Otherwise -> we want to be in cash.
The order is filled at the NEXT day's open, so we never use information we couldn't have had.
"""
import pandas as pd


def sma_signals(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    if fast >= slow:
        raise ValueError("fast must be smaller than slow")
    out = prices.copy()
    out["sma_fast"] = out["Close"].rolling(fast).mean()
    out["sma_slow"] = out["Close"].rolling(slow).mean()
    # 1 = want to hold, 0 = want cash. NaN warm-up period counts as cash.
    out["want_long"] = (out["sma_fast"] > out["sma_slow"]).astype(int)
    out.loc[out["sma_slow"].isna(), "want_long"] = 0
    return out

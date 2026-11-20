"""Download daily price history (Yahoo Finance) and cache it to CSV."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_prices(ticker: str, start: str = "2005-01-01", refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame with columns Open, High, Low, Close, Volume indexed by date."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"{ticker.upper().replace('^', '_')}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(
            f"No data for '{ticker}'. Check the symbol (Canadian stocks end in .TO, e.g. SHOP.TO) "
            "and your internet connection."
        )
    if isinstance(df.columns, pd.MultiIndex):  # yfinance returns (field, ticker) columns
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    df.to_csv(cache)
    return df


def demo_prices(years: int = 15, seed: int = 7) -> pd.DataFrame:
    """Made-up random-walk prices, for testing without internet. NOT real data."""
    rng = np.random.default_rng(seed)
    n = years * 252
    dates = pd.bdate_range("2010-01-04", periods=n)
    rets = rng.normal(0.0003, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=pd.Index(dates, name="Date"),
    )

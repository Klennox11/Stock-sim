"""Run a strategy on past prices and measure how it did, after fees and slippage."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import Costs


@dataclass
class Trade:
    date: pd.Timestamp
    side: str
    shares: int
    price: float
    fee: float


@dataclass
class Result:
    equity: pd.Series              # account value each day
    trades: list[Trade] = field(default_factory=list)
    total_fees: float = 0.0


def run_backtest(sig: pd.DataFrame, start_cash: float = 10_000.0, costs: Costs | None = None) -> Result:
    """Simulate day by day. Decision at close of day t, order filled at open of day t+1."""
    costs = costs or Costs()
    cash, shares = start_cash, 0
    equity, trades, fees = [], [], 0.0
    opens, closes, want = sig["Open"].values, sig["Close"].values, sig["want_long"].values
    dates = sig.index

    for i in range(len(sig)):
        if i > 0:  # act on yesterday's decision at today's open
            target = want[i - 1]
            if target == 1 and shares == 0:
                px = costs.fill_price(opens[i], "BUY")
                n = int(cash // (px * 1.001))  # tiny buffer so the fee fits
                if n > 0:
                    fee = costs.commission(n, px)
                    cash -= n * px + fee
                    shares, fees = n, fees + fee
                    trades.append(Trade(dates[i], "BUY", n, px, fee))
            elif target == 0 and shares > 0:
                px = costs.fill_price(opens[i], "SELL")
                fee = costs.commission(shares, px)
                cash += shares * px - fee
                fees += fee
                trades.append(Trade(dates[i], "SELL", shares, px, fee))
                shares = 0
        equity.append(cash + shares * closes[i])

    return Result(pd.Series(equity, index=dates, name="equity"), trades, fees)


def buy_and_hold(prices: pd.DataFrame, start_cash: float = 10_000.0, costs: Costs | None = None) -> pd.Series:
    """Benchmark: buy on day 1, never sell. Pays the same entry cost."""
    costs = costs or Costs()
    px = costs.fill_price(prices["Open"].iloc[0], "BUY")
    n = int(start_cash // (px * 1.001))
    cash = start_cash - n * px - costs.commission(n, px)
    return (cash + n * prices["Close"]).rename("buy_hold")


def metrics(equity: pd.Series, trades: list[Trade] | None = None) -> dict:
    rets = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    out = {
        "Final value": equity.iloc[-1],
        "Total return": total,
        "Yearly return (CAGR)": cagr,
        "Max drawdown": max_dd,
        "Sharpe ratio": sharpe,
    }
    if trades is not None:
        # pair BUY/SELL into round trips to get win rate
        wins = rounds = 0
        buy = None
        for t in trades:
            if t.side == "BUY":
                buy = t
            elif buy is not None:
                rounds += 1
                pnl = (t.price - buy.price) * t.shares - t.fee - buy.fee
                wins += pnl > 0
                buy = None
        out["Round trips"] = rounds
        out["Win rate"] = wins / rounds if rounds else 0.0
    return out

"""Step 2: 'Was it luck?' -- test many moving-average settings on many tickers.

For every (ticker, fast, slow) we compare the strategy against buy & hold on:
  * yearly return (CAGR) difference, in percentage points
  * Sharpe ratio difference (return per unit of risk)

Then an honest 'walk-forward' check: pick the BEST setting using only the first 70%
of history, and see how that same setting did on the last 30% it never saw.
"""
from dataclasses import dataclass

import pandas as pd

from .backtest import buy_and_hold, metrics, run_backtest
from .costs import Costs
from .strategy import sma_signals

FASTS = [10, 20, 50]
SLOWS = [50, 100, 150, 200]


@dataclass
class Cell:
    ticker: str
    fast: int
    slow: int
    d_cagr: float      # strategy CAGR - buy&hold CAGR (full period)
    d_sharpe: float    # strategy Sharpe - buy&hold Sharpe (full period)
    train_sharpe: float
    test_d_cagr: float
    test_d_sharpe: float


def _rebased(s: pd.Series) -> pd.Series:
    return s / s.iloc[0]


def sweep_ticker(ticker: str, prices: pd.DataFrame, costs: Costs, cash: float = 10_000) -> list[Cell]:
    bh = buy_and_hold(prices, cash, costs)
    cut = int(len(prices) * 0.7)
    bh_full, bh_test = metrics(bh), metrics(_rebased(bh.iloc[cut:]))
    cells = []
    for f in FASTS:
        for s in SLOWS:
            if f >= s:
                continue
            eq = run_backtest(sma_signals(prices, f, s), cash, costs).equity
            m_full = metrics(eq)
            m_train = metrics(_rebased(eq.iloc[:cut]))
            m_test = metrics(_rebased(eq.iloc[cut:]))
            cells.append(Cell(
                ticker, f, s,
                m_full["Yearly return (CAGR)"] - bh_full["Yearly return (CAGR)"],
                m_full["Sharpe ratio"] - bh_full["Sharpe ratio"],
                m_train["Sharpe ratio"],
                m_test["Yearly return (CAGR)"] - bh_test["Yearly return (CAGR)"],
                m_test["Sharpe ratio"] - bh_test["Sharpe ratio"],
            ))
    return cells

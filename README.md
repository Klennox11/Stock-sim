# STOCK-SIM

A paper-trading stock simulator (no real money). Learning / portfolio project.

**Strategy:** moving-average crossover. Own the stock when the 50-day average price is above the
200-day average, otherwise hold cash. Decisions are made at the close and filled at the next day's
open, so the backtest never "sees the future".

**Honesty features**
- Commissions modelled on Interactive Brokers fixed pricing ($0.005/share, $1 minimum).
- Slippage (default 0.05% per trade, adjustable with `--slippage`).
- Always compared against simply buying and holding.
- Results split into the first 70% and the last 30% of history, so you can see whether it holds up.

## Run it (Windows PowerShell)

```powershell
uv run main.py backtest SPY
uv run main.py backtest XIU.TO --fast 20 --slow 100
uv run main.py backtest --demo      # fake prices, no internet
uv run main.py sweep                # Step 2: many settings x many tickers
uv run main.py sweep SPY XIU.TO ZEB.TO
```

## Roadmap
1. [x] Backtest with fees, slippage, benchmark and out-of-sample check
2. [x] Parameter sweep + walk-forward check (is the result luck or robust?)
   - Lesson: on purely random fake prices (`sweep --demo`), ~40% of settings still "beat"
     buy & hold. Some winners always appear by luck, so one good backtest proves nothing.
3. [ ] Live paper trading with a simulated account + live HUD dashboard
4. [ ] Connect to Alpaca / Interactive Brokers paper accounts

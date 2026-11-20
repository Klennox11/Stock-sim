"""STOCK-SIM command line.

Examples:
    uv run main.py backtest SPY
    uv run main.py backtest SHOP.TO --fast 20 --slow 100
    uv run main.py backtest --demo        (fake prices, no internet needed)
"""
import argparse

from rich.columns import Columns
from rich.text import Text

from stock_sim.backtest import buy_and_hold, metrics, run_backtest
from stock_sim.costs import Costs
from stock_sim.data import demo_prices, load_prices
from stock_sim.hud import compare_table, console, panel, sparkline


def cmd_backtest(a: argparse.Namespace) -> None:
    name = "DEMO (fake data)" if a.demo else a.ticker.upper()
    with console.status(f"[hud]Loading price history for {name}...[/]"):
        prices = demo_prices() if a.demo else load_prices(a.ticker, start=a.start, refresh=a.refresh)

    from stock_sim.strategy import sma_signals

    costs = Costs(slippage_bps=a.slippage)
    sig = sma_signals(prices, a.fast, a.slow)
    res = run_backtest(sig, a.cash, costs)
    bh = buy_and_hold(prices, a.cash, costs)

    console.print(panel(
        Text.assemble(
            ("STOCK-SIM ", "bold cyan"), ("// BACKTEST ", "bold green"), (f"// {name}\n", "bold white"),
            (f"Strategy: buy when {a.fast}-day average > {a.slow}-day average, else cash. "
             f"Fills at next day's open.\n", "white"),
            (f"Period: {prices.index[0]:%Y-%m-%d} -> {prices.index[-1]:%Y-%m-%d}   "
             f"Start cash: ${a.cash:,.0f}   Slippage: {a.slippage} bps   "
             f"Fees paid: ${res.total_fees:,.2f}", "grey50"),
        ), "MISSION"))

    # Full period
    s_full = metrics(res.equity, res.trades)
    s_full["Time in market"] = float(sig["want_long"].mean())
    tables = [compare_table("FULL PERIOD", s_full, metrics(bh))]

    # Honesty check: split into first 70% ("in-sample") and last 30% ("out-of-sample")
    cut = int(len(prices) * 0.7)
    split_date = prices.index[cut]
    for label, sl in (("FIRST 70%", slice(None, cut)), ("LAST 30% (unseen)", slice(cut, None))):
        e, b = res.equity.iloc[sl], bh.iloc[sl]
        e, b = e / e.iloc[0] * a.cash, b / b.iloc[0] * a.cash  # restart each half at start cash
        tables.append(compare_table(label, metrics(e), metrics(b)))
    console.print(Columns(tables))

    console.print(panel(
        Text.assemble(
            ("Strategy   ", "bold cyan"), (sparkline(res.equity) + "\n", "cyan"),
            ("Buy & Hold ", "bold green"), (sparkline(bh), "green"),
        ), "EQUITY CURVES"))

    log = Text()
    for t in res.trades[-10:]:
        color = "green" if t.side == "BUY" else "cyan"
        log.append(f"{t.date:%Y-%m-%d}  {t.side:<4}  {t.shares:>6} sh @ ${t.price:>9,.2f}  fee ${t.fee:.2f}\n", color)
    console.print(panel(log or Text("No trades.", "grey50"), f"LAST 10 OF {len(res.trades)} TRADES"))

    # Plain-English verdict on the unseen part
    s_test, b_test = metrics(res.equity.iloc[cut:]), metrics(bh.iloc[cut:])
    beat_ret = s_test["Yearly return (CAGR)"] > b_test["Yearly return (CAGR)"]
    beat_risk = s_test["Sharpe ratio"] > b_test["Sharpe ratio"]
    if beat_ret and beat_risk:
        verdict, style = "Beat buy & hold on BOTH return and risk-adjusted return in the unseen period.", "bold green"
    elif beat_risk:
        verdict, style = "Lower return than buy & hold, but better risk-adjusted (smoother ride).", "bold cyan"
    elif beat_ret:
        verdict, style = "Higher return than buy & hold, but a bumpier ride (worse Sharpe).", "bold cyan"
    else:
        verdict, style = "No edge: buy & hold did better in the unseen period. (This is the most common result!)", "bold red"
    console.print(panel(Text(f"Unseen period starts {split_date:%Y-%m-%d}. {verdict}", style=style), "VERDICT", "green"))


def main() -> None:
    p = argparse.ArgumentParser(prog="stock-sim", description="Paper-trading stock simulator")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="test the strategy on past prices")
    b.add_argument("ticker", nargs="?", default="SPY", help="symbol, e.g. SPY, AAPL, SHOP.TO, XIU.TO")
    b.add_argument("--fast", type=int, default=50, help="fast moving-average length in days (default 50)")
    b.add_argument("--slow", type=int, default=200, help="slow moving-average length in days (default 200)")
    b.add_argument("--cash", type=float, default=10_000, help="starting cash (default 10000)")
    b.add_argument("--slippage", type=float, default=5.0, help="slippage in basis points (default 5 = 0.05%%)")
    b.add_argument("--start", default="2005-01-01", help="first date to download (default 2005-01-01)")
    b.add_argument("--refresh", action="store_true", help="re-download prices instead of using the saved copy")
    b.add_argument("--demo", action="store_true", help="use fake prices (no internet)")
    b.set_defaults(func=cmd_backtest)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

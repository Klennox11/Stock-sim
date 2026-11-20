"""Step 3: live paper trading with a team of 4 bots and a HUD dashboard.

  DATA bot      downloads the latest price bars for the watchlist
  STRATEGY bot  checks the moving-average crossover on each ticker -> BUY / SELL / HOLD
  RISK bot      vetoes orders that break the rules (market hours, size, loss limit, cooldown)
  TRADER bot    fills approved orders in the fake account (with fees + slippage)

Modes:
  day   = 5-minute bars, 10/30 crossover, all positions closed before the 4pm bell
  swing = daily bars, 50/200 crossover, positions can be held for weeks
"""
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .hud import console, panel, sparkline
from .paper import DB, DEMO_DB, PaperAccount
from .strategy import sma_signals

ET = ZoneInfo("America/New_York")
MODES = {
    "day": dict(interval="5m", period="5d", fast=10, slow=30, poll=60),
    "swing": dict(interval="1d", period="2y", fast=50, slow=200, poll=300),
}


@dataclass
class Rules:
    max_position_pct: float = 0.25   # never put more than 25% of the account in one stock
    daily_loss_limit: float = 0.03   # stop buying for the day after a 3% loss
    cooldown_min: int = 15           # wait 15 min after trading a ticker before trading it again
    flatten_at: tuple = (15, 55)     # day mode: sell everything at 3:55pm ET
    no_new_after: tuple = (15, 45)   # day mode: no new buys after 3:45pm ET


@dataclass
class State:
    logs: dict = field(default_factory=lambda: {b: deque(maxlen=25) for b in ("DATA", "STRATEGY", "RISK", "TRADER")})
    status: dict = field(default_factory=lambda: {b: "starting" for b in ("DATA", "STRATEGY", "RISK", "TRADER")})
    prices: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    equity_hist: list = field(default_factory=list)
    next_cycle: float = 0.0
    cycle: int = 0
    day_start_equity: float | None = None
    day: object = None
    last_trade: dict = field(default_factory=dict)
    clock: str = ""

    def log(self, bot: str, msg: str, style: str = "white") -> None:
        self.logs[bot].append((datetime.now().strftime("%H:%M:%S"), msg, style))


# ---------------------------------------------------------------- data feeds
class YahooFeed:
    def __init__(self, interval: str, period: str):
        self.interval, self.period = interval, period

    def now(self) -> datetime:
        return datetime.now(ET)

    def market_open(self) -> bool:
        n = self.now()
        return n.weekday() < 5 and (9, 30) <= (n.hour, n.minute) < (16, 0)  # ignores holidays

    def bars(self, ticker: str) -> pd.DataFrame:
        import yfinance as yf
        df = yf.download(ticker, period=self.period, interval=self.interval, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


class DemoFeed:
    """Fake random-walk market that is always open. Each cycle = one new 5-minute bar."""

    def __init__(self, tickers: list[str]):
        self.rng = np.random.default_rng(42)
        self.t = datetime(2026, 1, 5, 9, 30, tzinfo=ET)
        idx = pd.date_range(end=self.t, periods=120, freq="5min")
        self.data = {}
        for i, tk in enumerate(tickers):
            close = (50 + 50 * i) * np.exp(np.cumsum(self.rng.normal(0, 0.003, len(idx))))
            self.data[tk] = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                                          "Volume": 1000}, index=idx)
        self.ticked = set()

    def now(self) -> datetime:
        return self.t

    def market_open(self) -> bool:
        return True

    def bars(self, ticker: str) -> pd.DataFrame:
        df = self.data[ticker]
        if ticker not in self.ticked:  # one new bar per ticker per cycle
            last = df["Close"].iloc[-1] * np.exp(self.rng.normal(0.0002, 0.004))
            df.loc[df.index[-1] + timedelta(minutes=5)] = [last, last, last, last, 1000]
            self.ticked.add(ticker)
        return df

    def advance(self) -> None:
        self.t += timedelta(minutes=5)
        if (self.t.hour, self.t.minute) >= (16, 0):  # after the bell, jump to next morning's open
            self.t = (self.t + timedelta(days=1)).replace(hour=9, minute=30)
            while self.t.weekday() >= 5:
                self.t += timedelta(days=1)
        self.ticked.clear()


# ---------------------------------------------------------------- the bots
def data_bot(feed, tickers, st: State) -> dict[str, pd.DataFrame]:
    st.status["DATA"] = "downloading"
    out = {}
    for t in tickers:
        try:
            df = feed.bars(t)
            out[t] = df
            st.prices[t] = float(df["Close"].iloc[-1])
        except Exception as e:
            st.log("DATA", f"{t}: download failed ({type(e).__name__})", "red")
    st.log("DATA", f"got {len(out)}/{len(tickers)} tickers, last bar {feed.now():%H:%M} ET", "cyan")
    st.status["DATA"] = "idle"
    return out


def strategy_bot(bars, fast, slow, held, st: State) -> list[tuple[str, str]]:
    st.status["STRATEGY"] = "thinking"
    orders = []
    for t, df in bars.items():
        if len(df) < slow + 1:
            st.signals[t] = "WAIT"
            continue
        want = sma_signals(df, fast, slow)["want_long"].iloc[-1]
        st.signals[t] = "LONG" if want else "CASH"
        if want and t not in held:
            orders.append((t, "BUY"))
            st.log("STRATEGY", f"{t}: {fast}-bar avg is ABOVE {slow}-bar avg -> BUY", "green")
        elif not want and t in held:
            orders.append((t, "SELL"))
            st.log("STRATEGY", f"{t}: {fast}-bar avg is BELOW {slow}-bar avg -> SELL", "cyan")
    if not orders:
        st.log("STRATEGY", "no crossovers -> hold", "grey50")
    st.status["STRATEGY"] = "idle"
    return orders


def risk_bot(orders, feed, acct: PaperAccount, mode, rules: Rules, st: State) -> list[tuple[str, str]]:
    st.status["RISK"] = "checking"
    now = feed.now()
    equity = acct.equity(st.prices)
    approved = []
    for t, side in orders:
        if side == "SELL":  # selling always allowed: it reduces risk
            approved.append((t, side))
            continue
        reason = None
        if not feed.market_open():
            reason = "market closed"
        elif st.day_start_equity and equity < st.day_start_equity * (1 - rules.daily_loss_limit):
            reason = f"daily loss limit hit (-{rules.daily_loss_limit:.0%})"
        elif mode == "day" and (now.hour, now.minute) >= rules.no_new_after:
            reason = "too close to the closing bell"
        elif t in st.last_trade and now - st.last_trade[t] < timedelta(minutes=rules.cooldown_min):
            reason = f"cooldown ({rules.cooldown_min} min)"
        if reason:
            st.log("RISK", f"VETO {t} BUY: {reason}", "red")
        else:
            approved.append((t, side))
            st.log("RISK", f"OK {t} BUY (max {rules.max_position_pct:.0%} of account)", "green")
    # Day traders don't hold overnight
    if mode == "day" and feed.market_open() and (now.hour, now.minute) >= rules.flatten_at:
        for t in acct.positions():
            if (t, "SELL") not in approved:
                approved.append((t, "SELL"))
                st.log("RISK", f"closing bell: force SELL {t}", "cyan")
    st.status["RISK"] = "idle"
    return approved


def trader_bot(approved, acct: PaperAccount, feed, rules: Rules, st: State) -> None:
    st.status["TRADER"] = "trading" if approved else "idle"
    now = feed.now()
    equity = acct.equity(st.prices)
    for t, side in approved:
        if side == "SELL":
            msg = acct.sell(t, st.prices[t], now)
        else:
            msg = acct.buy(t, equity * rules.max_position_pct, st.prices[t], now)
        if msg:
            st.last_trade[t] = now
            st.log("TRADER", msg, "green" if side == "BUY" else "cyan")
        else:
            st.log("TRADER", f"{side} {t}: not enough cash", "red")
    if not approved:
        st.log("TRADER", "no orders", "grey50")
    st.status["TRADER"] = "idle"


# ---------------------------------------------------------------- dashboard
def render(acct: PaperAccount, st: State, title: str, poll: int) -> Layout:
    equity = acct.equity(st.prices)
    pnl = equity - acct.start_cash
    realized, n_closed, n_win = acct.stats()
    color = "bold green" if pnl >= 0 else "bold red"
    secs = max(0, int(st.next_cycle - time.time()))
    head = Text.assemble(
        ("STOCK-SIM ", "bold cyan"), ("// LIVE PAPER TRADING ", "bold green"), (f"// {title}\n", "bold white"),
        ("Equity ", "grey50"), (f"${equity:,.2f}  ", "bold white"),
        ("P&L ", "grey50"), (f"${pnl:+,.2f} ({pnl / acct.start_cash:+.2%})  ", color),
        ("Cash ", "grey50"), (f"${acct.cash:,.2f}  ", "white"),
        ("Realized ", "grey50"), (f"${realized:+,.2f}  ", "green" if realized >= 0 else "red"),
        ("Wins ", "grey50"), (f"{n_win}/{n_closed}  ", "white"),
        ("Fees ", "grey50"), (f"${acct.fees:,.2f}\n", "white"),
        ("Equity curve ", "grey50"), (sparkline(pd.Series(st.equity_hist or [equity]), 50), "cyan"),
        (f"   {st.clock}  cycle #{st.cycle}, next in {secs}s   (Ctrl+C to stop)", "grey50"),
    )

    def bot_panel(name: str) -> object:
        txt = Text()
        fit = max(3, (console.size.height - 17) // 2 - 2)  # only the newest lines that fit on screen
        for ts, msg, style in list(st.logs[name])[-fit:]:
            txt.append(f"{ts} ", "grey50")
            txt.append(msg + "\n", style)
        return panel(txt, f"{name} BOT ({st.status[name]})", "green" if st.status[name] != "idle" else "cyan")

    pos = Table(border_style="cyan", header_style="bold white", style="on black", expand=True)
    for c in ("Ticker", "Signal", "Shares", "Avg cost", "Price", "P&L"):
        pos.add_column(c, justify="right" if c not in ("Ticker", "Signal") else "left")
    held = acct.positions()
    for t in sorted(set(st.prices) | set(held)):
        sig = st.signals.get(t, "-")
        px = st.prices.get(t, 0)
        if t in held:
            s, avg, _ = held[t]
            p = (px - avg) * s
            pos.add_row(t, sig, str(s), f"${avg:,.2f}", f"${px:,.2f}", Text(f"${p:+,.2f}", "green" if p >= 0 else "red"))
        else:
            pos.add_row(t, Text(sig, "grey50"), "-", "-", f"${px:,.2f}", "-")

    trades = Text()
    for ts, t, side, s, px, fee, p in acct.trades(8):
        trades.append(f"{ts[5:16].replace('T', ' ')} {side:<4} {s:>4} {t:<6} @ ${px:,.2f}", "green" if side == "BUY" else "cyan")
        trades.append(f"  {'' if p is None else f'P&L ${p:+,.2f}'}\n", "white")

    lay = Layout()
    lay.split_column(Layout(panel(head, "ACCOUNT"), size=5), Layout(name="bots"), Layout(name="bottom", size=12))
    lay["bots"].split_row(*(Layout(name=b) for b in ("l", "r")))
    lay["bots"]["l"].split_column(Layout(bot_panel("DATA")), Layout(bot_panel("STRATEGY")))
    lay["bots"]["r"].split_column(Layout(bot_panel("RISK")), Layout(bot_panel("TRADER")))
    lay["bottom"].split_row(Layout(panel(pos, "WATCHLIST & POSITIONS")), Layout(panel(trades, "RECENT TRADES")))
    return lay


def run_live(tickers: list[str], mode: str, demo: bool, cash: float, costs, poll: int | None = None,
             max_cycles: int | None = None) -> None:
    cfg = MODES[mode]
    poll = poll or (2 if demo else cfg["poll"])
    feed = DemoFeed(tickers) if demo else YahooFeed(cfg["interval"], cfg["period"])
    acct_file = DEMO_DB if demo else DB
    acct = PaperAccount(cash, costs, acct_file)
    if demo:  # the fake market restarts from scratch, so the demo account does too
        acct.reset()
        acct = PaperAccount(cash, costs, acct_file)  # demo trades never touch your real paper account
    rules, st = Rules(), State()
    title = f"{mode.upper()} MODE // {cfg['fast']}/{cfg['slow']} on {cfg['interval']} bars" + (" // DEMO DATA" if demo else "")

    with Live(console=console, screen=False, refresh_per_second=2,
              get_renderable=lambda: render(acct, st, title, poll)):
        try:
            while max_cycles is None or st.cycle < max_cycles:
                st.cycle += 1
                today = feed.now().date()
                if st.day != today:  # new trading day -> reset the daily loss limit
                    st.day, st.day_start_equity = today, None
                st.clock = f"{feed.now():%a %b %d %H:%M} ET"
                bars = data_bot(feed, tickers, st)
                if st.day_start_equity is None and st.prices:
                    st.day_start_equity = acct.equity(st.prices)
                orders = strategy_bot(bars, cfg["fast"], cfg["slow"], acct.positions(), st)
                approved = risk_bot(orders, feed, acct, mode, rules, st)
                trader_bot(approved, acct, feed, rules, st)
                st.equity_hist.append(acct.equity(st.prices))
                if not feed.market_open():
                    st.log("RISK", f"market closed ({feed.now():%a %H:%M} ET) - watching only", "grey50")
                st.next_cycle = time.time() + poll
                if demo:
                    feed.advance()
                while time.time() < st.next_cycle:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            pass
    console.print(f"[hud]Stopped. Your paper account is saved in {acct_file.name} - run the same command to continue.[/]")

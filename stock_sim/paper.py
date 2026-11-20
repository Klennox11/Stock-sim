"""The fake brokerage account. Everything is saved in paper_account.db (SQLite),
so you can stop the bot and start it again tomorrow with the same balance."""
import functools
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from .costs import Costs

DB = Path(__file__).resolve().parent.parent / "paper_account.db"
DEMO_DB = DB.with_name("paper_account_demo.db")
_LOCK = threading.RLock()  # the dashboard thread and the bots share one database


def locked(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with _LOCK:
            return fn(*a, **k)
    return wrapper


class PaperAccount:
    def __init__(self, start_cash: float = 10_000.0, costs: Costs | None = None, db: Path = DB):
        self.costs = costs or Costs()
        self.con = sqlite3.connect(db, check_same_thread=False)
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS account (key TEXT PRIMARY KEY, value REAL);
            CREATE TABLE IF NOT EXISTS positions (ticker TEXT PRIMARY KEY, shares INTEGER, avg_price REAL, opened TEXT);
            CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, ts TEXT, ticker TEXT, side TEXT,
                                               shares INTEGER, price REAL, fee REAL, pnl REAL);
        """)
        if self._get("cash") is None:
            self._set("cash", start_cash)
            self._set("start_cash", start_cash)
            self._set("fees", 0.0)
        self.con.commit()

    # --- tiny key/value helpers
    @locked
    def _get(self, k):
        r = self.con.execute("SELECT value FROM account WHERE key=?", (k,)).fetchone()
        return r[0] if r else None

    def _set(self, k, v):
        self.con.execute("INSERT OR REPLACE INTO account VALUES (?,?)", (k, v))

    @property
    def cash(self) -> float:
        return self._get("cash")

    @property
    def start_cash(self) -> float:
        return self._get("start_cash")

    @property
    def fees(self) -> float:
        return self._get("fees")

    @locked
    def positions(self) -> dict[str, tuple[int, float, str]]:
        return {t: (s, p, o) for t, s, p, o in self.con.execute("SELECT * FROM positions")}

    @locked
    def trades(self, limit: int = 10) -> list[tuple]:
        return self.con.execute(
            "SELECT ts, ticker, side, shares, price, fee, pnl FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    @locked
    def stats(self) -> tuple[float, int, int]:
        """realized P&L, number of closed trades, number of winners"""
        r = self.con.execute(
            "SELECT COALESCE(SUM(pnl),0), COUNT(*), COALESCE(SUM(pnl>0),0) FROM trades WHERE side='SELL'").fetchone()
        return r[0], r[1], r[2]

    @locked
    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(s * prices.get(t, p) for t, (s, p, _) in self.positions().items())

    # --- orders (filled instantly at price +/- slippage, plus commission)
    @locked
    def buy(self, ticker: str, dollars: float, price: float, ts: datetime) -> str | None:
        px = self.costs.fill_price(price, "BUY")
        shares = int(min(dollars, self.cash) // (px * 1.001))
        if shares <= 0:
            return None
        fee = self.costs.commission(shares, px)
        self._set("cash", self.cash - shares * px - fee)
        self._set("fees", self.fees + fee)
        self.con.execute("INSERT INTO positions VALUES (?,?,?,?)", (ticker, shares, px, ts.isoformat()))
        self.con.execute("INSERT INTO trades (ts,ticker,side,shares,price,fee,pnl) VALUES (?,?,?,?,?,?,NULL)",
                         (ts.isoformat(), ticker, "BUY", shares, px, fee))
        self.con.commit()
        return f"BUY {shares} {ticker} @ ${px:,.2f} (fee ${fee:.2f})"

    @locked
    def sell(self, ticker: str, price: float, ts: datetime) -> str | None:
        pos = self.positions().get(ticker)
        if not pos:
            return None
        shares, avg, _ = pos
        px = self.costs.fill_price(price, "SELL")
        fee = self.costs.commission(shares, px)
        buy_fee = self.costs.commission(shares, avg)
        pnl = (px - avg) * shares - fee - buy_fee
        self._set("cash", self.cash + shares * px - fee)
        self._set("fees", self.fees + fee)
        self.con.execute("DELETE FROM positions WHERE ticker=?", (ticker,))
        self.con.execute("INSERT INTO trades (ts,ticker,side,shares,price,fee,pnl) VALUES (?,?,?,?,?,?,?)",
                         (ts.isoformat(), ticker, "SELL", shares, px, fee, pnl))
        self.con.commit()
        return f"SELL {shares} {ticker} @ ${px:,.2f}  P&L ${pnl:+,.2f}"

    @locked
    def reset(self) -> None:
        self.con.executescript("DELETE FROM account; DELETE FROM positions; DELETE FROM trades;")
        self.con.commit()

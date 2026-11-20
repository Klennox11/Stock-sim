"""Terminal 'HUD' look: black background, cyan / green / white text."""
import os
import sys

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

THEME = Theme({"hud": "bold cyan", "good": "bold green", "bad": "bold red", "dim": "grey50", "txt": "white"})
console = Console(theme=THEME, style="white on black", highlight=False)

# Windows Terminal (sets WT_SESSION) and Mac/Linux can draw block bars.
# The old blue PowerShell window can't, so it gets plain characters instead.
FANCY = sys.platform != "win32" or "WT_SESSION" in os.environ
BLOCKS = "▁▂▃▄▅▆▇█" if FANCY else "_.-~=+*#"


def sparkline(series: pd.Series, width: int = 64) -> str:
    """Squash a price/equity series into one line of block characters."""
    vals = series.values
    idx = np.linspace(0, len(vals) - 1, min(width, len(vals))).astype(int)
    pts = vals[idx]
    lo, hi = pts.min(), pts.max()
    span = hi - lo or 1
    return "".join(BLOCKS[int((p - lo) / span * (len(BLOCKS) - 1))] for p in pts)


def fmt(key: str, val) -> str:
    if isinstance(val, int):
        return f"{val}"
    if key in ("Final value",):
        return f"${val:,.0f}"
    if key in ("Sharpe ratio",):
        return f"{val:.2f}"
    return f"{val * 100:,.1f}%"


def panel(body, title: str, style: str = "cyan") -> Panel:
    return Panel(body, title=f"[hud]// {title}[/]", border_style=style, style="on black")


def compare_table(title: str, strat: dict, bench: dict) -> Table:
    t = Table(title=f"[hud]{title}[/]", border_style="cyan", header_style="bold white", style="on black")
    t.add_column("Metric", style="txt")
    t.add_column("Strategy", justify="right")
    t.add_column("Buy & Hold", justify="right")
    higher_is_better = {"Final value", "Total return", "Yearly return (CAGR)", "Max drawdown", "Sharpe ratio"}
    for k, v in strat.items():
        b = bench.get(k)
        if b is None:
            t.add_row(k, f"[hud]{fmt(k, v)}[/]", "[dim]-[/]")
            continue
        win = v > b if k in higher_is_better else None
        s_style = "good" if win else "bad"
        t.add_row(k, f"[{s_style}]{fmt(k, v)}[/]", f"[txt]{fmt(k, b)}[/]")
    return t


def cell_style(d_cagr: float, d_sharpe: float) -> str:
    """green = beat buy&hold on both, cyan = one of the two, red = neither."""
    wins = (d_cagr > 0) + (d_sharpe > 0)
    return {2: "bold green", 1: "cyan", 0: "red"}[wins]


def sweep_grid(ticker: str, cells, fasts, slows) -> Table:
    t = Table(title=f"[hud]{ticker}[/]", border_style="cyan", header_style="bold white", style="on black")
    t.add_column("fast \\ slow", style="txt")
    for s in slows:
        t.add_column(f"{s}d", justify="center")
    lookup = {(c.fast, c.slow): c for c in cells}
    for f in fasts:
        row = [f"{f}d"]
        for s in slows:
            c = lookup.get((f, s))
            if c is None:
                row.append("[dim]  -  [/]")
            else:
                row.append(f"[{cell_style(c.d_cagr, c.d_sharpe)}]{c.d_cagr * 100:+.1f}% | {c.d_sharpe:+.2f}[/]")
        t.add_row(*row)
    return t

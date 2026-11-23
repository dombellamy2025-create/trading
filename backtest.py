"""Backtest: 6 months of Alpaca history, every strategy, with the live bot's risk rules.

    python backtest.py                    # download (and cache) 6 months, run, report, chart
    python backtest.py --refresh          # ignore the cache and re-download
    python backtest.py --months 3         # different window
    python backtest.py --synthetic        # offline demo on generated data (NOT real results)

Data: SPY/QQQ 15-min, BTC/USD 1-hour, GLD/USO 4-hour bars from Alpaca, plus daily bars for
the 14-day sizing ATR. Extra history before the window is fetched so indicators (e.g. the
200 EMA on 4-hour bars) are warmed up before the first simulated trade.

Simulation (event-driven, same strategy classes and risk_manager as the live bot):
  * Signals are judged on CLOSED bars and filled at the NEXT bar's open (no look-ahead).
  * Slippage 0.05% against you on every fill (entries, exits, stops); commission $0 on stocks.
  * Crypto fee 0.25% of notional per fill (Alpaca's base-tier taker fee; CRYPTO_FEE_RATE overrides).
  * Equities: signals only on bars that close during regular hours (9:30-16:00 ET), and
    orders fill only at the open of a bar inside regular hours. A signal at the close
    waits for the next session, so overnight gaps are realistic. BTC/USD trades 24/7.
  * ATR sizing: 1 daily-ATR(14) move = 1% of equity, capped at MAX_ALLOC_PER_SYMBOL and
    MAX_GROSS_EXPOSURE; crypto is cash-only (non-marginable); shorts are whole shares.
  * Hard stop at a 1%-of-equity loss, triggered intrabar on the bar's high/low (equities only
    during regular hours, like Alpaca stop orders). Gaps through the stop fill at the open.
  * Strategy exits and trailing stops are evaluated at bar close and filled next open.
  * Daily loss limit (3%): flatten everything and stop entries for the rest of the NY day.
  * Reversals close and reopen in the opposite direction at the same fill.
  * Open positions are closed at the final bar so every trade is counted.

Report: per instrument (each on its own $100k account), then the combined $100k portfolio
trading all five together with the SPY/QQQ -> BTC correlation filter active.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from itertools import groupby
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from bot import risk_manager as risk
from bot.strategies import FLAT, LONG, SHORT, Action, Context, atr
from config import CORRELATION_RULES, CRYPTO_FEE_RATE, INSTRUMENTS, Instrument, Settings, project_path

NY = ZoneInfo("America/New_York")
SLIPPAGE = 0.0005      # 0.05% per fill
COMMISSION = 0.0       # Alpaca stocks are commission-free
CRYPTO_FEE = CRYPTO_FEE_RATE   # fraction of notional per crypto fill (config.CRYPTO_FEE_RATE)
CAPITAL = 100_000.0
RTH_OPEN, RTH_CLOSE = dtime(9, 30), dtime(16, 0)


# ------------------------------------------------------------------ market hours
def in_rth(ts: pd.Timestamp) -> bool:
    """True if `ts` is inside US regular trading hours."""
    et = ts.tz_convert(NY)
    return et.weekday() < 5 and RTH_OPEN <= et.time() < RTH_CLOSE


def closes_in_rth(ts: pd.Timestamp, minutes: int) -> bool:
    """True if a bar starting at `ts` closes while the market is open (incl. the 16:00 close)."""
    et = (ts + pd.Timedelta(minutes=minutes)).tz_convert(NY)
    return et.weekday() < 5 and RTH_OPEN < et.time() <= RTH_CLOSE


# ------------------------------------------------------------------ data
@dataclass
class MarketData:
    bars: pd.DataFrame    # strategy timeframe, OHLCV, UTC index (bar start)
    daily: pd.DataFrame   # daily bars for the sizing ATR


def _read_cache(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_data(broker, inst: Instrument, start: datetime, cache_dir: str, refresh: bool) -> MarketData:
    st = inst.strategy
    days = (datetime.now(timezone.utc) - start).days + st.lookback_days + 5
    out = {}
    for name, minutes, d in (("bars", st.timeframe_minutes, days), ("daily", 1440, days + 40)):
        path = os.path.join(cache_dir, f"{inst.key}_{minutes}m.csv")
        if os.path.exists(path) and not refresh:
            out[name] = _read_cache(path)
            continue
        print(f"  downloading {inst.symbol} {minutes}-min bars ({d} days)...")
        df = broker.bars(inst, minutes, d)
        df.to_csv(path)
        out[name] = df
    return MarketData(**out)


def synthetic_data(inst: Instrument, start: datetime, seed: int) -> MarketData:
    """Random-walk bars with realistic sessions, for exercising the pipeline offline."""
    rng = np.random.default_rng(seed)
    st = inst.strategy
    tf = st.timeframe_minutes
    first = start - timedelta(days=st.lookback_days)
    idx = pd.date_range(first, datetime.now(timezone.utc), freq=f"{tf}min", tz="UTC")
    if not inst.is_crypto:  # IEX-like regular-session bars only
        idx = idx[[in_rth(t) for t in idx]]
    vol = {"SPY": 0.010, "QQQ": 0.013, "BTC/USD": 0.030, "GLD": 0.011, "USO": 0.020}[inst.symbol]
    bars_per_day = 26 if tf == 15 else 24 if inst.is_crypto else 2
    sigma = vol / math.sqrt(bars_per_day)
    price0 = {"SPY": 600, "QQQ": 520, "BTC/USD": 95_000, "GLD": 250, "USO": 75}[inst.symbol]
    rets = rng.standard_t(4, len(idx)) * sigma * 0.7 + rng.normal(0.00002, sigma * 0.1, len(idx))
    close = price0 * np.exp(np.cumsum(rets))
    opn = np.concatenate([[price0], close[:-1]]) * (1 + rng.normal(0, sigma * 0.2, len(idx)))
    hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, sigma * 0.5, len(idx))))
    lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, sigma * 0.5, len(idx))))
    volu = rng.lognormal(10, 0.5, len(idx)) * (1 + 3 * (np.abs(rets) > 2 * sigma))
    bars = pd.DataFrame({"open": opn, "high": hi, "low": lo, "close": close, "volume": volu}, index=idx)
    daily = bars.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                     "close": "last", "volume": "sum"}).dropna()
    return MarketData(bars, daily)


# ------------------------------------------------------------------ simulation
@dataclass
class Trade:
    instrument: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float
    profit_loss: float
    exit_reason: str


@dataclass
class Pos:
    side: int = FLAT
    qty: float = 0.0
    entry: float = 0.0
    entry_time: pd.Timestamp | None = None
    entry_equity: float = 0.0
    extreme: float | None = None
    hard: float | None = None
    entry_fee: float = 0.0


class _Inst:
    """Pre-computed per-instrument arrays for a fast event loop."""

    def __init__(self, inst: Instrument, md: MarketData, s: Settings):
        self.inst, self.key, self.st = inst, inst.key, inst.strategy
        self.tf = self.st.timeframe_minutes
        self.ind = self.st.indicators(md.bars)
        self.index = md.bars.index
        self.o, self.h, self.l, self.c = (md.bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.min_bars = self.st.min_bars()
        if s.sizing_atr_timeframe == "daily":
            d = md.daily
            self.atr_vals = np.array(atr(d, s.atr_period), dtype=float)  # writable copy
            self.atr_ends = (d.index + pd.Timedelta(days=1)).asi8  # ns since epoch; known once the day closes
            self.atr_vals[: s.atr_period] = np.nan
        else:
            self.atr_vals = np.array(atr(md.bars, s.atr_period), dtype=float)
            self.atr_ends = (self.index + pd.Timedelta(minutes=self.tf)).asi8

    def sizing_atr(self, ts: pd.Timestamp) -> float | None:
        i = np.searchsorted(self.atr_ends, pd.Timestamp(ts).as_unit("ns").value, side="right") - 1
        if i < 0 or not np.isfinite(self.atr_vals[i]):
            return None
        return float(self.atr_vals[i])


class Simulator:
    def __init__(self, s: Settings, instruments: list[Instrument], data: dict[str, MarketData],
                 start: datetime, capital: float = CAPITAL, rules=(), slippage: float = SLIPPAGE,
                 commission: float = COMMISSION, crypto_fee: float = CRYPTO_FEE):
        self.s, self.start, self.capital, self.rules = s, pd.Timestamp(start), capital, list(rules)
        self.slip, self.commission, self.crypto_fee = slippage, commission, crypto_fee
        self.fees = 0.0          # crypto fees paid
        self.inst = {i.key: _Inst(i, data[i.key], s) for i in instruments}
        gated = {b for _, b in self.rules}
        self.order = {k: (1 if k in gated else 0) for k in self.inst}  # gated symbols go last
        self.cash = capital
        self.pos = {k: Pos() for k in self.inst}
        self.pending: dict[str, dict] = {}
        self.last_px: dict[str, float] = {}
        self.last_entry_bar: dict[str, pd.Timestamp] = {}
        self.trades: list[Trade] = []
        self.curve: list[tuple[pd.Timestamp, float]] = []
        self.blocked = 0          # entries stopped by the correlation filter
        self.hard_stops = 0
        self.daily_halts = 0
        self._day, self._day_start_eq, self._halted = None, capital, False

    # ---- accounting
    def equity(self) -> float:
        return self.cash + sum(p.qty * p.side * self.last_px.get(k, p.entry) for k, p in self.pos.items())

    def _notional(self, k):
        p = self.pos[k]
        return p.qty * self.last_px.get(k, p.entry)

    def _fill(self, raw: float, buying: bool) -> float:
        return raw * (1 + self.slip) if buying else raw * (1 - self.slip)

    def _open(self, k: str, side: int, raw: float, ts, signal_bar):
        I = self.inst[k]
        a = I.sizing_atr(ts)
        if not a:
            return
        px = self._fill(raw, buying=side == LONG)
        eq = self.equity()
        gross_other = sum(self._notional(j) for j in self.inst if j != k)
        budget = max(0.0, self.s.max_gross_exposure * eq - gross_other)
        if I.inst.is_crypto:  # non-marginable: only unlevered cash
            long_other = sum(self._notional(j) for j, p in self.pos.items() if j != k and p.side == LONG)
            budget = min(budget, max(0.0, eq - long_other))
        fractional = I.inst.is_crypto or (self.s.fractional_shares and side == LONG)
        qty, _ = risk.position_size(eq, budget, px, a, self.s, fractional=fractional)
        if qty <= 0:
            return
        fee = qty * px * self.crypto_fee if I.inst.is_crypto else 0.0
        self.fees += fee
        self.cash -= side * qty * px + self.commission + fee
        self.pos[k] = Pos(side, qty, px, ts, eq, px,
                          risk.hard_stop_price(side, px, qty, eq, self.s.hard_stop_pct), fee)
        self.last_entry_bar[k] = signal_bar

    def _close(self, k: str, raw: float, ts, reason: str):
        p = self.pos[k]
        if p.side == FLAT:
            return
        px = self._fill(raw, buying=p.side == SHORT)
        fee = p.qty * px * self.crypto_fee if self.inst[k].inst.is_crypto else 0.0
        self.fees += fee
        self.cash += p.side * p.qty * px - self.commission - fee
        pnl = (px - p.entry) * p.qty * p.side - 2 * self.commission - p.entry_fee - fee
        self.trades.append(Trade(self.inst[k].inst.symbol, "LONG" if p.side == LONG else "SHORT",
                                 p.entry_time, ts, p.entry, px, p.qty, pnl, reason))
        self.pos[k] = Pos()

    def _longs(self) -> set[str]:
        held = {k for k, p in self.pos.items() if p.side == LONG}
        pend = {k for k, o in self.pending.items() if o.get("open") == LONG}
        return held | pend

    # ---- main loop
    def run(self):
        events = []
        for k, I in self.inst.items():
            events += [(ts, self.order[k], k, i) for i, ts in enumerate(I.index)]
        events.sort(key=lambda e: (e[0], e[1]))
        for ts, group in groupby(events, key=lambda e: e[0]):
            for _, _, k, i in group:
                self._bar(k, i, ts)
            if ts >= self.start:
                self._daily_limit(ts)
                self.curve.append((ts, self.equity()))
        for k, I in self.inst.items():  # close out at the end so every trade is counted
            if self.pos[k].side != FLAT:
                self._close(k, self.last_px[k], I.index[-1], "end of backtest")
        if self.curve:
            self.curve[-1] = (self.curve[-1][0], self.equity())
        return self

    def _bar(self, k: str, i: int, ts):
        I = self.inst[k]
        crypto = I.inst.is_crypto
        o, h, l, c = I.o[i], I.h[i], I.l[i], I.c[i]
        live = ts >= self.start
        tradable = crypto or in_rth(ts)

        if live and tradable:
            # 1) pending orders fill at this bar's open
            order = self.pending.pop(k, None)
            if order:
                if order.get("close"):
                    self._close(k, o, ts, order["reason"])
                if order.get("open") and not self._halted:
                    self._open(k, order["open"], o, ts, order["signal_bar"])
            # 2) hard stop, intrabar (gap through the stop fills at the open)
            p = self.pos[k]
            if p.side == LONG and l <= p.hard:
                self._close(k, min(o, p.hard), ts, "hard stop (1% equity)")
                self.hard_stops += 1
            elif p.side == SHORT and h >= p.hard:
                self._close(k, max(o, p.hard), ts, "hard stop (1% equity)")
                self.hard_stops += 1

        self.last_px[k] = c
        p = self.pos[k]
        if p.side != FLAT:
            p.extreme = max(p.extreme, c) if p.side == LONG else min(p.extreme, c)

        # 3) strategy decision on the closed bar; executes at the next tradable open
        if not live or i + 1 < I.min_bars or self._halted or k in self.pending:
            return
        if not (crypto or closes_in_rth(ts, I.tf)):
            return
        sig = I.st.decide(I.ind.iloc[i:i + 1], Context(p.side, c, p.extreme))
        if sig.action is Action.EXIT and p.side != FLAT:
            self.pending[k] = {"close": True, "reason": sig.reason}
        elif sig.action in (Action.LONG, Action.SHORT):
            target = LONG if sig.action is Action.LONG else SHORT
            if p.side == target or (p.side == FLAT and self.last_entry_bar.get(k) == ts):
                return
            if target == LONG and risk.correlation_blocked(k, self._longs(), self.rules):
                self.blocked += 1
                if p.side != FLAT:
                    self.pending[k] = {"close": True, "reason": f"reversal: {sig.reason}"}
                return
            order = {"open": target, "signal_bar": ts}
            if p.side != FLAT:
                order.update(close=True, reason=f"reversal: {sig.reason}")
            self.pending[k] = order

    def _daily_limit(self, ts):
        day = ts.tz_convert(NY).date()
        eq = self.equity()
        if day != self._day:
            self._day, self._day_start_eq, self._halted = day, (self.curve[-1][1] if self.curve else eq), False
        if self._halted or eq > self._day_start_eq * (1 - self.s.max_daily_loss):
            return
        self._halted = True
        self.daily_halts += 1
        self.pending = {k: {"close": True, "reason": "daily loss limit"}
                        for k, p in self.pos.items() if p.side != FLAT}
        for k in list(self.pending):  # crypto (and equities in session) flatten right away
            if self.inst[k].inst.is_crypto or in_rth(ts):
                self.pending.pop(k)
                self._close(k, self.last_px[k], ts, "daily loss limit")


# ------------------------------------------------------------------ metrics
def metrics(sim: Simulator, annualization: int, bh_return: float | None = None) -> dict:
    pnls = np.array([t.profit_loss for t in sim.trades])
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    curve = pd.Series([e for _, e in sim.curve], index=pd.DatetimeIndex([t for t, _ in sim.curve]))
    if curve.empty:
        curve = pd.Series([sim.capital], index=[sim.start])
    peak = curve.cummax()
    dd = curve / peak - 1
    daily = curve.resample("1D").last()
    daily = daily.ffill() if annualization == 365 else daily.dropna()
    rets = daily.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * math.sqrt(annualization)) if len(rets) > 1 and rets.std() > 0 else 0.0
    gross_win, gross_loss = wins.sum(), -losses.sum()
    return {
        "trades": len(pnls),
        "win_rate": len(wins) / len(pnls) if len(pnls) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0),
        "max_drawdown": float(dd.min()),
        "max_drawdown_usd": float((curve - peak).min()),
        "sharpe": sharpe,
        "total_return": float(curve.iloc[-1] / sim.capital - 1),
        "final_equity": float(curve.iloc[-1]),
        "hard_stops": sim.hard_stops,
        "fees": getattr(sim, "fees", 0.0),
        "buy_hold": bh_return,
        "curve": curve,
        "drawdown": dd,
    }


def buy_and_hold(md: MarketData, start) -> float | None:
    b = md.bars[md.bars.index >= pd.Timestamp(start)]
    return float(b["close"].iloc[-1] / b["open"].iloc[0] - 1) if len(b) > 1 else None


# ------------------------------------------------------------------ reporting
def _fmt_pf(x):
    return "inf" if x == math.inf else f"{x:.2f}"


def summary_table(rows: list[tuple[str, dict]]) -> str:
    head = (f"{'Instrument':<26}{'Trades':>7}{'Win %':>8}{'Avg win':>11}{'Avg loss':>11}{'PF':>7}"
            f"{'Max DD':>9}{'Sharpe':>8}{'Return':>9}{'Buy&Hold':>10}")
    lines = [head, "-" * len(head)]
    for name, m in rows:
        if name.startswith("PORTFOLIO"):
            lines.append("-" * len(head))
        bh = f"{m['buy_hold']:>10.2%}" if m["buy_hold"] is not None else f"{'—':>10}"
        lines.append(f"{name:<26}{m['trades']:>7}{m['win_rate']:>8.1%}{m['avg_win']:>11,.0f}"
                     f"{m['avg_loss']:>11,.0f}{_fmt_pf(m['profit_factor']):>7}{m['max_drawdown']:>9.2%}"
                     f"{m['sharpe']:>8.2f}{m['total_return']:>9.2%}{bh}")
    return "\n".join(lines)


# Reference palette, light mode (dataviz skill): one series per panel, so slot 1 throughout.
SURFACE, INK, INK2, GRID, SERIES, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6", "#9a9994"


def plot(results: dict[str, dict], portfolio: dict, path: str, title_note: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.titlecolor": INK, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(15, 13), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 3, height_ratios=[2.2, 0.9, 1.6, 1.6], hspace=0.55, wspace=0.22,
                          top=0.9, bottom=0.04)

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    # Portfolio equity
    ax0 = fig.add_subplot(gs[0, :])
    style(ax0)
    eq = portfolio["curve"]
    ax0.plot(eq.index, eq.values, color=SERIES, linewidth=2)
    ax0.axhline(CAPITAL, color=NEUTRAL, linewidth=1, linestyle=(0, (4, 3)))
    ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax0.xaxis.set_major_locator(mdates.MonthLocator())
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax0.set_title(f"Combined portfolio: all 5 instruments on one $100k account, correlation filter on   "
                  f"return {portfolio['total_return']:+.2%} · max DD {portfolio['max_drawdown']:.2%} · "
                  f"Sharpe {portfolio['sharpe']:.2f}", loc="left")
    ax0.annotate(f"${eq.iloc[-1]:,.0f}", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0),
                 textcoords="offset points", va="center", color=INK, fontsize=10)

    # Portfolio drawdown
    ax = fig.add_subplot(gs[1, :], sharex=ax0)
    style(ax)
    dd = portfolio["drawdown"]
    ax.fill_between(dd.index, dd.values, 0, color=SERIES, alpha=0.25, linewidth=0)
    ax.plot(dd.index, dd.values, color=SERIES, linewidth=1.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_title("Portfolio drawdown from peak", loc="left")

    # Per-instrument panels (each on its own $100k account)
    slots = [gs[2, 0], gs[2, 1], gs[2, 2], gs[3, 0], gs[3, 1]]
    for (name, m), slot in zip(results.items(), slots):
        ax = fig.add_subplot(slot)
        style(ax)
        r = m["curve"] / CAPITAL - 1
        ax.plot(r.index, r.values, color=SERIES, linewidth=2, zorder=3)
        if m.get("bh_curve") is not None:
            ax.plot(m["bh_curve"].index, m["bh_curve"].values, color=NEUTRAL, linewidth=1.2,
                    linestyle=(0, (4, 3)))
        ax.axhline(0, color=NEUTRAL, linewidth=0.8, zorder=1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(f"{name}   {m['total_return']:+.1%} · {m['trades']} trades", loc="left", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Table view in the sixth slot
    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    rows = [[n.split(" ")[0], f"{m['win_rate']:.0%}", _fmt_pf(m["profit_factor"]),
             f"{m['max_drawdown']:.1%}", f"{m['sharpe']:.2f}"] for n, m in results.items()]
    rows.append(["Portfolio", f"{portfolio['win_rate']:.0%}", _fmt_pf(portfolio["profit_factor"]),
                 f"{portfolio['max_drawdown']:.1%}", f"{portfolio['sharpe']:.2f}"])
    t = ax.table(cellText=rows, colLabels=["", "Win %", "PF", "Max DD", "Sharpe"],
                 loc="center", cellLoc="right", colLoc="right")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        cell.set_text_props(color=INK if r else INK2, weight="bold" if r == len(rows) else "normal")

    handles = [plt.Line2D([], [], color=SERIES, linewidth=2, label="Strategy (each on its own $100k account)"),
               plt.Line2D([], [], color=NEUTRAL, linewidth=1.2, linestyle=(0, (4, 3)), label="Buy & hold")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.452), ncol=2,
               frameon=False, fontsize=9)
    fig.suptitle("Backtest results" + title_note, x=0.06, y=0.978, ha="left", fontsize=15, weight="bold")
    fig.text(0.06, 0.947, f"Fills at next bar open · 0.05% slippage per fill · $0 stock commission · {CRYPTO_FEE:.2%} crypto fee per fill · "
             "ATR sizing (1 daily ATR = 1% equity) · 1% hard stops · equities in regular hours only",
             color=INK2, fontsize=9.5)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def expectations(results: dict, portfolio: dict, start, end) -> dict:
    """Numbers the live evening report compares against (saved as backtest_summary.json)."""
    def daily_stats(m):
        d = m["curve"].resample("1D").last().ffill().pct_change().dropna()
        return float(d.mean()), float(d.std())

    days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days)
    mean, std = daily_stats(portfolio)
    out = {"window": [str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())],
           "portfolio": {"win_rate": portfolio["win_rate"], "trades_per_day": portfolio["trades"] / days,
                         "daily_return_mean": mean, "daily_return_std": std,
                         "max_drawdown": portfolio["max_drawdown"], "total_return": portfolio["total_return"]},
           "instruments": {}}
    for name, m in results.items():
        out["instruments"][name.split(" ")[0]] = {"win_rate": m["win_rate"], "trades_per_day": m["trades"] / days,
                                                 "avg_win": m["avg_win"], "avg_loss": m["avg_loss"]}
    return out


def run_backtest(s: Settings, data: dict[str, MarketData], start: datetime, instruments=INSTRUMENTS):
    results = {}
    for inst in instruments:
        sim = Simulator(s, [inst], {inst.key: data[inst.key]}, start).run()
        m = metrics(sim, 365 if inst.is_crypto else 252, buy_and_hold(data[inst.key], start))
        b = data[inst.key].bars
        b = b[b.index >= pd.Timestamp(start)]
        m["bh_curve"] = b["close"] / b["open"].iloc[0] - 1 if len(b) else None
        m["sim"] = sim
        results[f"{inst.symbol} ({inst.strategy.name.replace('_', ' ')})"] = m
    port_sim = Simulator(s, instruments, data, start, rules=CORRELATION_RULES).run()
    portfolio = metrics(port_sim, 365)
    portfolio["sim"] = port_sim
    return results, portfolio


def main():
    ap = argparse.ArgumentParser(description="Backtest all strategies on Alpaca historical data")
    ap.add_argument("--months", type=float, default=6)
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    ap.add_argument("--synthetic", action="store_true", help="offline demo on generated data")
    ap.add_argument("--out", default="backtest_results.png")
    args = ap.parse_args()

    s = Settings()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=round(args.months * 30.44))
    print(f"Backtest window: {start:%Y-%m-%d} -> {end:%Y-%m-%d} ({args.months:g} months)")

    if args.synthetic:
        data = {i.key: synthetic_data(i, start, seed=n) for n, i in enumerate(INSTRUMENTS)}
        note = "   (SYNTHETIC DATA: pipeline demo, not real market results)"
    else:
        s.validate()
        from bot.broker import AlpacaBroker
        cache = project_path("data")
        os.makedirs(cache, exist_ok=True)
        broker = AlpacaBroker(s)
        print("Loading historical data from Alpaca (cached in data/):")
        data = {i.key: load_data(broker, i, start, cache, args.refresh) for i in INSTRUMENTS}
        note = ""
    for i in INSTRUMENTS:
        b = data[i.key].bars
        print(f"  {i.symbol:<8} {i.strategy.timeframe_minutes:>4}-min bars: {len(b):>6} "
              f"({b.index[0]:%Y-%m-%d} -> {b.index[-1]:%Y-%m-%d}, incl. warm-up)")

    results, portfolio = run_backtest(s, data, start)

    trades = [t.__dict__ | {"run": name} for name, m in results.items() for t in m["sim"].trades]
    trades += [t.__dict__ | {"run": "portfolio"} for t in portfolio["sim"].trades]
    pd.DataFrame(trades).to_csv(project_path("backtest_trades.csv"), index=False)
    if not args.synthetic:  # never let demo data become the live benchmark
        with open(project_path("backtest_summary.json"), "w") as f:
            json.dump(expectations(results, portfolio, start, end), f, indent=2)
    out = project_path(args.out)
    plot(results, portfolio, out, note)

    print()
    print(summary_table(list(results.items()) + [("PORTFOLIO (corr. filter on)", portfolio)]))
    ps = portfolio["sim"]
    print(f"\nPortfolio: ${CAPITAL:,.0f} -> ${portfolio['final_equity']:,.0f} | max drawdown "
          f"${portfolio['max_drawdown_usd']:,.0f} ({portfolio['max_drawdown']:.2%}) | "
          f"hard stops {ps.hard_stops} | daily-loss halts {ps.daily_halts} | crypto fees ${ps.fees:,.0f} | "
          f"BTC entries blocked by correlation filter: {ps.blocked}")
    print("Per instrument: each on its own $100k account with the same risk rules (no correlation filter).")
    print("Sharpe: daily returns, rf = 0, annualized x sqrt(252) for equities, sqrt(365) for BTC/portfolio.")
    print(f"Chart: {out}\nTrades: {project_path('backtest_trades.csv')}")
    if args.synthetic:
        print("\n*** SYNTHETIC DATA: these numbers only demonstrate the pipeline. ***")


if __name__ == "__main__":
    main()

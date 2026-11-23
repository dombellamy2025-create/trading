"""Evening performance report, sent to Telegram.

    python -m bot.evening_report             # build and send
    python -m bot.evening_report --dry-run   # print only
    python -m bot.evening_report --date 2026-11-23

Reads today's rows of trades.csv and daily_pnl.csv (written by the bot), current equity
from Alpaca (falls back to the CSV), and backtest_summary.json (written by backtest.py) to
judge whether live results match the backtest.

Schedule after the US close, e.g. weekdays 16:30 ET:
    30 16 * * 1-5  cd /path/to/alpaca_bot && python -m bot.evening_report
"""
from __future__ import annotations

if __package__ in (None, ""):  # run as a script: put the project root on sys.path
    import pathlib
    import sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from bot.briefing import NY, _money, ny_date, read_csv, send_telegram
from config import Settings, project_path

STOP_TOLERANCE = 1.25   # a 1% stop may lose up to 1.25% (slippage); more = gap / failure
MIN_TRADES, MIN_DAYS = 20, 10   # below this, divergence can't be judged meaningfully
Z_DIVERGE = 2.0
MAX_WORDS = 200


@dataclass
class StopCheck:
    instrument: str
    loss_pct: float        # loss as % of equity (positive number)
    reason: str
    verdict: str


@dataclass
class EveningData:
    day: date
    trades: list[dict] = field(default_factory=list)
    pnl: float | None = None
    pnl_pct: float | None = None
    equity: float | None = None
    stops: list[StopCheck] = field(default_factory=list)
    verdict: str = ""
    details: list[str] = field(default_factory=list)


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def check_stops(trades: list[dict], ref_equity: float, hard_stop_pct: float) -> list[StopCheck]:
    """Did each stopped-out (or large) loss stay within the 1%-of-equity limit?"""
    out = []
    for t in trades:
        pnl = _f(t.get("profit_loss"))
        if pnl is None or pnl >= 0 or not ref_equity:
            continue
        loss = -pnl / ref_equity
        reason = (t.get("exit_reason") or "").lower()
        is_hard = "hard stop" in reason or "broker stop" in reason
        if not is_hard and loss <= hard_stop_pct * STOP_TOLERANCE:
            continue  # ordinary strategy exit within limits
        if loss <= hard_stop_pct * STOP_TOLERANCE:
            verdict = "worked"
        elif is_hard:
            verdict = f"overshot ({loss:.2%} vs {hard_stop_pct:.0%}: gap/slippage)"
        else:
            verdict = f"FAILED: lost {loss:.2%} without a stop exit"
        out.append(StopCheck(t["instrument"], loss, t.get("exit_reason", ""), verdict))
    return out


def compare_to_backtest(summary: dict | None, all_trades: list[dict], daily_rows: list[dict],
                        today_ret: float | None) -> tuple[str, list[str]]:
    if not summary:
        return "No benchmark: run backtest.py first", []
    exp = summary["portfolio"]
    notes, diverging = [], []

    pnls = [_f(t.get("profit_loss")) for t in all_trades if _f(t.get("profit_loss")) is not None]
    p = exp["win_rate"]
    if len(pnls) >= MIN_TRADES and 0 < p < 1:
        wr = sum(x > 0 for x in pnls) / len(pnls)
        z = (wr - p) / math.sqrt(p * (1 - p) / len(pnls))
        notes.append(f"Win rate {wr:.0%} vs {p:.0%} backtest (n={len(pnls)})")
        if abs(z) > Z_DIVERGE:
            diverging.append("win rate")

    rets = [_f(r.get("pnl_pct")) / 100 for r in daily_rows if _f(r.get("pnl_pct")) is not None]
    mu, sd = exp["daily_return_mean"], exp["daily_return_std"]
    if len(rets) >= MIN_DAYS and sd > 0:
        live_mu = sum(rets) / len(rets)
        z = (live_mu - mu) / (sd / math.sqrt(len(rets)))
        notes.append(f"Avg day {live_mu:+.2%} vs {mu:+.2%} backtest ({len(rets)}d)")
        if abs(z) > Z_DIVERGE:
            diverging.append("daily returns")
    if today_ret is not None and sd > 0 and (today_ret - mu) / sd < -Z_DIVERGE:
        notes.append(f"Today was a {abs((today_ret - mu) / sd):.1f}σ down day")

    if diverging:
        return "DIVERGING: " + " and ".join(diverging) + " outside backtest range", notes
    if len(pnls) < MIN_TRADES and len(rets) < MIN_DAYS:
        n, k = len(pnls), len(rets)
        return f"Too early to judge ({n} trade{'s' * (n != 1)}, {k} day{'s' * (k != 1)})", notes
    return "On track with backtest", notes


def collect(s: Settings, day: date, broker=None, summary_path: str | None = None) -> EveningData:
    d = EveningData(day=day)
    trades = read_csv(s.trades_csv)
    daily = read_csv(s.daily_pnl_csv)
    d.trades = [t for t in trades if ny_date(t["timestamp"]) == day]
    row = next((r for r in daily if r["date"] == day.isoformat()), None)
    start_eq = _f(row.get("start_equity")) if row else None
    if row:
        d.pnl, d.equity = _f(row.get("pnl")), _f(row.get("end_equity"))
        d.pnl_pct = d.pnl / start_eq if d.pnl is not None and start_eq else None
    if broker is not None:
        try:
            d.equity = broker.account().equity
        except Exception:
            d.details.append("Alpaca unavailable: equity from daily_pnl.csv")
    d.stops = check_stops(d.trades, start_eq or d.equity or 0, s.hard_stop_pct)

    summary = None
    path = summary_path or project_path("backtest_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            summary = json.load(f)
    cutoff = day - timedelta(days=30)
    recent = [t for t in trades if cutoff < ny_date(t["timestamp"]) <= day]
    d.verdict, notes = compare_to_backtest(summary, recent, daily, d.pnl_pct)
    d.details += notes
    return d


def _trade_line(t: dict) -> str:
    return (f"{t['instrument']} {t['direction']} {_f(t['position_size'], 0):g} "
            f"{_f(t['entry_price'], 0):,.2f}→{_f(t['exit_price'], 0):,.2f} "
            f"{_money(_f(t['profit_loss'], 0))} ({t.get('exit_reason', '')})")


def compose(d: EveningData) -> str:
    L = [f"🌙 Evening report {d.day:%a %b %d}"]
    by_inst: dict[str, int] = {}
    for t in d.trades:
        by_inst[t["instrument"]] = by_inst.get(t["instrument"], 0) + 1
    L.append(f"Trades: {len(d.trades)}" + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(by_inst.items())) + ")"
                                             if by_inst else ""))
    if d.pnl is not None:
        pct = f" ({d.pnl_pct:+.2%})" if d.pnl_pct is not None else ""
        L.append(f"P&L: {_money(d.pnl)}{pct}")
    else:
        L.append("P&L: n/a (no daily_pnl row for today)")
    L.append(f"Equity: ${d.equity:,.0f}" if d.equity is not None else "Equity: n/a")

    scored = [t for t in d.trades if _f(t.get("profit_loss")) is not None]
    if scored:
        best = max(scored, key=lambda t: _f(t["profit_loss"]))
        worst = min(scored, key=lambda t: _f(t["profit_loss"]))
        L.append(f"\nBest: {_trade_line(best)}")
        if worst is not best:
            L.append(f"Worst: {_trade_line(worst)}")

    L.append("\nSTOPS")
    if d.stops:
        L += [f"{c.instrument}: -{c.loss_pct:.2%} of equity, {c.verdict}" for c in d.stops]
    else:
        L.append("No stop-loss exits today")

    L.append("\nVS BACKTEST")
    L.append(d.verdict)
    L += d.details

    msg = "\n".join(L)
    if len(msg.split()) > MAX_WORDS:  # trim the detail lines first
        msg = "\n".join(L[: L.index("\nVS BACKTEST") + 2])
    return msg


def main():
    ap = argparse.ArgumentParser(description="Send the evening performance report to Telegram")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, New York time)")
    args = ap.parse_args()
    s = Settings()
    day = date.fromisoformat(args.date) if args.date else datetime.now(timezone.utc).astimezone(NY).date()
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")
    if not args.dry_run and (not token or not chat or "your_" in token + chat):
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env (or use --dry-run)")
    if not os.path.exists(s.trades_csv) and not os.path.exists(s.daily_pnl_csv):
        raise SystemExit(f"No bot output yet ({s.trades_csv}, {s.daily_pnl_csv}). Start the bot first.")

    broker = None
    try:
        s.validate()
        from bot.broker import AlpacaBroker
        broker = AlpacaBroker(s)
    except SystemExit:
        pass  # no valid keys: report from the CSVs alone
    msg = compose(collect(s, day, broker))
    print(msg)
    print(f"\n({len(msg.split())} words)")
    if not args.dry_run:
        send_telegram(msg, token, chat)
        print("Sent to Telegram.")


if __name__ == "__main__":
    main()

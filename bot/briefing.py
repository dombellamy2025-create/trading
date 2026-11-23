"""Morning briefing: bot results + market conditions + risk flags, sent to Telegram.

    python -m bot.briefing             # build and send to Telegram
    python -m bot.briefing --dry-run   # print only

Sources
  * Alpaca: account equity, open positions (entry price, unrealized P&L), daily bars
  * trades.csv / daily_pnl.csv / bot_state.json written by the bot
  * VIX: Cboe's public delayed-quote feed (Alpaca has no index data)

Needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env (create a bot with @BotFather,
send it a message, then read your chat id from https://api.telegram.org/bot<TOKEN>/getUpdates).
Schedule it with cron, e.g. weekdays 8:45 ET:  45 8 * * 1-5  cd /path/to/alpaca_bot && python -m bot.briefing
"""
from __future__ import annotations

if __package__ in (None, ""):  # run as a script: put the project root on sys.path
    import pathlib
    import sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import CORRELATION_RULES, INSTRUMENTS, Settings

NY = ZoneInfo("America/New_York")
VIX_URL = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_VIX.json"
VIX_ELEVATED = 20.0
ADX_TRENDING, ADX_RANGING = 25.0, 20.0
UNUSUAL_VOLUME = 1.5        # last 24h vs 30-day average daily volume
STOP_WARN = 0.75            # flag positions that have used >= 75% of their 1% stop
DRAWDOWN_FLAG = 0.05
MAX_WORDS = 200


# ------------------------------------------------------------------ data model
@dataclass
class PositionView:
    symbol: str
    side: str
    qty: float
    entry: float
    unrealized: float
    stop_used: float | None = None    # fraction of the 1% hard-stop budget already lost


@dataclass
class BriefingData:
    day: date
    equity: float | None = None
    positions: list[PositionView] = field(default_factory=list)
    yesterday_total: float | None = None
    yesterday_by_symbol: dict[str, float] = field(default_factory=dict)
    win_rate_7d: tuple[int, int] | None = None   # (wins, trades)
    vix: float | None = None
    regimes: dict[str, tuple[str, float]] = field(default_factory=dict)  # SPY/QQQ -> (label, adx)
    btc_volume_ratio: float | None = None
    corr_blocking: bool = False
    drawdown: float | None = None
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ calculations
def adx(df: pd.DataFrame, n: int = 14) -> float:
    """Wilder's ADX on OHLC bars."""
    h, l, c = df["high"], df["low"], df["close"]
    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    w = lambda s: s.ewm(alpha=1 / n, adjust=False).mean()  # noqa: E731
    atr_ = w(tr)
    plus_di, minus_di = 100 * w(plus_dm) / atr_, 100 * w(minus_dm) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return float(w(dx.fillna(0)).iloc[-1])


def regime(adx_value: float, closes: pd.Series) -> str:
    if adx_value >= ADX_TRENDING:
        return "trending " + ("up" if closes.iloc[-1] > closes.iloc[-20] else "down")
    if adx_value < ADX_RANGING:
        return "ranging"
    return "weak trend"


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ny_date(ts: str) -> date:
    return pd.Timestamp(ts).tz_convert(NY).date() if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).date()


def previous_session(today: date) -> date:
    d = today - timedelta(days=1)
    while d.weekday() >= 5:  # skip weekends for "yesterday"
        d -= timedelta(days=1)
    return d


def trade_stats(trades: list[dict], today: date):
    y = previous_session(today)
    by_symbol: dict[str, float] = {}
    wins = total = 0
    for t in trades:
        if t.get("profit_loss") in ("", None):
            continue
        pnl, d = float(t["profit_loss"]), ny_date(t["timestamp"])
        if d == y or (y < d < today):  # include weekend crypto exits with Friday
            by_symbol[t["instrument"]] = by_symbol.get(t["instrument"], 0.0) + pnl
        if today - timedelta(days=7) <= d < today:
            total += 1
            wins += pnl > 0
    return by_symbol, ((wins, total) if total else None)


def daily_stats(rows: list[dict], today: date, equity: float | None):
    y = previous_session(today)
    yrows = [r for r in rows if y.isoformat() <= r["date"] < today.isoformat()]
    total = sum(float(r["pnl"]) for r in yrows) if yrows else None
    ends = [float(r["end_equity"]) for r in rows if r.get("end_equity")]
    if equity is not None:
        ends.append(equity)
    dd = (ends[-1] / max(ends) - 1) if ends else None
    return total, dd


# ------------------------------------------------------------------ collection
def fetch_vix() -> float | None:
    try:
        r = requests.get(VIX_URL, timeout=10, headers={"User-Agent": "trading-briefing/1.0"})
        r.raise_for_status()
        return float(r.json()["data"]["current_price"])
    except Exception:
        return None


def collect(s: Settings, broker, now: datetime | None = None) -> BriefingData:
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(NY).date()
    data = BriefingData(day=today)
    state = {}
    if os.path.exists(s.state_file):
        with open(s.state_file) as f:
            state = json.load(f)
    entry_equity = state.get("entry_equity", {})

    # Alpaca: account + positions
    try:
        acct = broker.account()
        data.equity = acct.equity
        positions = broker.positions()
        symbols = {i.key: i.symbol for i in INSTRUMENTS}
        for key, p in positions.items():
            if key not in symbols:
                continue
            budget = s.hard_stop_pct * float(entry_equity.get(key, acct.equity))
            used = max(0.0, -p.unrealized_pl) / budget if budget else None
            data.positions.append(PositionView(symbols[key], "LONG" if p.qty > 0 else "SHORT",
                                               abs(p.qty), p.avg_entry, p.unrealized_pl, used))
        longs = {k for k, p in positions.items() if p.qty > 0}
        data.corr_blocking = any(set(g) <= longs for g, _ in CORRELATION_RULES)
    except Exception as e:
        data.notes.append(f"Alpaca unavailable ({type(e).__name__})")

    # Bot logs
    data.yesterday_by_symbol, data.win_rate_7d = trade_stats(read_csv(s.trades_csv), today)
    data.yesterday_total, data.drawdown = daily_stats(read_csv(s.daily_pnl_csv), today, data.equity)

    # Market conditions
    data.vix = fetch_vix()
    by_sym = {i.symbol: i for i in INSTRUMENTS}
    for sym in ("SPY", "QQQ"):
        try:
            d = broker.bars(by_sym[sym], 1440, 90)
            a = adx(d)
            data.regimes[sym] = (regime(a, d["close"]), a)
        except Exception:
            pass
    try:
        h = broker.bars(by_sym["BTC/USD"], 60, 31)
        last24 = h["volume"].iloc[-24:].sum()
        prior = h["volume"].iloc[:-24]
        daily_avg = prior.sum() / (len(prior) / 24) if len(prior) else 0
        data.btc_volume_ratio = float(last24 / daily_avg) if daily_avg else None
    except Exception:
        pass
    return data


# ------------------------------------------------------------------ message
def _money(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


def compose(d: BriefingData) -> str:
    L = [f"☀️ Morning briefing {d.day:%a %b %d}"]
    if d.equity is not None:
        L.append(f"Equity ${d.equity:,.0f}")

    L.append("\nPOSITIONS")
    if d.positions:
        for p in d.positions:
            L.append(f"{p.symbol} {p.side} {p.qty:g} @ {p.entry:,.2f}  {_money(p.unrealized)}")
    else:
        L.append("None open")

    L.append("\nYESTERDAY")
    total = "n/a" if d.yesterday_total is None else _money(d.yesterday_total)
    L.append(f"Total {total}")
    per = ", ".join(f"{k} {_money(v)}" for k, v in sorted(d.yesterday_by_symbol.items()))
    L.append(f"Realized: {per or 'no closed trades'}")
    wr = d.win_rate_7d
    L.append(f"7-day win rate: {wr[0] / wr[1]:.0%} ({wr[0]}/{wr[1]})" if wr else "7-day win rate: no trades")

    L.append("\nMARKET")
    if d.vix is not None:
        L.append(f"VIX {d.vix:.1f} ({'ELEVATED' if d.vix >= VIX_ELEVATED else 'normal'})")
    else:
        L.append("VIX n/a")
    for sym, (lab, a) in d.regimes.items():
        L.append(f"{sym} {lab} (ADX {a:.0f})")
    if d.btc_volume_ratio is not None:
        unusual = " UNUSUAL" if d.btc_volume_ratio >= UNUSUAL_VOLUME else ""
        L.append(f"BTC 24h volume {d.btc_volume_ratio:.1f}x avg{unusual}")

    flags = []
    for p in d.positions:
        if p.stop_used is not None and p.stop_used >= STOP_WARN:
            flags.append(f"{p.symbol} at {p.stop_used:.0%} of 1% stop")
    if d.corr_blocking:
        flags.append("Correlation filter ON: new BTC longs blocked")
    if d.drawdown is not None and d.drawdown <= -DRAWDOWN_FLAG:
        flags.append(f"Drawdown {d.drawdown:.1%} from peak")
    L.append("\nRISK FLAGS")
    L += [f"⚠️ {f}" for f in flags] or ["None"]
    if d.drawdown is not None:
        L.append(f"Drawdown from peak: {d.drawdown:.1%}")
    L += [f"Note: {n}" for n in d.notes]

    msg = "\n".join(L)
    words = len(msg.split())
    if words > MAX_WORDS:  # keep it short: trim per-instrument detail first
        msg = msg.replace("Realized: " + per, f"Realized: {len(d.yesterday_by_symbol)} instruments")
    return msg


# ------------------------------------------------------------------ Telegram
def send_telegram(text: str, token: str, chat_id: str) -> None:
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
                      timeout=15)
    if r.status_code != 200 or not r.json().get("ok"):
        raise SystemExit(f"Telegram send failed: HTTP {r.status_code} {r.text[:200]}")


def main():
    ap = argparse.ArgumentParser(description="Send the morning trading briefing to Telegram")
    ap.add_argument("--dry-run", action="store_true", help="print instead of sending")
    args = ap.parse_args()
    s = Settings()
    s.validate()
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")
    if not args.dry_run and (not token or not chat or "your_" in token + chat):
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env (or use --dry-run)")

    from bot.broker import AlpacaBroker
    msg = compose(collect(s, AlpacaBroker(s)))
    print(msg)
    print(f"\n({len(msg.split())} words)")
    if not args.dry_run:
        send_telegram(msg, token, chat)
        print("Sent to Telegram.")


if __name__ == "__main__":
    main()

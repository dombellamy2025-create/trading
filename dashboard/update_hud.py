"""Collect live data for the HUD and write it in the same shape as MOCK_DATA in hud.html.

    python dashboard/update_hud.py              # one snapshot
    python dashboard/update_hud.py --loop 60    # refresh every 60 s (run next to the bot)
    python dashboard/update_hud.py --backtest   # recompute dashboard/backtest_stats.json first

Writes (both git-ignored, they hold account data):
    dashboard/hud_data.js    window.HUD_DATA = {...}; hud.html loads it and refreshes every 30 s
    dashboard/hud_live.html  hud.html with the snapshot inlined: one self-contained file to open or send

Sources: the Alpaca account (equity, positions, orders, clock, prices, bars, portfolio history),
the bot's process and logs/bot.log, bot_state.json, trades.csv, and backtest_stats.json.
Read-only: it never places, changes or cancels orders.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from bot.broker import AlpacaBroker  # noqa: E402
from config import ALL_INSTRUMENTS, ENABLED_SYMBOLS, INSTRUMENTS, Settings  # noqa: E402

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
LOG = ROOT / "logs" / "bot.log"
SUPERVISOR_LOG = ROOT / "logs" / "supervisor.out"
BACKTEST_STATS = HERE / "backtest_stats.json"
LOG_ROTATE_KB = 5 * 1024
KEEPALIVE_SCHEDULE = "HOURLY :42"

# Log/describe() keys -> HUD labels; values in RATIO_KEYS are shown as "1.80x".
LABELS = {"hh": "high20", "ll": "low20", "vol": "vol ratio", "z": "z-score"}
RATIO_KEYS = {"vol"}

LINE_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ (\w+)\s+(.*)$")
SIGNAL_RE = re.compile(r"\[\s*(?P<sym>[^\]]+?)\] (?P<strat>\w+)(?P<new> \[new bar\])? px=(?P<px>[\d.]+) "
                       r"(?P<desc>.*?) pos=(?P<pos>[-\d.e]+) -> (?P<act>\w+) \((?P<reason>[^)]*)\)")


# ---------------------------------------------------------------- helpers
def r2(x, n=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


class Timer:
    """Times each Alpaca call so the HUD shows real request latency."""
    def __init__(self):
        self.samples: list[dict] = []

    def __call__(self, name, fn, *a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            self.samples.append({"name": name, "ms": round((time.perf_counter() - t0) * 1000)})


def read_log_lines(max_bytes=400_000):
    if not LOG.exists():
        return []
    with open(LOG, "rb") as f:
        f.seek(max(0, LOG.stat().st_size - max_bytes))
        text = f.read().decode("utf-8", "replace")
    out = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            out.append((ts, m.group(2), m.group(3)))
    return out


def bot_process():
    """(pid, uptime minutes) of the running bot, or (None, 0)."""
    try:
        out = subprocess.run(["pgrep", "-f", "python -m bot.main"], capture_output=True, text=True).stdout.split()
        if not out:
            return None, 0
        pid = int(out[0])
        et = subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return pid, int(et or 0) // 60
    except Exception:
        return None, 0


def closed_bars(df: pd.DataFrame, minutes: int, now: datetime) -> pd.DataFrame:
    """Drop a still-forming last bar (index = bar start), as the bot does."""
    if df.empty:
        return df
    return df[df.index + pd.Timedelta(minutes=minutes) <= pd.Timestamp(now)]


def next_bar_close(minutes: int, now: datetime, delay_s: int) -> datetime:
    epoch = int(now.timestamp())
    step = minutes * 60
    return datetime.fromtimestamp((epoch // step + 1) * step + delay_s, timezone.utc)


# ---------------------------------------------------------------- backtest stats
def build_backtest_stats(s: Settings) -> dict:
    """Per-instrument stats for every strategy, plus the enabled and the all-five portfolios."""
    import backtest as bt

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=round(6 * 30.44))
    broker = AlpacaBroker(s)
    cache = str(ROOT / "data")
    os.makedirs(cache, exist_ok=True)
    data = {i.key: bt.load_data(broker, i, start, cache, False) for i in ALL_INSTRUMENTS}

    def summary(m):
        return {"trades": m["trades"], "winRate": r2(m["win_rate"], 4), "sharpe": r2(m["sharpe"]),
                "ret": r2(m["total_return"], 4), "maxDD": r2(m["max_drawdown"], 4)}

    per = {}
    for inst in ALL_INSTRUMENTS:
        sim = bt.Simulator(s, [inst], {inst.key: data[inst.key]}, start).run()
        per[inst.symbol] = summary(bt.metrics(sim, 365 if inst.is_crypto else 252))

    def portfolio(insts):
        sim = bt.Simulator(s, insts, data, start, rules=bt.CORRELATION_RULES).run()
        m = bt.metrics(sim, 365)
        pnls = [t.profit_loss for t in sim.trades]
        wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p < 0]
        d = m["curve"].resample("1D").last().ffill().pct_change().dropna()
        mid = pd.Timestamp(start + (end - start) / 2)

        def sh(x):
            return r2(x.mean() / x.std() * math.sqrt(365)) if len(x) > 1 and x.std() > 0 else 0.0
        return summary(m) | {
            "startEquity": bt.CAPITAL, "endEquity": r2(m["final_equity"], 0), "maxDDUsd": r2(m["max_drawdown_usd"], 0),
            "profitFactor": r2(m["profit_factor"]), "avgWin": r2(sum(wins) / len(wins) if wins else 0, 0),
            "avgLoss": r2(sum(losses) / len(losses) if losses else 0, 0), "hardStops": sim.hard_stops,
            "dailyHalts": sim.daily_halts, "sharpeFirstHalf": sh(d[d.index < mid]), "sharpeSecondHalf": sh(d[d.index >= mid]),
        }

    stats = {
        "window": f"{start:%Y-%m-%d} → {end:%Y-%m-%d}",
        "computedUtc": f"{end:%Y-%m-%d %H:%M}",
        "instruments": per,
        "enabled": {"config": " + ".join(ENABLED_SYMBOLS), **portfolio(INSTRUMENTS)},
        "allFive": {"config": "ALL FIVE", **portfolio(ALL_INSTRUMENTS)},
    }
    BACKTEST_STATS.write_text(json.dumps(stats, indent=1))
    return stats


# ---------------------------------------------------------------- snapshot
def snapshot(s: Settings, broker: AlpacaBroker) -> dict:
    now = datetime.now(timezone.utc)
    now_et = now.astimezone(ET)
    T = Timer()

    acct = T("account", broker.account)
    raw_acct = T("account (raw)", broker.api.get_account)
    clock = T("clock", broker.clock)
    positions = T("positions", broker.positions)
    orders = T("orders", broker.open_orders)
    try:
        hist = T("portfolio history", broker.api.get_portfolio_history, period="1D", timeframe="1H", extended_hours=True)
    except Exception:
        hist = None

    # ---- bot process + log
    lines = read_log_lines()
    pid, uptime = bot_process()
    hour_ago, day_ago = now - timedelta(hours=1), now - timedelta(days=1)
    starts = [ts for ts, _, msg in lines if msg.startswith("Starting bot")]
    last_start_line = next((msg for ts, _, msg in reversed(lines) if msg.startswith("Starting bot")), "")
    try:
        state = json.loads((ROOT / "bot_state.json").read_text())
    except Exception:
        state = {}
    daily = state.get("daily", {})
    halted = state.get("halted_date") == now_et.strftime("%Y-%m-%d")
    sup_restarts = SUPERVISOR_LOG.read_text(errors="replace").count("restarting in") if SUPERVISOR_LOG.exists() else 0

    # last decision per symbol from the log
    last_sig = {}
    for ts, _, msg in lines:
        m = SIGNAL_RE.search(msg)
        if m:
            last_sig[m.group("sym").strip()] = (ts, m)

    # ---- instruments
    instruments = []
    gross = 0.0
    for inst in ALL_INSTRUMENTS:
        st = inst.strategy
        enabled = inst.symbol in ENABLED_SYMBOLS
        item = {"symbol": inst.symbol, "strategy": st.name.replace("_", " ").upper(),
                "timeframe": f"{st.timeframe_minutes // 60}H" if st.timeframe_minutes >= 60 else f"{st.timeframe_minutes}M",
                "session": "24/7" if inst.is_crypto else "US HOURS", "enabled": enabled,
                "price": None, "changePct24h": None, "side": "OFF", "qty": 0,
                "signal": "OFF", "reason": "DISABLED", "nextSignal": None, "indicators": {}, "signalTimeUtc": None}
        pos = positions.get(inst.key)
        if pos:
            item["side"] = "LONG" if pos.qty > 0 else "SHORT"
            item["qty"] = r2(abs(pos.qty), 6)
            item["unrealizedPl"] = r2(pos.unrealized_pl)
            gross += abs(pos.market_value)
        if enabled:
            item["side"] = item["side"] if pos else "FLAT"
            item["price"] = r2(T(f"price {inst.symbol}", broker.latest_price, inst))
            bars = T(f"bars {inst.symbol}", broker.bars, inst, st.timeframe_minutes, st.lookback_days)
            closed = closed_bars(bars, st.timeframe_minutes, now)
            if len(closed) >= st.min_bars():
                row = st.indicators(closed).iloc[-1]
                for k, v in re.findall(r"(\w+)=([-+\d.]+)", st.describe(row)):
                    item["indicators"][LABELS.get(k, k)] = f"{float(v):.2f}x" if k in RATIO_KEYS else float(v)
            daily_bars = T(f"daily {inst.symbol}", broker.bars, inst, 1440, 10)
            if inst.is_crypto and len(bars) > 24 and item["price"]:
                ref = float(bars["close"].iloc[-25])
                item["changePct24h"] = r2(item["price"] / ref - 1, 5)
            elif len(daily_bars) >= 2 and item["price"]:
                prev = daily_bars[daily_bars.index.date < now_et.date()]
                if len(prev):
                    item["changePct24h"] = r2(item["price"] / float(prev["close"].iloc[-1]) - 1, 5)
            ls = last_sig.get(inst.symbol)
            if ls:
                ts, m = ls
                item["signal"] = m.group("act")
                item["reason"] = m.group("reason").upper()
                item["signalTimeUtc"] = f"{ts:%H:%M}"
            else:
                item["signal"], item["reason"] = "WAIT", "NO SIGNAL YET"
            if inst.is_crypto or clock.is_open:
                nb = next_bar_close(st.timeframe_minutes, now, s.bar_delay_seconds)
                item["nextSignal"] = f"{nb:%H:%M} UTC"
            else:
                item["signal"], item["reason"] = ("WAIT", "MARKET CLOSED") if not ls else (item["signal"], item["reason"])
                item["nextSignal"] = f"{clock.next_open.astimezone(ET):%a %H:%M} ET".upper()
        instruments.append(item)
    instruments.sort(key=lambda x: not x["enabled"])   # enabled first, config order otherwise

    # ---- backtest
    bts = json.loads(BACKTEST_STATS.read_text()) if BACKTEST_STATS.exists() else None
    for item in instruments:
        item["backtest"] = (bts or {}).get("instruments", {}).get(item["symbol"]) or \
            {"trades": 0, "winRate": 0, "sharpe": 0, "ret": 0, "maxDD": 0}
    if bts:
        e, a5 = bts["enabled"], bts["allFive"]
        backtest = {"window": bts["window"], "config": e["config"], **{k: v for k, v in e.items() if k != "config"},
                    "previous": {"config": a5["config"], "ret": a5["ret"], "sharpe": a5["sharpe"],
                                 "maxDD": a5["maxDD"], "trades": a5["trades"]}}
    else:
        backtest = None

    # ---- equity history (last 24 hourly points; hours before the account existed count as flat)
    eq = [e for e in (hist.equity if hist else []) if e is not None]
    base = float(getattr(hist, "base_value", None) or acct.last_equity or acct.equity)
    eq = ([base] * max(0, 24 - len(eq)) + eq)[-24:]
    eq[-1] = acct.equity
    pnl = [0.0] + [round(eq[i] - eq[i - 1], 2) for i in range(1, len(eq))]

    # ---- trades
    trades = []
    tcsv = ROOT / "trades.csv"
    if tcsv.exists():
        with open(tcsv) as f:
            for r in list(csv.DictReader(f))[-10:][::-1]:
                trades.append({"time": r["timestamp"][11:16] + " UTC", "symbol": r["instrument"],
                               "side": r["direction"].upper(), "entry": float(r["entry_price"]),
                               "exit": float(r["exit_price"]), "qty": float(r["position_size"]),
                               "pnl": float(r["profit_loss"]), "reason": r["exit_reason"].upper()})

    day_pnl = acct.equity - acct.last_equity
    lat = [x["ms"] for x in T.samples if x["name"] == "account"]
    return {
        "meta": {"title": "TRADING HUD", "source": "LIVE", "mode": "PAPER" if s.base_url.startswith("https://paper") else "LIVE",
                 "asOfUtc": f"{now:%Y-%m-%d %H:%M:%S} UTC", "clockEt": f"{now_et:%H:%M}",
                 "dateEt": f"{now_et:%a %d %b}".upper()},
        "account": {"status": str(raw_acct.status).upper(), "equity": r2(acct.equity), "cash": r2(float(raw_acct.cash)),
                    "buyingPower": r2(acct.buying_power), "startEquityToday": r2(acct.last_equity),
                    "dayPnl": r2(day_pnl), "dayPnlPct": r2(day_pnl / acct.last_equity if acct.last_equity else 0, 6),
                    "realizedToday": r2(daily.get("realized_pnl", 0)), "tradesToday": daily.get("trades", 0),
                    "winsToday": daily.get("wins", 0), "lossesToday": daily.get("losses", 0),
                    "openPositions": len(positions), "openOrders": len(orders)},
        "market": {"equitiesOpen": clock.is_open, "cryptoOpen": True,
                   "nextOpen": f"{clock.next_open.astimezone(ET):%a %H:%M} ET".upper(),
                   "nextClose": f"{clock.next_close.astimezone(ET):%a %H:%M} ET".upper(), "feed": s.stock_feed.upper()},
        "bot": {"running": pid is not None, "pid": pid, "paper": "paper=True" in last_start_line,
                "dryRun": "dry_run=True" in last_start_line, "uptimeMin": uptime, "loopSeconds": s.loop_seconds,
                "lastLogUtc": f"{lines[-1][0]:%H:%M:%S}" if lines else "—",
                "apiLatencyMs": lat[0] if lat else None, "apiLatencyMaxMs": 1000,
                "apiErrors1h": sum(1 for ts, lvl, _ in lines if ts >= hour_ago and lvl in ("ERROR", "WARNING")),
                "reconnects24h": sum(1 for ts, _, msg in lines if ts >= day_ago and "reachable again" in msg),
                "startsToday": sum(1 for ts in starts if ts.astimezone(ET).date() == now_et.date()),
                "supervisorRestarts": sup_restarts,
                "logSizeKb": round(LOG.stat().st_size / 1024) if LOG.exists() else 0, "logRotateKb": LOG_ROTATE_KB,
                "latencySamples": T.samples,
                "keepAlive": {"schedule": KEEPALIVE_SCHEDULE,
                              "lastStartUtc": f"{starts[-1]:%H:%M}" if starts else "—"}},
        "risk": {"riskPerTrade": s.risk_per_trade, "hardStopPct": s.hard_stop_pct, "maxDailyLoss": s.max_daily_loss,
                 "dailyLossUsedPct": r2(max(0.0, -day_pnl) / (s.max_daily_loss * acct.last_equity) if acct.last_equity else 0, 4),
                 "grossExposure": r2(gross / acct.equity if acct.equity else 0, 3), "grossCap": s.max_gross_exposure,
                 "maxAllocPerSymbol": s.max_alloc_per_symbol, "brokerStopsLive": sum(1 for o in orders if o.is_stop),
                 "halted": halted,
                 "correlationBlocksToday": sum(1 for ts, _, msg in lines
                                               if ts.astimezone(ET).date() == now_et.date() and "blocked by correlation" in msg)},
        "instruments": instruments,
        "backtest": backtest,
        "hourlyPnl24h": pnl,
        "equityCurve24h": [r2(x) for x in eq],
        "recentTrades": trades,
    }


def write(data: dict):
    blob = json.dumps(data, separators=(",", ":"))
    (HERE / "hud_data.js").write_text(f"window.HUD_DATA = {blob};\n")
    page = (HERE / "hud.html").read_text()
    inline = f"<script>window.HUD_DATA = {blob};</script>"
    page = page.replace('<script src="hud_data.js"></script>', inline, 1)
    (HERE / "hud_live.html").write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--loop", type=int, default=0, help="refresh every N seconds")
    ap.add_argument("--backtest", action="store_true", help="recompute backtest_stats.json first")
    args = ap.parse_args()
    s = Settings()
    s.validate()
    if args.backtest or not BACKTEST_STATS.exists():
        print("computing backtest stats ...")
        build_backtest_stats(s)
    broker = AlpacaBroker(s)
    while True:
        try:
            write(snapshot(s, broker))
            print(f"{datetime.now(timezone.utc):%H:%M:%S} HUD data written")
        except Exception as e:  # keep looping through API hiccups
            print(f"{datetime.now(timezone.utc):%H:%M:%S} HUD update failed: {type(e).__name__}: {e}", file=sys.stderr)
            if not args.loop:
                raise
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()

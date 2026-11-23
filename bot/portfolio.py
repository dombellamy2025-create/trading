"""Portfolio bookkeeping: persisted bot state, the trade log, and daily P&L.

  State        JSON file (bot_state.json) that survives restarts: trailing-stop extremes,
               entry equity for hard stops, open trades, daily P&L running totals.
  TradeJournal trades.csv. Trades are tracked by reconciling Alpaca positions every
               cycle rather than by hooking individual orders, so *every* exit is logged
               no matter what caused it: a strategy exit, the broker-side hard stop, the
               daily loss limit, or a manual close in the Alpaca dashboard. Exit prices come
               from Alpaca's fill activity (volume-weighted across partial fills), falling
               back to the last seen price if fills aren't visible.
  DailyPnL     daily_pnl.csv, one row per New York calendar day.
"""
from __future__ import annotations


import csv
import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")


class State:
    """Small JSON state file that survives restarts.

    extreme:         best price since entry per symbol (for trailing stops)
    entry_equity:    account equity when the position was opened (hard-stop reference)
    last_entry_bar:  bar timestamp that triggered the last entry (no double entries per signal)
    open_trades / exit_reason / daily:  trade-log and daily-P&L bookkeeping
    halted_date:     date the daily loss limit fired
    """

    TABLES = ("extreme", "entry_equity", "last_entry_bar", "open_trades", "exit_reason")

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        self.data = {t: {} for t in self.TABLES} | {"halted_date": None, "daily": None}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    loaded = json.load(f)
                for k in self.data:
                    if k in loaded:
                        self.data[k] = loaded[k]
            except (OSError, json.JSONDecodeError):
                log.warning("Could not read %s, starting fresh", path)

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def get(self, table, key):
        with self.lock:
            return self.data[table].get(key)

    def set(self, table, key, value):
        with self.lock:
            if value is None:
                self.data[table].pop(key, None)
            else:
                self.data[table][key] = value
            self._save()

    def clear_position(self, key):
        with self.lock:
            for t in ("extreme", "entry_equity"):
                self.data[t].pop(key, None)
            self._save()

    def update(self, **kv):
        with self.lock:
            self.data.update(kv)
            self._save()

TRADE_FIELDS = ["timestamp", "instrument", "direction", "entry_price", "exit_price",
                "profit_loss", "position_size", "entry_time", "exit_reason", "exit_price_source"]
DAILY_FIELDS = ["date", "start_equity", "end_equity", "pnl", "pnl_pct",
                "realized_pnl", "trades", "wins", "losses"]


def _sign(q: float) -> int:
    return 1 if q > 0 else -1 if q < 0 else 0


class TradeJournal:
    def __init__(self, path: str, state, broker, symbols: dict[str, str]):
        """symbols: {normalized key: display symbol}, e.g. {"BTCUSD": "BTC/USD"}."""
        self.path, self.state, self.broker, self.symbols = path, state, broker, symbols
        with state.lock:
            state.data.setdefault("open_trades", {})
            state.data.setdefault("exit_reason", {})

    def note_exit_reason(self, key: str, reason: str):
        self.state.set("exit_reason", key, reason)

    def reconcile(self, positions: dict, last_prices: dict[str, float], now: datetime | None = None) -> list[dict]:
        """Diff positions against open trades; returns the trade rows closed this call."""
        now = now or datetime.now(timezone.utc)
        open_trades = self.state.data["open_trades"]
        closed = []
        for key in self.symbols:
            pos = positions.get(key)
            ot = open_trades.get(key)
            side = _sign(pos.qty) if pos else 0
            if ot and side != ot["direction"]:
                closed.append(self._close(key, ot, last_prices.get(key), now))
                ot = None
            if pos and side != 0:
                if ot is None:
                    self.state.set("open_trades", key, {
                        "direction": side, "entry_price": pos.avg_entry, "qty": abs(pos.qty),
                        "entry_time": now.isoformat()})
                    log.info("[%7s] trade opened: %s %g @ %.4f", self.symbols[key],
                             "LONG" if side > 0 else "SHORT", abs(pos.qty), pos.avg_entry)
                elif abs(abs(pos.qty) - ot["qty"]) > 1e-12 or abs(pos.avg_entry - ot["entry_price"]) > 1e-9:
                    ot = dict(ot, qty=abs(pos.qty), entry_price=pos.avg_entry)  # partial fills
                    self.state.set("open_trades", key, ot)
        return closed

    def _exit_price(self, key: str, ot: dict, fallback: float | None) -> tuple[float | None, str]:
        exit_side = "sell" if ot["direction"] > 0 else "buy"
        since = datetime.fromisoformat(ot["entry_time"]) - timedelta(minutes=1)
        try:
            fills = [f for f in self.broker.fills_since(since) if f.symbol == key and f.side == exit_side]
        except Exception as e:
            log.warning("[%7s] could not load fills for exit price (%s)", self.symbols[key], e)
            fills = []
        need, got, notional = ot["qty"], 0.0, 0.0
        for f in fills:
            take = min(f.qty, need - got)
            got += take
            notional += take * f.price
            if got >= need - 1e-9:
                break
        if got > 0:
            return notional / got, "fills"
        return fallback, "last_price"

    def _close(self, key: str, ot: dict, fallback: float | None, now: datetime) -> dict:
        exit_px, source = self._exit_price(key, ot, fallback)
        entry = ot["entry_price"]
        pnl = (exit_px - entry) * ot["qty"] * ot["direction"] if exit_px is not None else None
        reason = self.state.get("exit_reason", key) or "broker stop / external"
        row = {
            "timestamp": now.isoformat(timespec="seconds"),
            "instrument": self.symbols[key],
            "direction": "LONG" if ot["direction"] > 0 else "SHORT",
            "entry_price": round(entry, 6),
            "exit_price": round(exit_px, 6) if exit_px is not None else "",
            "profit_loss": round(pnl, 2) if pnl is not None else "",
            "position_size": ot["qty"],
            "entry_time": ot["entry_time"],
            "exit_reason": reason,
            "exit_price_source": source,
        }
        self._append(row)
        self.state.set("open_trades", key, None)
        self.state.set("exit_reason", key, None)
        log.info("[%7s] trade closed: %s %g  %.4f -> %s  P/L %s  (%s)", row["instrument"],
                 row["direction"], ot["qty"], entry, row["exit_price"], row["profit_loss"], reason)
        return row

    def _append(self, row: dict):
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


class DailyPnL:
    """One row per New York calendar day in daily_pnl.csv (upserted as the day goes on).

    start_equity is the equity at the start of the day (the previous day's last reading
    if the bot ran through midnight, otherwise the first reading of the day).
    """

    def __init__(self, path: str, state, flush_every: timedelta = timedelta(minutes=5)):
        self.path, self.state, self.flush_every = path, state, flush_every
        self._last_flush: datetime | None = None
        with state.lock:
            state.data.setdefault("daily", None)

    def update(self, equity: float, now: datetime | None = None):
        now = now or datetime.now(timezone.utc)
        today = now.astimezone(NY).date().isoformat()
        d = self.state.data["daily"]
        if d is None or d["date"] != today:
            if d is not None:
                self._write(d)  # finalize yesterday
                start = d["end_equity"]
            else:
                start = equity
            d = {"date": today, "start_equity": start, "end_equity": equity,
                 "realized_pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
            self._set(d)
            self._write(d)
            self._last_flush = now
            return
        d = dict(d, end_equity=equity)
        self._set(d)
        if self._last_flush is None or now - self._last_flush >= self.flush_every:
            self._write(d)
            self._last_flush = now

    def record_trade(self, pnl):
        d = self.state.data.get("daily")
        if d is None or pnl in ("", None):
            return
        d = dict(d, realized_pnl=round(d["realized_pnl"] + pnl, 2), trades=d["trades"] + 1,
                 wins=d["wins"] + (pnl > 0), losses=d["losses"] + (pnl < 0))
        self._set(d)
        self._write(d)

    def flush(self):
        d = self.state.data.get("daily")
        if d:
            self._write(d)

    def _set(self, d):
        self.state.update(daily=d)

    def _write(self, d: dict):
        pnl = d["end_equity"] - d["start_equity"]
        row = {"date": d["date"], "start_equity": round(d["start_equity"], 2),
               "end_equity": round(d["end_equity"], 2), "pnl": round(pnl, 2),
               "pnl_pct": round(pnl / d["start_equity"] * 100, 4) if d["start_equity"] else 0.0,
               "realized_pnl": d["realized_pnl"], "trades": d["trades"],
               "wins": d["wins"], "losses": d["losses"]}
        rows = []
        if os.path.exists(self.path):
            with open(self.path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if r.get("date") != d["date"]]
        rows.append(row)
        rows.sort(key=lambda r: r["date"])
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DAILY_FIELDS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, self.path)

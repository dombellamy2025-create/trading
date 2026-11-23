"""Entry point and main trading loop.

    python -m bot.main              # paper trading (default)
    python -m bot.main --dry-run    # compute signals and log them, place no orders
    python -m bot.main --once       # run a single cycle and exit
    python -m bot.main --live       # real money: also requires the live ALPACA_BASE_URL

Scheduling
  * Every LOOP_SECONDS (30s): read account / positions / orders / clock, reconcile the
    trade log, enforce hard stops, maintain broker stop orders, and check exits and
    trailing stops against the latest traded price.
  * Strategy bars are fetched once per bar of each strategy's own timeframe, shortly
    after the bar closes: SPY/QQQ every 15 min, BTC/USD hourly, GLD/USO every 4 hours.
    Entry signals therefore change only when a new bar closes.

Market hours
  * Equities (SPY, QQQ, GLD, USO) are traded only while Alpaca's clock says the US market
    is open (holidays and early closes included). Their broker-side stops stay live
    regardless. BTC/USD is traded 24/7.

Disconnections
  * Transient API failures are retried inside the broker. If Alpaca stays unreachable,
    the cycle is abandoned, the loop backs off exponentially (5s up to 5 min),
    rebuilds the HTTP session every few failures, and resumes when Alpaca answers.
"""
from __future__ import annotations

if __package__ in (None, ""):  # run as a script: put the project root on sys.path
    import pathlib
    import sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from bot import risk_manager as risk
from bot.broker import AlpacaBroker, BrokerUnavailable
from bot.portfolio import DailyPnL, State, TradeJournal
from bot.strategies import FLAT, LONG, SHORT, Action, Context, Signal, atr
from config import CORRELATION_RULES, INSTRUMENTS, Instrument, Settings, project_path

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")


def _side(qty: float) -> int:
    return LONG if qty > 0 else SHORT if qty < 0 else FLAT


class _BarCache:
    __slots__ = ("bucket", "closed", "ind", "bar_ts")

    def __init__(self, bucket, closed, ind, bar_ts):
        self.bucket, self.closed, self.ind, self.bar_ts = bucket, closed, ind, bar_ts


class TradingBot:
    def __init__(self, settings: Settings, broker, instruments: list[Instrument] = INSTRUMENTS,
                 dry_run: bool = False, correlation_rules=CORRELATION_RULES):
        self.s = settings
        self.broker = broker
        self.instruments = instruments
        self.dry_run = dry_run
        self.rules = correlation_rules
        self.state = State(settings.state_file)
        self.journal = TradeJournal(settings.trades_csv, self.state, broker,
                                    {i.key: i.symbol for i in instruments})
        self.daily = DailyPnL(settings.daily_pnl_csv, self.state)
        self._lock = threading.Lock()   # guards budget + longs during a cycle
        self._budget = {"stock": 0.0, "crypto": 0.0}
        self._longs: set[str] = set()
        self._bars: dict[str, _BarCache] = {}
        self._last_price: dict[str, float] = {}
        self._daily_atr: dict[str, tuple[str, float]] = {}  # key -> (date, ATR), refreshed daily
        self._market_open: bool | None = None
        self._shorting_enabled = True
        self._stop = threading.Event()

    # ------------------------------------------------------------------ loop
    def run(self):
        log.info("Starting bot | paper=%s dry_run=%s | sizing: 1 %s ATR(%d) = %.2f%% equity | "
                 "hard stop %.2f%% equity | gross cap %.1fx | check every %ds",
                 self.s.paper, self.dry_run, self.s.sizing_atr_timeframe, self.s.atr_period,
                 self.s.risk_per_trade * 100, self.s.hard_stop_pct * 100,
                 self.s.max_gross_exposure, self.s.loop_seconds)
        for i in self.instruments:
            st = i.strategy
            log.info("  %-8s %-18s signals every %3d min | %s | shorting=%s", i.symbol, st.name,
                     st.timeframe_minutes, "24/7" if i.is_crypto else "US market hours",
                     st.allow_short and not i.is_crypto)
        for group, blocked in self.rules:
            log.info("  correlation filter: no new %s longs while %s are all long",
                     blocked, "+".join(sorted(group)))
        log.info("  trade log: %s | daily P&L: %s", self.s.trades_csv, self.s.daily_pnl_csv)

        failures = 0
        while not self._stop.is_set():
            started = time.monotonic()
            wait = self.s.loop_seconds
            try:
                self.run_cycle()
                if failures:
                    log.warning("Alpaca API reachable again after %d failed attempt(s); resuming", failures)
                failures = 0
            except BrokerUnavailable as e:
                failures += 1
                wait = min(300, 5 * 2 ** (failures - 1))
                log.error("Alpaca API unreachable (%s). Attempt %d; retrying in %ds. "
                          "Broker-side stop orders remain active.", e, failures, wait)
                if failures % 3 == 0 and hasattr(self.broker, "connect"):
                    log.info("Rebuilding Alpaca HTTP session")
                    try:
                        self.broker.connect()
                    except Exception:
                        log.exception("Reconnect failed")
            except Exception:
                log.exception("Cycle failed; will retry next loop")
            self._stop.wait(max(1.0, wait - (time.monotonic() - started)))
        self.daily.flush()
        log.info("Bot stopped")

    def stop(self):
        self._stop.set()

    def run_cycle(self, now: datetime | None = None):
        now = now or datetime.now(timezone.utc)
        acct = self.broker.account()
        clock = self.broker.clock()
        positions = self.broker.positions()
        orders = self.broker.open_orders()
        self._log_market_transition(clock)

        # Bookkeeping first, so every exit (including broker stops) lands in trades.csv.
        for row in self.journal.reconcile(positions, self._last_price, now):
            self.daily.record_trade(row["profit_loss"])
        self.daily.update(acct.equity, now)

        if acct.trading_blocked:
            log.error("Account is blocked from trading; skipping cycle")
            return
        today = now.astimezone(NY).date().isoformat()
        if self.state.data.get("halted_date") == today:
            log.debug("Halted for today after daily loss limit; equity=%.2f", acct.equity)
            return
        if risk.daily_loss_breached(acct.equity, acct.last_equity, self.s.max_daily_loss):
            log.critical("Daily loss limit hit (equity %.2f vs prior close %.2f). Flattening all.",
                         acct.equity, acct.last_equity)
            for key in positions:
                self.journal.note_exit_reason(key, "daily loss limit")
            if not self.dry_run:
                self.broker.close_all()
            self.state.update(halted_date=today, extreme={}, entry_equity={})
            return

        self._shorting_enabled = acct.shorting_enabled
        # Shared budget so concurrent entries can't oversubscribe the account:
        # limited by buying power and by the gross-exposure cap.
        gross_now = sum(abs(p.market_value) for p in positions.values())
        headroom = max(0.0, self.s.max_gross_exposure * acct.equity - gross_now)
        self._budget = {"stock": min(acct.buying_power, headroom),
                        "crypto": min(acct.crypto_buying_power, headroom)}
        # Longs held, plus pending (non-stop) buy orders that aren't covering a short.
        self._longs = {k for k, p in positions.items() if p.qty > 0} | {
            o.symbol for o in orders
            if o.side == "buy" and not o.is_stop and not (o.symbol in positions and positions[o.symbol].qty < 0)}

        # Phase 1: everything not gated by a correlation rule; phase 2: gated symbols,
        # so they see this cycle's new SPY/QQQ entries.
        gated = {blocked for _, blocked in self.rules}
        phases = [[i for i in self.instruments if i.key not in gated],
                  [i for i in self.instruments if i.key in gated]]
        unavailable, attempted = [], 0
        for phase in phases:
            if not phase:
                continue
            with ThreadPoolExecutor(max_workers=len(phase)) as pool:
                futures = {pool.submit(self.process, inst, acct.equity, clock.is_open,
                                       positions.get(inst.key),
                                       [o for o in orders if o.symbol == inst.key], now): inst
                           for inst in phase}
                for fut, inst in futures.items():
                    attempted += 1
                    try:
                        fut.result()
                    except BrokerUnavailable as e:
                        log.warning("[%7s] skipped this cycle: Alpaca unreachable (%s)", inst.symbol, e)
                        unavailable.append(e)
                    except Exception:
                        log.exception("[%7s] processing failed", inst.symbol)
        if unavailable and len(unavailable) == attempted:
            raise unavailable[0]  # nothing got through: treat as a disconnection and back off

    def _log_market_transition(self, clock):
        if clock.is_open == self._market_open:
            return
        if clock.is_open:
            log.info("US equity market OPEN (closes %s). Trading SPY/QQQ/GLD/USO and BTC/USD.",
                     clock.next_close.astimezone(NY).strftime("%a %H:%M ET") if clock.next_close else "?")
        else:
            log.info("US equity market CLOSED (next open %s). Equities paused, their broker stops "
                     "stay live; BTC/USD keeps trading 24/7.",
                     clock.next_open.astimezone(NY).strftime("%a %b %d %H:%M ET") if clock.next_open else "?")
        self._market_open = clock.is_open

    # ------------------------------------------------------------ per symbol
    def process(self, inst: Instrument, equity: float, market_open: bool, pos, orders: list,
                now: datetime | None = None):
        now = now or datetime.now(timezone.utc)
        tag = f"[{inst.symbol:>7}]"
        st = inst.strategy
        qty = pos.qty if pos else 0.0
        side = _side(qty)
        stops = [o for o in orders if o.is_stop]
        working = [o for o in orders if not o.is_stop]
        tradable_now = inst.is_crypto or market_open

        # ---- hard stop: every position, every cycle, before anything else ----
        if side != FLAT:
            entry_eq = self.state.get("entry_equity", inst.key)
            if entry_eq is None:
                entry_eq = equity  # position predates the bot / state was lost
                self.state.set("entry_equity", inst.key, entry_eq)
            if tradable_now and risk.hard_stop_breached(pos.unrealized_pl, entry_eq, self.s.hard_stop_pct):
                log.warning("%s HARD STOP: unrealized P/L %.2f <= -%.2f%% of equity (%.2f)", tag,
                            pos.unrealized_pl, self.s.hard_stop_pct * 100, entry_eq * self.s.hard_stop_pct)
                self._close(inst, tag, "hard stop (1% equity)", stops)
                return
            if not working:
                self._ensure_broker_stop(inst, tag, pos, entry_eq, stops)
        elif stops:
            # Position gone (e.g. stop filled) but an orphan stop remains: cancel it.
            for o in stops:
                log.info("%s cancelling orphan stop order %s", tag, o.id)
                if not self.dry_run:
                    self.broker.cancel(o.id)

        if not tradable_now:
            return
        if working:
            log.info("%s open order pending; waiting", tag)
            return

        # ---- strategy bars: refreshed once per bar of this strategy's timeframe ----
        cache, new_bar = self._refresh_bars(inst, now)
        if cache is None:
            return
        price = self.broker.latest_price(inst) or float(cache.closed["close"].iloc[-1])
        self._last_price[inst.key] = price
        ind, row, bar_ts = cache.ind, cache.ind.iloc[-1], cache.bar_ts

        # Track the best price since entry for trailing stops.
        extreme = self.state.get("extreme", inst.key)
        if side == FLAT:
            if extreme is not None or self.state.get("entry_equity", inst.key) is not None:
                self.state.clear_position(inst.key)  # closed by broker stop / externally
            extreme = None
        else:
            extreme = price if extreme is None else (max(extreme, price) if side == LONG else min(extreme, price))
            self.state.set("extreme", inst.key, extreme)

        sig = st.decide(ind, Context(side=side, price=price, extreme=extreme))
        level = logging.INFO if new_bar or sig.action is not Action.HOLD else logging.DEBUG
        log.log(level, "%s %s%s px=%.2f %s pos=%g -> %s (%s)%s", tag, st.name,
                " [new bar]" if new_bar else "", price, st.describe(row), qty, sig.action.value,
                sig.reason, f" trail={sig.stop:.2f}" if sig.stop and side != FLAT else "")

        if sig.action is Action.EXIT and side != FLAT:
            self._close(inst, tag, sig.reason, stops)
        elif sig.action in (Action.LONG, Action.SHORT):
            target = LONG if sig.action is Action.LONG else SHORT
            if side == target:
                return
            if side != FLAT:
                # Reversal: flatten now; the opposite entry is placed next cycle once flat.
                self._close(inst, tag, f"reversal: {sig.reason}", stops)
            elif self.state.get("last_entry_bar", inst.key) == bar_ts:
                log.debug("%s already entered on bar %s; waiting for a new signal", tag, bar_ts)
            else:
                self._enter(inst, tag, equity, sig, target, bar_ts, cache.closed)

    def _refresh_bars(self, inst: Instrument, now: datetime) -> tuple[_BarCache | None, bool]:
        """Fetch bars only when a new bar of this strategy's timeframe should have closed."""
        st = inst.strategy
        tf_s = st.timeframe_minutes * 60
        bucket = int((now.timestamp() - self.s.bar_delay_seconds) // tf_s)
        cache = self._bars.get(inst.key)
        if cache is not None and cache.bucket == bucket:
            return cache, False
        df = self.broker.bars(inst, st.timeframe_minutes, st.lookback_days)
        if df is None or df.empty:
            log.warning("[%7s] no bar data returned", inst.symbol)
            return cache, False
        closed = self._closed_bars(df, st.timeframe_minutes, now)
        if len(closed) < st.min_bars():
            log.warning("[%7s] only %d closed %d-min bars, need %d", inst.symbol, len(closed),
                        st.timeframe_minutes, st.min_bars())
            return None, False
        age = now - closed.index[-1].to_pydatetime()
        if inst.is_crypto and age > timedelta(minutes=st.timeframe_minutes * 3):
            log.warning("[%7s] latest bar is %s old; data feed may be stale", inst.symbol, age)
        ind = st.indicators(closed)
        bar_ts = str(ind.index[-1])
        new = cache is None or cache.bar_ts != bar_ts
        cache = _BarCache(bucket, closed, ind, bar_ts)
        self._bars[inst.key] = cache
        return cache, new

    def _sizing_atr(self, inst: Instrument, closed: pd.DataFrame) -> float | None:
        """14-period ATR used for position sizing (daily bars by default, cached per day)."""
        n = self.s.atr_period
        if self.s.sizing_atr_timeframe == "strategy":
            return float(atr(closed, n).iloc[-1])
        today = datetime.now(timezone.utc).date().isoformat()
        cached = self._daily_atr.get(inst.key)
        if cached and cached[0] == today:
            return cached[1]
        daily = self._closed_bars(self.broker.bars(inst, 1440, n * 3 + 10), 1440)
        if daily is None or len(daily) < n + 1:
            return None
        value = float(atr(daily, n).iloc[-1])
        self._daily_atr[inst.key] = (today, value)
        return value

    @staticmethod
    def _closed_bars(df: pd.DataFrame, minutes: int, now: datetime | None = None) -> pd.DataFrame:
        """Drop the still-forming bar (timestamp + bar length in the future)."""
        now = now or datetime.now(timezone.utc)
        ends = pd.to_datetime(df.index, utc=True) + timedelta(minutes=minutes)
        return df[ends <= now]

    # ------------------------------------------------------------ orders
    def _ensure_broker_stop(self, inst, tag, pos, entry_eq, stops):
        """Keep exactly one broker-side stop at the 1%-of-equity loss level."""
        side = _side(pos.qty)
        qty = abs(pos.qty)
        stop_px = risk.hard_stop_price(side, pos.avg_entry, qty, entry_eq, self.s.hard_stop_pct)
        if stop_px <= 0:
            log.error("%s computed hard stop %.2f is invalid; relying on software stop", tag, stop_px)
            return
        exit_side = "sell" if side == LONG else "buy"
        ok = [o for o in stops if o.side == exit_side and abs(o.qty - qty) < 1e-9
              and o.stop_price is not None and abs(o.stop_price - stop_px) < 0.011]
        if len(ok) == 1 and len(stops) == 1:
            return
        for o in stops:
            if not self.dry_run:
                self.broker.cancel(o.id)
        log.info("%s placing hard stop: %s %g @ %.2f (max loss %.2f = %.2f%% equity)", tag, exit_side,
                 qty, stop_px, entry_eq * self.s.hard_stop_pct, self.s.hard_stop_pct * 100)
        if not self.dry_run:
            try:
                self.broker.place_stop(inst, qty, exit_side, stop_px)
            except BrokerUnavailable:
                raise
            except Exception:
                log.exception("%s could not place broker stop; software hard stop still active", tag)

    def _enter(self, inst: Instrument, tag: str, equity: float, sig: Signal, target: int,
               bar_ts: str, closed: pd.DataFrame):
        if target == SHORT:
            if not self._shorting_enabled:
                log.warning("%s SHORT signal but shorting is disabled on this account", tag)
                return
            if not self.broker.can_short(inst):
                log.warning("%s SHORT signal but %s is not shortable / easy-to-borrow", tag, inst.symbol)
                return
        atr14 = self._sizing_atr(inst, closed)
        if not atr14 or atr14 <= 0:
            log.warning("%s %s signal but no sizing ATR available; skipping", tag, sig.action.value)
            return
        bucket = "crypto" if inst.is_crypto else "stock"
        # Alpaca does not allow fractional short sales.
        fractional = inst.is_crypto or (self.s.fractional_shares and target == LONG)
        with self._lock:
            if target == LONG:
                blocked = risk.correlation_blocked(inst.key, self._longs, self.rules)
                if blocked:
                    log.info("%s LONG signal blocked by correlation filter: %s", tag, blocked)
                    return
            qty, capped = risk.position_size(equity, self._budget[bucket], sig.price, atr14, self.s,
                                             fractional=fractional)
            if qty <= 0:
                log.info("%s %s signal but size rounds to 0", tag, sig.action.value)
                return
            cost = qty * sig.price
            for k in self._budget:
                self._budget[k] -= cost
            if target == LONG:
                self._longs.add(inst.key)
        risk_per_atr = qty * atr14 / equity
        stop_px = risk.hard_stop_price(target, sig.price, qty, equity, self.s.hard_stop_pct)
        verb = "BUY" if target == LONG else "SELL SHORT"
        log.info("%s %s %g @ ~%.2f (≈$%.2f) | ATR %.4f, 1 ATR = %.2f%% equity%s | hard stop ~%.2f | %s",
                 tag, verb, qty, sig.price, cost, atr14, risk_per_atr * 100,
                 " (capped by allocation/buying power/gross exposure)" if capped else "", stop_px, sig.reason)
        if self.dry_run:
            return
        order = self.broker.buy(inst, qty) if target == LONG else self.broker.sell_short(inst, qty)
        log.info("%s order submitted id=%s; broker stop placed once filled", tag, getattr(order, "id", "?"))
        self.state.set("extreme", inst.key, sig.price)
        self.state.set("entry_equity", inst.key, equity)
        self.state.set("last_entry_bar", inst.key, bar_ts)

    def _close(self, inst: Instrument, tag: str, why: str, stops):
        log.info("%s CLOSE position (%s)", tag, why)
        if self.dry_run:
            return
        self.journal.note_exit_reason(inst.key, why)
        self.broker.close(inst, [o.id for o in stops])
        self.state.clear_position(inst.key)


# ------------------------------------------------------------------ CLI
LOG_FILE = project_path(os.path.join("logs", "bot.log"))


def setup_logging(verbose: bool):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in (logging.StreamHandler(sys.stdout),
              RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)):
        h.setFormatter(fmt)
        root.addHandler(h)
    for noisy in ("urllib3", "alpaca_trade_api"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main():
    p = argparse.ArgumentParser(description="Alpaca multi-strategy bot (SPY, QQQ, BTC/USD, GLD, USO)")
    p.add_argument("--dry-run", action="store_true", help="log signals only, place no orders")
    p.add_argument("--once", action="store_true", help="run one cycle and exit")
    p.add_argument("--live", action="store_true", help="allow live (real-money) trading")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    setup_logging(args.verbose)
    s = Settings()
    s.validate()
    if not s.paper and not args.live:
        raise SystemExit("ALPACA_BASE_URL is the LIVE endpoint but --live not given. Refusing to trade real money.")
    if s.paper and args.live:
        raise SystemExit("--live given but ALPACA_BASE_URL is the paper endpoint. "
                         "Set ALPACA_BASE_URL=https://api.alpaca.markets for live trading.")

    bot = TradingBot(s, AlpacaBroker(s), dry_run=args.dry_run)
    signal.signal(signal.SIGINT, lambda *_: bot.stop())
    signal.signal(signal.SIGTERM, lambda *_: bot.stop())

    if args.once:
        try:
            bot.run_cycle()
        except BrokerUnavailable as e:
            raise SystemExit(f"Alpaca API unreachable: {e}")
        bot.daily.flush()
    else:
        bot.run()


if __name__ == "__main__":
    main()

"""Broker layer on top of the `alpaca-trade-api` SDK.

Responsibilities:
  * account, clock, positions, open orders, fills, latest prices, bars
  * market / stop / stop-limit orders, closing positions
  * resilience: request timeouts, retries with backoff on transient errors
    (connection drops, timeouts, 429, 5xx), reconnecting the HTTP session, and
    idempotent order submission via client_order_id so a dropped connection
    can never create a duplicate order.

Transient failures that persist are raised as BrokerUnavailable, which the bot
treats as "API disconnected" (it backs off and reconnects instead of crashing).
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from alpaca_trade_api.rest import REST, APIError, RetryException, TimeFrame, TimeFrameUnit
from requests.adapters import HTTPAdapter

from config import Instrument, Settings

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = (5, 20)  # (connect, read) seconds; the SDK has no timeout by default


class BrokerUnavailable(Exception):
    """Alpaca could not be reached (network down, timeouts, 5xx, rate limited)."""


@dataclass
class AccountSnapshot:
    equity: float
    last_equity: float
    buying_power: float
    crypto_buying_power: float  # crypto is non-marginable on Alpaca
    trading_blocked: bool
    shorting_enabled: bool = True


@dataclass
class Clock:
    is_open: bool
    next_open: datetime | None = None
    next_close: datetime | None = None


@dataclass
class PositionInfo:
    qty: float              # signed: negative for shorts
    avg_entry: float
    unrealized_pl: float
    market_value: float = 0.0   # signed notional


@dataclass
class OpenOrder:
    id: str
    symbol: str             # normalized (no slash)
    side: str               # "buy" / "sell"
    qty: float
    is_stop: bool           # protective stop / stop-limit order
    stop_price: float | None


@dataclass
class Fill:
    time: datetime
    symbol: str             # normalized
    side: str               # "buy", "sell", "sell_short"
    qty: float
    price: float


class _TimeoutAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = REQUEST_TIMEOUT
        return super().send(request, **kwargs)


def is_transient(e: Exception) -> bool:
    if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                      RetryException, ConnectionError, TimeoutError)):
        return True
    if isinstance(e, APIError):
        return e.status_code in (429, 500, 502, 503, 504)
    return False


def norm(symbol: str) -> str:
    return symbol.replace("/", "")


def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def timeframe(minutes: int) -> TimeFrame:
    if minutes % 1440 == 0:
        return TimeFrame(minutes // 1440, TimeFrameUnit.Day)
    if minutes % 60 == 0:
        return TimeFrame(minutes // 60, TimeFrameUnit.Hour)
    return TimeFrame(minutes, TimeFrameUnit.Minute)


class AlpacaBroker:
    def __init__(self, s: Settings):
        self.s = s
        self._shortable: dict[str, bool] = {}
        self._lock = threading.Lock()
        self.connect()

    # ---- connection --------------------------------------------------------
    def connect(self):
        """(Re)create the REST client and its HTTP session."""
        self.api = REST(self.s.api_key, self.s.secret_key,
                        base_url=self.s.base_url, api_version="v2")
        adapter = _TimeoutAdapter(max_retries=0)
        self.api._session.mount("https://", adapter)
        self.api._session.mount("http://", adapter)

    def _call(self, method: str, *args, attempts: int = 3, **kwargs):
        """Call a REST method by name, retrying transient failures with backoff."""
        for i in range(attempts):
            try:
                return getattr(self.api, method)(*args, **kwargs)
            except Exception as e:
                if not is_transient(e):
                    raise
                if i == attempts - 1:
                    raise BrokerUnavailable(f"{method}: {type(e).__name__}: {e}") from e
                wait = 2 ** i
                log.warning("Alpaca %s failed (%s); retrying in %ss", method, type(e).__name__, wait)
                time.sleep(wait)

    # ---- account -----------------------------------------------------------
    def account(self) -> AccountSnapshot:
        a = self._call("get_account")
        return AccountSnapshot(
            equity=_f(a.equity),
            last_equity=_f(a.last_equity),
            buying_power=_f(a.buying_power),
            crypto_buying_power=_f(getattr(a, "non_marginable_buying_power", None), _f(a.cash)),
            trading_blocked=bool(a.trading_blocked or a.account_blocked),
            shorting_enabled=bool(getattr(a, "shorting_enabled", True)),
        )

    def clock(self) -> Clock:
        c = self._call("get_clock")
        return Clock(bool(c.is_open), pd.Timestamp(c.next_open).to_pydatetime(),
                     pd.Timestamp(c.next_close).to_pydatetime())

    def market_open(self) -> bool:
        return self.clock().is_open

    def positions(self) -> dict[str, PositionInfo]:
        """{normalized symbol: PositionInfo}; crypto keys have no slash."""
        out = {}
        for p in self._call("list_positions"):
            qty = abs(_f(p.qty))
            out[norm(p.symbol)] = PositionInfo(
                qty=-qty if p.side == "short" else qty,
                avg_entry=_f(p.avg_entry_price),
                unrealized_pl=_f(p.unrealized_pl),
                market_value=_f(p.market_value),
            )
        return out

    def open_orders(self) -> list[OpenOrder]:
        out = []
        for o in self._call("list_orders", status="open", limit=500):
            otype = getattr(o, "order_type", None) or o.type
            out.append(OpenOrder(id=str(o.id), symbol=norm(o.symbol), side=o.side, qty=_f(o.qty),
                                 is_stop=otype in ("stop", "stop_limit"),
                                 stop_price=_f(o.stop_price) if o.stop_price is not None else None))
        return out

    def fills_since(self, after: datetime) -> list[Fill]:
        """Executions since `after` (oldest first), used to price exits in trades.csv."""
        acts = self._call("get_activities", activity_types="FILL", after=_ts(after),
                          direction="asc", page_size=100)
        return [Fill(pd.Timestamp(a.transaction_time).to_pydatetime(), norm(a.symbol), a.side,
                     _f(a.qty), _f(a.price)) for a in acts]

    def can_short(self, inst: Instrument) -> bool:
        """Alpaca only lets you short easy-to-borrow US equities (never crypto)."""
        if inst.is_crypto:
            return False
        with self._lock:
            if inst.key not in self._shortable:
                a = self._call("get_asset", inst.symbol)
                self._shortable[inst.key] = bool(a.shortable and a.easy_to_borrow)
            return self._shortable[inst.key]

    def latest_price(self, inst: Instrument) -> float | None:
        if inst.is_crypto:
            trades = self._call("get_latest_crypto_trades", [inst.symbol])
            t = trades.get(inst.symbol) if hasattr(trades, "get") else trades[inst.symbol]
        else:
            t = self._call("get_latest_trade", inst.symbol, feed=self.s.stock_feed)
        return _f(getattr(t, "price", None), None) if t is not None else None

    # ---- orders ------------------------------------------------------------
    def _submit(self, inst: Instrument, **order):
        """Submit an order exactly once, even if the connection drops mid-request.

        A client_order_id is attached; after a transient failure we look the order up
        by that id before retrying, so a request that did reach Alpaca isn't sent twice.
        """
        cid = f"bot-{inst.key}-{uuid.uuid4().hex[:16]}"
        for attempt in range(3):
            try:
                return self.api.submit_order(symbol=inst.symbol, client_order_id=cid, **order)
            except Exception as e:
                if not is_transient(e):
                    raise
                log.warning("submit_order %s failed (%s); checking whether it reached Alpaca",
                            inst.symbol, type(e).__name__)
                time.sleep(1 + attempt)
                try:
                    return self.api.get_order_by_client_order_id(cid)
                except APIError as lookup:
                    if lookup.status_code != 404:
                        raise BrokerUnavailable(f"order state unknown for {cid}") from e
                except Exception as lookup:
                    if is_transient(lookup):
                        raise BrokerUnavailable(f"order state unknown for {cid}") from e
                    raise
                # 404: the order never arrived; safe to retry with the same id.
        raise BrokerUnavailable(f"could not submit order for {inst.symbol}")

    def buy(self, inst: Instrument, qty: float):
        # Crypto requires gtc/ioc; fractional equity orders require day.
        return self._submit(inst, qty=qty, side="buy", type="market",
                            time_in_force="gtc" if inst.is_crypto else "day")

    def sell_short(self, inst: Instrument, qty: float):
        return self._submit(inst, qty=qty, side="sell", type="market", time_in_force="day")

    def place_stop(self, inst: Instrument, qty: float, side: str, stop_price: float):
        """Broker-side protective stop that stays live even if the bot goes down.

        Equities: stop (market) order, GTC (DAY for fractional qty, as Alpaca requires;
        the bot re-places it each session). Crypto: Alpaca has no plain stop orders for
        crypto, so a stop-limit with a 1% limit buffer is used.
        """
        if inst.is_crypto:
            limit = stop_price * (0.99 if side == "sell" else 1.01)
            return self._submit(inst, qty=qty, side=side, type="stop_limit", time_in_force="gtc",
                                stop_price=f"{stop_price:.2f}", limit_price=f"{limit:.2f}")
        tif = "gtc" if float(qty).is_integer() else "day"
        return self._submit(inst, qty=qty, side=side, type="stop", time_in_force=tif,
                            stop_price=f"{stop_price:.2f}")

    def cancel(self, order_id: str):
        try:
            self._call("cancel_order", order_id)
        except APIError as e:
            log.warning("cancel %s failed: %s", order_id, e)

    def close(self, inst: Instrument, stop_order_ids=()):
        """Flatten a long or short position, cancelling its protective stop first."""
        for oid in stop_order_ids:
            self.cancel(oid)
        for i in range(6):  # a cancel can take a moment to release the held qty
            try:
                return self._call("close_position", inst.key)
            except APIError as e:
                if e.status_code == 404:
                    log.info("close %s: no position (already closed)", inst.symbol)
                    return None
                if i == 5 or e.status_code not in (403, 422):
                    raise
                time.sleep(0.5)

    def close_all(self):
        self._call("cancel_all_orders")
        return self._call("close_all_positions")

    # ---- data --------------------------------------------------------------
    def bars(self, inst: Instrument, minutes: int, lookback_days: int) -> pd.DataFrame:
        """OHLCV bars for the last `lookback_days`, oldest first. Last row may still be forming."""
        start = _ts(datetime.now(timezone.utc) - timedelta(days=lookback_days))
        tf = timeframe(minutes)
        if inst.is_crypto:
            b = self._call("get_crypto_bars", inst.symbol, tf, start=start)
        else:
            b = self._call("get_bars", inst.symbol, tf, start=start, feed=self.s.stock_feed,
                           adjustment="raw")
        df = b.df
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if "symbol" in df.columns:
            df = df[df["symbol"] == inst.symbol]
        df.index = pd.to_datetime(df.index, utc=True)
        return df[["open", "high", "low", "close", "volume"]].sort_index()

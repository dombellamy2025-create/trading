"""Offline tests: strategies, risk manager, portfolio logs, and full bot cycles against a fake broker."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import risk_manager as risk  # noqa: E402
from bot.main import TradingBot  # noqa: E402
from bot.broker import AccountSnapshot, BrokerUnavailable, Clock, Fill, OpenOrder, PositionInfo  # noqa: E402
from config import CORRELATION_RULES, INSTRUMENTS, Settings  # noqa: E402
from bot.strategies import (FLAT, LONG, SHORT, Action, Context, MeanReversion,  # noqa: E402
                                  MomentumBreakout, TrendFollowing)


def bars(closes, minutes=60, volume=None) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    # last bar ends one bar ago, so it is fully closed
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0) - pd.Timedelta(minutes=minutes)
    idx = pd.date_range(end=end, periods=len(closes), freq=f"{minutes}min", tz="UTC")
    vol = np.full(len(closes), 1000.0) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame({"open": closes, "high": closes * 1.002, "low": closes * 0.998,
                         "close": closes, "volume": vol}, index=idx)


def settings(tmp_path) -> Settings:
    s = Settings(api_key="k", secret_key="s")
    s.state_file = str(tmp_path / "state.json")
    s.trades_csv = str(tmp_path / "trades.csv")
    s.daily_pnl_csv = str(tmp_path / "daily_pnl.csv")
    return s


rng = np.random.default_rng(1)
FLATISH = 100 + rng.normal(0, 0.2, 60)


# ------------------------------------------------------------ mean reversion
def test_mean_reversion_long_short_and_exit():
    mr = MeanReversion(period=20, entry_std=1.5)
    dip = bars(np.append(FLATISH, 98.0), 15)
    ind = mr.indicators(dip)
    assert mr.decide(ind, Context(FLAT, 98.0, None)).action is Action.LONG
    pop = mr.indicators(bars(np.append(FLATISH, 102.0), 15))
    assert mr.decide(pop, Context(FLAT, 102.0, None)).action is Action.SHORT
    assert mr.decide(mr.indicators(bars(FLATISH, 15)), Context(FLAT, 100.0, None)).action is Action.HOLD
    # Exit when price returns to the SMA
    sma = float(ind["sma"].iloc[-1])
    assert mr.decide(ind, Context(LONG, sma + 0.01, 98.0)).action is Action.EXIT
    assert mr.decide(ind, Context(LONG, sma - 0.5, 98.0)).action is Action.HOLD
    assert mr.decide(pop, Context(SHORT, float(pop["sma"].iloc[-1]) - 0.01, 102)).action is Action.EXIT


def test_mean_reversion_threshold_per_symbol():
    # ~1.6 std move: SPY (1.5) enters, QQQ (1.8) does not.
    base = bars(FLATISH, 15)
    sma = base["close"].tail(19).mean()
    probe = None
    for px in np.arange(99.9, 98.0, -0.01):
        ind = MeanReversion(entry_std=1.5).indicators(bars(np.append(FLATISH, px), 15))
        z = ind["z"].iloc[-1]
        if -1.8 < z < -1.5:
            probe = px
            break
    assert probe is not None and sma > probe
    ctx = Context(FLAT, probe, None)
    spy, qqq = MeanReversion(entry_std=1.5), MeanReversion(entry_std=1.8)
    df = bars(np.append(FLATISH, probe), 15)
    assert spy.decide(spy.indicators(df), ctx).action is Action.LONG
    assert qqq.decide(qqq.indicators(df), ctx).action is Action.HOLD


# ------------------------------------------------------------ breakout
def test_breakout_requires_volume():
    mb = MomentumBreakout(allow_short=False)
    closes = np.append(FLATISH, 103.0)
    low_vol = mb.indicators(bars(closes))
    assert mb.decide(low_vol, Context(FLAT, 103, None)).action is Action.HOLD
    vol = np.append(np.full(60, 1000.0), 1600.0)
    hi_vol = mb.indicators(bars(closes, volume=vol))
    sig = mb.decide(hi_vol, Context(FLAT, 103, None))
    assert sig.action is Action.LONG and sig.stop < 103


def test_breakdown_exits_long_when_shorting_not_allowed():
    mb = MomentumBreakout(allow_short=False)
    vol = np.append(np.full(60, 1000.0), 2000.0)
    ind = mb.indicators(bars(np.append(FLATISH, 99.0), volume=vol))
    assert mb.decide(ind, Context(LONG, 99.0, 99.2)).action is Action.EXIT
    assert mb.decide(ind, Context(FLAT, 99.0, None)).action is Action.HOLD  # no crypto shorts


def test_breakout_trailing_stop_2x_atr():
    mb = MomentumBreakout()
    ind = mb.indicators(bars(FLATISH))
    a = float(ind["atr"].iloc[-1])
    assert mb.decide(ind, Context(LONG, 110 - 2.1 * a, 110)).action is Action.EXIT
    assert mb.decide(ind, Context(LONG, 110 - 1.9 * a, 110)).action is Action.HOLD


# ------------------------------------------------------------ trend following
def trend_series(cross_up_at_end: bool):
    """Long downtrend then sharp rally so EMA50 crosses EMA200 on the final bar (or not)."""
    down = np.linspace(120, 100, 300)
    up = np.linspace(100, 125, 60)
    s = np.concatenate([down, up])
    tf = TrendFollowing()
    for n in range(300, 600):
        ind = tf.indicators(bars(s[:n] if n <= len(s) else np.append(s, np.full(n - len(s), 125)), 240))
        if ind["cross_up"].iloc[-1]:
            data = s[:n] if n <= len(s) else np.append(s, np.full(n - len(s), 125))
            return data if cross_up_at_end else np.append(data, data[-1] * 1.001)
    raise AssertionError("no cross found")


def test_trend_entries_only_on_crossover():
    tf = TrendFollowing()
    ind = tf.indicators(bars(trend_series(True), 240))
    px = float(ind["close"].iloc[-1])
    assert tf.decide(ind, Context(FLAT, px, None)).action is Action.LONG
    later = tf.indicators(bars(trend_series(False), 240))
    assert tf.decide(later, Context(FLAT, px, None)).action is Action.HOLD  # cross was a bar ago


def test_trend_short_on_cross_down_and_3x_atr_stop():
    tf = TrendFollowing()
    s = 200 - trend_series(True)  # mirror image -> cross down on last bar
    ind = tf.indicators(bars(s, 240))
    px = float(ind["close"].iloc[-1])
    assert tf.decide(ind, Context(FLAT, px, None)).action is Action.SHORT
    assert tf.decide(ind, Context(LONG, px, px)).action is Action.SHORT  # reversal
    a = float(ind["atr"].iloc[-1])
    assert tf.decide(ind, Context(SHORT, px + 3.1 * a, px)).action is Action.EXIT
    assert TrendFollowing(allow_short=False).decide(ind, Context(FLAT, px, None)).action is Action.HOLD



# ------------------------------------------------------------ risk
def test_atr_sizing_one_atr_equals_one_percent():
    s = Settings(api_key="k", secret_key="s")
    s.max_alloc_per_symbol = 1.0
    # quiet instrument: ATR 0.50 -> 2000 units; volatile: ATR 5.00 -> 200 units
    quiet, capped = risk.position_size(100_000, 1e9, 40, 0.5, s, fractional=False)
    loud, _ = risk.position_size(100_000, 1e9, 40, 5.0, s, fractional=False)
    assert (quiet, loud, capped) == (2000, 200, False)
    assert quiet * 0.5 == loud * 5.0 == 1000  # 1 ATR = 1% of 100k for both
    # crypto, fractional: ATR $2,000 -> 0.5 BTC
    q, _ = risk.position_size(100_000, 1e9, 90_000, 2000, s, fractional=True)
    assert q == 0.5


def test_sizing_caps_never_exceed_one_percent():
    s = Settings(api_key="k", secret_key="s")
    s.max_alloc_per_symbol = 0.2
    q, capped = risk.position_size(100_000, 1e9, 100, 0.25, s, fractional=False)
    assert capped and q == 200 and q * 0.25 < 1000          # allocation cap binds
    q, capped = risk.position_size(100_000, 5_000, 100, 5, s, fractional=False)
    assert capped and q == 49                                 # buying power binds
    assert risk.position_size(100_000, 1e9, 100, 0, s, fractional=False)[0] == 0


def test_hard_stop_price_is_one_percent_of_equity():
    # uncapped ATR sizing: stop lands exactly 1 ATR from entry
    qty = 1000 / 2.0
    assert risk.hard_stop_price(LONG, 100, qty, 100_000, 0.01) == 98.0
    assert risk.hard_stop_price(SHORT, 100, qty, 100_000, 0.01) == 102.0
    # capped size (fewer shares): stop is further away, loss still exactly 1%
    stop = risk.hard_stop_price(LONG, 100, 200, 100_000, 0.01)
    assert stop == 95.0 and (100 - stop) * 200 == 1000
    # rounding never lets the loss exceed 1%
    stop = risk.hard_stop_price(LONG, 100.003, 300, 100_000, 0.01)
    assert (100.003 - stop) * 300 <= 1000
    assert risk.hard_stop_breached(-1000, 100_000, 0.01)
    assert not risk.hard_stop_breached(-999, 100_000, 0.01)


def test_correlation_rule():
    rules = CORRELATION_RULES
    assert risk.correlation_blocked("BTCUSD", {"SPY", "QQQ"}, rules)
    assert not risk.correlation_blocked("BTCUSD", {"SPY"}, rules)
    assert not risk.correlation_blocked("GLD", {"SPY", "QQQ"}, rules)
    assert risk.daily_loss_breached(96_000, 100_000, 0.03)
    assert not risk.daily_loss_breached(98_000, 100_000, 0.03)


# ------------------------------------------------------------ bot cycle
def daily_bars(price=100.0, rng_half=2.0, n=60):
    """Daily bars with a constant true range of 2 x rng_half -> ATR(14) = 2 x rng_half."""
    df = bars(np.full(n, price), 1440)
    df["high"], df["low"] = price + rng_half, price - rng_half
    return df


class FakeBroker:
    def __init__(self, data, positions=None, market_open=True, equity=100_000, last_equity=100_000,
                 shortable=True):
        self.data, self._open, self._shortable = data, market_open, shortable
        self._pos = {k: (v if isinstance(v, PositionInfo) else PositionInfo(v, 100.0, 0.0))
                     for k, v in (positions or {}).items()}
        self.equity, self.last_equity = equity, last_equity
        self.orders, self.closed, self.closed_all = [], [], False
        self.stops: list[OpenOrder] = []
        self.cancelled: list[str] = []
        self.daily = {}  # symbol -> daily bars; default ATR = 4.0
        self.fills: list[Fill] = []
        self.prices: dict[str, float] = {}
        self.bar_calls = 0

    def account(self):
        return AccountSnapshot(self.equity, self.last_equity, 200_000, 100_000, False, True)

    def clock(self): return Clock(self._open)
    def latest_price(self, inst):
        return self.prices.get(inst.symbol, float(self.data[inst.symbol]["close"].iloc[-1]))
    def fills_since(self, after): return [f for f in self.fills if f.time >= after]
    def positions(self): return dict(self._pos)
    def open_orders(self): return list(self.stops)
    def can_short(self, inst): return self._shortable and not inst.is_crypto
    def bars(self, inst, minutes, days):
        self.bar_calls += minutes != 1440
        if minutes == 1440:
            return self.daily.get(inst.symbol, daily_bars())
        return self.data[inst.symbol]
    def buy(self, inst, qty): self.orders.append((inst.symbol, "buy", qty))
    def sell_short(self, inst, qty): self.orders.append((inst.symbol, "short", qty))

    def place_stop(self, inst, qty, side, stop_price):
        self.stops.append(OpenOrder(f"stop{len(self.stops)}", inst.key, side, qty, True, stop_price))

    def cancel(self, oid):
        self.cancelled.append(oid)
        self.stops = [o for o in self.stops if o.id != oid]

    def close(self, inst, stop_ids=()):
        for oid in stop_ids:
            self.cancel(oid)
        self.closed.append(inst.symbol)
        self._pos.pop(inst.key, None)

    def close_all(self): self.closed_all = True


def with_forming_bar(df, minutes):
    """Append a forming bar so the bot must ignore it for signals."""
    nxt = df.index[-1] + pd.Timedelta(minutes=minutes)
    extra = df.iloc[[-1]].copy()
    extra.index = [nxt]
    return pd.concat([df, extra])


def market_data(spy_px=98.0, qqq_px=100.0):
    vol = np.append(np.full(60, 1000.0), 2000.0)
    return {
        "SPY": with_forming_bar(bars(np.append(FLATISH, spy_px), 15), 15),
        "QQQ": with_forming_bar(bars(np.append(FLATISH, qqq_px), 15), 15),
        "BTC/USD": with_forming_bar(bars(np.append(FLATISH, 103.0), volume=vol), 60),
        "GLD": with_forming_bar(bars(trend_series(True), 240), 240),
        "USO": with_forming_bar(bars(200 - trend_series(True), 240), 240),
    }


def only(*symbols):
    return [i for i in INSTRUMENTS if i.symbol in symbols]


def test_cycle_routes_each_strategy(tmp_path):
    b = FakeBroker(market_data(spy_px=98.0, qqq_px=100.0))
    TradingBot(settings(tmp_path), b).run_cycle()
    got = {sym: side for sym, side, _ in b.orders}
    assert got == {"SPY": "buy", "BTC/USD": "buy", "GLD": "buy", "USO": "short"}
    assert all(q == int(q) for sym, side, q in b.orders if sym != "BTC/USD")  # whole ETF shares


def test_entry_size_is_atr_based(tmp_path):
    # daily ATR 4.0 -> 1% of $100k / 4 = 250 units; ATR 10 -> 100 units
    b = FakeBroker(market_data())
    b.daily = {"BTC/USD": daily_bars(100, 5.0), "SPY": daily_bars(100, 2.0)}
    TradingBot(settings(tmp_path), b).run_cycle()
    sizes = {sym: q for sym, _, q in b.orders}
    assert sizes["BTC/USD"] == 100 and sizes["SPY"] == 250  # volatile -> smaller
    assert sizes["BTC/USD"] * 10 == sizes["SPY"] * 4 == 1000  # same $ risk per ATR


def test_strategy_timeframe_atr_option(tmp_path):
    s = settings(tmp_path)
    s.sizing_atr_timeframe = "strategy"
    s.max_alloc_per_symbol = s.max_gross_exposure = 4.0
    data = market_data()
    b = FakeBroker({"BTC/USD": data["BTC/USD"]})
    TradingBot(s, b, instruments=only("BTC/USD")).run_cycle()
    from bot.strategies import atr
    a = float(atr(data["BTC/USD"].iloc[:-1], 14).iloc[-1])
    wanted = 1000 / a                       # ~1,520 units ~= $157k on a $100k account
    qty = b.orders[0][2]
    assert wanted * 103 > 100_000           # 1-hour ATR sizing would need more than the account
    assert qty < wanted and abs(qty * 103 - 100_000 * 0.98) < 1  # capped at crypto buying power


def test_gross_exposure_cap(tmp_path):
    s = settings(tmp_path)
    s.max_gross_exposure = 1.0
    held = {"GLD": PositionInfo(800, 100, 0, market_value=80_000)}  # 80% of equity already used
    b = FakeBroker(market_data(), positions=held)
    TradingBot(s, b, instruments=only("SPY")).run_cycle()
    assert b.orders[0][2] * 98 <= 20_000  # only the $20k headroom is available


def test_market_closed_only_crypto(tmp_path):
    b = FakeBroker(market_data(), market_open=False)
    TradingBot(settings(tmp_path), b).run_cycle()
    assert [o[0] for o in b.orders] == ["BTC/USD"]


def test_correlation_filter_blocks_btc_when_spy_and_qqq_long(tmp_path):
    held = {"SPY": PositionInfo(10, 100, 0), "QQQ": PositionInfo(10, 100, 0)}
    b = FakeBroker(market_data(spy_px=100.0), positions=held)
    TradingBot(settings(tmp_path), b).run_cycle()
    assert "BTC/USD" not in [o[0] for o in b.orders]
    # only SPY long -> BTC allowed
    b = FakeBroker(market_data(spy_px=100.0), positions={"SPY": PositionInfo(10, 100, 0)})
    (tmp_path / "x").mkdir()
    TradingBot(settings(tmp_path / "x"), b).run_cycle()
    assert "BTC/USD" in [o[0] for o in b.orders]


def test_correlation_filter_sees_same_cycle_entries(tmp_path):
    # SPY already long, QQQ enters long this cycle -> BTC must be blocked in the same cycle
    b = FakeBroker(market_data(spy_px=100.0, qqq_px=98.0), positions={"SPY": PositionInfo(10, 100, 0)})
    TradingBot(settings(tmp_path), b).run_cycle()
    syms = [o[0] for o in b.orders]
    assert "QQQ" in syms and "BTC/USD" not in syms


def test_broker_stop_placed_at_one_percent(tmp_path):
    s = settings(tmp_path)
    data = market_data()
    b = FakeBroker({"GLD": data["GLD"]}, positions={"GLD": PositionInfo(200, 100.0, -50.0)})
    bot = TradingBot(s, b, instruments=only("GLD"))
    bot.run_cycle()
    assert len(b.stops) == 1
    st = b.stops[0]
    assert (st.side, st.qty, st.stop_price) == ("sell", 200, 95.0)  # 200 sh x $5 = $1,000 = 1%
    bot.run_cycle()
    assert len(b.stops) == 1 and not b.cancelled  # not re-placed every cycle
    # partial fill grows the position -> stop is replaced to match
    b._pos["GLD"] = PositionInfo(250, 100.0, 0.0)
    bot.run_cycle()
    assert len(b.stops) == 1 and b.stops[0].qty == 250 and b.stops[0].stop_price == 96.0


def test_short_gets_buy_stop_above_entry(tmp_path):
    b = FakeBroker({"USO": market_data()["USO"]}, positions={"USO": PositionInfo(-100, 50.0, 0.0)})
    TradingBot(settings(tmp_path), b, instruments=only("USO")).run_cycle()
    assert (b.stops[0].side, b.stops[0].stop_price) == ("buy", 60.0)


def test_software_hard_stop_closes_position(tmp_path):
    data = market_data()
    b = FakeBroker({"GLD": data["GLD"]}, positions={"GLD": PositionInfo(200, 100.0, -1000.0)})
    b.stops = [OpenOrder("s1", "GLD", "sell", 200, True, 95.0)]
    TradingBot(settings(tmp_path), b, instruments=only("GLD")).run_cycle()
    assert b.closed == ["GLD"] and "s1" in b.cancelled and not b.orders


def test_orphan_stop_cancelled_when_flat(tmp_path):
    b = FakeBroker({"GLD": market_data()["GLD"]})
    b.stops = [OpenOrder("old", "GLD", "sell", 200, True, 95.0)]
    TradingBot(settings(tmp_path), b, instruments=only("GLD")).run_cycle()
    assert b.cancelled == ["old"]


def test_reversal_closes_then_reopens_next_cycle(tmp_path):
    data = market_data()
    b = FakeBroker({"USO": data["USO"]}, positions={"USO": 10.0})
    bot = TradingBot(settings(tmp_path), b, instruments=only("USO"))
    bot.run_cycle()
    assert b.closed == ["USO"] and not b.orders and not b.stops
    bot.run_cycle()  # now flat, same crossover bar -> short
    assert b.orders and b.orders[0][:2] == ("USO", "short")
    bot.run_cycle()  # no duplicate entry on the same signal bar
    assert len(b.orders) == 1


def test_short_skipped_when_not_shortable(tmp_path):
    b = FakeBroker(market_data(spy_px=102.0), shortable=False)
    TradingBot(settings(tmp_path), b).run_cycle()
    assert not any(side == "short" for _, side, _ in b.orders)


def test_dry_run_and_kill_switch(tmp_path):
    b = FakeBroker(market_data())
    TradingBot(settings(tmp_path), b, dry_run=True).run_cycle()
    assert not b.orders
    b2 = FakeBroker(market_data(), equity=90_000)
    bot = TradingBot(settings(tmp_path), b2)
    bot.run_cycle()
    assert b2.closed_all and not b2.orders
    b2.equity = 100_000
    bot.run_cycle()
    assert not b2.orders  # stays halted today


def test_config_matches_spec():
    by = {i.symbol: i.strategy for i in INSTRUMENTS}
    assert by["SPY"].entry_std == 1.5 and by["QQQ"].entry_std == 1.8
    assert by["SPY"].timeframe_minutes == 15 and by["SPY"].period == 20
    assert by["BTC/USD"].timeframe_minutes == 60 and by["BTC/USD"].atr_mult == 2.0
    assert by["BTC/USD"].vol_mult == 1.5 and not by["BTC/USD"].allow_short
    for sym in ("GLD", "USO"):
        assert by[sym].timeframe_minutes == 240 and (by[sym].fast, by[sym].slow) == (50, 200)
        assert by[sym].atr_mult == 3.0
    s = Settings(api_key="k", secret_key="s")
    assert (s.atr_period, s.risk_per_trade, s.hard_stop_pct) == (14, 0.01, 0.01)


# ------------------------------------------------------------ trade + P&L logs
import csv  # noqa: E402
import json  # noqa: E402
from datetime import date, timedelta  # noqa: E402

from bot.portfolio import DailyPnL  # noqa: E402


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_trade_logged_with_fill_price_and_pnl(tmp_path):
    s = settings(tmp_path)
    data = market_data()
    b = FakeBroker({"GLD": data["GLD"]}, positions={"GLD": PositionInfo(200, 100.0, 0.0)})
    bot = TradingBot(s, b, instruments=only("GLD"))
    t0 = datetime.now(timezone.utc)
    bot.run_cycle(t0)                       # trade opened in journal
    b._pos = {}                             # broker stop filled...
    b.fills = [Fill(t0 + timedelta(minutes=5), "GLD", "sell", 120, 95.10),
               Fill(t0 + timedelta(minutes=5), "GLD", "sell", 80, 94.85)]
    bot.run_cycle(t0 + timedelta(minutes=6))
    rows = read_csv(s.trades_csv)
    assert len(rows) == 1
    r = rows[0]
    assert (r["instrument"], r["direction"], r["position_size"]) == ("GLD", "LONG", "200")
    exit_px = (120 * 95.10 + 80 * 94.85) / 200
    assert float(r["entry_price"]) == 100.0 and abs(float(r["exit_price"]) - exit_px) < 1e-6
    assert abs(float(r["profit_loss"]) - (exit_px - 100) * 200) < 0.01
    assert r["exit_reason"] == "broker stop / external" and r["exit_price_source"] == "fills"
    assert list(r)[:7] == ["timestamp", "instrument", "direction", "entry_price", "exit_price",
                           "profit_loss", "position_size"]
    daily = read_csv(s.daily_pnl_csv)
    assert daily[-1]["trades"] == "1" and daily[-1]["losses"] == "1"


def test_short_trade_logged_with_bot_exit_reason(tmp_path):
    s = settings(tmp_path)
    # short 100 @ 50 with a $1,000 loss = 1% of equity -> hard stop
    b = FakeBroker({"USO": market_data()["USO"]}, positions={"USO": PositionInfo(-100, 50.0, -1000.0)})
    bot = TradingBot(s, b, instruments=only("USO"))
    t0 = datetime.now(timezone.utc)
    bot.run_cycle(t0)                                        # opened + hard-stop close
    b.fills = [Fill(t0 + timedelta(seconds=5), "USO", "buy", 100, 60.0)]
    bot.run_cycle(t0 + timedelta(seconds=30))
    r = read_csv(s.trades_csv)[0]
    assert (r["direction"], float(r["profit_loss"])) == ("SHORT", -1000.0)
    assert r["exit_reason"].startswith("hard stop")


def test_daily_pnl_rows_and_rollover(tmp_path):
    s = settings(tmp_path)
    from bot.portfolio import State
    st = State(s.state_file)
    d = DailyPnL(s.daily_pnl_csv, st)
    day1 = datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc)
    d.update(100_000, day1)
    d.record_trade(250.0)
    d.update(100_400, day1 + timedelta(hours=6))
    d.update(100_300, day1 + timedelta(days=1))              # next NY day: finalizes day 1
    rows = read_csv(s.daily_pnl_csv)
    assert [r["date"] for r in rows] == ["2026-11-20", "2026-11-21"]
    assert (rows[0]["start_equity"], rows[0]["end_equity"], rows[0]["pnl"]) == ("100000", "100400", "400")
    assert rows[0]["realized_pnl"] == "250.0" and rows[0]["wins"] == "1"
    assert rows[1]["start_equity"] == "100400" and rows[1]["pnl"] == "-100"


# ------------------------------------------------------------ scheduling
def test_bars_fetched_once_per_strategy_bar(tmp_path):
    b = FakeBroker(market_data())
    bot = TradingBot(settings(tmp_path), b)
    t0 = datetime.now(timezone.utc).replace(minute=1, second=0, microsecond=0)
    bot.run_cycle(t0)
    first = b.bar_calls
    assert first == 5
    bot.run_cycle(t0 + timedelta(seconds=30))
    assert b.bar_calls == first                              # same bars: nothing refetched
    bot.run_cycle(t0 + timedelta(minutes=15))                # next 15-min bar: only SPY/QQQ
    assert b.bar_calls == first + 2


def test_market_closed_keeps_equity_stops_but_trades_crypto(tmp_path):
    held = {"GLD": PositionInfo(200, 100.0, 0.0)}
    b = FakeBroker(market_data(), positions=held, market_open=False)
    TradingBot(settings(tmp_path), b).run_cycle()
    assert [o[0] for o in b.orders] == ["BTC/USD"]
    assert [st.symbol for st in b.stops] == ["GLD"]          # stop still maintained when closed


# ------------------------------------------------------------ disconnections
def test_run_loop_backs_off_and_recovers(tmp_path, monkeypatch):
    b = FakeBroker(market_data())
    calls = {"n": 0, "connect": 0}
    real = b.account

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise BrokerUnavailable("connection reset")
        return real()

    b.account = flaky
    b.connect = lambda: calls.__setitem__("connect", calls["connect"] + 1)
    bot = TradingBot(settings(tmp_path), b)
    waits = []

    def fake_wait(t):
        waits.append(round(t))
        if len(waits) >= 4:
            bot.stop()
        return False

    monkeypatch.setattr(bot._stop, "wait", fake_wait)
    bot.run()
    assert waits[:3] == [5, 10, 20]                          # exponential backoff
    assert calls["connect"] == 1                             # session rebuilt after 3 failures
    assert b.orders                                          # recovered and traded


class _FakeREST:
    """Stands in for alpaca_trade_api.REST to test order idempotency."""

    def __init__(self, submit_errors, exists_after_error):
        self.submit_errors, self.exists = list(submit_errors), exists_after_error
        self.submitted = []

    def submit_order(self, **kw):
        self.submitted.append(kw["client_order_id"])
        if self.submit_errors:
            raise self.submit_errors.pop(0)
        return type("O", (), {"id": "ok", "client_order_id": kw["client_order_id"]})

    def get_order_by_client_order_id(self, cid):
        if self.exists:
            return type("O", (), {"id": "found", "client_order_id": cid})
        import requests
        from alpaca_trade_api.rest import APIError
        resp = requests.Response()
        resp.status_code = 404
        raise APIError({"message": "not found"}, requests.HTTPError(response=resp))


def _broker_with(fake):
    from bot.broker import AlpacaBroker
    br = AlpacaBroker.__new__(AlpacaBroker)
    br.s, br.api = Settings(api_key="k", secret_key="s"), fake
    return br


def test_submit_not_duplicated_when_order_reached_alpaca(monkeypatch):
    import requests
    monkeypatch.setattr("time.sleep", lambda *_: None)
    fake = _FakeREST([requests.exceptions.ConnectionError("reset")], exists_after_error=True)
    order = _broker_with(fake).buy(INSTRUMENTS[0], 10)
    assert order.id == "found" and len(fake.submitted) == 1


def test_submit_retried_with_same_id_when_order_never_arrived(monkeypatch):
    import requests
    monkeypatch.setattr("time.sleep", lambda *_: None)
    fake = _FakeREST([requests.exceptions.Timeout("timeout")], exists_after_error=False)
    order = _broker_with(fake).buy(INSTRUMENTS[0], 10)
    assert order.id == "ok" and len(fake.submitted) == 2 and len(set(fake.submitted)) == 1


def test_transient_error_classification():
    import requests
    from alpaca_trade_api.rest import APIError
    from bot.broker import is_transient

    def api_err(code):
        r = requests.Response()
        r.status_code = code
        return APIError({"message": "x"}, requests.HTTPError(response=r))

    assert is_transient(requests.exceptions.ConnectionError())
    assert is_transient(api_err(503)) and is_transient(api_err(429))
    assert not is_transient(api_err(403)) and not is_transient(ValueError())


# ------------------------------------------------------------ backtester
import math  # noqa: E402

import backtest as bt  # noqa: E402
from bot.strategies import Signal, Strategy  # noqa: E402
from config import AssetClass, Instrument  # noqa: E402


class Scripted(Strategy):
    """Emits pre-set actions at given bar timestamps; HOLD otherwise."""
    name = "scripted"

    def __init__(self, script, tf):
        self.script, self.timeframe_minutes, self.lookback_days = script, tf, 1

    def min_bars(self):
        return 1

    def indicators(self, df):
        return df.copy()

    def decide(self, ind, ctx):
        a = self.script.get(ind.index[-1], Action.HOLD)
        return Signal(a, "scripted", ctx.price)


def md_from(closes, start, tf, opens=None, highs=None, lows=None, rth_only=False):
    idx = pd.date_range(start, periods=len(closes) * (4 if rth_only else 1), freq=f"{tf}min", tz="UTC")
    if rth_only:
        idx = idx[[bt.in_rth(t) for t in idx]][: len(closes)]
    c = np.asarray(closes, float)
    o = np.asarray(opens if opens is not None else c, float)
    df = pd.DataFrame({"open": o, "high": highs if highs is not None else np.maximum(o, c) * 1.001,
                       "low": lows if lows is not None else np.minimum(o, c) * 0.999,
                       "close": c, "volume": 1000.0}, index=idx[: len(c)])
    daily_idx = pd.date_range(idx[0] - pd.Timedelta(days=30), periods=40, freq="1D", tz="UTC")
    daily = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1.0},
                         index=daily_idx)  # constant daily ATR = 4
    return bt.MarketData(df, daily)


def sim_settings():
    s = Settings(api_key="k", secret_key="s")
    s.max_alloc_per_symbol, s.max_gross_exposure = 1.0, 2.0
    return s


T0 = pd.Timestamp("2026-06-01 00:00", tz="UTC")  # Monday


def test_fill_next_bar_open_with_slippage():
    md = md_from([100, 100, 101, 102, 103, 104], T0, 60, opens=[100, 100, 100.5, 101.5, 102.5, 103.5])
    idx = md.bars.index
    inst = Instrument("BTC/USD", AssetClass.CRYPTO, Scripted({idx[1]: Action.LONG, idx[3]: Action.EXIT}, 60))
    sim = bt.Simulator(sim_settings(), [inst], {"BTCUSD": md}, T0).run()
    t = sim.trades[0]
    assert t.entry_time == idx[2] and t.entry_price == 100.5 * 1.0005   # next open + 0.05%
    assert t.exit_time == idx[4] and t.exit_price == 102.5 * 0.9995     # next open - 0.05%
    assert t.position_size == 250                                       # 1% of $100k / ATR 4
    assert abs(t.profit_loss - (t.exit_price - t.entry_price) * 250) < 1e-9


def test_equity_orders_wait_for_regular_hours():
    # 15-min SPY bars over a full day incl. pre/post market: signal on the 15:45 bar
    # (closes 16:00) must fill at the next day's 9:30 open, not after hours.
    idx = pd.date_range("2026-06-01 13:30", "2026-06-02 20:00", freq="15min", tz="UTC")  # 9:30 ET open
    c = np.linspace(100, 101, len(idx))
    md = bt.MarketData(pd.DataFrame({"open": c, "high": c + .1, "low": c - .1, "close": c, "volume": 1.0},
                                    index=idx), md_from([100], T0, 15).daily)
    sig_bar = pd.Timestamp("2026-06-01 19:45", tz="UTC")  # 15:45 ET
    inst = Instrument("SPY", AssetClass.STOCK, Scripted({sig_bar: Action.LONG}, 15))
    sim = bt.Simulator(sim_settings(), [inst], {"SPY": md}, idx[0]).run()
    assert sim.trades[0].entry_time == pd.Timestamp("2026-06-02 13:30", tz="UTC")


def test_hard_stop_gap_fills_at_open():
    # long 250 @ ~100 -> hard stop ~96; next bar gaps open at 90
    md = md_from([100, 100, 100, 91, 91], T0, 60, opens=[100, 100, 100, 90, 91],
                 lows=[99, 99, 99, 89, 90], highs=[101, 101, 101, 92, 92])
    idx = md.bars.index
    inst = Instrument("BTC/USD", AssetClass.CRYPTO, Scripted({idx[1]: Action.LONG}, 60))
    sim = bt.Simulator(sim_settings(), [inst], {"BTCUSD": md}, T0).run()
    t = sim.trades[0]
    assert t.exit_reason.startswith("hard stop") and t.exit_price == 90 * 0.9995
    assert t.profit_loss < -1000  # a gap can exceed the 1% stop: reported honestly


def test_hard_stop_normal_fill_near_one_percent():
    md = md_from([100, 100, 100, 97, 97], T0, 60, lows=[99, 99, 99, 95, 96])
    idx = md.bars.index
    inst = Instrument("BTC/USD", AssetClass.CRYPTO, Scripted({idx[1]: Action.LONG}, 60))
    t = bt.Simulator(sim_settings(), [inst], {"BTCUSD": md}, T0).run().trades[0]
    exit_slip = t.position_size * (t.exit_price / 0.9995) * 0.0005   # 0.05% of exit notional
    assert abs(t.profit_loss - (-1000 - exit_slip)) < 0.01             # 1% of equity + exit slippage


def test_correlation_filter_in_portfolio_backtest():
    idx_start = pd.Timestamp("2026-06-01 13:30", tz="UTC")
    eq = md_from(np.full(30, 100.0), idx_start, 15)
    btc = md_from(np.full(30, 100.0), idx_start, 60)
    e_idx, b_idx = eq.bars.index, btc.bars.index
    spy = Instrument("SPY", AssetClass.STOCK, Scripted({e_idx[0]: Action.LONG}, 15))
    qqq = Instrument("QQQ", AssetClass.STOCK, Scripted({e_idx[0]: Action.LONG}, 15))
    b = Instrument("BTC/USD", AssetClass.CRYPTO, Scripted({b_idx[2]: Action.LONG}, 60))
    data = {"SPY": eq, "QQQ": eq, "BTCUSD": btc}
    on = bt.Simulator(sim_settings(), [spy, qqq, b], data, idx_start, rules=CORRELATION_RULES).run()
    off = bt.Simulator(sim_settings(), [spy, qqq, b], data, idx_start).run()
    assert on.blocked == 1 and "BTC/USD" not in {t.instrument for t in on.trades}
    assert "BTC/USD" in {t.instrument for t in off.trades}


def test_metrics_formulas():
    sim = bt.Simulator.__new__(bt.Simulator)
    sim.capital, sim.start, sim.hard_stops = 100_000, T0, 0
    sim.trades = [bt.Trade("X", "LONG", T0, T0, 1, 1, 1, p, "") for p in (300, -100, 200, -100)]
    days = pd.date_range(T0, periods=5, freq="1D")
    sim.curve = list(zip(days, [100_000, 110_000, 99_000, 104_500, 100_300]))
    m = bt.metrics(sim, 365)
    assert m["trades"] == 4 and m["win_rate"] == 0.5
    assert (m["avg_win"], m["avg_loss"], m["profit_factor"]) == (250, -100, 2.5)
    assert abs(m["max_drawdown"] - (99_000 / 110_000 - 1)) < 1e-12        # -10%
    assert abs(m["total_return"] - 0.003) < 1e-12
    r = pd.Series([100_000, 110_000, 99_000, 104_500, 100_300]).pct_change().dropna()
    assert abs(m["sharpe"] - r.mean() / r.std() * math.sqrt(365)) < 1e-9


def test_full_synthetic_run_and_chart(tmp_path):
    s = Settings(api_key="k", secret_key="s")
    start = datetime.now(timezone.utc) - timedelta(days=60)
    data = {i.key: bt.synthetic_data(i, start, seed=n) for n, i in enumerate(INSTRUMENTS)}
    results, portfolio = bt.run_backtest(s, data, start)
    assert len(results) == 5
    for m in list(results.values()) + [portfolio]:
        for k in ("trades", "win_rate", "avg_win", "avg_loss", "profit_factor", "max_drawdown", "sharpe",
                  "total_return"):
            assert k in m
        assert m["max_drawdown"] <= 0
    out = tmp_path / "chart.png"
    bt.plot(results, portfolio, str(out))
    assert out.stat().st_size > 50_000
    assert "PORTFOLIO" in bt.summary_table(list(results.items()) + [("PORTFOLIO (x)", portfolio)])


def test_base_url_controls_paper_and_live(monkeypatch):
    from config import LIVE_URL, PAPER_URL
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/")
    assert Settings().paper and Settings().base_url == PAPER_URL
    monkeypatch.setenv("ALPACA_BASE_URL", LIVE_URL)
    assert not Settings().paper
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.market")  # typo
    s = Settings(api_key="k", secret_key="s")
    with pytest.raises(SystemExit):
        s.validate()


def test_placeholder_keys_rejected():
    with pytest.raises(SystemExit):
        Settings(api_key="your_paper_trading_key", secret_key="your_paper_trading_secret").validate()


# ------------------------------------------------------------ morning briefing
from bot import briefing  # noqa: E402


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_briefing_end_to_end(tmp_path, monkeypatch):
    s = settings(tmp_path)
    now = datetime(2026, 11, 23, 13, 30, tzinfo=timezone.utc)  # Monday 8:30 ET
    _write_csv(s.trades_csv, [
        {"timestamp": "2026-11-20T18:00:00+00:00", "instrument": "SPY", "profit_loss": "-420.5"},
        {"timestamp": "2026-11-21T03:00:00+00:00", "instrument": "BTC/USD", "profit_loss": "610"},
        {"timestamp": "2026-11-18T15:00:00+00:00", "instrument": "GLD", "profit_loss": "300"},
        {"timestamp": "2026-11-02T15:00:00+00:00", "instrument": "USO", "profit_loss": "-50"},  # > 7 days
    ])
    _write_csv(s.daily_pnl_csv, [
        {"date": "2026-11-19", "end_equity": "106000", "pnl": "500"},
        {"date": "2026-11-20", "end_equity": "101500", "pnl": "-4500"},
        {"date": "2026-11-21", "end_equity": "100500", "pnl": "-1000"},
    ])
    with open(s.state_file, "w") as f:
        json.dump({"entry_equity": {"SPY": 100_000, "QQQ": 100_000}}, f)
    b = FakeBroker({}, positions={"SPY": PositionInfo(100, 700.0, -850.0),
                                  "QQQ": PositionInfo(50, 600.0, 120.0)})
    b.equity = 100_000
    daily = bars(np.linspace(700, 650, 90), 1440)
    btc_h = bars(np.full(31 * 24, 60_000.0), 60, volume=np.r_[np.full(30 * 24, 100.0), np.full(24, 300.0)])
    b.bars = lambda inst, m, d: btc_h if inst.is_crypto else daily
    monkeypatch.setattr(briefing, "fetch_vix", lambda: 24.3)

    d = briefing.collect(s, b, now)
    assert d.yesterday_by_symbol == {"SPY": -420.5, "BTC/USD": 610.0}
    assert d.yesterday_total == -5500.0                      # Fri + Sat rows
    assert d.win_rate_7d == (2, 3)
    assert d.corr_blocking
    assert abs(d.drawdown - (100_000 / 106_000 - 1)) < 1e-9
    assert abs(d.btc_volume_ratio - 3.0) < 1e-9
    assert d.regimes["SPY"][0] == "trending down"
    msg = briefing.compose(d)
    for needle in ("SPY LONG 100 @ 700.00  -$850", "VIX 24.3 (ELEVATED)", "BTC 24h volume 3.0x avg UNUSUAL",
                   "SPY at 85% of 1% stop", "Correlation filter ON", "Drawdown -5.7% from peak",
                   "7-day win rate: 67% (2/3)", "Total -$5,500"):
        assert needle in msg, needle
    assert len(msg.split()) <= 200


def test_briefing_without_alpaca_or_logs(tmp_path, monkeypatch):
    s = settings(tmp_path)

    class Down:
        def account(self): raise BrokerUnavailable("down")
        def bars(self, *a): raise BrokerUnavailable("down")

    monkeypatch.setattr(briefing, "fetch_vix", lambda: None)
    msg = briefing.compose(briefing.collect(s, Down()))
    assert "None open" in msg and "Alpaca unavailable" in msg and "VIX n/a" in msg


def test_adx_trend_vs_range():
    trend = bars(np.linspace(100, 160, 80), 1440)
    chop = bars(100 + np.tile([0, 1.5, 0, -1.5], 20), 1440)
    chop["high"], chop["low"] = chop["close"] + 1, chop["close"] - 1
    assert briefing.adx(trend) > 25 > 20 > briefing.adx(chop)


def test_telegram_send(monkeypatch):
    sent = {}

    class R:
        status_code = 200
        def json(self): return {"ok": True}

    monkeypatch.setattr(briefing.requests, "post", lambda url, json, timeout: sent.update(url=url, **json) or R())
    briefing.send_telegram("hello", "TOKEN", "123")
    assert sent["url"].endswith("/botTOKEN/sendMessage") and sent["chat_id"] == "123" and sent["text"] == "hello"


# ------------------------------------------------------------ evening report
from bot import evening_report as er  # noqa: E402

TODAY = date(2026, 11, 23)


def _trade(inst, pnl, reason, ts="2026-11-23T17:00:00+00:00", d="LONG", size=10, entry=100, exit_=99):
    return {"timestamp": ts, "instrument": inst, "direction": d, "entry_price": entry, "exit_price": exit_,
            "profit_loss": pnl, "position_size": size, "exit_reason": reason}


def _summary(tmp_path, win_rate=0.5, mu=0.0005, sd=0.008):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"portfolio": {"win_rate": win_rate, "daily_return_mean": mu,
                                           "daily_return_std": sd}}))
    return str(p)


def test_evening_report_today(tmp_path):
    s = settings(tmp_path)
    _write_csv(s.trades_csv, [
        _trade("SPY", 640.0, "reverted to SMA"),
        _trade("QQQ", -1010.0, "hard stop (1% equity)"),                    # worked (1.01%)
        _trade("GLD", -1600.0, "broker stop / external"),                   # gap: 1.6%
        _trade("BTC/USD", 220.0, "trailing stop hit", d="LONG", size=0.05),
        _trade("USO", -50.0, "x", ts="2026-11-22T17:00:00+00:00"),          # yesterday
    ])
    _write_csv(s.daily_pnl_csv, [{"date": "2026-11-23", "start_equity": "100000", "end_equity": "98250",
                                  "pnl": "-1750", "pnl_pct": "-1.75"}])
    d = er.collect(s, TODAY, summary_path=_summary(tmp_path))
    assert len(d.trades) == 4
    assert d.pnl == -1750 and abs(d.pnl_pct + 0.0175) < 1e-12 and d.equity == 98250
    verdicts = {c.instrument: c.verdict for c in d.stops}
    assert verdicts["QQQ"] == "worked" and verdicts["GLD"].startswith("overshot")
    assert d.verdict.startswith("Too early")
    assert any("2.2σ down day" in n for n in d.details)
    msg = er.compose(d)
    for needle in ("Trades: 4 (BTC/USD 1, GLD 1, QQQ 1, SPY 1)", "P&L: -$1,750 (-1.75%)", "Equity: $98,250",
                   "Best: SPY LONG 10", "+$640", "Worst: GLD", "QQQ: -1.01% of equity, worked"):
        assert needle in msg, needle
    assert len(msg.split()) <= 200


def test_stop_failure_detected():
    checks = er.check_stops([_trade("SPY", -2000.0, "reverted to SMA")], 100_000, 0.01)
    assert checks[0].verdict.startswith("FAILED")


def test_backtest_divergence(tmp_path):
    summary = json.loads(open(_summary(tmp_path, win_rate=0.55)).read())
    losers = [_trade("SPY", -100.0, "x") for _ in range(25)] + [_trade("SPY", 100.0, "x") for _ in range(5)]
    days = [{"pnl_pct": "0.05"} for _ in range(12)]
    verdict, notes = er.compare_to_backtest(summary, losers, days, None)
    assert verdict.startswith("DIVERGING") and "win rate" in verdict      # 17% vs 55% on 30 trades
    good = [_trade("SPY", 100.0 if i % 2 else -100.0, "x") for i in range(30)]
    verdict, _ = er.compare_to_backtest(summary, good, days, None)
    assert verdict == "On track with backtest"
    assert er.compare_to_backtest(None, good, days, None)[0].startswith("No benchmark")


def test_backtest_writes_expectations():
    s = Settings(api_key="k", secret_key="s")
    start = datetime.now(timezone.utc) - timedelta(days=40)
    data = {i.key: bt.synthetic_data(i, start, seed=n) for n, i in enumerate(INSTRUMENTS)}
    results, portfolio = bt.run_backtest(s, data, start)
    e = bt.expectations(results, portfolio, start, datetime.now(timezone.utc))
    assert {"win_rate", "daily_return_mean", "daily_return_std", "trades_per_day"} <= set(e["portfolio"])
    assert set(e["instruments"]) == {"SPY", "QQQ", "BTC/USD", "GLD", "USO"}


def test_base_url_with_v2_suffix(monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    s = Settings(api_key="k", secret_key="s")
    assert s.paper and s.base_url == "https://paper-api.alpaca.markets"
    s.validate()

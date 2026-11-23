"""Shared building blocks for all strategies: signal types, indicators, trailing stops.

Each strategy (one file per strategy in this package) is a small class with:
  timeframe_minutes / lookback_days  -> what bars the bot fetches
  indicators(df)                      -> df with indicator columns (closed bars only)
  decide(ind, ctx)                    -> Signal (LONG / SHORT / EXIT / HOLD)

Entries are judged on the last CLOSED bar. Exits and trailing stops are checked
against the latest price (ctx.price) so they react within a bar.

The live bot (bot/main.py) and backtest.py share these classes, so both use the same logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

LONG, FLAT, SHORT = 1, 0, -1


class Action(str, Enum):
    LONG = "LONG"     # be long (from flat: buy; from short: cover first)
    SHORT = "SHORT"   # be short (from flat: sell short; from long: sell first)
    EXIT = "EXIT"     # go flat
    HOLD = "HOLD"


@dataclass
class Signal:
    action: Action
    reason: str
    price: float
    stop: float | None = None   # protective stop level, used for sizing and logging


@dataclass
class Context:
    side: int                   # LONG / FLAT / SHORT
    price: float                # latest price (may be inside a forming bar)
    extreme: float | None       # best price since entry: high for longs, low for shorts


# ---------------------------------------------------------------- indicators
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()  # Wilder smoothing


def trail_exit(ctx: Context, a: float, mult: float) -> Signal | None:
    """Exit signal if the ATR trailing stop is hit, else None."""
    if ctx.side == LONG:
        stop = ctx.extreme - mult * a
        if ctx.price <= stop:
            return Signal(Action.EXIT, f"trailing stop hit ({ctx.price:.2f} <= {stop:.2f})", ctx.price, stop)
    elif ctx.side == SHORT:
        stop = ctx.extreme + mult * a
        if ctx.price >= stop:
            return Signal(Action.EXIT, f"trailing stop hit ({ctx.price:.2f} >= {stop:.2f})", ctx.price, stop)
    return None


def trail_level(ctx: Context, a: float, mult: float) -> float | None:
    if ctx.side == LONG:
        return ctx.extreme - mult * a
    if ctx.side == SHORT:
        return ctx.extreme + mult * a
    return None


class Strategy:
    name = "base"
    timeframe_minutes = 60
    lookback_days = 30
    allow_short = True

    def min_bars(self) -> int:
        raise NotImplementedError

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def decide(self, ind: pd.DataFrame, ctx: Context) -> Signal:
        raise NotImplementedError

    def describe(self, row: pd.Series) -> str:
        return ""

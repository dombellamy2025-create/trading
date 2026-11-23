"""Strategy 2 - Momentum breakout on 1-hour bars (BTC/USD)."""
from __future__ import annotations

from .base import FLAT, LONG, SHORT, Action, Signal, Strategy, atr, trail_exit, trail_level


class MomentumBreakout(Strategy):
    """1-hour Donchian breakout with volume confirmation (BTC/USD).

    LONG  when a bar closes above the prior `period`-bar high on volume >= vol_mult x avg
    SHORT (or EXIT long) on the mirror-image breakdown below the prior `period`-bar low
    Trailing stop: atr_mult x ATR from the best price since entry.
    """
    name = "momentum_breakout"

    def __init__(self, period=20, vol_mult=1.5, atr_mult=2.0, atr_period=14,
                 timeframe_minutes=60, lookback_days=15, allow_short=True):
        self.period, self.vol_mult, self.atr_mult, self.atr_period = period, vol_mult, atr_mult, atr_period
        self.timeframe_minutes, self.lookback_days, self.allow_short = \
            timeframe_minutes, lookback_days, allow_short

    def min_bars(self):
        return max(self.period, self.atr_period) * 2

    def indicators(self, df):
        out = df.copy()
        # shift(1): compare each bar to the channel formed by the bars BEFORE it
        out["hh"] = out["high"].rolling(self.period).max().shift(1)
        out["ll"] = out["low"].rolling(self.period).min().shift(1)
        out["vol_avg"] = out["volume"].rolling(self.period).mean().shift(1)
        out["atr"] = atr(out, self.atr_period)
        return out

    def describe(self, row):
        vr = row["volume"] / row["vol_avg"] if row["vol_avg"] else float("nan")
        return f"hh={row['hh']:.2f} ll={row['ll']:.2f} vol={vr:.2f}x atr={row['atr']:.2f}"

    def decide(self, ind, ctx):
        row = ind.iloc[-1]
        a, px = float(row["atr"]), ctx.price
        vol_ok = row["vol_avg"] > 0 and row["volume"] >= self.vol_mult * row["vol_avg"]
        up = vol_ok and row["close"] > row["hh"]
        down = vol_ok and row["close"] < row["ll"]

        if ctx.side != FLAT:
            hit = trail_exit(ctx, a, self.atr_mult)
            if hit:
                return hit
            stop = trail_level(ctx, a, self.atr_mult)
            if ctx.side == LONG and down:
                return Signal(Action.SHORT if self.allow_short else Action.EXIT,
                              "breakdown below 20-bar low on volume", px, stop)
            if ctx.side == SHORT and up:
                return Signal(Action.LONG, "breakout above 20-bar high on volume", px, stop)
            return Signal(Action.HOLD, "riding trend", px, stop)

        if up:
            return Signal(Action.LONG, "breakout above 20-bar high on volume", px, px - self.atr_mult * a)
        if down and self.allow_short:
            return Signal(Action.SHORT, "breakdown below 20-bar low on volume", px, px + self.atr_mult * a)
        return Signal(Action.HOLD, "no breakout", px)

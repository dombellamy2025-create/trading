"""Strategy 3 - Trend following on 4-hour bars (GLD, USO)."""
from __future__ import annotations

from .base import FLAT, LONG, SHORT, Action, Signal, Strategy, atr, ema, trail_exit, trail_level


class TrendFollowing(Strategy):
    """4-hour EMA crossover (GLD, USO).

    LONG  when EMA(50) crosses above EMA(200)
    SHORT when EMA(50) crosses below EMA(200); a long held while EMA(50) < EMA(200) is exited
          even if the cross itself was missed
    Trailing stop: atr_mult x ATR from the best price since entry.
    Entries happen only on the crossover bar; after a stop-out the bot waits for the next cross.
    """
    name = "trend_following"

    def __init__(self, fast=50, slow=200, atr_mult=3.0, atr_period=14,
                 timeframe_minutes=240, lookback_days=400, allow_short=True):
        self.fast, self.slow, self.atr_mult, self.atr_period = fast, slow, atr_mult, atr_period
        self.timeframe_minutes, self.lookback_days, self.allow_short = \
            timeframe_minutes, lookback_days, allow_short

    def min_bars(self):
        return int(self.slow * 1.5)  # let the 200 EMA settle

    def indicators(self, df):
        out = df.copy()
        out["ema_fast"] = ema(out["close"], self.fast)
        out["ema_slow"] = ema(out["close"], self.slow)
        out["atr"] = atr(out, self.atr_period)
        above = out["ema_fast"] > out["ema_slow"]
        prev = above.shift(1, fill_value=False).astype(bool)
        out["cross_up"] = above & ~prev
        out["cross_down"] = ~above & prev
        return out

    def describe(self, row):
        return f"ema{self.fast}={row['ema_fast']:.2f} ema{self.slow}={row['ema_slow']:.2f} atr={row['atr']:.2f}"

    def decide(self, ind, ctx):
        row = ind.iloc[-1]
        a, px = float(row["atr"]), ctx.price
        bullish = row["ema_fast"] > row["ema_slow"]

        if ctx.side != FLAT:
            hit = trail_exit(ctx, a, self.atr_mult)
            if hit:
                return hit
            stop = trail_level(ctx, a, self.atr_mult)
            if ctx.side == LONG and not bullish:
                if row["cross_down"] and self.allow_short:
                    return Signal(Action.SHORT, "EMA50 crossed below EMA200", px, stop)
                return Signal(Action.EXIT, "EMA50 below EMA200", px, stop)
            if ctx.side == SHORT and bullish:
                if row["cross_up"]:
                    return Signal(Action.LONG, "EMA50 crossed above EMA200", px, stop)
                return Signal(Action.EXIT, "EMA50 above EMA200", px, stop)
            return Signal(Action.HOLD, "in trend", px, stop)

        if row["cross_up"]:
            return Signal(Action.LONG, "EMA50 crossed above EMA200", px, px - self.atr_mult * a)
        if row["cross_down"] and self.allow_short:
            return Signal(Action.SHORT, "EMA50 crossed below EMA200", px, px + self.atr_mult * a)
        return Signal(Action.HOLD, "no crossover", px)

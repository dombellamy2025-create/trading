"""Strategy 1 - Mean reversion on 15-minute bars (SPY, QQQ)."""
from __future__ import annotations

from .base import LONG, SHORT, Action, Signal, Strategy


class MeanReversion(Strategy):
    """15-min Bollinger-style reversion (SPY, QQQ).

    LONG  when the bar closes more than `entry_std` std-devs BELOW the SMA
    SHORT when the bar closes more than `entry_std` std-devs ABOVE the SMA
    EXIT  when price returns to the SMA
    Optional `stop_std`: also exit if price stretches this many std-devs from the SMA
    against the position (0 = off, which matches the spec).
    """
    name = "mean_reversion"

    def __init__(self, period=20, entry_std=1.5, stop_std=0.0, timeframe_minutes=15,
                 lookback_days=10, allow_short=True):
        self.period, self.entry_std, self.stop_std = period, entry_std, stop_std
        self.timeframe_minutes, self.lookback_days, self.allow_short = \
            timeframe_minutes, lookback_days, allow_short

    def min_bars(self):
        return self.period + 1

    def indicators(self, df):
        out = df.copy()
        out["sma"] = out["close"].rolling(self.period).mean()
        out["std"] = out["close"].rolling(self.period).std()
        out["z"] = (out["close"] - out["sma"]) / out["std"]
        return out

    def describe(self, row):
        return f"sma={row['sma']:.2f} std={row['std']:.3f} z={row['z']:+.2f}"

    def decide(self, ind, ctx):
        row = ind.iloc[-1]
        px, sma, sd = ctx.price, float(row["sma"]), float(row["std"])
        stop_dist = self.stop_std * sd if self.stop_std > 0 else None

        if ctx.side == LONG:
            if px >= sma:
                return Signal(Action.EXIT, "reverted to SMA", px)
            if stop_dist and px <= sma - stop_dist:
                return Signal(Action.EXIT, f"stop: {self.stop_std} std below SMA", px)
            return Signal(Action.HOLD, "waiting for reversion", px, sma - stop_dist if stop_dist else None)
        if ctx.side == SHORT:
            if px <= sma:
                return Signal(Action.EXIT, "reverted to SMA", px)
            if stop_dist and px >= sma + stop_dist:
                return Signal(Action.EXIT, f"stop: {self.stop_std} std above SMA", px)
            return Signal(Action.HOLD, "waiting for reversion", px, sma + stop_dist if stop_dist else None)

        z = float(row["z"])
        # Require price to still be stretched now, not already back at the mean.
        if z < -self.entry_std and px < sma:
            return Signal(Action.LONG, f"z={z:.2f} < -{self.entry_std}", px,
                          sma - stop_dist if stop_dist else None)
        if z > self.entry_std and px > sma and self.allow_short:
            return Signal(Action.SHORT, f"z={z:.2f} > +{self.entry_std}", px,
                          sma + stop_dist if stop_dist else None)
        return Signal(Action.HOLD, "within bands", px)

"""Risk management: ATR position sizing, 1%-of-equity hard stops, correlation filter,
and the daily loss limit. Pure functions, shared by the live bot and the backtester."""
from __future__ import annotations

import math

LONG, SHORT = 1, -1


def position_size(equity: float, buying_power: float, price: float, atr: float, s,
                  fractional: bool, decimals: int = 6) -> tuple[float, bool]:
    """ATR-based size: a 1-ATR move against the position = risk_per_trade x equity.

        qty = (equity x risk_per_trade) / ATR

    A quiet instrument (small ATR) gets more units, a volatile one fewer, and the
    dollar risk per ATR is the same for all. Hard caps still apply, because an order
    can't exceed what the account can fund: max_alloc_per_symbol of equity and the
    remaining budget (buying power and gross-exposure headroom). When a cap binds,
    risk per ATR is *below* 1%, never above.

    Returns (qty, capped).
    """
    if price <= 0 or equity <= 0 or atr <= 0:
        return 0.0, False
    atr_qty = (equity * s.risk_per_trade) / atr
    cap_qty = min((equity * s.max_alloc_per_symbol) / price, (buying_power * 0.98) / price)
    qty = max(0.0, min(atr_qty, cap_qty))
    capped = cap_qty < atr_qty
    if fractional:
        factor = 10 ** decimals
        return math.floor(qty * factor) / factor, capped
    return float(math.floor(qty)), capped


def hard_stop_price(side: int, avg_entry: float, qty: float, equity_at_entry: float,
                    hard_stop_pct: float) -> float:
    """Price at which the position has lost exactly hard_stop_pct of equity.

    Long:  entry - (equity x pct) / qty     Short: entry + (equity x pct) / qty
    Rounded to the cent in the protective direction, so the loss is never above the limit.
    With uncapped ATR sizing this sits exactly 1 ATR from entry.
    """
    dist = (equity_at_entry * hard_stop_pct) / abs(qty)
    if side == LONG:
        return math.ceil((avg_entry - dist) * 100) / 100
    return math.floor((avg_entry + dist) * 100) / 100


def hard_stop_breached(unrealized_pl: float, equity_at_entry: float, hard_stop_pct: float) -> bool:
    return unrealized_pl <= -(equity_at_entry * hard_stop_pct)


def daily_loss_breached(equity: float, last_equity: float, max_daily_loss: float) -> bool:
    """True when equity has fallen more than max_daily_loss since the prior close."""
    if last_equity <= 0:
        return False
    return (last_equity - equity) / last_equity >= max_daily_loss


def correlation_blocked(symbol_key: str, longs: set[str], rules) -> str | None:
    """Return a reason if a new long on symbol_key is blocked by a correlation rule.

    rules: iterable of (group_of_keys, blocked_key). If every symbol in the group is
    already long, no new long is opened on blocked_key.
    """
    for group, blocked in rules:
        if symbol_key == blocked and set(group) <= longs:
            return f"{' and '.join(sorted(group))} already long (risk-on exposure)"
    return None

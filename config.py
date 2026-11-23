"""Project configuration: instruments, their strategies, correlation rules, and settings.

Secrets (API keys) and optional overrides are read from .env in the project root."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv

from bot.strategies import MeanReversion, MomentumBreakout, Strategy, TrendFollowing

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def project_path(p: str) -> str:
    """Relative output paths are resolved against the project root, not the cwd."""
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def _env(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    raw = raw.split("#", 1)[0].strip()
    if cast is bool:
        return raw.lower() in ("1", "true", "yes", "y", "on")
    return cast(raw)


class AssetClass(str, Enum):
    STOCK = "stock"    # US equities / ETFs: regular market hours only
    CRYPTO = "crypto"  # 24/7, cannot be shorted on Alpaca


@dataclass
class Instrument:
    symbol: str          # as used for orders/data, e.g. "SPY" or "BTC/USD"
    asset_class: AssetClass
    strategy: Strategy

    @property
    def key(self) -> str:
        """Alpaca reports crypto positions without the slash ("BTCUSD")."""
        return self.symbol.replace("/", "")

    @property
    def is_crypto(self) -> bool:
        return self.asset_class is AssetClass.CRYPTO


ALLOW_SHORT = _env("ALLOW_SHORT", True, bool)
MR_STOP_STD = _env("MR_STOP_STD", 0.0, float)  # optional mean-reversion stop, 0 = off
# Alpaca crypto trading fee per fill, as a fraction of notional (base tier: 0.25% taker, 0.15% maker).
# Taken out of the asset received, so it shows in equity but not in fill prices. Stocks: commission-free.
CRYPTO_FEE_RATE = _env("CRYPTO_FEE_RATE", 0.0025, float)

ALL_INSTRUMENTS: list[Instrument] = [
    # Strategy 1: mean reversion, 15-min bars, 20-period SMA / std-dev
    Instrument("SPY", AssetClass.STOCK,
               MeanReversion(period=20, entry_std=1.5, stop_std=MR_STOP_STD, allow_short=ALLOW_SHORT)),
    Instrument("QQQ", AssetClass.STOCK,
               MeanReversion(period=20, entry_std=1.8, stop_std=MR_STOP_STD, allow_short=ALLOW_SHORT)),
    # Strategy 2: momentum breakout, 1-hour bars. Alpaca does not allow crypto shorts,
    # so a downside breakout exits the long instead of opening a short.
    Instrument("BTC/USD", AssetClass.CRYPTO,
               MomentumBreakout(period=20, vol_mult=1.5, atr_mult=2.0, allow_short=False)),
    # Strategy 3: trend following, 4-hour bars, EMA 50/200 crossover
    Instrument("GLD", AssetClass.STOCK, TrendFollowing(fast=50, slow=200, atr_mult=3.0, allow_short=ALLOW_SHORT)),
    Instrument("USO", AssetClass.STOCK, TrendFollowing(fast=50, slow=200, atr_mult=3.0, allow_short=ALLOW_SHORT)),
]

# Which of ALL_INSTRUMENTS actually trade (and are backtested). Comma-separated symbols.
# Default BTC/USD + GLD: over the May-Nov 2026 backtest SPY, QQQ and USO had negative Sharpe
# ratios at these settings, while BTC/USD + GLD alone returned +6.9% (Sharpe 1.76, max DD -2.0%).
# All five: ENABLED_SYMBOLS=SPY,QQQ,BTC/USD,GLD,USO
ENABLED_SYMBOLS = [x.strip().upper() for x in _env("ENABLED_SYMBOLS", "BTC/USD,GLD").split(",") if x.strip()]
_unknown = set(ENABLED_SYMBOLS) - {i.symbol for i in ALL_INSTRUMENTS}
if _unknown or not ENABLED_SYMBOLS:
    raise SystemExit(f"ENABLED_SYMBOLS: unknown or empty {sorted(_unknown)}; "
                     f"choose from {[i.symbol for i in ALL_INSTRUMENTS]}")
INSTRUMENTS: list[Instrument] = [i for i in ALL_INSTRUMENTS if i.symbol in ENABLED_SYMBOLS]


# Correlation filter: (group, blocked). If every symbol in the group is already long,
# no NEW long is opened on the blocked symbol. Keys use the slash-less form ("BTCUSD").
CORRELATION_RULES: list[tuple[frozenset[str], str]] = [
    (frozenset({"SPY", "QQQ"}), "BTCUSD"),  # don't stack BTC on top of long SPY + QQQ
]


PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: _env("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: _env("ALPACA_SECRET_KEY", ""))
    # Trading endpoint. The paper URL (default) = paper account; the live URL = real money,
    # which also requires the --live flag. ALPACA_PAPER=false is accepted as a legacy alias.
    base_url: str = field(default_factory=lambda: _env(
        "ALPACA_BASE_URL", PAPER_URL if _env("ALPACA_PAPER", True, bool) else LIVE_URL).rstrip("/")
        .removesuffix("/v2"))  # the SDK appends /v2 itself
    stock_feed: str = field(default_factory=lambda: _env("STOCK_DATA_FEED", "iex"))

    # Risk
    atr_period: int = field(default_factory=lambda: _env("ATR_PERIOD", 14, int))
    # Size so a 1-ATR adverse move = this fraction of equity
    risk_per_trade: float = field(default_factory=lambda: _env("RISK_PER_TRADE", 0.01, float))
    # Which bars the sizing ATR uses: "daily" (14 trading days) or "strategy"
    # (14 bars of each strategy's own timeframe: 15m / 1h / 4h).
    sizing_atr_timeframe: str = field(default_factory=lambda: _env("SIZING_ATR_TIMEFRAME", "daily"))
    # Hard stop on every position: close when the loss reaches this fraction of equity
    hard_stop_pct: float = field(default_factory=lambda: _env("HARD_STOP_PCT", 0.01, float))
    # Notional caps: an ATR-sized order can't exceed what the account can fund.
    max_alloc_per_symbol: float = field(default_factory=lambda: _env("MAX_ALLOC_PER_SYMBOL", 1.0, float))
    max_gross_exposure: float = field(default_factory=lambda: _env("MAX_GROSS_EXPOSURE", 2.0, float))
    max_daily_loss: float = field(default_factory=lambda: _env("MAX_DAILY_LOSS", 0.03, float))
    fractional_shares: bool = field(default_factory=lambda: _env("FRACTIONAL_SHARES", False, bool))

    # Runtime
    # How often stops/exits are checked against the latest price. Strategy signals are
    # evaluated once per bar of each strategy's own timeframe (15m / 1h / 4h).
    loop_seconds: int = field(default_factory=lambda: _env("LOOP_SECONDS", 30, int))
    # Wait this long after a bar closes before fetching it (lets Alpaca finalize the bar).
    bar_delay_seconds: int = field(default_factory=lambda: _env("BAR_DELAY_SECONDS", 15, int))
    state_file: str = field(default_factory=lambda: project_path(_env("STATE_FILE", "bot_state.json")))
    trades_csv: str = field(default_factory=lambda: project_path(_env("TRADES_CSV", "trades.csv")))
    daily_pnl_csv: str = field(default_factory=lambda: project_path(_env("DAILY_PNL_CSV", "daily_pnl.csv")))

    @property
    def paper(self) -> bool:
        return self.base_url == PAPER_URL

    def validate(self) -> None:
        if self.base_url not in (PAPER_URL, LIVE_URL):
            raise SystemExit(f"ALPACA_BASE_URL must be {PAPER_URL} (paper) or {LIVE_URL} (live), "
                             f"got {self.base_url!r}")
        if not self.api_key or not self.secret_key or "your_" in self.api_key + self.secret_key:
            raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env (see .env.example)")
        if not 0 < self.hard_stop_pct <= 0.05 or not 0 < self.risk_per_trade <= 0.05:
            raise SystemExit("HARD_STOP_PCT and RISK_PER_TRADE must be between 0 and 0.05")
        if self.sizing_atr_timeframe not in ("daily", "strategy"):
            raise SystemExit("SIZING_ATR_TIMEFRAME must be 'daily' or 'strategy'")
        if not 0 < self.max_alloc_per_symbol <= self.max_gross_exposure <= 4:
            raise SystemExit("Need 0 < MAX_ALLOC_PER_SYMBOL <= MAX_GROSS_EXPOSURE <= 4")

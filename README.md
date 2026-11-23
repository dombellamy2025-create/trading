# Alpaca Multi-Strategy Bot

Trades **SPY, QQQ, BTC/USD, GLD and USO** at the same time on Alpaca, with a separate strategy
and timeframe for each group. Built on the [`alpaca-trade-api`](https://github.com/alpacahq/alpaca-trade-api-python) SDK.
It uses **paper trading by default**.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # recommended: the SDK pins old dependencies
pip install -r requirements.txt
# edit .env: replace the placeholder keys with your Alpaca paper API key and secret
python -m bot.main --dry-run --once   # one cycle, logs signals only, places no orders
python -m bot.main                    # run continuously on paper (python bot/main.py also works)
python backtest.py                    # 6-month backtest on Alpaca data -> table + backtest_results.png
python -m pytest -q tests         # offline tests, no API keys needed
```

Live trading needs both `ALPACA_BASE_URL=https://api.alpaca.markets` **and** the `--live` flag.

API keys are read from `.env`, which is git-ignored. They are never hard-coded.

## How it runs

The bot runs in a continuous loop with two speeds:

| What | How often |
|---|---|
| Account, positions and orders sync; trade-log reconciliation; daily P&L | every `LOOP_SECONDS` (30s) |
| Hard stops, broker stop maintenance, exits and trailing stops (against the latest traded price) | every 30s |
| Strategy signals (new bars fetched, indicators recomputed) | once per bar of each strategy: **SPY/QQQ every 15 min, BTC/USD hourly, GLD/USO every 4 h**, `BAR_DELAY_SECONDS` after the bar closes |

**Market hours.** Equities trade only while Alpaca's market clock says the US market is open, so
weekends, holidays and early closes are handled automatically. The bot logs each open and close.
While the market is closed, equity positions stay protected by their broker-side stop orders.
**BTC/USD trades 24/7.**

**API disconnections.** Every request has a timeout; the SDK sets none, so without one a dropped
connection could hang the bot forever.
- Connection errors, timeouts, HTTP 429 and 5xx responses are retried with backoff.
- If Alpaca stays unreachable, the bot skips the cycle and backs off 5s → 10s → 20s … up to 5 min.
  It rebuilds the HTTP session every 3 failures and resumes automatically, logging the outage and the recovery.
- Broker-side stops keep protecting positions during an outage.
- Orders carry a `client_order_id`. If a submit request drops mid-flight, the bot checks whether
  Alpaca received it before retrying, so a flaky connection can't create a duplicate order.

## Output files

**`trades.csv`**: one row per completed trade (round trip):

| timestamp | instrument | direction | entry_price | exit_price | profit_loss | position_size | entry_time | exit_reason | exit_price_source |
|---|---|---|---|---|---|---|---|---|---|

- Trades are detected by reconciling Alpaca positions every cycle, so **every** exit is logged:
  strategy exits, broker hard stops, the daily loss limit, and manual closes in the Alpaca dashboard.
- The exit price is the volume-weighted average of the actual fills from Alpaca's fill activity.
  It falls back to the last seen price if no fills are visible (`exit_price_source`).
- `timestamp` is when the exit was recorded; `entry_time` is when the position was first seen.
- `profit_loss` is `(exit − entry) × size` for longs, reversed for shorts, before crypto fees.

**`daily_pnl.csv`**: one row per New York calendar day, updated every 5 minutes and when the day ends:

| date | start_equity | end_equity | pnl | pnl_pct | realized_pnl | trades | wins | losses |
|---|---|---|---|---|---|---|---|---|

- `pnl` is the change in total account equity, open positions included.
- `realized_pnl` sums the trades closed that day.

Operational logs go to `logs/bot.log`, rotated at 5 MB.

## Strategies

| Instrument | Strategy | Bars | Long | Short | Exit |
|---|---|---|---|---|---|
| SPY | Mean reversion | 15 min | close < SMA20 − **1.5σ** | close > SMA20 + **1.5σ** | price returns to SMA20 |
| QQQ | Mean reversion | 15 min | close < SMA20 − **1.8σ** | close > SMA20 + **1.8σ** | price returns to SMA20 |
| BTC/USD | Momentum breakout | 1 hour | close > prior 20-bar high, volume ≥ 1.5× 20-bar avg | *not allowed for crypto*: exits the long instead | breakdown below 20-bar low on volume, or **2× ATR** trailing stop |
| GLD, USO | Trend following | 4 hour | EMA50 crosses above EMA200 | EMA50 crosses below EMA200 | opposite cross, or **3× ATR** trailing stop |

**Enabled by default: BTC/USD and GLD only** (`ENABLED_SYMBOLS` in `.env`). Over the May–Nov 2026 backtest, SPY, QQQ and USO had negative Sharpe ratios at these settings (−3.98, −1.21, −0.99), and no SPY variant tested was positive. BTC/USD + GLD alone returned +6.9% (Sharpe 1.76, max drawdown −2.0%) against −12.1% for all five. To trade all five: `ENABLED_SYMBOLS=SPY,QQQ,BTC/USD,GLD,USO`.

How the rules are applied:
- **Entries** are judged on the last **closed** bar.
- **Exits** (return to the moving average, trailing stops) are checked against the latest traded
  price every 30s, so they react within a bar.
- **Trailing stops** follow the best price since entry: the highest price for longs, the lowest for shorts.
- **Reversals** (e.g. long GLD, then EMA50 crosses below EMA200): the bot closes the position, then opens
  the opposite one on the next cycle, about 30s later, once Alpaca confirms the close.
- **One entry per signal bar.** After a stop-out, it won't re-enter on the same bar.
  For the trend strategy that means waiting for the next crossover.
- Parameters are set in `config.py`.

## Risk controls

**ATR position sizing (all instruments).** Each position is sized so that a 1-ATR move against it
equals 1% of account equity:

    qty = (equity x 1%) / ATR(14)

A quiet instrument gets a larger position and a volatile one a smaller one, so the dollar risk per
ATR is the same for all five. ATR is the **14-day ATR** by default (`SIZING_ATR_TIMEFRAME=daily`).
Here's why: ATR on 15-minute or 1-hour bars is only about 0.1–1% of price, so "1 ATR = 1%" would mean
positions several times your account size, and a 1-ATR stop would sit inside a single bar's noise.
Set `SIZING_ATR_TIMEFRAME=strategy` to size on each strategy's own bars instead.

**Hard stop on every trade, 1% of equity.** A stop price is computed from the actual fill:
`entry ∓ (1% of equity at entry) / qty`. With uncapped ATR sizing it lands exactly 1 ATR from entry.
It's enforced two ways:
1. **Broker-side stop order** at Alpaca, placed as soon as the entry fills and re-checked every cycle.
   If the position size changes (e.g. a partial fill), the stop is replaced to match. It stays live
   even if the bot goes down. Equities use a GTC stop order. Crypto uses a stop-limit with a 1% buffer,
   because Alpaca has no plain stop orders for crypto.
2. **Software backstop:** if unrealized loss reaches 1% of equity, the bot closes the position.
   This covers a stop-limit that didn't fill, or a broker stop that failed to place.

The strategy trailing stops (2× ATR for BTC, 3× ATR for GLD/USO) still apply on top. They start out
wider than the 1-ATR hard stop, so they only take over once a trade has moved about 1–2 ATR in your favour.

**Correlation filter.** If SPY and QQQ are both long (held, or with a buy order pending), no new
BTC/USD long is opened. BTC is evaluated after the ETFs in each cycle, so a QQQ entry in the same cycle
counts too. Existing BTC positions aren't touched. Rules are in `CORRELATION_RULES` in `config.py`.

**Notional caps.** An ATR-sized order can't exceed what the account can fund:
- `MAX_ALLOC_PER_SYMBOL` (1× equity per instrument)
- `MAX_GROSS_EXPOSURE` (2× equity in total, about Reg-T overnight margin)
- available buying power. Crypto can only use cash.

When a cap binds, risk per ATR comes out *below* 1%, never above, and the log line says
`(capped ...)`. The hard stop is computed from the actual quantity, so the max loss is still 1%.

**Account level.**
- A 3% drop from the prior close (`MAX_DAILY_LOSS`) closes everything and halts trading for the day.
- Before shorting, the bot confirms the ETF is shortable and easy to borrow.
- It skips any symbol that has an open order, so it never places duplicates.
- State is saved in `bot_state.json`; logs go to `logs/bot.log`.

## Things to know

- **`alpaca-trade-api` is deprecated.** Alpaca has replaced it with `alpaca-py` and no longer
  updates it (the last release is 3.2.0). It still works, and it installs and runs on Python 3.13, but
  it pins old dependencies (`urllib3` 1.26, `websockets` 10, `msgpack` 1.0.3), so use a dedicated
  virtualenv. New Alpaca API features won't reach it. All SDK calls are in `bot/broker.py`, so moving
  back to `alpaca-py` later only means changing that one file.

- **Shorting needs a margin account.** Alpaca requires at least $2,000 equity for margin, and paper
  accounts have it by default. Set `ALLOW_SHORT=false` to run long-only.
- **Leverage:** with daily-ATR sizing, a typical uncapped position is roughly 0.3× (BTC) to about 1×
  (SPY) equity in notional value. With several positions open, total exposure can approach the
  2× gross cap. The 1% hard stops limit each loss, but **not overnight or weekend gaps**: a stop
  order fills at the next available price. Lower `MAX_GROSS_EXPOSURE` if that's more than you want.
- **Overnight gaps:** SPY/QQQ/GLD/USO are only checked during regular market hours, so positions held
  overnight are exposed to the gap at the open.
- **BTC volume:** the bot sees only the volume traded on Alpaca's own crypto venue, not the whole market,
  so the 1.5× volume filter is based on Alpaca's activity alone.
- **Free data feed:** with the free IEX feed, ETF volumes are thinner than the full SIP feed.
- **If the bot stops,** the 1% hard stops stay live at Alpaca, but trailing stops and strategy exits
  only run while the bot is running.
- **USO** tracks oil futures and loses value over time from rolling those contracts (roll decay).
- **Not proven to make money.** Backtest and paper trade before considering real money.

## Backtesting

```bash
python backtest.py              # 6 months of Alpaca data (cached in data/), all 5 instruments
python backtest.py --refresh    # re-download instead of using the cache
python backtest.py --months 12  # longer window
python backtest.py --synthetic  # offline pipeline demo on generated data (not real results)
```

**Data:** 6 months of SPY/QQQ 15-min, BTC/USD 1-hour and GLD/USO 4-hour bars, plus daily bars for the
sizing ATR. Extra history before the window is downloaded so the indicators are warmed up before the
first trade; the 200 EMA on 4-hour bars needs months of it.

**Simulated trading conditions.** The backtest uses the same strategy classes and risk manager as the
live bot:
- Signals use closed bars and fill at the **next bar's open**, so it never uses information it
  couldn't have had at the time.
- **0.05% slippage against you on every fill**, entries, exits and stops alike. **$0 commission.**
- Equities only signal and fill during regular hours (9:30–16:00 ET). A signal at the close waits
  for the next open, so overnight gaps are included. BTC/USD trades 24/7.
- Position sizing uses the daily ATR, with the per-instrument cap, the 2× gross exposure cap, and
  crypto limited to cash. Shorts are whole shares.
- The 1% hard stop triggers within the bar using its high and low. A gap through the stop fills at
  the open, which can lose more than 1%, and the report shows that.
- The daily loss limit applies. Reversals flip at the same fill.
- Positions still open at the end are closed so every trade is counted.

**Report:**
- **Per instrument:** each runs on its own $100k account. Reports total trades, win rate, average
  win and loss, profit factor, max drawdown, Sharpe ratio, total return and buy-and-hold.
- **Combined portfolio:** all five instruments on one $100k account with the **correlation filter
  active**, plus how many BTC entries it blocked.
- **Sharpe** uses daily returns with a 0% risk-free rate, annualized with √252 for equities and √365
  for BTC and the portfolio.

**Outputs:**
- `backtest_results.png`: portfolio equity and drawdown, plus one panel per instrument against buy-and-hold
- `backtest_trades.csv`: every simulated trade
- a summary table printed at the end

**Caveats:**
- Six months is a short sample. GLD/USO only trade on 50/200 EMA crossovers, which may happen 0–2
  times in six months, so their statistics mean little on their own.
- Fills at the next open are realistic for 15-minute bars. For 1-hour and 4-hour bars the live bot
  reacts to stops and exits within about 30 seconds, so the backtest is somewhat pessimistic on those exits.
- BTC volume is Alpaca's own venue volume, the same data the live bot sees.

## Morning briefing (Telegram)

```bash
python -m bot.briefing --dry-run   # print it
python -m bot.briefing             # send to Telegram
```

Under 200 words. It reports:
- open positions with entry price and unrealized P&L
- yesterday's total P&L (from `daily_pnl.csv`) and realized P&L per instrument (from `trades.csv`)
- 7-day win rate
- VIX (from Cboe; 20 or above counts as elevated)
- SPY/QQQ trending or ranging, using daily ADX(14): 25 or above trending, below 20 ranging
- BTC volume over the last 24 hours against its 30-day average (1.5× or more is unusual)

**Risk flags:**
- a position that has used 75% or more of its 1% stop
- the correlation filter currently blocking BTC longs
- drawdown more than 5% from peak equity

Needs `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`. To run it every weekday at 8:45 ET, add a cron entry:
`45 8 * * 1-5  cd /path/to/alpaca_bot && python -m bot.briefing`

## Evening report (Telegram)

```bash
python -m bot.evening_report --dry-run   # print it
python -m bot.evening_report             # send to Telegram (same TELEGRAM_* settings)
```

Under 200 words. It reports:
- today's trades by instrument
- P&L in dollars and as a percentage of the day's starting equity
- current equity
- the best and worst trade, with entry, exit, size, P&L and exit reason
- **stop check:** every hard-stop exit, and any loss above 1%, is checked against the 1%-of-equity limit:
  - `worked`: loss up to 1.25%, allowing for slippage
  - `overshot`: a gap went through the stop
  - `FAILED`: a loss above the limit with no stop exit, which means something is wrong

**Compared with the backtest:**
- `backtest.py` saves its expected win rate and daily-return mean and spread to `backtest_summary.json`.
- The report tests the last 30 days of live results against those numbers. More than 2 standard
  errors away counts as **DIVERGING**.
- With fewer than 20 trades and fewer than 10 days, it reports "too early to judge" rather than
  reading meaning into noise. It also flags any single day worse than −2σ.

To run it every weekday at 4:30 ET, add a cron entry:
`30 16 * * 1-5  cd /path/to/alpaca_bot && python -m bot.evening_report`

## Project structure

```
alpaca_bot/
├── .env                          # API keys + optional overrides (git-ignored; template in .env.example)
├── config.py                     # instruments → strategies, correlation rules, settings from .env
├── bot/
│   ├── main.py                   # entry point + main loop: scheduling, market hours, reconnects
│   ├── strategies/
│   │   ├── base.py               # shared Signal/Action types, EMA/ATR, trailing-stop helpers
│   │   ├── mean_reversion.py     # Strategy 1: SPY, QQQ, 15-min
│   │   ├── momentum_breakout.py  # Strategy 2: BTC/USD, 1-hour
│   │   └── trend_following.py    # Strategy 3: GLD, USO, 4-hour
│   ├── risk_manager.py           # ATR sizing, 1% hard stops, correlation filter, daily loss limit
│   ├── portfolio.py              # saved state, trades.csv, daily_pnl.csv
│   └── broker.py                 # alpaca-trade-api wrapper: orders, stops, data, retries, timeouts
├── backtest.py                   # 6-month backtest: Alpaca data, realistic fills, report + chart
├── tests/test_bot.py             # offline tests (no API keys or network needed)
└── requirements.txt
```

Generated at runtime in the project folder: `trades.csv`, `daily_pnl.csv`, `bot_state.json`, `logs/bot.log`.

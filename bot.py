"""
Trading 212 Trend-Following Bot
Strategy: EMA 20/50 crossover with RSI filter and ATR-based position sizing
Supports: Paper trading (demo) and Live trading
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from state import StopLossStore
from t212_client import (
    T212AuthError,
    T212Client,
    T212NotFoundError,
    T212PermissionError,
    T212RateLimitError,
)

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler(stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)),
    ],
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these values to customise the bot
# ════════════════════════════════════════════════════════════════════════════

class Config:
    # ── Account mode ─────────────────────────────────────────────────────
    # Set via T212_LIVE_MODE in .env. Real money only trades once you ALSO
    # set I_UNDERSTAND_LIVE_RISK=yes — see the safety guard in run_bot().
    LIVE_MODE: bool = os.environ.get("T212_LIVE_MODE", "false").strip().lower() in ("1", "true", "yes")

    # ── Instruments to trade (Trading 212 ticker format) ─────────────────
    # Find the correct ticker in the T212 app (tap an instrument → Details).
    # Validated against T212's instrument list at startup — a typo here
    # used to only surface as a failed order placement mid-run.
    WATCHLIST: list[str] = [
        "SPY_US_EQ",    # S&P 500 ETF
        "QQQ_US_EQ",    # Nasdaq 100 ETF
        "JPM_US_EQ",    # JPMorgan
        "NVDA_US_EQ",   # Nvidia
        "META_US_EQ",   # Meta
        "AAPL_US_EQ",   # Apple
        "AVGO_US_EQ",   # Broadcom
        "AMZN_US_EQ",   # Amazon
    ]

    # ── Risk management ───────────────────────────────────────────────────
    # Risk is a percentage of total account value (cash + invested), not just
    # free cash — otherwise the amount risked per trade shrinks every time
    # capital gets deployed into an open position, which understates risk
    # relative to your actual account size.
    RISK_PER_TRADE_PCT: float = 1.0       # 1% of account value per trade
    MAX_OPEN_POSITIONS: int = 3           # Never hold more than this many stocks
    MAX_POSITION_PCT: float = 10.0        # Single position ≤ 10% of account
    MIN_ORDER_VALUE: float = 1.0          # Skip trades smaller than this (currency units)

    # ── Strategy parameters ───────────────────────────────────────────────
    EMA_FAST: int = 20    # Fast EMA period
    EMA_SLOW: int = 50    # Slow EMA period
    RSI_PERIOD: int = 14  # RSI smoothing period
    RSI_OVERBOUGHT: float = 70.0   # Don't buy above this RSI
    RSI_OVERSOLD: float = 30.0     # Oversold threshold for exits
    ATR_PERIOD: int = 14           # ATR for stop-loss sizing
    ATR_STOP_MULTIPLIER: float = 2.0  # Stop = 2× ATR below entry

    # ── Timing ────────────────────────────────────────────────────────────
    # The strategy runs on DAILY bars, which only change once a day — a
    # 5-minute poll re-downloads 6 months of daily history per ticker for a
    # signal that's almost always identical to the last check. 15 minutes
    # keeps stop-loss monitoring reasonably responsive without hammering
    # both yfinance and the T212 API for no benefit.
    CHECK_INTERVAL_SECONDS: int = 900     # Check signals every 15 minutes
    NYSE_TIMEZONE = ZoneInfo("America/New_York")
    NYSE_OPEN_LOCAL = (9, 30)   # 9:30 AM local (handles DST automatically)
    NYSE_CLOSE_LOCAL = (16, 0)  # 4:00 PM local

    # ── State / instrument cache ────────────────────────────────────────
    STOP_LOSS_STATE_PATH: str = "stop_loss_state.json"
    INSTRUMENT_CACHE_PATH: str = "instrument_cache.json"
    INSTRUMENT_CACHE_TTL_HOURS: float = 24.0


# ════════════════════════════════════════════════════════════════════════════
#  MARKET DATA  (Yahoo Finance via yfinance — no API key needed)
# ════════════════════════════════════════════════════════════════════════════

def get_price_data(ticker: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from Yahoo Finance.
    T212 tickers like 'AAPL_US_EQ' map to Yahoo tickers like 'AAPL'.
    """
    try:
        import yfinance as yf

        # Convert T212 ticker → Yahoo ticker
        yahoo_ticker = ticker.split("_")[0]
        df = yf.download(yahoo_ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            log.warning("Not enough data for %s (%d bars)", ticker, len(df))
            return None
        # Newer yfinance returns a MultiIndex — flatten it to simple lowercase strings
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        log.error("Failed to fetch data for %s: %s", ticker, e)
        return None


# ════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = calc_ema(df["close"], cfg.EMA_FAST)
    df["ema_slow"] = calc_ema(df["close"], cfg.EMA_SLOW)
    df["rsi"] = calc_rsi(df["close"], cfg.RSI_PERIOD)
    df["atr"] = calc_atr(df, cfg.ATR_PERIOD)
    return df


# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL LOGIC
# ════════════════════════════════════════════════════════════════════════════

def get_signal(df: pd.DataFrame, cfg: Config) -> str:
    """
    Returns 'BUY', 'SELL', or 'HOLD'.

    BUY  — EMA fast crosses above EMA slow AND RSI is not overbought
    SELL — EMA fast crosses below EMA slow OR RSI is oversold (stop-loss handled separately)
    """
    if len(df) < cfg.EMA_SLOW + 5:
        return "HOLD"

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    fast_above_slow_now = latest["ema_fast"] > latest["ema_slow"]
    fast_above_slow_prev = prev["ema_fast"] > prev["ema_slow"]
    rsi = latest["rsi"]

    # Golden cross: fast EMA crosses above slow EMA
    bullish_cross = fast_above_slow_now and not fast_above_slow_prev
    # Death cross: fast EMA crosses below slow EMA
    bearish_cross = not fast_above_slow_now and fast_above_slow_prev

    if bullish_cross and rsi < cfg.RSI_OVERBOUGHT:
        return "BUY"
    if bearish_cross or rsi < cfg.RSI_OVERSOLD:
        return "SELL"
    return "HOLD"


# ════════════════════════════════════════════════════════════════════════════
#  POSITION SIZING
# ════════════════════════════════════════════════════════════════════════════

def calc_quantity(
    account_total: float,
    price: float,
    atr: float,
    cfg: Config,
) -> float:
    """
    Risk-based position sizing:
      risk_amount = account_total × RISK_PER_TRADE_PCT
      stop_distance = ATR × ATR_STOP_MULTIPLIER
      quantity = risk_amount / stop_distance

    Also enforces MAX_POSITION_PCT cap. Returns 0.0 if the properly-sized
    position would round to nothing — the caller should skip the trade
    rather than forcing a token minimum quantity that ignores the risk
    calculation entirely.
    """
    if price <= 0 or atr <= 0 or account_total <= 0:
        return 0.0

    risk_amount = account_total * (cfg.RISK_PER_TRADE_PCT / 100)
    stop_distance = atr * cfg.ATR_STOP_MULTIPLIER
    qty_by_risk = risk_amount / stop_distance

    # Cap: don't let a single position exceed MAX_POSITION_PCT of account
    max_value = account_total * (cfg.MAX_POSITION_PCT / 100)
    qty_by_cap = max_value / price

    quantity = min(qty_by_risk, qty_by_cap)
    return round(quantity, 4)


# ════════════════════════════════════════════════════════════════════════════
#  MARKET HOURS CHECK
# ════════════════════════════════════════════════════════════════════════════

def is_market_open(cfg: Config) -> bool:
    """
    Check if NYSE is currently open, using America/New_York local time so
    DST transitions (which shift the UTC open/close by an hour twice a
    year) are handled automatically instead of approximated.
    """
    now_local = datetime.now(cfg.NYSE_TIMEZONE)
    if now_local.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    now_minutes = now_local.hour * 60 + now_local.minute
    open_minutes = cfg.NYSE_OPEN_LOCAL[0] * 60 + cfg.NYSE_OPEN_LOCAL[1]
    close_minutes = cfg.NYSE_CLOSE_LOCAL[0] * 60 + cfg.NYSE_CLOSE_LOCAL[1]
    return open_minutes <= now_minutes < close_minutes


# ════════════════════════════════════════════════════════════════════════════
#  WATCHLIST VALIDATION  (catches ticker typos at startup, not mid-run)
# ════════════════════════════════════════════════════════════════════════════

def _load_instrument_cache(cfg: Config) -> Optional[set[str]]:
    if not os.path.exists(cfg.INSTRUMENT_CACHE_PATH):
        return None
    try:
        with open(cfg.INSTRUMENT_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        age_hours = (time.time() - cached["fetched_at"]) / 3600
        if age_hours > cfg.INSTRUMENT_CACHE_TTL_HOURS:
            return None
        return set(cached["tickers"])
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def _save_instrument_cache(cfg: Config, tickers: set[str]):
    with open(cfg.INSTRUMENT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": time.time(), "tickers": sorted(tickers)}, f)


def validate_watchlist(client: T212Client, cfg: Config) -> list[str]:
    """
    Checks every WATCHLIST ticker against T212's instrument list, so a typo
    is caught once at startup with a clear message instead of failing a
    market order deep into a run. The instrument endpoint returns a large,
    heavily rate-limited payload, so the result is cached to disk.
    """
    tickers = _load_instrument_cache(cfg)
    if tickers is None:
        log.info("Fetching instrument metadata to validate watchlist (cached for %.0fh)...", cfg.INSTRUMENT_CACHE_TTL_HOURS)
        try:
            instruments = client.get_instruments()
            tickers = {i.get("ticker") for i in instruments if i.get("ticker")}
            _save_instrument_cache(cfg, tickers)
        except Exception as e:
            log.warning("Could not fetch instrument list to validate watchlist (%s) — skipping validation", e)
            return list(cfg.WATCHLIST)

    valid = [t for t in cfg.WATCHLIST if t in tickers]
    invalid = [t for t in cfg.WATCHLIST if t not in tickers]
    for t in invalid:
        log.error("WATCHLIST ticker %r not found in T212 instrument list — skipping it. "
                   "Check the exact ticker in the T212 app (tap the instrument → Details).", t)
    if not valid:
        raise ValueError("No valid tickers left in WATCHLIST after validation — check your ticker symbols.")
    return valid


# ════════════════════════════════════════════════════════════════════════════
#  MAIN BOT LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_bot():
    cfg = Config()

    # Load credentials from .env
    api_key = os.environ.get("T212_API_KEY")
    api_secret = os.environ.get("T212_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError(
            "Missing T212_API_KEY or T212_API_SECRET. "
            "Copy .env.example to .env and fill in your credentials."
        )

    # ── Live-trading safety guard ────────────────────────────────────────
    # A config flag alone is too easy to leave on by accident. Real money
    # only trades once this second, explicit env var is also set.
    if cfg.LIVE_MODE and os.environ.get("I_UNDERSTAND_LIVE_RISK", "").strip().lower() != "yes":
        raise SystemExit(
            "T212_LIVE_MODE=true but I_UNDERSTAND_LIVE_RISK is not set to 'yes' in .env.\n"
            "This bot is about to place real orders with real money. Set "
            "I_UNDERSTAND_LIVE_RISK=yes in .env only once you've paper-traded successfully "
            "and are sure you want that."
        )

    client = T212Client(api_key, api_secret, live=cfg.LIVE_MODE)
    stop_loss_store = StopLossStore(cfg.STOP_LOSS_STATE_PATH)

    # ── Startup connectivity check ───────────────────────────────────────
    # Fail fast on bad credentials instead of silently sleeping through
    # market-closed hours and only discovering the problem later.
    log.info("Testing API connection...")
    try:
        cash = client.get_cash()
        log.info("Connected OK — free=%.2f total=%.2f", float(cash.get("free", 0)), float(cash.get("total", 0)))
    except (T212AuthError, T212PermissionError, T212NotFoundError) as e:
        log.error("Startup connectivity check failed: %s", e)
        raise
    except ConnectionError as e:
        log.error(
            "Startup connectivity check failed: %s\n"
            "If you're running this inside a sandboxed/cloud environment with restricted "
            "network egress, that will block trading212.com outright — run the bot from a "
            "machine with normal internet access instead.",
            e,
        )
        raise

    watchlist = validate_watchlist(client, cfg)

    log.info("=" * 60)
    log.info("Bot started  |  mode=%s  |  watchlist=%s", "LIVE" if cfg.LIVE_MODE else "PAPER", watchlist)
    log.info("Strategy: EMA%d/EMA%d crossover + RSI%d filter", cfg.EMA_FAST, cfg.EMA_SLOW, cfg.RSI_PERIOD)
    log.info("Risk: %.1f%% per trade  |  max %d positions", cfg.RISK_PER_TRADE_PCT, cfg.MAX_OPEN_POSITIONS)
    log.info("=" * 60)

    while True:
        try:
            cycle_start = datetime.now(timezone.utc)
            log.info("── Cycle start: %s ──", cycle_start.strftime("%Y-%m-%d %H:%M:%S UTC"))

            if not is_market_open(cfg):
                log.info("Market closed — sleeping 60 s")
                time.sleep(60)
                continue

            # ── Account state ─────────────────────────────────────────────
            cash = client.get_cash()
            free_cash = float(cash.get("free", 0))
            total_value = float(cash.get("total", free_cash))
            positions = client.get_positions()
            open_tickers = {p["ticker"] for p in positions}
            n_open = len(open_tickers)

            stop_loss_store.reconcile(open_tickers)

            log.info(
                "Account: free=%.2f  total=%.2f  open positions=%d",
                free_cash, total_value, n_open,
            )

            # ── Scan watchlist ────────────────────────────────────────────
            for ticker in watchlist:
                log.info("Analysing %s …", ticker)

                df = get_price_data(ticker)
                if df is None:
                    continue

                df = add_indicators(df, cfg)
                signal = get_signal(df, cfg)
                latest = df.iloc[-1]
                price = float(latest["close"])
                atr = float(latest["atr"])

                log.info(
                    "%s | price=%.4f EMA%d=%.4f EMA%d=%.4f RSI=%.1f ATR=%.4f → %s",
                    ticker, price,
                    cfg.EMA_FAST, latest["ema_fast"],
                    cfg.EMA_SLOW, latest["ema_slow"],
                    latest["rsi"], atr, signal,
                )

                in_position = ticker in open_tickers

                # ── Stop-loss check ───────────────────────────────────────
                stop_price = stop_loss_store.get(ticker) if in_position else None
                if in_position and stop_price is not None and price <= stop_price:
                    log.warning("STOP-LOSS triggered for %s: price=%.4f stop=%.4f", ticker, price, stop_price)
                    position = client.get_position(ticker, positions=positions)
                    if position:
                        qty = abs(float(position.get("quantity", 0)))
                        if qty > 0:
                            client.place_market_order(ticker, -qty)
                            open_tickers.discard(ticker)
                            stop_loss_store.clear(ticker)
                            n_open -= 1
                            log.info("STOPPED OUT %s qty=%.4f", ticker, qty)
                    continue

                # ── Entry ─────────────────────────────────────────────────
                if signal == "BUY" and not in_position and n_open < cfg.MAX_OPEN_POSITIONS:
                    qty = calc_quantity(total_value, price, atr, cfg)
                    cost = qty * price
                    if qty <= 0 or cost < cfg.MIN_ORDER_VALUE:
                        log.info("Signal-sized order for %s too small to place (qty=%.4f, cost=%.2f), skipping", ticker, qty, cost)
                        continue
                    if cost > free_cash:
                        log.warning("Insufficient cash for %s (need %.2f, have %.2f)", ticker, cost, free_cash)
                        continue
                    client.place_market_order(ticker, qty)
                    stop_loss_store.set(ticker, price - (atr * cfg.ATR_STOP_MULTIPLIER))
                    open_tickers.add(ticker)
                    n_open += 1
                    free_cash -= cost
                    log.info("BOUGHT %s qty=%.4f @ ~%.4f  cost=%.2f", ticker, qty, price, cost)

                # ── Exit ──────────────────────────────────────────────────
                elif signal == "SELL" and in_position:
                    position = client.get_position(ticker, positions=positions)
                    if position:
                        qty = abs(float(position.get("quantity", 0)))
                        if qty > 0:
                            client.place_market_order(ticker, -qty)
                            open_tickers.discard(ticker)
                            stop_loss_store.clear(ticker)
                            n_open -= 1
                            log.info("SOLD %s qty=%.4f @ ~%.4f", ticker, qty, price)

                time.sleep(1)  # Polite rate-limiting between instrument calls

        except KeyboardInterrupt:
            log.info("Bot stopped by user (Ctrl+C)")
            break
        except T212RateLimitError as e:
            wait = e.retry_after if e.retry_after else 60
            log.warning("Rate limited: %s — waiting %.0f s", e, wait)
            time.sleep(wait)
        except (T212AuthError, T212PermissionError) as e:
            log.error("Credential/permission error — this will not fix itself on retry: %s", e)
            raise
        except T212NotFoundError as e:
            log.error("API endpoint not found — T212 may have changed their API: %s", e)
            raise
        except (ConnectionError, requests.exceptions.HTTPError) as e:
            log.error("API connection error: %s — retrying in 60 s", e)
            time.sleep(60)
        except Exception as e:
            log.exception("Unexpected error: %s — retrying in 30 s", e)
            time.sleep(30)

        log.info("Sleeping %d s until next cycle …\n", cfg.CHECK_INTERVAL_SECONDS)
        time.sleep(cfg.CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_bot()

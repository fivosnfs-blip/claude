# Trading 212 Trend-Following Bot

EMA(20/50) crossover strategy with an RSI filter and ATR-based position
sizing/stop-loss, driven against the Trading 212 public API. Built to
paper-trade on T212's demo environment before ever touching a live account.

## Files

- `bot.py` — main loop: strategy, sizing, order placement.
- `t212_client.py` — Trading 212 REST client (auth, error handling).
- `state.py` — persists stop-loss levels to disk so a restart can't lose them.
- `test_connection.py` — fast standalone script to verify API credentials
  and connectivity before running the full bot.
- `.env.example` — copy to `.env` and fill in your credentials.

## Setup

1. **Run this on a machine with normal internet access** — your own
   computer, a Raspberry Pi, a small VPS, etc. It will **not** work from a
   sandboxed cloud coding environment with restricted network egress;
   `trading212.com` gets blocked outright there.

2. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. In the T212 app: **switch to your Practice/Demo account**, then go to
   Settings → API (Beta) → generate an API key. You'll get a key and a
   secret — a key generated under demo will not work against live, and
   vice versa, so double check which account you're in first.

4. Copy `.env.example` to `.env` and fill in `T212_API_KEY` /
   `T212_API_SECRET`. Leave `T212_LIVE_MODE=false`.

5. **Verify the connection before anything else:**
   ```bash
   python test_connection.py --demo
   ```
   This should print your paper account's cash balance and open positions.
   If it fails, the error message tells you which of 401 (bad
   credentials)/403 (missing scope)/404 (endpoint changed)/network issue
   you're hitting — fix that before running the bot itself.

## Paper trading plan (do this before real money)

1. Run `test_connection.py --demo` and confirm it passes cleanly.
2. Start the bot: `python bot.py` (with `T212_LIVE_MODE=false`).
3. Let it run for **several consecutive trading days**, watching `bot.log`.
   Check specifically for:
   - Orders actually filling on the demo account (cross-check against the
     T212 app's Portfolio/Orders tabs, not just the log).
   - Stop-losses triggering correctly and staying set across a bot restart
     (deliberately restart the bot once with a position open, and confirm
     `stop_loss_state.json` still has that ticker's stop level after restart).
   - No repeated errors in the log (rate limiting, unexpected exceptions).
   - Position sizing looks sane relative to your account size and
     `RISK_PER_TRADE_PCT`/`MAX_POSITION_PCT` settings.
4. Only after that, in the T212 app switch to your **live** account and
   generate a **new** API key+secret there (the demo one won't work).
5. Update `.env`: new `T212_API_KEY`/`T212_API_SECRET`, `T212_LIVE_MODE=true`,
   and `I_UNDERSTAND_LIVE_RISK=yes`. The bot deliberately refuses to start
   in live mode without that last variable set — it's a second, explicit
   confirmation you meant to do this.
6. Start small. Consider lowering `RISK_PER_TRADE_PCT` and
   `MAX_OPEN_POSITIONS` for the first live days regardless of how paper
   trading went — paper fills are instant and slippage-free in a way real
   fills aren't.

## Notes / known limitations

- The strategy runs on **daily** bars from Yahoo Finance (`yfinance`), so
  entries/exits only really change once a day even though the bot polls
  every 15 minutes by default (`CHECK_INTERVAL_SECONDS`) — that cadence
  exists mainly to keep stop-loss checks reasonably current.
- `t212_client.py` targets Trading 212's public API as best documented; it
  is officially in beta and endpoint paths have changed before. If you see
  a `T212NotFoundError` (404) in the logs, check
  https://t212public-api-docs.redoc.ly for the current path and update the
  `T212_PATHS` dict at the top of `t212_client.py`.
- This is a simple trend-following strategy with no backtesting harness
  included — past behavior of EMA/RSI crossovers on your chosen watchlist
  is not validated here. Consider backtesting before relying on it.

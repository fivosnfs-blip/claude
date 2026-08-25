"""
Persistent stop-loss state.

Stop-loss levels used to live only in an in-memory dict, which meant a
restart (crash, redeploy, server reboot) silently dropped protection on any
position still open. This stores them in a small JSON file next to the bot
and reconciles against the broker's actual open positions on load, so a
restart can never leave a position "invisibly" unprotected or "phantom"
protected.
"""

import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_STATE_PATH = "stop_loss_state.json"


class StopLossStore:
    def __init__(self, path: str = DEFAULT_STATE_PATH):
        self.path = path
        self._levels: dict[str, float] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._levels = json.load(f)
            log.info("Loaded stop-loss state for %d ticker(s) from %s", len(self._levels), self.path)
        except (json.JSONDecodeError, OSError) as e:
            log.error("Could not read stop-loss state file %s (%s) — starting empty", self.path, e)
            self._levels = {}

    def _save(self):
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._levels, f, indent=2)
        os.replace(tmp_path, self.path)

    def reconcile(self, open_tickers: set[str]):
        """Drop stop-loss entries for tickers we no longer hold, and warn about
        open positions that have no recorded stop (e.g. bought outside the bot)."""
        stale = [t for t in self._levels if t not in open_tickers]
        for t in stale:
            log.info("Dropping stale stop-loss entry for %s (position no longer open)", t)
            del self._levels[t]
        if stale:
            self._save()

        unprotected = open_tickers - self._levels.keys()
        for t in unprotected:
            log.warning(
                "Position in %s has no recorded stop-loss (opened outside this bot, or state was "
                "lost) — it will not be auto-stopped-out until the strategy generates a fresh signal.",
                t,
            )

    def set(self, ticker: str, stop_price: float):
        self._levels[ticker] = stop_price
        self._save()

    def get(self, ticker: str) -> Optional[float]:
        return self._levels.get(ticker)

    def clear(self, ticker: str):
        if ticker in self._levels:
            del self._levels[ticker]
            self._save()

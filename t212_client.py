"""
Trading 212 REST API client.

Auth: HTTP Basic, with your API key as the username and your API secret as
the password (base64-encoded "key:secret" in the Authorization header).
Generate a key+secret pair in the app under Settings -> API (Beta) -- make
sure you generate it while your account is switched to the mode you want
to hit (paper/demo vs. live), since a key created in one mode will not
authenticate against the other.

NOTE ON ENDPOINT PATHS: Trading 212's public API is officially in beta and
has changed shape before. This client targets the endpoints below as best
verified; if you see a 404 on a specific call, T212 has likely renamed that
route -- check https://t212public-api-docs.redoc.ly for the current path
and update T212_PATHS below rather than hunting through the whole file.
"""

import json
import logging
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

# Centralised so a path rename only needs one edit.
T212_PATHS = {
    "account_cash": "/equity/account/cash",
    "account_info": "/equity/account/info",
    "portfolio": "/equity/portfolio",
    "order_market": "/equity/orders/market",
    "instruments": "/equity/metadata/instruments",
}


class T212AuthError(Exception):
    """401 -- credentials are wrong, or a key generated for the other mode (demo/live)."""


class T212PermissionError(Exception):
    """403 -- credentials are valid but the key lacks a required scope."""


class T212NotFoundError(Exception):
    """404 -- endpoint path doesn't exist. T212's API is beta and paths do change."""


class T212RateLimitError(Exception):
    """429 -- too many requests. Caller should back off (retry_after seconds if known)."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class T212Client:
    """Minimal Trading 212 REST API client (HTTP Basic Auth)."""

    DEMO_BASE = "https://demo.trading212.com/api/v0"
    LIVE_BASE = "https://live.trading212.com/api/v0"

    def __init__(self, api_key: str, api_secret: str, live: bool = False, timeout: int = 15):
        self.base = self.LIVE_BASE if live else self.DEMO_BASE
        self.live = live
        self.timeout = timeout
        self._session = requests.Session()
        self._session.auth = (api_key, api_secret)
        self._session.headers.update({"Content-Type": "application/json"})
        log.info("T212 client initialised — %s mode (%s)", "LIVE" if live else "PAPER", self.base)

    def _request(self, method: str, path: str, params: dict = None, body: dict = None) -> dict:
        url = f"{self.base}{path}"
        try:
            r = self._session.request(method, url, params=params, json=body, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Network error calling {method} {path}: {e}") from e

        if r.status_code == 401:
            raise T212AuthError(
                f"401 Unauthorized on {method} {path}. Check T212_API_KEY/T212_API_SECRET, "
                f"and make sure the key was generated for {'LIVE' if self.live else 'DEMO/paper'} mode "
                f"(a demo key will not work against live, and vice versa)."
            )
        if r.status_code == 403:
            raise T212PermissionError(
                f"403 Forbidden on {method} {path}. Your API key is valid but is missing a "
                f"required scope for this endpoint — check the key's permissions in the T212 app."
            )
        if r.status_code == 404:
            raise T212NotFoundError(
                f"404 Not Found on {method} {path}. This endpoint path may have changed — "
                f"check https://t212public-api-docs.redoc.ly for the current route."
            )
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            raise T212RateLimitError(
                f"429 Rate limited on {method} {path}.",
                retry_after=float(retry_after) if retry_after else None,
            )
        r.raise_for_status()
        if not r.content:
            return {}
        return r.json()

    def _get(self, path: str, params: dict = None) -> dict:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, body=body)

    # ── Account ───────────────────────────────────────────────────────────
    def get_cash(self) -> dict:
        """Returns the raw cash dict: free, total, invested, ppl, blocked, result."""
        return self._get(T212_PATHS["account_cash"])

    def get_account_info(self) -> dict:
        return self._get(T212_PATHS["account_info"])

    # ── Positions ─────────────────────────────────────────────────────────
    def get_positions(self) -> list[dict]:
        return self._get(T212_PATHS["portfolio"])

    def get_position(self, ticker: str, positions: Optional[list[dict]] = None) -> Optional[dict]:
        """Pass an already-fetched `positions` list to avoid re-fetching the whole portfolio."""
        if positions is None:
            positions = self.get_positions()
        for p in positions:
            if p.get("ticker") == ticker:
                return p
        return None

    # ── Orders ────────────────────────────────────────────────────────────
    def place_market_order(self, ticker: str, quantity: float) -> dict:
        """Positive quantity = BUY, negative quantity = SELL."""
        body = {"ticker": ticker, "quantity": quantity}
        log.info(
            "Placing market order: %s qty=%.4f (%s)",
            ticker,
            quantity,
            "BUY" if quantity > 0 else "SELL",
        )
        result = self._post(T212_PATHS["order_market"], body)
        log.info("Order result: %s", json.dumps(result))
        return result

    # ── Instruments ───────────────────────────────────────────────────────
    def get_instruments(self) -> list[dict]:
        """Full instrument list. Large payload and tightly rate-limited by T212 —
        callers should cache this to disk rather than calling it on every cycle."""
        return self._get(T212_PATHS["instruments"])

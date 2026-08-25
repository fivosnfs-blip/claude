"""
Standalone connectivity check for the Trading 212 API.

Run this BEFORE running bot.py, and any time you're not sure the bot's API
setup is correct. It makes just three quick calls and tells you clearly
what passed and what didn't, instead of making you wait through bot.py's
5-minute cycle to find out your credentials are wrong.

Usage:
    python test_connection.py         # tests whichever mode is in .env (T212_LIVE_MODE)
    python test_connection.py --live  # force live mode
    python test_connection.py --demo  # force demo/paper mode
"""

import argparse
import sys

from dotenv import load_dotenv
import os

from t212_client import (
    T212Client,
    T212AuthError,
    T212PermissionError,
    T212NotFoundError,
    T212RateLimitError,
)

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Test Trading 212 API connectivity.")
    parser.add_argument("--live", action="store_true", help="Force live mode.")
    parser.add_argument("--demo", action="store_true", help="Force demo/paper mode.")
    args = parser.parse_args()

    if args.live and args.demo:
        print("Pass only one of --live / --demo.")
        sys.exit(2)

    if args.live:
        live = True
    elif args.demo:
        live = False
    else:
        live = os.environ.get("T212_LIVE_MODE", "false").strip().lower() in ("1", "true", "yes")

    api_key = os.environ.get("T212_API_KEY")
    api_secret = os.environ.get("T212_API_SECRET")

    print(f"Mode: {'LIVE (real money)' if live else 'DEMO (paper trading)'}")

    if not api_key or not api_secret:
        print("FAIL: T212_API_KEY / T212_API_SECRET not set. Copy .env.example to .env and fill them in.")
        sys.exit(1)

    client = T212Client(api_key, api_secret, live=live)

    ok = True

    print("\n[1/3] Fetching account cash...")
    try:
        cash = client.get_cash()
        print(f"  OK — free={cash.get('free')} total={cash.get('total')} invested={cash.get('invested')}")
    except Exception as e:
        ok = False
        print(f"  FAIL — {type(e).__name__}: {e}")

    print("\n[2/3] Fetching open positions...")
    try:
        positions = client.get_positions()
        print(f"  OK — {len(positions)} open position(s)")
        for p in positions[:10]:
            print(f"    {p.get('ticker')}: qty={p.get('quantity')}")
    except Exception as e:
        ok = False
        print(f"  FAIL — {type(e).__name__}: {e}")

    print("\n[3/3] Fetching instrument metadata for a known ticker (AAPL_US_EQ)...")
    try:
        instruments = client.get_instruments()
        found = any(i.get("ticker") == "AAPL_US_EQ" for i in instruments)
        print(f"  OK — {len(instruments)} instruments returned, AAPL_US_EQ {'found' if found else 'NOT found'}")
    except Exception as e:
        ok = False
        print(f"  FAIL — {type(e).__name__}: {e}")

    print()
    if ok:
        print("All checks passed. Your API setup looks good.")
    else:
        print("Some checks failed — see messages above.")
        print("Common causes:")
        print("  401 -> wrong key/secret, or a key generated for the other mode (demo vs live)")
        print("  403 -> key is missing a required permission scope in the T212 app")
        print("  404 -> T212 changed this endpoint path (it's a beta API) — check")
        print("         https://t212public-api-docs.redoc.ly and update t212_client.py T212_PATHS")
        print("  ConnectionError / timeout -> network/firewall issue reaching trading212.com from")
        print("         wherever this script is running (this will NOT work from a sandboxed cloud")
        print("         coding session with restricted egress — run it from your own machine/VPS)")
        sys.exit(1)


if __name__ == "__main__":
    main()

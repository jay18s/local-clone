#!/usr/bin/env python3
"""
ROX Edge Engine — Fyers Daily Login Helper
==========================================
Run this script ONCE every morning before market open to generate
a fresh access token.  The token is saved to your .env file
automatically so python main.py plan picks it up without extra steps.

Usage:
    python fyers_login.py

Requirements:
    pip install fyers-apiv3
"""

import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ── locate .env ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
ENV_PATH = PROJECT_ROOT / ".env"

def _load_env(path: Path) -> dict:
    """Parse .env file into a dict."""
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _save_env_key(path: Path, key: str, value: str):
    """Update or insert a single key=value in the .env file."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def main():
    print("=" * 60)
    print("  ROX Edge Engine — Fyers Daily Login")
    print("=" * 60)

    # ── load existing .env ─────────────────────────────────────────────────
    env = _load_env(ENV_PATH)

    client_id  = env.get("FYERS_APP_ID", "").strip()
    secret_key = env.get("FYERS_SECRET_KEY", "").strip()

    if not client_id:
        client_id = input("\nEnter your Fyers App ID (e.g. ABC123-100): ").strip()
    if not secret_key:
        secret_key = input("Enter your Fyers Secret Key: ").strip()

    if not client_id or not secret_key:
        print("\nERROR: App ID and Secret Key are required.")
        print("Get them from https://myapi.fyers.in → My Apps")
        sys.exit(1)

    try:
        from fyers_apiv3 import fyersModel
    except ImportError:
        print("\nERROR: fyers-apiv3 not installed.")
        print("Run:  pip install fyers-apiv3")
        sys.exit(1)

    # ── Step 1: Generate auth URL ──────────────────────────────────────────
    redirect_uri ="http://127.0.0.1:5000/"

    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )

    auth_url = session.generate_authcode()
    print(f"\n[1] Opening Fyers login page in your browser...")
    print(f"    URL: {auth_url}\n")

    try:
        webbrowser.open(auth_url)
    except Exception:
        print("    (Could not open browser automatically — copy the URL above manually)")

    # ── Step 2: Capture auth code from redirect ────────────────────────────
    print("[2] After logging in, Fyers will redirect you to a URL that looks like:")
    print("    https://trade.fyers.in/api-login/redirect-uri/index.html?auth_code=XXXX&state=None\n")
    redirect_url = input("    Paste the full redirect URL here:\n    > ").strip()

    try:
        parsed   = urlparse(redirect_url)
        params   = parse_qs(parsed.query)
        auth_code = params.get("auth_code", params.get("code", [None]))[0]
        if not auth_code:
            raise ValueError("auth_code not found in URL")
    except Exception as e:
        print(f"\nERROR parsing redirect URL: {e}")
        auth_code = input("    Or enter the auth_code directly: ").strip()

    # ── Step 3: Exchange code for access token ─────────────────────────────
    print("\n[3] Generating access token...")
    try:
        session.set_token(auth_code)
        response = session.generate_token()

        if response.get("s") != "ok":
            print(f"\nERROR from Fyers: {response}")
            sys.exit(1)

        access_token = response["access_token"]
        print(f"    Access token generated successfully!")

    except Exception as e:
        print(f"\nERROR generating token: {e}")
        sys.exit(1)

    # ── Step 4: Save to .env ──────────────────────────────────────────────
    _save_env_key(ENV_PATH, "FYERS_APP_ID",       client_id)
    _save_env_key(ENV_PATH, "FYERS_SECRET_KEY",   secret_key)
    _save_env_key(ENV_PATH, "FYERS_ACCESS_TOKEN", access_token)
    _save_env_key(ENV_PATH, "FYERS_ENABLED",      "true")

    print(f"\n[4] Saved to .env:")
    print(f"    FYERS_APP_ID      = {client_id}")
    print(f"    FYERS_ACCESS_TOKEN = {access_token[:20]}...  (truncated)")
    print(f"    FYERS_ENABLED     = true")

    # ── Step 5: Quick verify ───────────────────────────────────────────────
    print("\n[5] Verifying connection...")
    try:
        fyers = fyersModel.FyersModel(
            client_id=client_id,
            token=access_token,
            log_path=str(PROJECT_ROOT / "logs"),
        )
        profile = fyers.get_profile()
        if profile.get("s") == "ok":
            name = profile.get("data", {}).get("name", "Unknown")
            print(f"    Connected as: {name}")
        else:
            print(f"    Warning: {profile}")
    except Exception as e:
        print(f"    Verify skipped: {e}")

    print("\n" + "=" * 60)
    print("  SUCCESS! You can now run:  python main.py plan")
    print("  NOTE: Access token expires daily — run this script each morning")
    print("=" * 60)


if __name__ == "__main__":
    main()

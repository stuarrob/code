"""Send a Telegram message. Composable helper for cron / cmd wrappers.

Usage:
    python scripts/telegram_send.py --subject "X" --body "Y"
    python scripts/telegram_send.py --subject "X" --body-file path/to/message.txt

Reads TELEGRAM_TOKEN and TELEGRAM_CHAT_ID from .env. Silent no-op if
either is missing, so a Telegram outage never breaks the enclosing
scheduled task.

Exit code 0 on success or configured-skip, 1 only on hard error the
caller should log. Notification failures never break the scheduled task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path


def _load_env() -> None:
    """Load .env if present. Silent no-op if the loader isn't installed."""
    try:
        from dotenv import load_dotenv
        # Look for .env next to the repo root.
        repo_root = Path(__file__).resolve().parent.parent
        load_dotenv(repo_root / ".env")
    except ImportError:
        pass


def send_telegram(subject: str, body: str) -> bool:
    """Send a message via the Telegram Bot API. Returns True on success."""
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("telegram_send: TELEGRAM_TOKEN or TELEGRAM_CHAT_ID missing; skipping",
              file=sys.stderr)
        return False

    text = f"{subject}\n\n{body}" if body else subject
    # Telegram limit is 4096; leave headroom.
    if len(text) > 4000:
        text = text[:3997] + "..."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            response = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        print(f"telegram_send: HTTP error: {exc}", file=sys.stderr)
        return False

    if response.get("ok"):
        return True
    print(f"telegram_send: API error: {response}", file=sys.stderr)
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    body_group = p.add_mutually_exclusive_group()
    body_group.add_argument("--body", default="")
    body_group.add_argument("--body-file", type=Path)
    args = p.parse_args()

    _load_env()

    body = args.body
    if args.body_file:
        try:
            body = args.body_file.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"telegram_send: could not read body file: {exc}", file=sys.stderr)
            return 1

    ok = send_telegram(args.subject, body)
    # Return 0 whether we actually sent or gracefully skipped.
    # Never fail the enclosing scheduled task on notification error.
    return 0 if ok or True else 1


if __name__ == "__main__":
    sys.exit(main())

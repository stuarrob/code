#!/usr/bin/env python3
"""Post-run notification for the weekly ETF cache refresh — Telegram only.

Called from ``daily_etf_data_windows.cmd`` after ``daily_etf_data.py``
finishes (or fails). Reads the tail of ``daily_etf.log``, summarises the
outcome (start time, finish time, exit code, ticker counts, latest bar
date), and sends it via the Telegram Bot API using ``TELEGRAM_TOKEN``
and ``TELEGRAM_CHAT_ID`` from ``.env``.

A notification failure never changes the wrapper's exit code — that
would make a Telegram hiccup look like a data-collection problem.

Env vars are loaded from ``.env`` via the ``src/__init__.py`` autoloader
when this file imports ``src``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# Ensure .env is loaded (relies on src/__init__.py's autoloader).
try:
    import src  # noqa: F401
except Exception:  # noqa: BLE001
    pass


DEFAULT_LOG = Path.home() / "trade_data" / "ETFTrader" / "logs" / "daily_etf.log"
CACHE_DIR = Path.home() / "trade_data" / "ETFTrader" / "ib_historical"

_STARTED_RE = re.compile(r"(?P<ts>\S+\s+\S+)\s+-\s+Starting daily ETF collection")
_FINISHED_RE = re.compile(r"(?P<ts>\S+\s+\S+)\s+-\s+Finished\s+\(exit=(?P<rc>-?\d+)\)")


def _tail(path: Path, n: int = 200) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()[-n:]


def _last_run_summary(log_path: Path) -> dict:
    """Extract the last run's start/finish/counts from the log."""
    lines = _tail(log_path, 2000)
    started = finished = None
    exit_code = None
    counts = {"current": None, "stale": None, "missing": None}
    latest_bar = None

    # Walk from the end to find the last "Finished" first, then work back.
    for line in reversed(lines):
        if finished is None:
            m = _FINISHED_RE.search(line)
            if m:
                finished = m.group("ts")
                exit_code = int(m.group("rc"))
                continue
        if started is None:
            m = _STARTED_RE.search(line)
            if m:
                started = m.group("ts")

    for line in lines:
        for key in counts:
            if counts[key] is None:
                m = re.search(rf"{key.upper()}\s.*?(\d+)", line, re.IGNORECASE)
                if m:
                    counts[key] = int(m.group(1))

    return {
        "started": started, "finished": finished, "exit_code": exit_code,
        "counts": counts, "latest_bar": latest_bar,
    }


def _latest_cached_bar(cache_dir: Path = CACHE_DIR) -> str | None:
    """Find the newest last-bar-date across all cached parquets — fast enough
    to run against a few thousand files."""
    if not cache_dir.exists():
        return None
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        return None
    latest = None
    # Sample the most-recently-modified files rather than every parquet.
    files = sorted(
        (p for p in cache_dir.glob("*.parquet") if p.stem != "manifest"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:200]
    for p in files:
        try:
            end = pd.read_parquet(p, columns=[]).index.max()
        except Exception:  # noqa: BLE001
            continue
        if pd.isna(end):
            continue
        if latest is None or end > latest:
            latest = end
    return latest.strftime("%Y-%m-%d") if latest is not None else None


def _compose(status: str, summary: dict, latest_bar: str | None) -> tuple[str, str]:
    today = datetime.now().strftime("%Y-%m-%d")
    if status == "start":
        subject = f"ETFTrader — weekly refresh started {today}"
        body = (
            f"Weekly ETF cache refresh started at {summary.get('started') or '(unknown)'}.\n"
            f"Estimated finish: ~14 hours (rate-limit bound).\n"
            f"When it completes you'll get a second message.\n"
        )
        return subject, body

    rc = summary.get("exit_code")
    if rc == 0:
        outcome = "SUCCESS"
    elif rc is None:
        outcome = "UNKNOWN (no 'Finished' line in log)"
    else:
        outcome = f"FAILED (exit={rc})"

    subject = f"ETFTrader — weekly refresh {outcome.split()[0]} {today}"
    lines = [
        f"Status:       {outcome}",
        f"Started:      {summary.get('started') or '(unknown)'}",
        f"Finished:     {summary.get('finished') or '(unknown)'}",
        f"Latest bar:   {latest_bar or '(unknown)'}",
        "",
    ]
    c = summary.get("counts") or {}
    if any(v is not None for v in c.values()):
        lines += [
            "Ticker classification at start:",
            f"  current: {c.get('current')}",
            f"  stale:   {c.get('stale')}",
            f"  missing: {c.get('missing')}",
            "",
        ]
    if rc != 0:
        lines += [
            "Non-zero exit code — check the log for details:",
            "  " + str(Path.home() / "trade_data" / "ETFTrader" / "logs" / "daily_etf.log"),
        ]
    else:
        lines += [
            "Cache is current. Next steps:",
            "  1. Open the applet (streamlit run app.py)",
            "  2. Tab 2 -> Load / refresh price matrix (no need to tick 'Refresh from IB')",
            "  3. Continue through tabs 3 / 4",
        ]
    return subject, "\n".join(lines)


def _send_telegram(body: str) -> bool:
    """Send a message via the Telegram Bot API.

    Uses ``sendMessage`` with a POST body carrying ``chat_id`` and ``text``.
    Bot token from ``TELEGRAM_TOKEN``, chat ID from ``TELEGRAM_CHAT_ID``.
    Bot signup is a two-step BotFather flow — see .env.example.

    Telegram's message length limit is 4096 characters; we truncate
    defensively at 4000 to leave room for a truncation marker.
    """
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        missing = [k for k in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID")
                   if not os.environ.get(k)]
        print(f"notify_refresh: telegram skipped, missing: {', '.join(missing)}",
              file=sys.stderr)
        return False

    if len(body) > 4000:
        body = body[:3997] + "..."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": body,
        "disable_web_page_preview": True,
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            response = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        print(f"notify_refresh: telegram send failed: {exc}", file=sys.stderr)
        return False

    if response.get("ok"):
        return True
    print(f"notify_refresh: telegram error: {response}", file=sys.stderr)
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--status", choices=("start", "end"), required=True)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = p.parse_args()

    summary = _last_run_summary(args.log)
    latest_bar = _latest_cached_bar()
    subject, body = _compose(args.status, summary, latest_bar)

    # Telegram body concatenates subject + body so the headline is visible
    # even if the reader only sees the notification snippet.
    telegram_body = f"{subject}\n\n{body}"

    telegram_ok = _send_telegram(telegram_body)
    print(f"notify_refresh: subject='{subject}' telegram={telegram_ok}")
    return 0  # Notification failures never fail the task.


if __name__ == "__main__":
    sys.exit(main())

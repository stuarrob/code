#!/usr/bin/env python3
"""Post-run notification for the weekly ETF cache refresh — email + WhatsApp.

Called from ``daily_etf_data_windows.cmd`` after ``daily_etf_data.py``
finishes (or fails). Reads the tail of ``daily_etf.log``, summarises the
outcome (start time, finish time, exit code, ticker counts, latest bar
date), and delivers it via:

  1. **SMTP email** using ``SMTP_HOST``/``SMTP_USER``/``SMTP_PASSWORD``/
     ``ALERT_EMAIL_TO`` from ``.env``. Fires only if all are populated.
  2. **WhatsApp via CallMeBot** using ``WHATSAPP_PHONE`` (with country code,
     e.g. ``+9665...``) and ``WHATSAPP_APIKEY`` from ``.env``. Fires only if
     both are populated.

Both channels are attempted; one silently succeeding is enough. Neither
channel failing causes the wrapper's exit code to change — that would
make notification hiccups look like data-collection problems.

Env vars are loaded from ``.env`` via the ``src/__init__.py`` autoloader
when this file imports ``src``. For iCloud SMTP you need an app-specific
password from appleid.apple.com — the regular Apple ID password will
not authenticate.
"""

from __future__ import annotations

import argparse
import os
import re
import smtplib
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from email.mime.text import MIMEText
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
            f"When it completes you'll get a second email.\n"
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


def _send_email(subject: str, body: str) -> bool:
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to = os.environ.get("ALERT_EMAIL_TO")
    if not all([host, user, password, to]):
        missing = [k for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO")
                   if not os.environ.get(k)]
        print(f"notify_refresh: email skipped, missing: {', '.join(missing)}",
              file=sys.stderr)
        return False

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    try:
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=30)
        else:
            server = smtplib.SMTP(host, port, timeout=30)
            server.ehlo()
            server.starttls()
            server.ehlo()
        server.login(user, password)
        server.sendmail(user, [to], msg.as_string())
        server.quit()
    except Exception as exc:  # noqa: BLE001
        print(f"notify_refresh: SMTP send failed: {exc}", file=sys.stderr)
        return False
    return True


def _send_whatsapp(body: str) -> bool:
    """Send a WhatsApp message via the CallMeBot personal-use API.

    CallMeBot's `/whatsapp.php` endpoint takes ``phone`` (with country code,
    no plus), ``text`` (URL-encoded, up to ~1000 chars), and ``apikey``.
    See https://www.callmebot.com/blog/free-api-whatsapp-messages/ for the
    signup flow (message their bot from your phone; they reply with a key).
    """
    phone = os.environ.get("WHATSAPP_PHONE")
    api_key = os.environ.get("WHATSAPP_APIKEY")
    if not phone or not api_key:
        missing = [k for k in ("WHATSAPP_PHONE", "WHATSAPP_APIKEY")
                   if not os.environ.get(k)]
        print(f"notify_refresh: whatsapp skipped, missing: {', '.join(missing)}",
              file=sys.stderr)
        return False

    # CallMeBot expects the phone with country code but no plus sign.
    phone_normalized = phone.lstrip("+").replace(" ", "")

    # CallMeBot has an ~1000-char limit; truncate defensively.
    if len(body) > 900:
        body = body[:897] + "..."

    params = urllib.parse.urlencode({
        "phone": phone_normalized,
        "text": body,
        "apikey": api_key,
    })
    url = f"https://api.callmebot.com/whatsapp.php?{params}"

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            response = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        print(f"notify_refresh: whatsapp send failed: {exc}", file=sys.stderr)
        return False

    # CallMeBot returns "Message queued" or similar on success; explicit error text on failure.
    if "queued" in response.lower() or "sent" in response.lower():
        return True
    print(f"notify_refresh: whatsapp response was: {response[:200]}", file=sys.stderr)
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--status", choices=("start", "end"), required=True)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = p.parse_args()

    summary = _last_run_summary(args.log)
    latest_bar = _latest_cached_bar()
    subject, body = _compose(args.status, summary, latest_bar)

    # WhatsApp body is the subject + body concatenated so the phone-length
    # limit doesn't hide the headline.
    whatsapp_body = f"{subject}\n\n{body}"

    email_ok = _send_email(subject, body)
    whatsapp_ok = _send_whatsapp(whatsapp_body)
    print(f"notify_refresh: subject='{subject}' email={email_ok} whatsapp={whatsapp_ok}")
    return 0  # Notification failures never fail the task.


if __name__ == "__main__":
    sys.exit(main())

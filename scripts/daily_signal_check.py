#!/usr/bin/env python3
"""
Daily Signal Check — Combined FX + Leveraged ETF Alert

Detects active signals from both strategies and sends a single
email alert. Designed to run via cron before NYSE open.

NO IB Gateway connection required — reads cached data only.

Outputs:
    Email to ALERT_EMAIL_TO with signal status
    Log messages to stdout (captured by shell wrapper)

Usage:
    python scripts/daily_signal_check.py           # detect + email
    python scripts/daily_signal_check.py --dry-run  # detect only, print to stdout
"""

import argparse
import logging
import os
import smtplib
import sys
import traceback
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

# src.* is importable via editable install; sibling script dirs need to be added
# because scripts/, signals/scripts/, notebooks/scripts/ are not installed packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "signals" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "notebooks" / "scripts"))

DATA_DIR = Path.home() / "trade_data" / "ETFTrader"
LIVE_DIR = PROJECT_ROOT / "fx_signals_live"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("signal_check")


# ---------------------------------------------------------------------------
# FX Signal Detection
# ---------------------------------------------------------------------------

def check_fx_signals():
    """Run FX skew divergence signal detection.

    Returns dict with keys: signals, positions, error
    """
    result = {"signals": [], "positions": {}, "error": None}

    try:
        from fx_signal_execute import detect_active_signals, check_fx_positions

        log.info("Checking FX skew divergence signals...")
        signals = detect_active_signals(data_dir=DATA_DIR)
        result["signals"] = signals
        log.info("FX signals: %d active", len(signals))

        if LIVE_DIR.exists():
            positions = check_fx_positions(live_dir=LIVE_DIR)
            result["positions"] = positions
            log.info(
                "FX positions: %d open, %d expired",
                positions.get("n_open", 0),
                positions.get("n_expired", 0),
            )
        else:
            log.info("No fx_signals_live directory — no open positions")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        log.error("FX signal detection failed: %s", e)
        log.error(traceback.format_exc())

    return result


# ---------------------------------------------------------------------------
# Leveraged ETF Signal Detection
# ---------------------------------------------------------------------------

def check_leveraged_signals():
    """Run leveraged ETF signal computation.

    Returns dict with keys: signals, error
    """
    result = {"signals": {}, "error": None}

    try:
        from s12_leveraged_execute import compute_live_signals

        log.info("Checking leveraged ETF signals...")
        ib_hist_dir = DATA_DIR / "ib_historical"
        signals = compute_live_signals(data_dir=ib_hist_dir)
        result["signals"] = signals

        for ticker, info in signals.items():
            state = info.get("signal_state", "unknown")
            log.info("  %s: %s", ticker, state.upper())

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        log.error("Leveraged ETF signal detection failed: %s", e)
        log.error(traceback.format_exc())

    return result


# ---------------------------------------------------------------------------
# Data Freshness
# ---------------------------------------------------------------------------

STALE_THRESHOLD = 3  # days

def check_data_freshness():
    """Check age of all critical data files."""
    files_to_check = {
        "SABR Params": DATA_DIR / "fx_sabr" / "sabr_params_historical.parquet",
        "EUR Spot": DATA_DIR / "fx_spot" / "EURUSD.parquet",
        "GBP Spot": DATA_DIR / "fx_spot" / "GBPUSD.parquet",
        "JPY Spot": DATA_DIR / "fx_spot" / "USDJPY.parquet",
        "EUR Surface": DATA_DIR / "fx_options_db" / "EUR_surface.parquet",
        "GBP Surface": DATA_DIR / "fx_options_db" / "GBP_surface.parquet",
        "JPY Surface": DATA_DIR / "fx_options_db" / "JPY_surface.parquet",
        "XLK (ref TECL)": DATA_DIR / "ib_historical" / "XLK.parquet",
        "ITB (ref NAIL)": DATA_DIR / "ib_historical" / "ITB.parquet",
    }

    freshness = {}
    now = datetime.now()

    for name, path in files_to_check.items():
        if not path.exists():
            freshness[name] = {
                "path": str(path),
                "last_modified": None,
                "age_days": None,
                "stale": True,
                "status": "MISSING",
            }
            continue

        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age_days = (now - mtime).total_seconds() / 86400

        # For SABR, also check latest data timestamp inside the file
        last_data_date = None
        if "sabr_params" in path.name:
            try:
                df = pd.read_parquet(path)
                if "timestamp" in df.columns:
                    last_data_date = pd.to_datetime(df["timestamp"]).max()
            except Exception:
                pass

        freshness[name] = {
            "path": str(path),
            "last_modified": mtime,
            "age_days": age_days,
            "stale": age_days > STALE_THRESHOLD,
            "status": "STALE" if age_days > STALE_THRESHOLD else "OK",
            "last_data_date": last_data_date,
        }

    return freshness


# ---------------------------------------------------------------------------
# Email Formatting
# ---------------------------------------------------------------------------

def format_email(fx_result, lev_result, freshness):
    """Format the combined alert email. Returns (subject, body)."""
    today = datetime.now().strftime("%Y-%m-%d")

    has_fx_signals = bool(fx_result.get("signals"))
    has_risk_on = any(
        info.get("signal_state") == "risk_on"
        for info in lev_result.get("signals", {}).values()
    )
    has_fx_expired = fx_result.get("positions", {}).get("n_expired", 0) > 0
    any_stale = any(f.get("stale") for f in freshness.values())
    has_errors = fx_result.get("error") or lev_result.get("error")

    if has_errors:
        subject = f"ETFTrader Signal Alert - ERRORS - {today}"
    elif has_fx_signals or has_fx_expired:
        subject = f"ETFTrader Signal Alert - ACTION - {today}"
    elif has_risk_on or any_stale:
        subject = f"ETFTrader Signal Alert - {today}"
    else:
        subject = f"ETFTrader - No Signals - {today}"

    lines = []
    lines.append(f"ETFTrader Daily Signal Check - {today}")
    lines.append("=" * 55)
    lines.append("")

    # --- FX Signals ---
    lines.append("FX SKEW DIVERGENCE SIGNALS")
    lines.append("-" * 40)

    if fx_result.get("error"):
        lines.append(f"  ERROR: {fx_result['error']}")
    elif not fx_result.get("signals"):
        lines.append("  No active signals.")
    else:
        for sig in fx_result["signals"]:
            pair = sig.get("pair", "?")
            direction = sig.get("direction", "?").upper()
            quality = sig.get("quality_score", 0)
            tercile = sig.get("quality_tercile", "?").upper()
            hold = sig.get("hold_days", "?")
            lines.append(f"  >> {pair}: {direction} signal")
            lines.append(f"     Quality: {quality:.2f} ({tercile})")
            lines.append(f"     Hold period: {hold} days")
            if sig.get("sabr_params"):
                sp = sig["sabr_params"]
                lines.append(
                    f"     SABR 1M: F={sp['forward']:.5f}, "
                    f"alpha={sp['alpha']:.4f}, rho={sp['rho']:.3f}"
                )
            lines.append("")

    lines.append("")

    # --- FX Open Positions ---
    lines.append("FX OPEN POSITIONS")
    lines.append("-" * 40)

    positions = fx_result.get("positions", {})
    open_pos = positions.get("open_positions", [])

    if not open_pos:
        lines.append("  No open positions.")
    else:
        for pos in open_pos:
            pair = pos.get("pair", "?")
            days_left = pos.get("days_left", "?")
            direction = pos.get("direction", "?").upper()
            expired = " ** EXPIRED **" if pos.get("expired") else ""
            lines.append(
                f"  {pair}: {direction}, {days_left} days remaining{expired}"
            )

    lines.append("")

    # --- Leveraged ETF ---
    lines.append("LEVERAGED ETF SIGNALS (TECL/NAIL)")
    lines.append("-" * 40)

    if lev_result.get("error"):
        lines.append(f"  ERROR: {lev_result['error']}")
    elif not lev_result.get("signals"):
        lines.append("  No signal data available.")
    else:
        for ticker, info in lev_result["signals"].items():
            if "error" in info:
                lines.append(f"  {ticker}: ERROR - {info['error']}")
                continue
            state = info.get("signal_state", "unknown").upper()
            price = info.get("price", 0)
            sma = info.get("sma_200")
            vol = info.get("ref_vol")
            above_sma = info.get("above_sma", False)
            data_date = info.get("data_date", "?")

            marker = ">>" if state == "RISK_ON" else "  "
            lines.append(f"  {marker} {ticker}: {state}")
            sma_str = f"${sma:.2f}" if sma else "N/A"
            lines.append(
                f"     Price: ${price:.2f} | SMA200: {sma_str} | "
                f"Above SMA: {'Yes' if above_sma else 'No'}"
            )
            vol_str = f"{vol:.1%}" if vol is not None else "N/A"
            lines.append(f"     Ref vol: {vol_str} | Data: {data_date}")
            lines.append("")

    lines.append("")

    # --- Data Freshness ---
    lines.append("DATA FRESHNESS")
    lines.append("-" * 40)

    for name, info in freshness.items():
        status = info.get("status", "?")
        if info.get("last_modified"):
            age = info["age_days"]
            mtime = info["last_modified"].strftime("%Y-%m-%d %H:%M")
            warn = " ** WARNING **" if info.get("stale") else ""
            lines.append(f"  {name}: {mtime} ({age:.1f}d ago) [{status}]{warn}")
            if info.get("last_data_date"):
                lines.append(f"     Latest data: {info['last_data_date']}")
        else:
            lines.append(f"  {name}: MISSING ** WARNING **")

    lines.append("")

    # --- Action Required ---
    lines.append("ACTION REQUIRED")
    lines.append("-" * 40)

    actions = []
    if has_fx_signals:
        for sig in fx_result["signals"]:
            pair = sig.get("pair", "?")
            direction = sig.get("direction", "?")
            tercile = sig.get("quality_tercile", "?")
            actions.append(
                f"  - Review {pair} {direction} signal ({tercile} quality) "
                f"in signals/03_fx_signal_execution.ipynb"
            )

    if has_fx_expired:
        n_exp = positions.get("n_expired", 0)
        actions.append(
            f"  - Close {n_exp} expired FX position(s) via execution notebook"
        )

    if has_risk_on:
        for ticker, info in lev_result.get("signals", {}).items():
            if info.get("signal_state") == "risk_on":
                actions.append(
                    f"  - {ticker} is RISK_ON: check position in "
                    f"notebooks/03_leveraged_execute.ipynb"
                )

    if any_stale:
        stale_names = [n for n, f in freshness.items() if f.get("stale")]
        actions.append(
            f"  - Stale data: {', '.join(stale_names)}. "
            f"Start IB Gateway and check data collection crons."
        )

    if has_errors:
        if fx_result.get("error"):
            actions.append("  - FX signal detection FAILED — check logs")
        if lev_result.get("error"):
            actions.append("  - Leveraged ETF detection FAILED — check logs")

    if not actions:
        actions.append("  No action needed. All clear.")

    lines.extend(actions)
    lines.append("")
    lines.append("-" * 55)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Log: ~/trade_data/ETFTrader/logs/daily_signal_check.log")

    return subject, "\n".join(lines)


# ---------------------------------------------------------------------------
# Email Sending
# ---------------------------------------------------------------------------

def send_email(subject, body):
    """Send email via SMTP. Config from environment variables."""
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    alert_to = os.environ.get("ALERT_EMAIL_TO")

    if not all([smtp_host, smtp_user, smtp_password, alert_to]):
        missing = [
            var
            for var in ["SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO"]
            if not os.environ.get(var)
        ]
        log.error("Cannot send email — missing env vars: %s", ", ".join(missing))
        log.info("Email subject: %s", subject)
        log.info("Email body:\n%s", body)
        return False

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = alert_to

    try:
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=30)
            server.ehlo()
            server.starttls()
            server.ehlo()

        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, [alert_to], msg.as_string())
        server.quit()
        log.info("Email sent to %s: %s", alert_to, subject)
        return True

    except Exception as e:
        log.error("FAILED TO SEND EMAIL: %s", e)
        log.error("Subject: %s", subject)
        log.error("Body:\n%s", body)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Daily signal check with email alert"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect signals and print email to stdout (don't send)",
    )
    args = parser.parse_args()

    log.info("Starting daily signal check...")

    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass  # Shell wrapper sources .env

    # 1. FX Signals
    fx_result = check_fx_signals()

    # 2. Leveraged ETF Signals
    lev_result = check_leveraged_signals()

    # 3. Data Freshness
    freshness = check_data_freshness()

    # 4. Format email
    subject, body = format_email(fx_result, lev_result, freshness)

    if args.dry_run:
        print(f"\nSubject: {subject}\n")
        print(body)
        log.info("Dry run complete — email not sent")
        return

    # 5. Send email
    success = send_email(subject, body)
    if not success:
        log.error("Email sending failed — check SMTP config in .env")
        sys.exit(1)

    log.info("Signal check complete")


if __name__ == "__main__":
    main()

"""
IB Gateway Connection Utility

Polls for IB Gateway availability with configurable retry logic.
Used by daily cron scripts to wait until IB is ready after login.

Usage:
    from ib_wait import wait_for_ib

    ib = wait_for_ib()  # blocks until connected, returns IB instance
    # ... do work ...
    ib.disconnect()
"""

import logging
import sys
import time
from pathlib import Path

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

log = logging.getLogger("ib_wait")


def wait_for_ib(
    host: str = "127.0.0.1",
    port: int = 4001,
    client_id: int = 20,
    readonly: bool = True,
    retry_interval: int = 60,
    max_retries: int = 120,
) -> "IB":
    """Poll for IB Gateway and return a connected IB instance.

    Args:
        host: IB Gateway host.
        port: IB Gateway port.
        client_id: IB client ID.
        readonly: Connect in readonly mode.
        retry_interval: Seconds between retries.
        max_retries: Maximum number of retries (default 120 = 2 hours).

    Returns:
        Connected IB instance.

    Raises:
        ConnectionError: If max retries exceeded.
    """
    from ib_insync import IB

    for attempt in range(1, max_retries + 1):
        try:
            ib = IB()
            ib.connect(host, port, clientId=client_id, readonly=readonly, timeout=10)
            accounts = ib.managedAccounts()
            log.info(
                "Connected to IB on attempt %d: %s",
                attempt, accounts[0] if accounts else "unknown",
            )
            return ib
        except Exception as e:
            if attempt == 1:
                log.info("IB Gateway not available (%s). Polling every %ds...", e, retry_interval)
            elif attempt % 10 == 0:
                log.info("Still waiting for IB... (attempt %d/%d)", attempt, max_retries)
            time.sleep(retry_interval)

    raise ConnectionError(
        f"IB Gateway not available after {max_retries} attempts "
        f"({max_retries * retry_interval / 60:.0f} minutes)"
    )


def is_ib_available(host: str = "127.0.0.1", port: int = 4001) -> bool:
    """Quick check if IB Gateway is reachable (no login required)."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False

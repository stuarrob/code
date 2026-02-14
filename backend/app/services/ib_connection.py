"""Interactive Brokers Gateway connection manager."""

import asyncio
import logging
from typing import Optional

from ib_insync import IB

from app.core.config import settings

logger = logging.getLogger(__name__)


class IBConnectionManager:
    """
    Singleton manager for Interactive Brokers Gateway connection.

    Handles connection, disconnection, and automatic reconnection.
    Uses ib.connectAsync() to work natively with FastAPI's asyncio event loop.
    """

    def __init__(self):
        self._ib: Optional[IB] = None
        self._connected: bool = False
        self._reconnect_task: Optional[asyncio.Task] = None

    @property
    def ib(self) -> Optional[IB]:
        """Get the IB client instance. Returns None if not connected."""
        if self._ib and self._ib.isConnected():
            return self._ib
        return None

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to IB Gateway."""
        return self._ib is not None and self._ib.isConnected()

    async def connect(self) -> bool:
        """
        Connect to IB Gateway.

        Returns:
            True if connection successful, False otherwise.
        """
        if not settings.IB_ENABLED:
            logger.info("IB Gateway integration is disabled (IB_ENABLED=False)")
            return False

        if self.is_connected:
            logger.info("Already connected to IB Gateway")
            return True

        try:
            self._ib = IB()
            self._ib.disconnectedEvent += self._on_disconnect

            await self._ib.connectAsync(
                host=settings.IB_HOST,
                port=settings.IB_PORT,
                clientId=settings.IB_CLIENT_ID,
                timeout=settings.IB_TIMEOUT,
                readonly=settings.IB_READONLY,
            )

            self._connected = True
            accounts = self._ib.managedAccounts()
            logger.info(
                f"Connected to IB Gateway at {settings.IB_HOST}:{settings.IB_PORT} "
                f"(client_id={settings.IB_CLIENT_ID}, readonly={settings.IB_READONLY})"
            )
            logger.info(f"Managed accounts: {accounts}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            self._connected = False
            self._start_reconnect_loop()
            return False

    async def disconnect(self) -> None:
        """Disconnect from IB Gateway and cancel reconnection."""
        self._cancel_reconnect()

        if self._ib:
            self._ib.disconnectedEvent -= self._on_disconnect
            self._ib.disconnect()
            self._ib = None

        self._connected = False
        logger.info("Disconnected from IB Gateway")

    def _on_disconnect(self) -> None:
        """Handle unexpected disconnection."""
        logger.warning("Disconnected from IB Gateway unexpectedly")
        self._connected = False
        self._start_reconnect_loop()

    def _start_reconnect_loop(self) -> None:
        """Start background reconnection task."""
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    def _cancel_reconnect(self) -> None:
        """Cancel any pending reconnection task."""
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            self._reconnect_task = None

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect at intervals until successful."""
        while not self.is_connected:
            logger.info(
                f"Attempting to reconnect to IB Gateway in "
                f"{settings.IB_RECONNECT_INTERVAL}s..."
            )
            await asyncio.sleep(settings.IB_RECONNECT_INTERVAL)

            try:
                if self._ib:
                    self._ib.disconnect()

                self._ib = IB()
                self._ib.disconnectedEvent += self._on_disconnect

                await self._ib.connectAsync(
                    host=settings.IB_HOST,
                    port=settings.IB_PORT,
                    clientId=settings.IB_CLIENT_ID,
                    timeout=settings.IB_TIMEOUT,
                    readonly=settings.IB_READONLY,
                )

                self._connected = True
                logger.info("Reconnected to IB Gateway successfully")
                return

            except asyncio.CancelledError:
                logger.info("Reconnection cancelled")
                return
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")

    def get_status(self) -> dict:
        """Get connection status summary."""
        return {
            "enabled": settings.IB_ENABLED,
            "connected": self.is_connected,
            "host": settings.IB_HOST,
            "port": settings.IB_PORT,
            "client_id": settings.IB_CLIENT_ID,
            "readonly": settings.IB_READONLY,
            "accounts": self._ib.managedAccounts() if self.is_connected else [],
        }


# Global singleton instance
ib_manager = IBConnectionManager()

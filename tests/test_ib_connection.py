"""Tests for IB connection manager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestIBConnectionManager:
    """Tests for IBConnectionManager."""

    def test_initial_state(self):
        """Connection manager starts disconnected."""
        from backend.app.services.ib_connection import IBConnectionManager

        manager = IBConnectionManager()
        assert not manager.is_connected
        assert manager.ib is None

    def test_get_status_disconnected(self):
        """Status shows disconnected state."""
        from backend.app.services.ib_connection import IBConnectionManager

        manager = IBConnectionManager()
        status = manager.get_status()
        assert status["connected"] is False
        assert status["accounts"] == []

    @pytest.mark.asyncio
    @patch("backend.app.services.ib_connection.settings")
    async def test_connect_disabled(self, mock_settings):
        """Connection returns False when IB_ENABLED is False."""
        mock_settings.IB_ENABLED = False
        from backend.app.services.ib_connection import IBConnectionManager

        manager = IBConnectionManager()
        result = await manager.connect()
        assert result is False

    @pytest.mark.asyncio
    @patch("backend.app.services.ib_connection.IB")
    @patch("backend.app.services.ib_connection.settings")
    async def test_connect_success(self, mock_settings, mock_ib_class):
        """Connection succeeds when Gateway is available."""
        mock_settings.IB_ENABLED = True
        mock_settings.IB_HOST = "127.0.0.1"
        mock_settings.IB_PORT = 4002
        mock_settings.IB_CLIENT_ID = 1
        mock_settings.IB_TIMEOUT = 10
        mock_settings.IB_READONLY = True

        mock_ib = MagicMock()
        mock_ib.connectAsync = AsyncMock()
        mock_ib.isConnected.return_value = True
        mock_ib.managedAccounts.return_value = ["DU12345"]
        mock_ib.disconnectedEvent = MagicMock()
        mock_ib_class.return_value = mock_ib

        from backend.app.services.ib_connection import IBConnectionManager

        manager = IBConnectionManager()
        result = await manager.connect()
        assert result is True
        assert manager.is_connected

    @pytest.mark.asyncio
    @patch("backend.app.services.ib_connection.IB")
    @patch("backend.app.services.ib_connection.settings")
    async def test_connect_failure_starts_reconnect(self, mock_settings, mock_ib_class):
        """Failed connection starts reconnect loop."""
        mock_settings.IB_ENABLED = True
        mock_settings.IB_HOST = "127.0.0.1"
        mock_settings.IB_PORT = 4002
        mock_settings.IB_CLIENT_ID = 1
        mock_settings.IB_TIMEOUT = 10
        mock_settings.IB_READONLY = True
        mock_settings.IB_RECONNECT_INTERVAL = 30

        mock_ib = MagicMock()
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError("refused"))
        mock_ib.disconnectedEvent = MagicMock()
        mock_ib_class.return_value = mock_ib

        from backend.app.services.ib_connection import IBConnectionManager

        manager = IBConnectionManager()
        result = await manager.connect()
        assert result is False
        assert not manager.is_connected

        # Clean up reconnect task
        manager._cancel_reconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Disconnect cleans up state."""
        from backend.app.services.ib_connection import IBConnectionManager

        manager = IBConnectionManager()
        await manager.disconnect()
        assert not manager.is_connected
        assert manager.ib is None

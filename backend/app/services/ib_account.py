"""Interactive Brokers account information service."""

import logging
from datetime import datetime
from typing import List, Optional

from app.core.config import settings
from app.services.ib_connection import ib_manager

logger = logging.getLogger(__name__)


def _safe_float(val) -> Optional[float]:
    """Convert IB value to float, returning None for NaN."""
    if val is None or val != val:
        return None
    return float(val)


def _get_account() -> Optional[str]:
    """Get the configured or first available IB account."""
    if settings.IB_ACCOUNT:
        return settings.IB_ACCOUNT

    ib = ib_manager.ib
    if ib:
        accounts = ib.managedAccounts()
        if accounts:
            return accounts[0]

    return None


class IBAccountService:
    """
    Service for fetching account information from Interactive Brokers.

    Provides account summary, positions, and P&L data.
    """

    async def get_account_summary(self) -> Optional[dict]:
        """
        Get account summary including balances and net liquidation value.

        Returns dict with account values or None if not connected.
        """
        ib = ib_manager.ib
        if not ib:
            return None

        try:
            account_values = ib.accountSummary()

            if not account_values:
                await ib.accountSummaryAsync()
                account_values = ib.accountSummary()

            summary = {
                "source": "ib_gateway",
                "timestamp": datetime.now().isoformat(),
                "values": {},
            }

            key_fields = {
                "NetLiquidation",
                "TotalCashValue",
                "GrossPositionValue",
                "AvailableFunds",
                "BuyingPower",
                "MaintMarginReq",
                "ExcessLiquidity",
                "UnrealizedPnL",
                "RealizedPnL",
            }

            for av in account_values:
                if av.tag in key_fields:
                    summary["values"][av.tag] = {
                        "value": float(av.value) if av.value else 0.0,
                        "currency": av.currency,
                        "account": av.account,
                    }

            summary["net_liquidation"] = self._extract_value(
                summary["values"], "NetLiquidation"
            )
            summary["total_cash"] = self._extract_value(
                summary["values"], "TotalCashValue"
            )
            summary["gross_position_value"] = self._extract_value(
                summary["values"], "GrossPositionValue"
            )
            summary["available_funds"] = self._extract_value(
                summary["values"], "AvailableFunds"
            )
            summary["buying_power"] = self._extract_value(
                summary["values"], "BuyingPower"
            )
            summary["unrealized_pnl"] = self._extract_value(
                summary["values"], "UnrealizedPnL"
            )
            summary["realized_pnl"] = self._extract_value(
                summary["values"], "RealizedPnL"
            )

            return summary

        except Exception as e:
            logger.error(f"Error fetching IB account summary: {e}")
            return None

    async def get_positions(self) -> List[dict]:
        """
        Get current positions from IB account.

        Returns list of position dicts with ticker, shares, avg_cost, market_value.
        """
        ib = ib_manager.ib
        if not ib:
            return []

        try:
            positions = ib.positions()

            return [
                {
                    "account": pos.account,
                    "ticker": pos.contract.symbol,
                    "security_type": pos.contract.secType,
                    "exchange": pos.contract.exchange,
                    "currency": pos.contract.currency,
                    "shares": float(pos.position),
                    "avg_cost": float(pos.avgCost),
                    "market_value": float(pos.position) * float(pos.avgCost),
                }
                for pos in positions
            ]

        except Exception as e:
            logger.error(f"Error fetching IB positions: {e}")
            return []

    async def get_pnl(self) -> Optional[dict]:
        """
        Get portfolio-level P&L from IB.

        Returns dict with daily_pnl, unrealized_pnl, realized_pnl.
        """
        ib = ib_manager.ib
        if not ib:
            return None

        try:
            account = _get_account()
            if not account:
                return None

            pnl = ib.reqPnL(account)
            await ib.sleep(2)

            result = {
                "source": "ib_gateway",
                "timestamp": datetime.now().isoformat(),
                "account": account,
                "daily_pnl": _safe_float(pnl.dailyPnL),
                "unrealized_pnl": _safe_float(pnl.unrealizedPnL),
                "realized_pnl": _safe_float(pnl.realizedPnL),
            }

            ib.cancelPnL(account)
            return result

        except Exception as e:
            logger.error(f"Error fetching IB PnL: {e}")
            return None

    async def get_position_pnl(self) -> List[dict]:
        """
        Get per-position P&L from IB.

        Returns list of position P&L dicts.
        """
        ib = ib_manager.ib
        if not ib:
            return []

        try:
            account = _get_account()
            if not account:
                return []

            positions = ib.positions()
            results = []

            for pos in positions:
                contract = pos.contract
                pnl_single = ib.reqPnLSingle(account, "", contract.conId)
                await ib.sleep(1)

                results.append(
                    {
                        "ticker": contract.symbol,
                        "shares": float(pos.position),
                        "avg_cost": float(pos.avgCost),
                        "daily_pnl": _safe_float(pnl_single.dailyPnL),
                        "unrealized_pnl": _safe_float(pnl_single.unrealizedPnL),
                        "realized_pnl": _safe_float(pnl_single.realizedPnL),
                        "market_value": _safe_float(pnl_single.value),
                    }
                )

                ib.cancelPnLSingle(account, "", contract.conId)

            return results

        except Exception as e:
            logger.error(f"Error fetching IB position PnL: {e}")
            return []

    @staticmethod
    def _extract_value(values: dict, key: str) -> Optional[float]:
        """Extract a float value from the account values dict."""
        if key in values:
            return values[key]["value"]
        return None


# Global singleton instance
ib_account = IBAccountService()

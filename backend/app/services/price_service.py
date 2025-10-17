"""Price service - loads latest ETF prices from data."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from decimal import Decimal

from app.core.config import settings


class PriceService:
    """Service for fetching latest ETF prices."""

    def __init__(self):
        self._cache = {}

    @property
    def data_dir(self) -> Path:
        """Resolve data directory dynamically."""
        return Path(settings.DATA_DIR)

    def load_latest_prices(self) -> Dict[str, Decimal]:
        """
        Load latest prices for all ETFs.

        Returns:
            Dict mapping ticker -> latest price
        """
        # Check cache
        if 'latest_prices' in self._cache:
            return self._cache['latest_prices']

        try:
            prices_file = self.data_dir / 'processed' / 'etf_prices_filtered.parquet'

            if not prices_file.exists():
                print(f"Price file not found: {prices_file}")
                return {}

            # Load price data
            df = pd.read_parquet(prices_file)

            # Get latest row (most recent date)
            latest_prices = df.iloc[-1]

            # Convert to dict of ticker -> price
            price_dict = {
                ticker: Decimal(str(price))
                for ticker, price in latest_prices.items()
                if pd.notna(price)
            }

            # Cache
            self._cache['latest_prices'] = price_dict

            print(f"Loaded {len(price_dict)} latest prices")
            return price_dict

        except Exception as e:
            print(f"Error loading prices: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_price(self, ticker: str) -> Optional[Decimal]:
        """
        Get latest price for specific ticker.

        Args:
            ticker: ETF ticker

        Returns:
            Latest price or None if not found
        """
        prices = self.load_latest_prices()
        return prices.get(ticker)


# Global instance
price_service = PriceService()

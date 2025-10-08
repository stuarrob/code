"""
Asset Class Mapper - Map ETF categories to broader asset classes

Creates a hierarchical asset class structure to prevent over-concentration
in correlated assets (e.g., multiple gold ETFs).
"""

from typing import Dict
import pandas as pd


class AssetClassMapper:
    """Map ETF categories to broader asset classes for diversification."""

    # Mapping from Morningstar-style categories to broader asset classes
    CATEGORY_TO_ASSET_CLASS = {
        # US Equity - Large Cap
        "Large Blend": "US Equity - Large Cap",
        "Large Growth": "US Equity - Large Cap",
        "Large Value": "US Equity - Large Cap",

        # US Equity - Mid Cap
        "Mid-Cap Blend": "US Equity - Mid Cap",
        "Mid-Cap Growth": "US Equity - Mid Cap",
        "Mid-Cap Value": "US Equity - Mid Cap",

        # US Equity - Small Cap
        "Small Blend": "US Equity - Small Cap",
        "Small Growth": "US Equity - Small Cap",
        "Small Value": "US Equity - Small Cap",

        # US Equity - Sector
        "Technology": "US Equity - Technology",
        "Health": "US Equity - Healthcare",
        "Healthcare": "US Equity - Healthcare",
        "Financials": "US Equity - Financials",
        "Financial": "US Equity - Financials",
        "Energy": "US Equity - Energy",
        "Energy Limited Partnership": "US Equity - Energy",
        "Real Estate": "US Equity - Real Estate",
        "Consumer Cyclical": "US Equity - Consumer",
        "Consumer Defensive": "US Equity - Consumer",
        "Industrials": "US Equity - Industrials",
        "Basic Materials": "US Equity - Materials",
        "Communication Services": "US Equity - Communication",
        "Utilities": "US Equity - Utilities",

        # International Equity - Developed
        "Foreign Large Blend": "International Equity - Developed",
        "Foreign Large Growth": "International Equity - Developed",
        "Foreign Large Value": "International Equity - Developed",
        "Europe Stock": "International Equity - Europe",
        "Japan Stock": "International Equity - Japan",
        "Pacific/Asia ex-Japan Stk": "International Equity - Asia Pacific",

        # International Equity - Emerging
        "Diversified Emerging Mkts": "International Equity - Emerging",
        "Emerging Markets": "International Equity - Emerging",
        "China Region": "International Equity - Emerging",
        "India Equity": "International Equity - Emerging",
        "Latin America Stock": "International Equity - Emerging",

        # Fixed Income - Government
        "Short Government": "Fixed Income - Government",
        "Intermediate Government": "Fixed Income - Government",
        "Long Government": "Fixed Income - Government",
        "Inflation-Protected Bond": "Fixed Income - TIPS",

        # Fixed Income - Corporate
        "Short-Term Bond": "Fixed Income - Corporate",
        "Intermediate Core Bond": "Fixed Income - Corporate",
        "Intermediate Core-Plus Bond": "Fixed Income - Corporate",
        "Corporate Bond": "Fixed Income - Corporate",
        "High Yield Bond": "Fixed Income - High Yield",

        # Fixed Income - Municipal
        "Muni National Long": "Fixed Income - Municipal",
        "Muni National Interm": "Fixed Income - Municipal",
        "Muni National Short": "Fixed Income - Municipal",

        # Commodities - Precious Metals (THIS IS KEY - ONE CATEGORY FOR ALL GOLD!)
        "Gold": "Commodities - Precious Metals",
        "Silver": "Commodities - Precious Metals",
        "Precious Metals": "Commodities - Precious Metals",

        # Commodities - Other
        "Commodities Broad Basket": "Commodities - Diversified",
        "Natural Resources": "Commodities - Diversified",
        "Energy Limited Partnership": "Commodities - Energy",

        # Alternatives
        "Long-Short Equity": "Alternatives - Hedge Strategies",
        "Market Neutral": "Alternatives - Hedge Strategies",
        "Multialternative": "Alternatives - Multi-Strategy",
        "Managed Futures": "Alternatives - Managed Futures",

        # Currency
        "Currency": "Alternatives - Currency",

        # Volatility
        "Volatility": "Alternatives - Volatility",

        # Crypto
        "Digital Assets": "Alternatives - Crypto",
        "Cryptocurrency": "Alternatives - Crypto",
    }

    def __init__(self, fundamentals_path: str = None):
        """
        Initialize asset class mapper.

        Parameters
        ----------
        fundamentals_path : str, optional
            Path to fundamentals.csv file
        """
        self.fundamentals_path = fundamentals_path
        self.asset_class_map = {}

        if fundamentals_path:
            self.load_from_fundamentals(fundamentals_path)

    def load_from_fundamentals(self, fundamentals_path: str) -> Dict[str, str]:
        """
        Load asset class mapping from fundamentals CSV.

        Returns
        -------
        dict
            Mapping from ticker to asset class
        """
        df = pd.read_csv(fundamentals_path)

        self.asset_class_map = {}

        for _, row in df.iterrows():
            ticker = row['ticker']
            category = row.get('category', 'Unknown')

            if pd.notna(category):
                # Map category to broader asset class
                asset_class = self.CATEGORY_TO_ASSET_CLASS.get(category, f"Other - {category}")
                self.asset_class_map[ticker] = asset_class

        return self.asset_class_map

    def get_asset_class(self, ticker: str) -> str:
        """
        Get asset class for a ticker.

        Parameters
        ----------
        ticker : str
            ETF ticker

        Returns
        -------
        str
            Asset class
        """
        return self.asset_class_map.get(ticker, "Unknown")

    def get_all_asset_classes(self) -> list:
        """Get list of all unique asset classes."""
        return sorted(set(self.asset_class_map.values()))

    def get_tickers_by_asset_class(self, asset_class: str) -> list:
        """Get all tickers in a given asset class."""
        return [ticker for ticker, ac in self.asset_class_map.items() if ac == asset_class]


def create_asset_class_map(fundamentals_path: str) -> Dict[str, str]:
    """
    Convenience function to create asset class mapping.

    Parameters
    ----------
    fundamentals_path : str
        Path to fundamentals.csv

    Returns
    -------
    dict
        Mapping from ticker to asset class
    """
    mapper = AssetClassMapper(fundamentals_path)
    return mapper.asset_class_map

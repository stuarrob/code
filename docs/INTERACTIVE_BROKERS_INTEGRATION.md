# Interactive Brokers Integration Guide

This guide explains how to access ETF data from your Interactive Brokers (IB) account for the ETFTrader system.

## Overview

Interactive Brokers provides several ways to access market data:
1. **TWS API** (Trader Workstation API) - Most comprehensive, real-time data
2. **IB Gateway** - Lightweight alternative to TWS
3. **IBKR Web API** - REST API for account data
4. **Flex Queries** - Historical data exports

## Recommended Approach: TWS API with ib_insync

### Prerequisites

1. **Interactive Brokers Account** ✅ (You have this)
2. **Market Data Subscription** (Check your IB account for active subscriptions)
3. **TWS or IB Gateway** installed on your machine

### Installation

```bash
# Install ib_insync - Modern Python wrapper for IB API
pip install ib_insync

# Optional: Install for async support
pip install nest-asyncio
```

### Market Data Subscriptions

Interactive Brokers requires specific market data subscriptions to access real-time and historical data:

- **US Securities Snapshot and Futures Value Bundle** - $10/month (includes most US ETFs)
- **US Equity and Options Add-On Streaming Bundle** - $4.50/month
- **NYSE (Network A/CTA)** - For NYSE-listed ETFs
- **NASDAQ (Network C/UTP)** - For NASDAQ-listed ETFs

**Note:** Check `Account Management > Settings > Market Data Subscriptions` to see what you currently have.

### Basic Setup

#### 1. Enable API Access in TWS

1. Open TWS (Trader Workstation) or IB Gateway
2. Go to `File > Global Configuration > API > Settings`
3. Enable these options:
   - ☑️ Enable ActiveX and Socket Clients
   - ☑️ Allow connections from localhost
   - ☑️ Read-Only API
   - Master API Client ID: (leave as default)
   - Socket port: **7497** (TWS paper) or **7496** (TWS live)

4. Add `127.0.0.1` to Trusted IP Addresses

#### 2. Connection Example

```python
from ib_insync import IB, Stock, util
import pandas as pd

# Connect to IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # 7497 = TWS paper, 7496 = TWS live

# Check connection
print(f"Connected: {ib.isConnected()}")

# Request ETF data
etf = Stock('SPY', 'SMART', 'USD')  # SPY on SMART routing
ib.qualifyContracts(etf)

# Get current price
ticker = ib.reqMktData(etf)
ib.sleep(2)  # Wait for data
print(f"SPY Price: {ticker.last}")

# Get historical data
bars = ib.reqHistoricalData(
    etf,
    endDateTime='',
    durationStr='1 Y',
    barSizeSetting='1 day',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=1
)

# Convert to DataFrame
df = util.df(bars)
print(df.head())

# Disconnect
ib.disconnect()
```

### ETFTrader Integration Module

Here's a complete module for fetching ETF data from Interactive Brokers:

```python
# src/data_collection/ib_data_provider.py

from ib_insync import IB, Stock, util
import pandas as pd
import logging
from typing import List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class IBDataProvider:
    """Interactive Brokers data provider for ETFTrader."""

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 7497,  # 7497 = paper, 7496 = live
        client_id: int = 1,
        connect_on_init: bool = True
    ):
        """
        Initialize IB data provider.

        Parameters
        ----------
        host : str
            IB Gateway/TWS host (default: localhost)
        port : int
            Port number (7497 for paper, 7496 for live)
        client_id : int
            Client ID for connection
        connect_on_init : bool
            Connect immediately on initialization
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()

        if connect_on_init:
            self.connect()

    def connect(self) -> bool:
        """Connect to Interactive Brokers."""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info(f"✅ Connected to IB at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to IB: {e}")
            return False

    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")

    def get_etf_historical_data(
        self,
        ticker: str,
        duration: str = '2 Y',
        bar_size: str = '1 day',
        exchange: str = 'SMART'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single ETF.

        Parameters
        ----------
        ticker : str
            ETF ticker symbol (e.g., 'SPY', 'QQQ')
        duration : str
            Historical data duration (e.g., '1 Y', '2 Y', '5 Y')
        bar_size : str
            Bar size (e.g., '1 day', '1 hour', '5 mins')
        exchange : str
            Exchange (usually 'SMART' for best routing)

        Returns
        -------
        pd.DataFrame or None
            OHLCV data with datetime index
        """
        try:
            # Create contract
            contract = Stock(ticker, exchange, 'USD')
            self.ib.qualifyContracts(contract)

            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            if not bars:
                logger.warning(f"No data returned for {ticker}")
                return None

            # Convert to DataFrame
            df = util.df(bars)
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            df = df.set_index('Date')

            logger.info(f"✅ {ticker}: {len(df)} bars retrieved")
            return df

        except Exception as e:
            logger.error(f"❌ Error fetching {ticker}: {e}")
            return None

    def get_multiple_etfs(
        self,
        tickers: List[str],
        duration: str = '2 Y',
        bar_size: str = '1 day',
        save_dir: Optional[str] = None
    ) -> dict:
        """
        Fetch historical data for multiple ETFs.

        Parameters
        ----------
        tickers : list
            List of ETF tickers
        duration : str
            Historical data duration
        bar_size : str
            Bar size
        save_dir : str, optional
            Directory to save CSV files

        Returns
        -------
        dict
            Dictionary mapping tickers to DataFrames
        """
        results = {}
        successful = 0
        failed = 0

        logger.info(f"Fetching data for {len(tickers)} ETFs...")

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] Fetching {ticker}...")

            df = self.get_etf_historical_data(ticker, duration, bar_size)

            if df is not None:
                results[ticker] = df
                successful += 1

                # Save to CSV if requested
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    filepath = os.path.join(save_dir, f"{ticker}.csv")
                    df.to_csv(filepath)
                    logger.info(f"  Saved to {filepath}")
            else:
                failed += 1

            # Rate limiting - IB has pacing restrictions
            self.ib.sleep(0.5)  # 500ms delay between requests

        logger.info(f"\n✅ Complete: {successful} successful, {failed} failed")
        return results

    def get_account_info(self) -> dict:
        """Get account information."""
        try:
            account = self.ib.managedAccounts()[0]
            summary = self.ib.accountSummary(account)

            info = {}
            for item in summary:
                info[item.tag] = {
                    'value': item.value,
                    'currency': item.currency
                }

            return info

        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Example usage
if __name__ == "__main__":
    # Connect to IB
    with IBDataProvider(port=7497) as ib_provider:
        # Fetch single ETF
        spy_data = ib_provider.get_etf_historical_data('SPY', duration='1 Y')
        print(spy_data.tail())

        # Fetch multiple ETFs
        etf_list = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
        data = ib_provider.get_multiple_etfs(
            etf_list,
            duration='2 Y',
            save_dir='data/raw/prices'
        )

        print(f"\nFetched {len(data)} ETFs")
```

## Integration with ETFTrader

### Step 1: Create IB Data Collection Script

Create `scripts/collect_etf_data_from_ib.py`:

```python
"""Collect ETF data from Interactive Brokers."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.ib_data_provider import IBDataProvider
import logging

logging.basicConfig(level=logging.INFO)

# Load ETF universe
universe_file = project_root / "data" / "raw" / "etf_universe.csv"
etf_universe = pd.read_csv(universe_file)
tickers = etf_universe['ticker'].tolist()

# Fetch from IB
with IBDataProvider(port=7497) as ib:
    data = ib.get_multiple_etfs(
        tickers=tickers,
        duration='5 Y',
        bar_size='1 day',
        save_dir=str(project_root / "data" / "raw" / "prices")
    )

print(f"✅ Downloaded {len(data)} ETFs from Interactive Brokers")
```

### Step 2: Update Data Collection Pipeline

Modify your existing data collection to prioritize IB data:

```python
def collect_etf_data_hybrid(tickers: List[str]):
    """
    Hybrid data collection: IB first, fallback to yfinance.

    Parameters
    ----------
    tickers : list
        List of ETF tickers to fetch
    """
    # Try IB first
    ib_data = {}
    failed_tickers = []

    try:
        with IBDataProvider(port=7497) as ib:
            ib_data = ib.get_multiple_etfs(tickers, duration='5 Y')
    except Exception as e:
        logger.warning(f"IB connection failed: {e}, falling back to yfinance")
        failed_tickers = tickers

    # Fallback to yfinance for failed tickers
    if failed_tickers:
        logger.info(f"Fetching {len(failed_tickers)} tickers from yfinance...")
        import yfinance as yf
        for ticker in failed_tickers:
            try:
                data = yf.download(ticker, period='5y', progress=False)
                # Save data...
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")

    return ib_data
```

## Rate Limiting and Best Practices

### IB Pacing Violations

Interactive Brokers has strict rate limits:
- **Historical data**: 60 requests per 600 seconds (1 per 10 seconds)
- **Market data**: 100 requests per second

**To avoid pacing violations:**

```python
import time

for ticker in tickers:
    data = ib_provider.get_etf_historical_data(ticker)
    time.sleep(10)  # Wait 10 seconds between requests
```

### Concurrent Connections

- Maximum 32 simultaneous connections per account
- Use one connection per trading system/strategy
- Avoid multiple connections from same machine

## Troubleshooting

### Common Issues

1. **"Failed to connect"**
   - Ensure TWS or IB Gateway is running
   - Check API settings are enabled
   - Verify port number (7497 paper, 7496 live)
   - Add 127.0.0.1 to trusted IPs

2. **"No market data permissions"**
   - Check `Account > Settings > Market Data Subscriptions`
   - Subscribe to required data feeds
   - May take 24 hours to activate after subscribing

3. **"Pacing violation"**
   - You're requesting data too quickly
   - Add delays between requests (10 seconds for historical data)
   - Use `ib.sleep(10)` between calls

4. **"No security definition found"**
   - Invalid ticker symbol
   - ETF may not be traded on specified exchange
   - Try 'SMART' exchange for automatic routing

## Cost Considerations

### Market Data Fees

| Subscription | Monthly Cost | Coverage |
|:-------------|-------------:|:---------|
| US Securities Snapshot Bundle | $10 | Most US ETFs (delayed) |
| US Equity Streaming Bundle | $4.50 | Real-time US ETFs |
| NYSE (Network A) | $4.50 | NYSE-listed securities |
| NASDAQ (UTP) | $4.50 | NASDAQ-listed securities |

### Recommendations

- **For ETFTrader**: US Securities Snapshot Bundle ($10/month) is sufficient
- **For real-time trading**: Add streaming bundle (+$4.50/month)
- **Total cost**: $10-15/month vs. free yfinance

## Comparison: IB vs. yfinance

| Feature | Interactive Brokers | yfinance |
|:--------|:-------------------:|:--------:|
| **Cost** | $10-15/month | Free |
| **Data Quality** | ✅ Exchange-grade | ⚠️ Best-effort |
| **Historical Data** | ✅ Clean, no gaps | ⚠️ Some gaps |
| **Real-time** | ✅ Yes (with subscription) | ❌ 15-min delay |
| **API Limits** | ⚠️ 60 req/10min | ⚠️ Rate limited |
| **Reliability** | ✅ High (99.99%) | ⚠️ Can break |
| **Setup Complexity** | ⚠️ Requires TWS/Gateway | ✅ Simple |

### Recommendation

**For ETFTrader:**
- **Development/Testing**: Use yfinance (free, simple)
- **Production Trading**: Use Interactive Brokers (reliable, accurate)
- **Hybrid Approach**: IB primary, yfinance fallback

## Next Steps

1. ✅ Enable API access in TWS (File > Global Configuration > API)
2. ✅ Install ib_insync: `pip install ib_insync`
3. ✅ Test connection with example code above
4. ✅ Create `src/data_collection/ib_data_provider.py` module
5. ✅ Update ETFTrader data collection pipeline
6. ✅ Monitor for pacing violations and adjust delays

## Support Resources

- **IB API Documentation**: https://interactivebrokers.github.io/tws-api/
- **ib_insync Documentation**: https://ib-insync.readthedocs.io/
- **IB API Support**: https://www.interactivebrokers.com/en/support/api.php
- **TWS API Forum**: https://groups.io/g/twsapi

---

*Last Updated: 2025-10-07*
*ETFTrader Version: 5.1*

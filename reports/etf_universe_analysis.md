# ETF Universe Analysis - Expansion Strategy

**Date:** 2025-10-05
**Current Universe:** 298 ETFs (with price data)
**Target Universe:** 2,000-4,000 ETFs
**Gap:** ~1,700-3,700 ETFs missing

---

## Current Situation

### What We Have:
- **298 ETFs** with successfully downloaded price data
- **424 ETF tickers** hardcoded in scraper
- **Bottleneck:** Using manually curated list instead of comprehensive data source

### Why So Few ETFs:

1. **Hardcoded List Limitation**
   - Current scraper (`etf_scraper.py`) uses a manually curated list of ~424 major ETFs
   - This is NOT a comprehensive universe - just popular/large-cap ETFs
   - Missing: Sector ETFs, thematic ETFs, international ETFs, smart-beta ETFs, etc.

2. **Data Download Failures**
   - Started with 424 tickers
   - Only 298 successfully downloaded and saved price data
   - ~126 ETFs failed (likely due to: delisted, data quality, yfinance issues)

3. **No Dynamic Discovery**
   - Not scraping from comprehensive ETF databases
   - Not using ETF screeners
   - Not accessing exchange listings

---

## Available Data Sources for Expansion

### 1. **ETF Database (etfdb.com)** - FREE ⭐ RECOMMENDED
   - **Comprehensive:** ~3,000+ US-listed ETFs
   - **Access:** Web scraping or unofficial API
   - **Data:** Ticker, name, AUM, expense ratio, category, holdings
   - **Coverage:** Best free source for US ETFs
   - **Limitation:** Rate limiting, may need scraping delays

### 2. **Yahoo Finance ETF Screener** - FREE
   - **Coverage:** ~2,500+ ETFs
   - **Access:** Can be scraped or accessed via yfinance
   - **Data:** Basic info, price data (already using for prices)
   - **Pro:** Already integrated with yfinance
   - **Con:** Screener results limited, need multiple queries

### 3. **Nasdaq ETF List** - FREE
   - **Coverage:** ~2,000+ ETFs listed on Nasdaq
   - **Access:** https://www.nasdaq.com/market-activity/etf/screener
   - **Format:** CSV download available
   - **Pro:** Official exchange data
   - **Con:** Only Nasdaq listings (missing NYSE Arca, CBOE)

### 4. **NYSE Arca ETF List** - FREE
   - **Coverage:** ~1,500+ ETFs
   - **Access:** https://www.nyse.com/listings_directory/etf
   - **Format:** Web scraping or data download
   - **Pro:** Official exchange data (largest ETF venue)
   - **Con:** Separate from Nasdaq

### 5. **CBOE ETF List** - FREE
   - **Coverage:** ~500+ ETFs
   - **Access:** https://www.cboe.com/us/etf/
   - **Pro:** Official exchange data
   - **Con:** Smaller coverage

### 6. **Alpha Vantage** - FREE (limited) / PAID
   - **Coverage:** Comprehensive ETF data
   - **Access:** API (500 calls/day free, unlimited paid)
   - **Data:** Fundamentals, price data
   - **Pro:** Structured API
   - **Con:** Rate limits on free tier

### 7. **Polygon.io** - PAID ($199/mo)
   - **Coverage:** Full universe of ETFs
   - **Access:** REST API + WebSocket
   - **Data:** Real-time + historical
   - **Pro:** Professional-grade, comprehensive
   - **Con:** Cost

### 8. **IEX Cloud** - FREE (limited) / PAID
   - **Coverage:** ~2,500 ETFs
   - **Access:** REST API (50k msg/month free)
   - **Data:** Price, fundamentals, reference data
   - **Pro:** Good free tier
   - **Con:** Limited free tier

---

## Recommended Expansion Strategy

### **Phase 1: Immediate (Free Sources - Target: 2,500+ ETFs)**

1. **Scrape ETF Database (etfdb.com)**
   - Most comprehensive free source
   - Pages to scrape:
     - Full ETF list: https://etfdb.com/etfs/
     - Can paginate through all results
     - Extract: ticker, name, AUM, expense ratio, category, inception date

2. **Combine Exchange Listings**
   - Nasdaq: CSV download
   - NYSE Arca: Web scraping
   - CBOE: API or scraping
   - **Benefit:** Official, accurate, no delisted ETFs

3. **Filter & Deduplicate**
   - Remove leveraged (already have function)
   - Remove inverse ETFs
   - Remove low AUM (<$10M) - likely to be delisted
   - Remove very new ETFs (<6 months) - insufficient history
   - Deduplicate across sources

### **Phase 2: Data Collection (Use yfinance - FREE)**
- Continue using yfinance for price data (already working)
- Implement batch downloading with error handling
- Parallel downloads (10-20 threads) for speed
- Retry logic for failed downloads
- Expected: ~2,000-2,500 ETFs with 3+ years data

### **Phase 3: Enhanced (Optional - If Needed)**
- If need >3,000 ETFs: Add Alpha Vantage API (free tier)
- If need better data quality: Consider IEX Cloud
- If professional use: Polygon.io or similar

---

## Implementation Plan

### Task 1: Expand ETF Universe Collection

**New Scraper Functions Needed:**

```python
# src/data_collection/etf_universe_builder.py

class ComprehensiveETFScraper:

    def scrape_etfdb_com(self) -> pd.DataFrame:
        """Scrape complete ETF list from ETF Database"""
        # Scrape all pages from etfdb.com
        # Return: ticker, name, AUM, expense_ratio, category, inception

    def scrape_nasdaq_listings(self) -> pd.DataFrame:
        """Download Nasdaq ETF listings"""
        # Get CSV from Nasdaq screener
        # Return: ticker, name, exchange

    def scrape_nyse_arca(self) -> pd.DataFrame:
        """Scrape NYSE Arca ETF listings"""
        # Scrape NYSE ETF directory
        # Return: ticker, name, exchange

    def merge_and_deduplicate(self, sources: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple sources and remove duplicates"""
        # Merge on ticker
        # Keep best data from each source
        # Remove duplicates

    def filter_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to create final universe"""
        # Remove leveraged/inverse
        # Remove low AUM (<$10M)
        # Remove new ETFs (<6 months old)
        # Remove non-USD
        # Return filtered list
```

### Task 2: Parallel Data Download

```python
# src/data_collection/parallel_downloader.py

class ParallelETFDownloader:

    def download_batch(self, tickers: List[str], max_workers: int = 20):
        """Download price data for multiple ETFs in parallel"""
        # Use ThreadPoolExecutor
        # Download with retries
        # Save successful downloads
        # Log failures

    def validate_data_quality(self, ticker: str, df: pd.DataFrame) -> bool:
        """Check if downloaded data meets quality requirements"""
        # Check: minimum 2 years data
        # Check: <10% missing data
        # Check: no obvious data errors
        # Return: True if passes
```

### Task 3: Incremental Updates

```python
def update_universe_incremental():
    """Add new ETFs to existing universe"""
    # Load existing universe
    # Scrape latest listings
    # Find new ETFs not in current universe
    # Download data for new ETFs only
    # Append to universe
```

---

## Expected Results After Expansion

| Metric | Current | After Expansion | Target |
|--------|---------|-----------------|--------|
| **Total ETFs Scraped** | 424 | 3,000+ | 2,000-4,000 |
| **ETFs with Price Data** | 298 | 2,000-2,500 | 2,000+ |
| **Data Success Rate** | 70% | 75-80% | 75%+ |
| **Coverage** | Large-cap only | Full spectrum | Comprehensive |

### Coverage by Category (After Expansion):

- **Equity ETFs:** ~1,500 (vs. current ~150)
- **Fixed Income ETFs:** ~400 (vs. current ~40)
- **Commodity ETFs:** ~100 (vs. current ~20)
- **International ETFs:** ~500 (vs. current ~50)
- **Sector/Thematic ETFs:** ~400 (vs. current ~30)
- **Smart Beta ETFs:** ~300 (vs. current ~8)

---

## Filtering Philosophy

### Include:
✅ Long-only ETFs
✅ Any asset class (equity, fixed income, commodity, currency, etc.)
✅ Any geography
✅ Any sector/theme
✅ AUM > $10M
✅ Inception > 6 months ago
✅ USD-denominated
✅ Sufficient liquidity (>$100k average daily volume)

### Exclude:
❌ Leveraged ETFs (2x, 3x)
❌ Inverse ETFs (short, bear)
❌ Low AUM (<$10M) - delisting risk
❌ Very new (<6 months) - insufficient data
❌ Non-USD denominated
❌ Very illiquid (<$100k ADV)

---

## Timeline

**Immediate (Today):**
1. Create comprehensive scraper (2-3 hours)
2. Scrape ETF Database + exchanges (1 hour)
3. Merge and filter (30 mins)

**Tomorrow:**
4. Download price data with parallel processing (4-6 hours for 2,000 ETFs)
5. Validate data quality (1 hour)
6. Generate universe statistics (30 mins)

**Result:** 2,000-2,500 ETF universe ready for portfolio optimization

---

## Cost Analysis

| Approach | Cost | ETFs | Notes |
|----------|------|------|-------|
| **Recommended: ETF Database + Exchanges + yfinance** | **$0** | **~2,500** | Best value |
| Alpha Vantage (premium) | $50/mo | ~3,000 | Better data quality |
| IEX Cloud (paid) | $9-79/mo | ~2,500 | Structured API |
| Polygon.io | $199/mo | ~3,000+ | Professional grade |

**Recommendation:** Start with free sources (should get 2,000-2,500 ETFs), only upgrade if needed.

---

## Next Steps

Would you like me to:

1. ✅ **Implement comprehensive ETF scraper** (recommended - I can do this now)
2. ✅ **Add parallel download capability** (will speed up 10x)
3. ⏸️ **Integrate paid API** (defer unless free sources insufficient)

I can have a 2,000-2,500 ETF universe ready within 6-8 hours of processing time.

**Question for you:**
- Should I proceed with implementing the comprehensive scraper now?
- Are there specific ETF categories you want to prioritize (e.g., sector, international, thematic)?
- Is 2,000-2,500 ETFs sufficient, or do you need the full 4,000?

# Web Application Plan - AQR Multi-Factor Strategy

## Overview

**Purpose**: Interactive web dashboard to review recommendations, enter trades manually, and monitor portfolio performance.

**Timeline**: Weeks 7-8 (after strategy validation complete)

**Future Enhancement**: Interactive Brokers integration for automated trading (Week 9+)

---

## Phase 1: Core Dashboard (Week 7)

### Tech Stack

**Backend**:
- **FastAPI** - Modern Python web framework
  - Fast, async, type hints
  - Automatic API documentation
  - Easy integration with existing Python code

**Frontend**:
- **Streamlit** - Rapid dashboard development
  - Pure Python (no JS required)
  - Built-in charting (Plotly)
  - Interactive widgets
  - Fast iteration

**Alternative** (if more customization needed):
- Backend: FastAPI
- Frontend: React + Recharts
- State: Redux

**Database**:
- **SQLite** - Simple, embedded
  - For portfolio state, trades, performance history
  - Easy backup/restore
  - No separate server needed

---

## Features: Week 7 (Core Dashboard)

### 1. Daily Recommendations View

**What You See**:
```
┌─────────────────────────────────────────────────────┐
│  📊 Portfolio Recommendations - October 8, 2025     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Current Portfolio Value: $1,234,567                │
│  Cash Available: $50,000                            │
│  Positions: 18 / 20 max                             │
│                                                      │
│  🔄 Rebalance Recommended: YES (drift 6.2%)         │
│                                                      │
├─────────────────────────────────────────────────────┤
│  SUGGESTED TRADES                                    │
├─────────────────────────────────────────────────────┤
│  BUY  │ VTI   │ 120 shares │ $12,000 │ Reason: ↑   │
│  BUY  │ QQQM  │ 80 shares  │ $8,000  │ New Entry   │
│  SELL │ XLE   │ 50 shares  │ $5,000  │ Stop-Loss   │
│  TRIM │ GLD   │ 30 shares  │ $3,000  │ Rebalance   │
│                                                      │
│  📈 Expected Impact: +0.3% portfolio weight adj     │
│  💰 Est. Transaction Cost: $45                      │
│                                                      │
│  [Review Factor Scores] [Execute Trades]            │
└─────────────────────────────────────────────────────┘
```

**Implementation**:
```python
# Backend: FastAPI endpoint
@app.get("/recommendations")
async def get_recommendations():
    # Run factor calculations
    factor_scores = calculate_all_factors()

    # Get current portfolio
    current = get_portfolio_state()

    # Generate target weights
    target = optimizer.optimize(factor_scores)

    # Calculate trades needed
    trades = rebalancer.generate_trades(current, target)

    return {
        "portfolio_value": current.value,
        "cash": current.cash,
        "positions": current.positions,
        "needs_rebalance": rebalancer.should_rebalance(),
        "trades": trades,
        "expected_cost": calculate_cost(trades)
    }
```

```python
# Frontend: Streamlit
import streamlit as st

st.title("📊 Portfolio Recommendations")

# Fetch recommendations
recs = fetch_recommendations()

# Display summary
col1, col2, col3 = st.columns(3)
col1.metric("Portfolio Value", f"${recs['portfolio_value']:,.0f}")
col2.metric("Cash", f"${recs['cash']:,.0f}")
col3.metric("Positions", f"{len(recs['positions'])} / 20")

# Show rebalance indicator
if recs['needs_rebalance']:
    st.warning("🔄 Rebalance Recommended")
else:
    st.success("✅ Portfolio Aligned")

# Display trades
st.subheader("Suggested Trades")
trades_df = pd.DataFrame(recs['trades'])
st.dataframe(trades_df, use_container_width=True)

# Action buttons
if st.button("Execute Trades"):
    show_trade_entry_form()
```

### 2. Manual Trade Entry

**Interface**:
```
┌─────────────────────────────────────────────────────┐
│  ✍️  Manual Trade Entry                             │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Date: [2025-10-08] [Today]                         │
│                                                      │
│  Action: ( ) Buy  (•) Sell                          │
│                                                      │
│  Ticker: [VTI    ]  [Search]                        │
│                                                      │
│  Shares: [120    ]                                  │
│                                                      │
│  Price:  [$245.67]  (auto-filled from market)      │
│                                                      │
│  Commission: [$1.00]  (default)                     │
│                                                      │
│  Notes: [Following rebalance recommendation]        │
│                                                      │
│  Total: $29,481.00                                  │
│                                                      │
│  [Cancel]  [Save Trade]                             │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Database Schema**:
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    action VARCHAR(4) NOT NULL,  -- BUY, SELL
    shares INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    commission DECIMAL(10, 2) DEFAULT 1.00,
    total DECIMAL(10, 2) NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE portfolio_state (
    id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    shares INTEGER NOT NULL,
    cost_basis DECIMAL(10, 2) NOT NULL,
    current_price DECIMAL(10, 2),
    current_value DECIMAL(10, 2),
    weight DECIMAL(5, 4),
    UNIQUE(date, ticker)
);
```

### 3. Factor Score Explorer

**View Factor Breakdown**:
```
┌─────────────────────────────────────────────────────┐
│  🔍 Factor Scores - October 8, 2025                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ETF: VTI (Vanguard Total Stock Market)             │
│                                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                      │
│  Momentum:     ████████████░░░░░  0.82 (Top 18%)   │
│  Quality:      ██████████████░░░  0.91 (Top 9%)    │
│  Value:        ████████░░░░░░░░░  0.56 (Top 44%)   │
│  Volatility:   ████████████████░  0.98 (Top 2%)    │
│                                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                      │
│  Composite:    ██████████████░░░  0.87 (Top 13%)   │
│                                                      │
│  [View Details] [Compare ETFs]                      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Comparison View**:
```
Compare ETFs:  [VTI]  [VOO]  [QQQM]  [Add ETF]

                VTI    VOO    QQQM
Momentum:      0.82   0.79   0.94
Quality:       0.91   0.88   0.72
Value:         0.56   0.54   0.21
Volatility:    0.98   0.97   0.68
──────────────────────────────────
Composite:     0.87   0.84   0.69
Rank:          #12    #18    #45
```

---

## Features: Week 8 (Performance Monitoring)

### 1. Portfolio Dashboard

**Overview**:
```
┌─────────────────────────────────────────────────────┐
│  📈 Portfolio Performance                            │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [Chart: Portfolio Value Over Time]                 │
│  $1.5M ┤                                  •          │
│  $1.4M ┤                             •               │
│  $1.3M ┤                        •                    │
│  $1.2M ┤                   •                         │
│  $1.1M ┤              •                              │
│  $1.0M ┤         •                                   │
│        └─────────────────────────────────────────   │
│         Jan   Feb   Mar   Apr   May   Jun   Jul     │
│                                                      │
├─────────────────────────────────────────────────────┤
│  METRICS                                             │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Total Return:    +23.4%     CAGR:        15.2%    │
│  Sharpe Ratio:     0.94      Sortino:      1.18    │
│  Max Drawdown:    -8.3%      Win Rate:     64%     │
│  Volatility:      12.1%      Beta:          0.87   │
│                                                      │
│  📊 vs S&P 500:   +5.1%      (Outperforming)       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 2. Holdings View

**Current Positions**:
```
┌─────────────────────────────────────────────────────┐
│  💼 Current Holdings - 18 Positions                 │
├─────────────────────────────────────────────────────┤
│ Ticker │Shares│Cost Basis│ Current │ Value  │Weight│
├────────┼──────┼──────────┼─────────┼────────┼──────┤
│ VTI    │ 520  │ $127,450 │$245.67  │$127,748│ 10.4%│
│ QQQM   │ 380  │  $46,740 │ $98.32  │ $37,361│  8.9%│
│ GLD    │ 180  │  $32,400 │$185.20  │ $33,336│  7.2%│
│ SCHD   │ 450  │  $23,850 │ $28.90  │ $13,005│  6.8%│
│ ...    │      │          │         │        │      │
├────────┼──────┼──────────┼─────────┼────────┼──────┤
│ TOTAL  │      │$972,450  │         │$1,234K │100.0%│
└─────────────────────────────────────────────────────┘

[Export to CSV] [View Stop-Loss Distances] [Rebalance History]
```

### 3. Trade History

**Trade Log**:
```
┌─────────────────────────────────────────────────────┐
│  📋 Trade History                                    │
├─────────────────────────────────────────────────────┤
│ Date       │ Action │ Ticker │ Shares │ Price │Total│
├────────────┼────────┼────────┼────────┼───────┼─────┤
│ 2025-10-08 │ BUY    │ VTI    │ 120    │$245.67│$29K │
│ 2025-10-08 │ SELL   │ XLE    │ 50     │ $98.50│ $5K │
│ 2025-10-01 │ BUY    │ QQQM   │ 80     │$120.45│$10K │
│ 2025-09-24 │ TRIM   │ GLD    │ 30     │$182.00│ $5K │
│ ...        │        │        │        │       │     │
└─────────────────────────────────────────────────────┘

Filters: [Last 30 days ▼] [All Actions ▼] [All Tickers ▼]
```

### 4. Performance Attribution

**What's Working**:
```
┌─────────────────────────────────────────────────────┐
│  🎯 Performance Attribution - YTD                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Factor Contribution:                               │
│                                                      │
│  Momentum:     +8.2%  ███████████████░░░░░         │
│  Quality:      +4.1%  ████████░░░░░░░░░░░          │
│  Volatility:   +2.8%  ██████░░░░░░░░░░░░░          │
│  Value:        +1.5%  ███░░░░░░░░░░░░░░░░          │
│  Selection:    -0.8%  ░░░░░░░░░░░░░░░░░░░          │
│                                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Total Alpha:  +15.8%                               │
│  Market:       +10.2%                               │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Total Return: +26.0%                               │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Directory Structure (Web App)

```
ETFTrader/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app
│   ├── api/
│   │   ├── __init__.py
│   │   ├── recommendations.py   # GET /recommendations
│   │   ├── trades.py            # POST /trades, GET /trades
│   │   ├── portfolio.py         # GET /portfolio
│   │   └── factors.py           # GET /factors/{ticker}
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py            # SQLAlchemy models
│   │   └── session.py           # DB connection
│   ├── services/
│   │   ├── __init__.py
│   │   ├── portfolio_service.py # Portfolio calculations
│   │   └── trade_service.py     # Trade execution
│   └── streamlit_app.py         # Streamlit dashboard
├── database/
│   └── portfolio.db             # SQLite database
└── requirements.txt             # Add: fastapi, streamlit, sqlalchemy
```

---

## Implementation Plan

### Week 7 Tasks

**Day 1-2: Backend API**
- [ ] Setup FastAPI application
- [ ] Create database models (trades, portfolio_state)
- [ ] Implement `/recommendations` endpoint
- [ ] Implement `/trades` CRUD endpoints
- [ ] Unit tests for API

**Day 3-4: Dashboard**
- [ ] Setup Streamlit app
- [ ] Build recommendations view
- [ ] Build manual trade entry form
- [ ] Build factor score explorer

**Day 5: Integration & Testing**
- [ ] Connect frontend to backend
- [ ] Test end-to-end workflow
- [ ] Error handling
- [ ] Documentation

### Week 8 Tasks

**Day 1-2: Performance Monitoring**
- [ ] Portfolio dashboard with charts
- [ ] Performance metrics calculation
- [ ] Holdings view with real-time prices

**Day 3-4: Advanced Features**
- [ ] Trade history with filtering
- [ ] Performance attribution by factor
- [ ] Export functionality (CSV, PDF)

**Day 5: Polish & Deploy**
- [ ] UI refinements
- [ ] Mobile responsiveness (Streamlit auto-handles)
- [ ] Deploy to local server or cloud
- [ ] User documentation

---

## Phase 2: Interactive Brokers Integration (Week 9+)

### Future Enhancements

**Data Feed**:
- Replace yfinance with IB real-time data
- More accurate pricing
- Lower latency

**Automated Trading**:
- One-click trade execution via IB API
- Order status tracking
- Fill confirmation

**Implementation**:
```python
from ib_insync import IB, Stock, MarketOrder

class IBTradeExecutor:
    def __init__(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)

    def execute_trade(self, ticker: str, action: str, shares: int):
        """Execute trade via Interactive Brokers."""
        contract = Stock(ticker, 'SMART', 'USD')
        order = MarketOrder(action, shares)

        trade = self.ib.placeOrder(contract, order)

        # Wait for fill
        while not trade.isDone():
            self.ib.sleep(1)

        return {
            'status': trade.orderStatus.status,
            'filled': trade.orderStatus.filled,
            'avg_price': trade.orderStatus.avgFillPrice
        }
```

---

## Technology Choices: Why?

### FastAPI vs Flask
- **FastAPI**: Modern, async, auto docs, type hints ✅
- **Flask**: Mature but older, sync only

### Streamlit vs React
**Phase 1: Streamlit** ✅
- Pure Python (no JS!)
- Rapid development (days not weeks)
- Built-in charts
- Interactive widgets
- Perfect for internal tools

**Phase 2: React** (if needed)
- More customization
- Better mobile support
- Separate frontend/backend
- Use only if Streamlit limitations hit

### SQLite vs PostgreSQL
**Phase 1: SQLite** ✅
- Embedded, no separate server
- Easy backup (one file)
- Sufficient for single user
- Can migrate to Postgres later if needed

---

## Sample Code: Complete Mini Example

```python
# app/main.py
from fastapi import FastAPI
from app.api import recommendations, trades

app = FastAPI(title="ETF Portfolio Manager")

app.include_router(recommendations.router, prefix="/api")
app.include_router(trades.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "ETF Portfolio Manager API"}
```

```python
# app/api/recommendations.py
from fastapi import APIRouter
from src.factors import calculate_all_factors
from src.portfolio import SimplePortfolioOptimizer

router = APIRouter()

@router.get("/recommendations")
def get_recommendations():
    # Calculate factors
    factors = calculate_all_factors()

    # Optimize
    optimizer = SimplePortfolioOptimizer()
    target = optimizer.optimize(factors)

    # Get current state
    current = get_current_portfolio()

    # Generate trades
    trades = generate_trades(current, target)

    return {
        "date": datetime.now().isoformat(),
        "current_value": current['value'],
        "trades": trades,
        "expected_impact": calculate_impact(trades)
    }
```

```python
# app/streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="ETF Portfolio", layout="wide")

st.title("📊 ETF Portfolio Manager")

# Fetch recommendations
recs = requests.get("http://localhost:8000/api/recommendations").json()

# Display
col1, col2 = st.columns(2)
col1.metric("Portfolio Value", f"${recs['current_value']:,.0f}")
col2.metric("Trades", len(recs['trades']))

# Show trades
st.subheader("Recommended Trades")
for trade in recs['trades']:
    with st.expander(f"{trade['action']} {trade['ticker']}"):
        st.write(f"Shares: {trade['shares']}")
        st.write(f"Amount: ${trade['amount']:,.0f}")
        st.write(f"Reason: {trade['reason']}")

        if st.button(f"Execute {trade['ticker']}", key=trade['ticker']):
            # Post trade
            response = requests.post(
                "http://localhost:8000/api/trades",
                json=trade
            )
            st.success("Trade recorded!")
```

---

## Running the App

### Development
```bash
# Terminal 1: Start FastAPI backend
uvicorn app.main:app --reload --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run app/streamlit_app.py --server.port 8501
```

### Access
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (auto-generated!)
- **Dashboard**: http://localhost:8501

---

## Summary

**Week 7**: Core dashboard for viewing recommendations and manual trade entry
**Week 8**: Performance monitoring and portfolio tracking
**Week 9+**: Interactive Brokers integration (future)

**Tech**: FastAPI + Streamlit + SQLite = Rapid development, pure Python, easy deployment

**Timeline**: 2 weeks to production-ready web interface after strategy validation complete.

---

*Ready to build the web app once backtesting validates the strategy!*

# ETF Trader Web Application - Architecture Design

**Version**: 1.0
**Date**: 2025-10-10
**Status**: Week 8 - Design Phase

---

## Executive Summary

This document outlines the architecture for the ETF Trader web application, which will enable:
- **Paper trading simulation** and performance monitoring
- **Real-time portfolio monitoring** with factor analysis
- **Automated portfolio generation** and rebalancing alerts
- **Foundation for Interactive Brokers integration**

### Design Philosophy
- **Start Simple**: Build core features first, add complexity gradually
- **Paper Trading First**: Simulate before executing real trades
- **Prepare for IB**: Design with Interactive Brokers integration in mind
- **Responsive & Fast**: Real-time updates, low latency

---

## Technology Stack

### Backend
- **Framework**: **FastAPI** (Python async framework)
  - Why: Fast, modern, async support, automatic OpenAPI docs
  - Integrates seamlessly with existing Python codebase
  - Native async for real-time updates

- **Database**: **PostgreSQL**
  - Why: ACID compliance, time-series support, mature
  - Tables: portfolios, positions, trades, factor_scores, prices

- **Cache**: **Redis**
  - Why: Fast in-memory cache for real-time data
  - Use for: Latest prices, factor scores, session data

- **Task Queue**: **Celery** (optional for Phase 2)
  - Why: Background tasks (data updates, portfolio generation)
  - Initially: Run scripts manually, add Celery in Phase 2

### Frontend
- **Framework**: **React** with **TypeScript**
  - Why: Component-based, strong typing, large ecosystem

- **UI Library**: **Ant Design** or **Material-UI**
  - Why: Professional components, charts, tables

- **Charts**: **Plotly.js** or **Recharts**
  - Why: Interactive, matches notebook visualizations

- **State Management**: **React Query** + **Zustand**
  - Why: Server state (React Query) + client state (Zustand)

### Deployment
- **Development**: Docker Compose (all services locally)
- **Production** (Future):
  - Backend: Docker on AWS ECS / DigitalOcean
  - Database: AWS RDS PostgreSQL
  - Frontend: Vercel / Netlify
  - Cache: AWS ElastiCache Redis

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Frontend (React)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Dashboard   │  │  Portfolio   │  │  Backtest    │          │
│  │    View      │  │   Manager    │  │   Results    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ REST API / WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      API Routes                           │  │
│  │  /portfolios  /positions  /trades  /factors  /backtest   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Business Logic                          │  │
│  │  Portfolio Service │ Trading Engine │ Factor Calculator   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Existing Python Code                      │  │
│  │  src/factors  │  src/portfolio  │  src/backtesting       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │   PostgreSQL     │        │      Redis       │
    │   (Persistent)   │        │     (Cache)      │
    └──────────────────┘        └──────────────────┘

                              ▼
                    ┌──────────────────┐
                    │  Data Sources    │
                    │  - yfinance      │
                    │  - IB (future)   │
                    └──────────────────┘
```

---

## Database Schema

### Core Tables

#### `portfolios`
```sql
CREATE TABLE portfolios (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    optimizer_type VARCHAR(50) NOT NULL,  -- 'mvo', 'rank_based', etc.
    num_positions INTEGER DEFAULT 20,
    initial_capital NUMERIC(15, 2) NOT NULL,
    current_value NUMERIC(15, 2),
    is_paper_trading BOOLEAN DEFAULT true,
    status VARCHAR(20) DEFAULT 'active',  -- 'active', 'closed'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### `positions`
```sql
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id),
    ticker VARCHAR(10) NOT NULL,
    target_weight NUMERIC(6, 4),  -- 0.0500 = 5%
    current_weight NUMERIC(6, 4),
    shares NUMERIC(12, 4),
    entry_price NUMERIC(10, 2),
    current_price NUMERIC(10, 2),
    unrealized_pnl NUMERIC(12, 2),
    stop_loss_price NUMERIC(10, 2),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(portfolio_id, ticker)
);
```

#### `trades`
```sql
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id),
    ticker VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'buy', 'sell'
    quantity NUMERIC(12, 4) NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    total_value NUMERIC(15, 2) NOT NULL,
    commission NUMERIC(10, 2) DEFAULT 0,
    slippage NUMERIC(10, 2) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'executed', 'failed'
    executed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_trades_portfolio ON trades(portfolio_id);
CREATE INDEX idx_trades_ticker ON trades(ticker);
```

#### `portfolio_values`
```sql
CREATE TABLE portfolio_values (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id),
    date DATE NOT NULL,
    total_value NUMERIC(15, 2) NOT NULL,
    cash NUMERIC(15, 2) NOT NULL,
    invested_value NUMERIC(15, 2) NOT NULL,
    daily_return NUMERIC(8, 6),
    cumulative_return NUMERIC(10, 6),

    UNIQUE(portfolio_id, date)
);

CREATE INDEX idx_portfolio_values_date ON portfolio_values(date);
```

#### `factor_scores`
```sql
CREATE TABLE factor_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    momentum NUMERIC(10, 6),
    quality NUMERIC(10, 6),
    value NUMERIC(10, 6),
    volatility NUMERIC(10, 6),
    composite NUMERIC(10, 6),

    UNIQUE(ticker, date)
);

CREATE INDEX idx_factor_scores_date ON factor_scores(date);
CREATE INDEX idx_factor_scores_ticker ON factor_scores(ticker);
```

#### `etf_prices`
```sql
CREATE TABLE etf_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open NUMERIC(10, 2),
    high NUMERIC(10, 2),
    low NUMERIC(10, 2),
    close NUMERIC(10, 2),
    adj_close NUMERIC(10, 2),
    volume BIGINT,

    UNIQUE(ticker, date)
);

CREATE INDEX idx_prices_ticker_date ON etf_prices(ticker, date);
```

#### `rebalance_events`
```sql
CREATE TABLE rebalance_events (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id),
    date DATE NOT NULL,
    reason VARCHAR(100),  -- 'drift_threshold', 'stop_loss', 'scheduled'
    max_drift NUMERIC(6, 4),
    num_trades INTEGER,
    total_cost NUMERIC(10, 2),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## API Design

### Authentication (Phase 2)
Initially: No authentication (local use only)
Future: JWT-based authentication

### REST API Endpoints

#### Portfolio Management

**GET /api/portfolios**
- List all portfolios
- Response: `[{id, name, optimizer_type, current_value, status, ...}]`

**POST /api/portfolios**
- Create new portfolio
- Body: `{name, optimizer_type, num_positions, initial_capital, is_paper_trading}`
- Response: `{id, ...created portfolio}`

**GET /api/portfolios/{id}**
- Get portfolio details
- Response: `{id, name, positions: [...], performance: {...}}`

**DELETE /api/portfolios/{id}**
- Close portfolio
- Sets status to 'closed'

#### Position Management

**GET /api/portfolios/{id}/positions**
- Get current positions for portfolio
- Response: `[{ticker, current_weight, target_weight, unrealized_pnl, ...}]`

**GET /api/portfolios/{id}/drift**
- Calculate current drift from targets
- Response: `{max_drift, total_drift, positions: [{ticker, drift, ...}]}`

**POST /api/portfolios/{id}/rebalance**
- Generate rebalancing recommendations
- Body: `{force: boolean}` (force even if drift < threshold)
- Response: `{trades: [{ticker, side, quantity, ...}], estimated_cost}`

**POST /api/portfolios/{id}/execute-rebalance**
- Execute recommended trades (paper trading)
- Body: `{trades: [...]}`
- Response: `{executed_trades: [...], new_positions: [...]}`

#### Trading

**GET /api/portfolios/{id}/trades**
- Get trade history
- Query params: `?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`
- Response: `[{id, ticker, side, quantity, price, executed_at, ...}]`

**POST /api/trades**
- Manual trade entry (paper trading)
- Body: `{portfolio_id, ticker, side, quantity, price}`
- Response: `{id, status, ...}`

#### Performance & Analytics

**GET /api/portfolios/{id}/performance**
- Get performance metrics
- Query params: `?period=1M|3M|6M|1Y|YTD|ALL`
- Response:
```json
{
  "total_return": 0.085,
  "annualized_return": 0.17,
  "sharpe_ratio": 1.07,
  "max_drawdown": -0.123,
  "win_rate": 0.59,
  "chart_data": [
    {"date": "2025-01-01", "value": 1000000, "return": 0.0},
    ...
  ]
}
```

**GET /api/portfolios/{id}/attribution**
- Factor attribution analysis
- Response:
```json
{
  "by_factor": {
    "momentum": 0.035,
    "quality": 0.028,
    "value": 0.012,
    "volatility": 0.010
  },
  "by_position": [
    {"ticker": "QQQ", "contribution": 0.015},
    ...
  ]
}
```

#### Factor Scores

**GET /api/factors/latest**
- Get latest factor scores for all ETFs
- Response: `[{ticker, momentum, quality, value, volatility, composite, rank}]`

**GET /api/factors/{ticker}**
- Get factor history for specific ETF
- Query params: `?days=90`
- Response:
```json
{
  "ticker": "QQQ",
  "history": [
    {"date": "2025-10-10", "momentum": 0.85, ...},
    ...
  ]
}
```

**GET /api/factors/recommendations**
- Get top N ETFs by composite score
- Query params: `?num_positions=20&optimizer=mvo`
- Response:
```json
{
  "recommendations": [
    {"ticker": "QQQ", "composite_score": 0.92, "target_weight": 0.08},
    ...
  ],
  "generation_date": "2025-10-10T12:00:00Z"
}
```

#### Risk Management

**GET /api/portfolios/{id}/risk**
- Get risk metrics and stop-loss status
- Response:
```json
{
  "vix_current": 18.5,
  "vix_regime": "Normal Volatility",
  "stop_loss_threshold": 0.12,
  "positions_at_risk": [
    {"ticker": "XLK", "current_dd": -0.095, "distance_to_stop": 0.025},
    ...
  ]
}
```

**GET /api/vix/history**
- Get VIX historical data
- Query params: `?days=252`
- Response: `[{date, value, regime}, ...]`

#### Data Management

**POST /api/data/update**
- Trigger data collection and factor calculation
- Background job (returns job_id in Phase 2)
- Response: `{status: "started", job_id: "..."}`

**GET /api/data/status**
- Check data freshness
- Response:
```json
{
  "latest_price_date": "2025-10-09",
  "days_old": 1,
  "num_etfs": 623,
  "latest_factor_date": "2025-10-09"
}
```

#### Backtesting (Read-Only)

**GET /api/backtests**
- List available backtest results
- Response: `[{id, name, period, optimizer, status, ...}]`

**GET /api/backtests/{id}**
- Get detailed backtest results
- Response: Full backtest metrics + chart data

---

## Frontend Pages

### 1. Dashboard (Home)
**Route**: `/`

**Components**:
- Portfolio summary cards (value, return, positions)
- Performance chart (cumulative return)
- Factor score radar chart
- Recent trades table
- Action alerts (rebalancing needed, stop-loss warnings)

### 2. Portfolio Manager
**Route**: `/portfolios`

**Components**:
- Portfolio list (create, view, delete)
- Position table (ticker, weight, drift, P&L, stop-loss distance)
- Target vs current weights chart
- Rebalance recommendations panel
- Manual trade entry form

### 3. Factor Analysis
**Route**: `/factors`

**Components**:
- ETF ranking table (sortable by factor)
- Factor score heatmap
- Time-series charts for selected ETFs
- Top recommendations panel
- Factor correlation matrix

### 4. Performance Analytics
**Route**: `/performance`

**Components**:
- Performance metrics summary
- Cumulative return chart (with benchmarks)
- Drawdown chart
- Rolling Sharpe ratio
- Factor attribution breakdown
- Monthly returns heatmap

### 5. Trade History
**Route**: `/trades`

**Components**:
- Trade history table (filterable)
- Transaction cost analysis
- Rebalancing frequency chart
- Trade distribution by ticker

### 6. Risk Monitor
**Route**: `/risk`

**Components**:
- Stop-loss dashboard
- VIX chart with regime indicators
- Position drawdown chart
- Portfolio heat map (by risk)
- Correlation matrix

### 7. Settings
**Route**: `/settings`

**Components**:
- Portfolio configuration
- Optimizer settings (MVO, RankBased, etc.)
- Risk parameters (stop-loss, position limits)
- Data update schedule
- IB connection settings (future)

---

## Paper Trading Engine

### Core Features

1. **Simulated Order Execution**
   - Use latest real prices + realistic slippage
   - Commission model: $1 per trade (as in backtests)
   - Slippage: 0.05% of trade value
   - Instant execution (no market impact modeling initially)

2. **Position Tracking**
   - Real-time P&L calculation
   - Mark-to-market using latest prices
   - Cost basis tracking
   - Realized vs unrealized P&L

3. **Rebalancing Simulation**
   - Weekly check for drift > 5%
   - Generate trade recommendations
   - One-click execution in paper mode
   - Track all rebalancing events

4. **Stop-Loss Management**
   - Daily check for stop-loss breaches
   - Automatic position exit in paper mode
   - Alert generation
   - VIX-based dynamic thresholds

### Implementation

```python
# app/services/paper_trading.py

class PaperTradingEngine:
    """
    Simulated trading engine for paper trading.
    """

    def __init__(self, db_session):
        self.db = db_session

    async def execute_trade(
        self,
        portfolio_id: int,
        ticker: str,
        side: str,
        quantity: float,
        price: float = None
    ) -> Trade:
        """
        Execute a simulated trade.

        Args:
            portfolio_id: Portfolio ID
            ticker: ETF ticker
            side: 'buy' or 'sell'
            quantity: Number of shares
            price: Execution price (if None, use latest market price)

        Returns:
            Trade object with execution details
        """
        # Get portfolio
        portfolio = await self.db.get_portfolio(portfolio_id)

        # Get current price if not provided
        if price is None:
            price = await self.get_latest_price(ticker)

        # Calculate slippage (0.05% of trade value)
        slippage = abs(quantity * price * 0.0005)

        # Commission ($1 flat)
        commission = 1.0

        # Adjust price for slippage
        if side == 'buy':
            execution_price = price * 1.0005  # Pay slippage
        else:
            execution_price = price * 0.9995  # Lose to slippage

        # Calculate total value
        total_value = abs(quantity * execution_price)

        # Create trade record
        trade = Trade(
            portfolio_id=portfolio_id,
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=execution_price,
            total_value=total_value,
            commission=commission,
            slippage=slippage,
            status='executed',
            executed_at=datetime.now()
        )

        # Update position
        await self.update_position(portfolio_id, ticker, trade)

        # Update cash
        if side == 'buy':
            portfolio.cash -= (total_value + commission)
        else:
            portfolio.cash += (total_value - commission)

        await self.db.save(trade)
        await self.db.save(portfolio)

        return trade

    async def check_stop_losses(self, portfolio_id: int) -> List[Dict]:
        """
        Check all positions for stop-loss breaches.

        Returns list of positions that should be exited.
        """
        positions = await self.db.get_positions(portfolio_id)
        current_prices = await self.get_latest_prices([p.ticker for p in positions])

        # Get current VIX for dynamic threshold
        vix = await self.get_latest_vix()
        stop_loss_threshold = self.get_stop_loss_threshold(vix)

        triggered = []
        for position in positions:
            current_price = current_prices[position.ticker]

            # Calculate drawdown from entry
            drawdown = (current_price - position.entry_price) / position.entry_price

            # Check if stop triggered
            if drawdown < -stop_loss_threshold:
                triggered.append({
                    'ticker': position.ticker,
                    'entry_price': position.entry_price,
                    'current_price': current_price,
                    'drawdown': drawdown,
                    'threshold': -stop_loss_threshold
                })

        return triggered
```

---

## Phase 1 Implementation Plan (Week 8)

### Day 1-2: Backend Setup
- [x] Design architecture (this document)
- [ ] Initialize FastAPI project structure
- [ ] Setup PostgreSQL database
- [ ] Create database models (SQLAlchemy)
- [ ] Create initial migration scripts
- [ ] Setup Docker Compose (backend + db)

### Day 3-4: Core API Development
- [ ] Portfolio CRUD endpoints
- [ ] Position tracking endpoints
- [ ] Paper trading engine implementation
- [ ] Factor score endpoints (read from existing data)
- [ ] Price data integration

### Day 5-6: Frontend Setup
- [ ] Initialize React + TypeScript project
- [ ] Setup routing (React Router)
- [ ] Create layout components (header, sidebar, main)
- [ ] Implement Dashboard page
- [ ] Implement Portfolio Manager page

### Day 7: Integration & Testing
- [ ] Connect frontend to backend API
- [ ] Test paper trading workflow end-to-end
- [ ] Create sample portfolio with paper trades
- [ ] Documentation and demo

---

## Future Enhancements (Weeks 9-10)

### Week 9: Advanced Frontend
- Real-time WebSocket updates
- Advanced charting (performance, attribution)
- Factor analysis page
- Risk monitoring page
- Export functionality (CSV, PDF reports)

### Week 10: IB Integration Prep
- IB TWS API research
- Authentication system
- Live data feed integration
- Order execution infrastructure
- Reconciliation system

---

## Development Environment

### Prerequisites
```bash
# Backend
Python 3.12+
PostgreSQL 15+
Redis 7+

# Frontend
Node.js 20+
npm or yarn
```

### Setup Commands
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev

# Database
docker-compose up postgres redis
```

---

## Security Considerations

### Phase 1 (Local Development)
- No authentication required
- Localhost only
- No sensitive data exposure

### Phase 2 (Multi-User / Production)
- JWT authentication
- HTTPS only
- CORS configuration
- Rate limiting
- SQL injection prevention (SQLAlchemy ORM)
- Input validation (Pydantic models)

---

## Testing Strategy

### Backend Testing
- **Unit Tests**: pytest for services and models
- **Integration Tests**: API endpoint testing with TestClient
- **Database Tests**: In-memory SQLite for fast tests

### Frontend Testing
- **Component Tests**: React Testing Library
- **Integration Tests**: Playwright or Cypress
- **Type Safety**: TypeScript strict mode

### End-to-End Testing
- Create portfolio → Generate recommendations → Execute trades → Monitor performance

---

## Monitoring & Logging

### Phase 1
- Python logging to console
- FastAPI automatic request logging
- Frontend console errors

### Phase 2
- Structured logging (JSON format)
- Application monitoring (Sentry)
- Performance monitoring (New Relic / DataDog)
- Database query monitoring

---

## Deployment Strategy

### Phase 1: Local Development
- Docker Compose for all services
- SQLite for local testing
- PostgreSQL for persistent local development

### Phase 2: Production
- **Backend**: Docker on AWS ECS or DigitalOcean App Platform
- **Database**: AWS RDS PostgreSQL (automated backups)
- **Frontend**: Vercel or Netlify (CDN, automatic deployments)
- **Redis**: AWS ElastiCache or Redis Cloud
- **CI/CD**: GitHub Actions

---

## Success Metrics

### Week 8 Goals
- ✅ Complete backend API with core endpoints
- ✅ Database schema implemented and tested
- ✅ Paper trading engine functional
- ✅ Basic frontend dashboard showing portfolio
- ✅ End-to-end workflow: Create portfolio → View positions → Execute trades

### Week 9 Goals
- Real-time updates working
- Complete all frontend pages
- Performance analytics functional
- Export capabilities

### Week 10 Goals
- IB data integration researched
- Authentication system designed
- Deployment pipeline created

---

**Next Step**: Initialize backend project structure and database

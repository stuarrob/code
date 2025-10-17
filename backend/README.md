# ETFTrader Backend API

FastAPI backend for the AQR Multi-Factor ETF Investment Strategy web application.

## Features

- **Portfolio Management**: Create, track, and manage portfolios
- **Paper Trading**: Simulated trading with realistic costs
- **Factor Analysis**: Real-time factor scores and recommendations
- **Performance Analytics**: Comprehensive performance metrics
- **Risk Management**: VIX-based dynamic stop-loss monitoring

## Tech Stack

- **FastAPI**: Modern async Python web framework
- **PostgreSQL**: Relational database for persistent storage
- **Redis**: In-memory cache for real-time data
- **SQLAlchemy**: Async ORM for database operations
- **Pydantic**: Data validation and serialization
- **Alembic**: Database migrations

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Option 1: Docker Compose (Recommended)

```bash
# From project root
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend

# API will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Option 2: Manual Setup

```bash
# 1. Install dependencies
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Setup database
# Start PostgreSQL and create database
createdb etftrader

# Run migrations
alembic upgrade head

# 3. Start Redis
redis-server

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Run development server
uvicorn app.main:app --reload

# API will be available at http://localhost:8000
```

## Project Structure

```
backend/
├── app/
│   ├── api/               # API route handlers
│   │   ├── portfolios.py  # Portfolio endpoints
│   │   ├── positions.py   # Position endpoints
│   │   ├── trades.py      # Trade endpoints
│   │   ├── factors.py     # Factor score endpoints
│   │   ├── performance.py # Performance analytics
│   │   ├── risk.py        # Risk management
│   │   └── data.py        # Data management
│   ├── core/              # Core configuration
│   │   ├── config.py      # Application settings
│   │   └── database.py    # Database connection
│   ├── models/            # SQLAlchemy models
│   │   ├── portfolio.py
│   │   ├── position.py
│   │   ├── trade.py
│   │   ├── portfolio_value.py
│   │   ├── factor_score.py
│   │   ├── etf_price.py
│   │   └── rebalance_event.py
│   ├── schemas/           # Pydantic schemas
│   │   └── portfolio.py
│   ├── services/          # Business logic
│   │   ├── portfolio_service.py
│   │   ├── paper_trading.py
│   │   └── factor_service.py
│   └── main.py            # FastAPI application
├── alembic/               # Database migrations
├── tests/                 # Test suite
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image definition
└── .env                  # Environment variables
```

## API Documentation

Once the server is running, visit:

- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Key Endpoints

### Portfolios

- `GET /api/portfolios` - List all portfolios
- `POST /api/portfolios` - Create new portfolio
- `GET /api/portfolios/{id}` - Get portfolio details
- `PATCH /api/portfolios/{id}` - Update portfolio
- `DELETE /api/portfolios/{id}` - Close portfolio

### Positions

- `GET /api/positions/{portfolio_id}` - Get portfolio positions
- `GET /api/positions/{portfolio_id}/drift` - Calculate position drift
- `POST /api/positions/{portfolio_id}/rebalance` - Generate rebalance recommendations

### Trades

- `GET /api/trades/{portfolio_id}` - Get trade history
- `POST /api/trades` - Execute trade (paper trading)

### Factors

- `GET /api/factors/latest` - Latest factor scores for all ETFs
- `GET /api/factors/{ticker}` - Factor history for specific ETF
- `GET /api/factors/recommendations` - Top ETF recommendations

### Performance

- `GET /api/performance/{portfolio_id}` - Performance metrics
- `GET /api/performance/{portfolio_id}/attribution` - Factor attribution

### Risk

- `GET /api/risk/{portfolio_id}` - Risk metrics and stop-loss status
- `GET /api/risk/vix/history` - VIX historical data

### Data

- `POST /api/data/update` - Trigger data collection
- `GET /api/data/status` - Check data freshness

## Database Management

### Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history
```

### Reset Database

```bash
# Drop and recreate (CAUTION: destroys all data)
dropdb etftrader
createdb etftrader
alembic upgrade head
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_portfolios.py
```

### Code Quality

```bash
# Format code
black app/

# Lint code
ruff check app/

# Type checking
mypy app/
```

## Environment Variables

See `.env.example` for all available configuration options.

Key variables:

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `DEBUG`: Enable debug mode
- `DEFAULT_OPTIMIZER`: Default portfolio optimizer (mvo, rank_based, minvar, simple)
- `DEFAULT_NUM_POSITIONS`: Default number of positions (20)
- `DEFAULT_DRIFT_THRESHOLD`: Rebalancing threshold (0.05 = 5%)

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
pg_isready

# Test connection
psql postgresql://etftrader:password@localhost:5432/etftrader
```

### Redis Connection Issues

```bash
# Check Redis is running
redis-cli ping
# Should return: PONG
```

### Import Errors

Make sure the project root and `src/` directory are in your PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:/home/stuar/code/ETFTrader:/home/stuar/code/ETFTrader/src"
```

## Next Steps

1. **Week 8 (Current)**: Core API implementation
   - Portfolio CRUD operations
   - Paper trading engine
   - Factor score endpoints

2. **Week 9**: Advanced features
   - Real-time WebSocket updates
   - Background task queue (Celery)
   - Advanced performance analytics

3. **Week 10**: IB Integration prep
   - Authentication system
   - Live data feed
   - Order execution infrastructure

## Contributing

See main project [OPERATIONS_MANUAL.md](../OPERATIONS_MANUAL.md) for development guidelines.

## License

Private - AQR Multi-Factor ETF Strategy Implementation

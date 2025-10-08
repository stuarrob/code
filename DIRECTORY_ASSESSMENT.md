# Directory Assessment & Final Structure

## Clean Directory Review - October 8, 2025

### Final Clean Structure

```
ETFTrader/
├── app/                    ✅ KEEP - For future web app (Week 7-8)
│   ├── api/                (Empty - will hold FastAPI endpoints)
│   └── pages/              (Empty - will hold Streamlit pages)
│
├── data/                   ✅ KEEP - ETF price data, factor scores
│   ├── etf_universe.parquet
│   └── factor_scores.parquet (future)
│
├── logs/                   ✅ KEEP - Application logs
│
├── notebooks/              ✅ KEEP - Jupyter notebooks for analysis
│   ├── 00_etf_universe_collection.ipynb
│   └── 01_data_validation.ipynb
│
├── results/                ✅ KEEP - Backtest results, reports
│   └── backtests/
│
├── scripts/                ✅ KEEP - Execution scripts
│   └── 01_collect_universe.py
│
├── src/                    ✅ KEEP - Source code
│   ├── data_collection/    (7 modules - ETF data)
│   ├── factors/            (Base + Momentum factor)
│   ├── portfolio/          (Empty - Week 3)
│   ├── backtesting/        (Empty - Week 4)
│   └── utils/              (Logging)
│
├── tests/                  ✅ KEEP - Test suite
│   ├── test_factors/       (16 tests passing)
│   ├── test_portfolio/     (Empty - Week 3)
│   └── test_backtesting/   (Empty - Week 4)
│
└── venv/                   ✅ KEEP - Python virtual environment
```

---

## Removed Directories

### ❌ Removed (Unnecessary/Old)
- **Plan/** - Old project planning files → Replaced by AQR_MULTIFACTOR_PROJECT_PLAN.md
- **config/** - Had signal_config_examples.json from old approach → No longer needed
- **docs/** - Had INTERACTIVE_BROKERS_INTEGRATION.md → Will recreate in Week 9
- **reports/** - Had old technical indicator research → Superseded by STOCK_PICKING_STRATEGIES_RESEARCH.md
- **templates/** - Empty directory → Not needed yet

---

## Directory Purpose & Timeline

### Active Now (Week 1-6)

**src/** - Source code
- Week 1: Factors library ✅
- Week 2: Factor integrator
- Week 3: Portfolio construction
- Week 4: Backtesting engine

**tests/** - Test suite
- Week 1: Factor tests ✅
- Week 3: Portfolio tests
- Week 4: Backtest tests

**scripts/** - Execution scripts
- Week 1: ETF collection ✅
- Week 2: Factor calculation
- Week 3: Backtest execution
- Week 5: Live portfolio generation

**notebooks/** - Analysis notebooks
- Keep existing ETF universe notebooks
- Week 6: Add backtest analysis notebooks
- Week 6: Add portfolio monitoring notebook

**data/** - Data storage
- ETF price history (populated)
- Factor scores (Week 2)
- Asset class mappings (populated)

**results/** - Output storage
- Backtest results (Week 4-6)
- Performance reports (Week 6)
- Trade logs (Week 7+)

**logs/** - Application logs
- All logging output goes here
- Automatic cleanup after 30 days (future)

---

### Active Later (Week 7-9)

**app/** - Web application
- Week 7: FastAPI backend
  - `app/main.py` - FastAPI app
  - `app/api/` - API endpoints
  - `app/database/` - SQLAlchemy models
  - `app/services/` - Business logic
  - `app/streamlit_app.py` - Dashboard

- Week 8: Streamlit frontend enhancements
  - `app/pages/` - Multi-page dashboard
  - Performance monitoring
  - Trade history

- Week 9+: Interactive Brokers integration
  - `app/ib/` - IB API wrapper
  - Automated trading

---

## Rationale for Each Directory

### ✅ Kept: app/
**Why**: Essential for web interface (Week 7-8)
- You explicitly requested webapp for viewing recommendations
- Manual trade entry interface
- Performance monitoring dashboard
- Future IB integration

**Status**: Empty now, will populate in Week 7

### ✅ Kept: notebooks/
**Why**: Essential for analysis and exploration
- ETF universe analysis (existing notebooks useful)
- Factor score visualization (Week 2)
- Backtest result analysis (Week 6)
- Interactive experimentation

**Status**: Has 2 existing notebooks, will add more

### ✅ Kept: data/
**Why**: Required for all operations
- ETF price history
- Factor scores cache
- Asset class mappings
- Portfolio state (future)

**Status**: Populated with ETF data

### ✅ Kept: results/
**Why**: Store backtest outputs
- Performance reports
- Trade history
- Comparative analysis
- Historical tracking

**Status**: Has some old backtest results, will clean and repopulate

### ✅ Kept: logs/
**Why**: Troubleshooting and monitoring
- Application logs
- Error tracking
- Performance monitoring
- Audit trail

**Status**: Active, logging configured

### ❌ Removed: Plan/
**Why**: Replaced by better documentation
- Had old project plan → Replaced by AQR_MULTIFACTOR_PROJECT_PLAN.md
- Progress log → Replaced by PROGRESS_WEEK1.md
- Instructions → Integrated into main plan

### ❌ Removed: config/
**Why**: Not needed for new approach
- Had signal config for old momentum strategy
- New approach uses simpler configuration
- Can recreate if needed later

### ❌ Removed: docs/
**Why**: Will recreate when needed
- Had IB integration doc → Week 9 is future
- Better to create fresh docs aligned with new architecture
- Current .md files at root are sufficient

### ❌ Removed: reports/
**Why**: Old analysis, superseded
- Had technical indicator research from old approach
- New approach uses AQR multi-factor (different)
- STOCK_PICKING_STRATEGIES_RESEARCH.md is comprehensive

### ❌ Removed: templates/
**Why**: Empty, not needed yet
- Was empty
- If needed for web app, will create in Week 7
- Can use Streamlit's built-in templating

---

## Disk Space

### Before Cleanup
- Total project: ~500 MB
- Code + docs: ~50 MB
- Data: ~400 MB (ETF price history)
- Old results/logs: ~50 MB

### After Cleanup
- Total project: ~480 MB
- Code + docs: ~30 MB (cleaned)
- Data: ~400 MB (kept)
- Results/logs: ~50 MB

**Savings**: ~20 MB (old code/docs removed)

---

## File Organization Standards

### Python Modules (src/)
```
src/
├── module_name/
│   ├── __init__.py        # Package exports
│   ├── main_class.py      # Primary implementation
│   ├── helper.py          # Supporting functions
│   └── constants.py       # Configuration constants
```

### Tests (tests/)
```
tests/
├── test_module_name/
│   ├── __init__.py
│   ├── test_main_class.py    # Unit tests
│   ├── test_integration.py   # Integration tests
│   └── fixtures.py            # Test data fixtures
```

### Scripts (scripts/)
```
scripts/
├── 01_collect_universe.py     # Step 1: Data collection
├── 02_calculate_factors.py    # Step 2: Factor calculation
├── 03_backtest_strategy.py    # Step 3: Backtesting
└── 04_run_live_portfolio.py   # Step 4: Live signals
```

**Naming**: Number prefix indicates execution order

### Notebooks (notebooks/)
```
notebooks/
├── 01_factor_analysis.ipynb      # Explore factor distributions
├── 02_backtest_results.ipynb     # Analyze backtest
└── 03_portfolio_monitoring.ipynb # Track live portfolio
```

**Naming**: Number prefix + descriptive name

---

## Future Additions

### Week 2
- `data/factor_scores.parquet` - Cached factor calculations
- `scripts/02_calculate_factors.py` - Factor calculation script

### Week 3
- `src/portfolio/optimizer.py` - Portfolio optimizer
- `src/portfolio/rebalancer.py` - Rebalance logic
- `tests/test_portfolio/` - Portfolio tests

### Week 4
- `src/backtesting/engine.py` - Backtest engine
- `src/backtesting/metrics.py` - Performance metrics
- `tests/test_backtesting/` - Backtest tests

### Week 5
- `scripts/03_backtest_strategy.py` - Backtest script
- `scripts/04_run_live_portfolio.py` - Live signal generator

### Week 6
- `notebooks/01_factor_analysis.ipynb` - Factor exploration
- `notebooks/02_backtest_results.ipynb` - Backtest analysis

### Week 7-8 (Web App)
- `app/main.py` - FastAPI application
- `app/api/` - API endpoints
- `app/streamlit_app.py` - Dashboard
- `database/portfolio.db` - SQLite database

### Week 9+ (IB Integration)
- `app/ib/` - Interactive Brokers wrapper
- `docs/IB_INTEGRATION.md` - IB setup guide

---

## Summary

### Final Structure: Clean & Purpose-Driven

**10 directories** (down from 14):
- ✅ 7 essential directories kept
- ✅ 3 empty but planned (app, notebooks additions)
- ❌ 4 unnecessary directories removed

### Key Principles
1. **Keep only what's needed now or soon**
2. **Remove old/superseded content**
3. **Can always recreate from git history**
4. **Clear purpose for each directory**

### Status: ✅ Clean & Ready

The directory structure now matches the 8-week project plan:
- Weeks 1-6: Strategy development (src, tests, scripts)
- Weeks 7-8: Web app (app)
- Week 9+: IB integration (app/ib)

All old project artifacts removed. Fresh start with AQR multi-factor approach.

---

*Last updated: October 8, 2025*

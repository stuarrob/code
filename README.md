# ETF Portfolio Optimization System

A comprehensive ETF portfolio optimization system for building long-only portfolios with high Sharpe ratio and low drawdown probability.

## Features

- **Automated ETF Data Collection**: Collect data from 2000-4000 ETFs using free data sources
- **Technical Signal Generation**: MACD, RSI, Bollinger Bands, and composite signal framework
- **Robust Portfolio Optimization**: Axioma-style mean-variance optimization with risk penalties
- **Multi-timeframe Analysis**: 3-month, 6-month, and 1-year backtesting windows
- **Web Interface**: Streamlit-based UI for weekly portfolio generation
- **Jupyter Notebooks**: Interactive analysis and component breakdown
- **Weekly Automation**: Scheduled portfolio rebalancing and reporting

## Project Structure

```
ETFTrader/
├── Plan/                    # Project planning and progress tracking
├── data/                    # CSV-based data storage
│   ├── raw/                 # Raw ETF data
│   ├── processed/           # Cleaned data
│   ├── indicators/          # Technical indicators
│   └── signals/             # Composite signals
├── src/                     # Source code modules
│   ├── data_collection/     # ETF data scrapers
│   ├── data_management/     # CSV utilities
│   ├── signals/             # Technical indicators
│   ├── optimization/        # Portfolio optimization
│   ├── backtesting/         # Performance testing
│   ├── analytics/           # Metrics and reporting
│   └── visualization/       # Charts and plots
├── notebooks/               # Jupyter analysis notebooks
├── app/                     # Streamlit web interface
├── tests/                   # Unit and integration tests
├── config/                  # Configuration files
├── results/                 # Optimization outputs
└── requirements.txt         # Python dependencies
```

## Installation

### Prerequisites

- Python 3.12+
- Ubuntu/WSL2 (tested on Ubuntu 24.04)

### Setup

1. **Activate virtual environment:**
   ```bash
   cd /home/stuar/code/ETFTrader
   source venv/bin/activate
   ```

2. **Install dependencies (already done):**
   ```bash
   pip install -r requirements.txt
   ```

### Installed Libraries

**Core Scientific Computing:**
- numpy 2.2.6
- pandas 2.3.3
- scipy 1.16.2
- scikit-learn 1.7.2

**Optimization:**
- cvxpy 1.7.3 (with OSQP, Clarabel, SCS solvers)

**Data Collection:**
- yfinance 0.2.66
- requests 2.32.5
- beautifulsoup4 4.14.2
- lxml 6.0.2

**Technical Analysis:**
- pandas-ta 0.4.71b0
- numba 0.61.2

**Web Framework:**
- Flask 3.1.2
- Streamlit 1.50.0

**Visualization:**
- matplotlib 3.10.6
- seaborn 0.13.2
- plotly 6.3.1

**Development:**
- jupyter 1.1.1
- ipywidgets 8.1.7
- pytest (to be added)

## Usage

### Quick Start - Data Collection

1. **Scrape ETF Universe:**
   ```bash
   cd /home/stuar/code/ETFTrader
   source venv/bin/activate
   python src/data_collection/etf_scraper.py
   ```
   Output: `data/raw/etf_universe.csv` with 298 ETFs

2. **Download Price Data:**
   ```bash
   python src/data_collection/price_downloader.py
   ```
   Output: 3 years of OHLCV data in `data/raw/prices/`

3. **Validate Data Quality:**
   ```bash
   python src/data_collection/data_validator.py
   ```
   Output: Validation report in `results/data_validation_report.csv`

### Jupyter Notebooks

**Setup the IPython Kernel (one-time):**

```bash
cd /home/stuar/code/ETFTrader
source venv/bin/activate
python -m ipykernel install --user --name=etftrader --display-name="Python (ETFTrader)"
```

**Run the Data Validation Notebook:**

```bash
jupyter notebook notebooks/01_data_validation.ipynb
```

**Important:** Select the **"Python (ETFTrader)"** kernel in Jupyter:
- Go to: **Kernel → Change Kernel → Python (ETFTrader)**

See [JUPYTER_SETUP.md](JUPYTER_SETUP.md) for detailed instructions.

### Available Notebooks

- ✅ **01_data_validation.ipynb** - ETF data quality dashboard (Phase 1 complete)
- 🔄 **02_signal_analysis.ipynb** - Technical indicators (Phase 2, coming soon)
- 🔄 **03_portfolio_optimization.ipynb** - Portfolio construction (Phase 3, coming soon)
- 🔄 **04_performance_analytics.ipynb** - Backtest results (Phase 4, coming soon)

### Run Streamlit Web App (coming soon)

```bash
streamlit run app/main.py
```

## Development Status

**Current Phase:** ✅ Phase 1 Complete - Phase 2 Starting
**Progress:**
- Phase 0: ✅ Planning & Setup Complete
- Phase 1: ✅ Data Infrastructure Complete (298 ETFs, 90.2/100 quality score)
- Phase 2: 🔄 Signal Generation (next)

See [Plan/PROJECT_PLAN.md](Plan/PROJECT_PLAN.md) for detailed implementation roadmap.

## Project Goals

- **Portfolio Size:** Maximum 20 ETF positions
- **Rebalancing:** Weekly
- **Target Sharpe Ratio:** >1.2
- **Max Drawdown:** <12%
- **Universe:** 2000-4000 ETFs (no leveraged ETFs)
- **Optimization Runtime:** <60 minutes for full universe

## License

Private project - All rights reserved

## Contact

Stuart - ETF Trader Project

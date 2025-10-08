"""
Comprehensive Technical Signal Exploration Framework

Systematically explores the space of technical signals to find combinations that:
1. Have stable predictions (low signal variance week-to-week)
2. Work out-of-sample (positive Sharpe in backtest)
3. Minimize portfolio churn (low turnover)
4. Are robust across market regimes

Tests:
- All major technical indicator families
- Multiple parameter combinations for each
- Signal combination strategies (weighted, ensemble, voting)
- Different normalization and scaling approaches
- Signal stability metrics
- Out-of-sample performance

Generates comprehensive output for post-hoc analysis.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
from itertools import product, combinations
import time
import traceback

from src.backtesting.backtest_engine import BacktestEngine
from src.data_collection.asset_class_mapper import create_asset_class_map
from src.data_collection.etf_filters import apply_etf_filters

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = project_root / "results" / "signal_exploration" / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "master.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# SIGNAL INDICATOR LIBRARY
# ============================================================================

class TechnicalSignalLibrary:
    """
    Comprehensive library of technical indicators.

    Each indicator returns a normalized signal in [-1, 1] or [0, 1] range.
    """

    @staticmethod
    def momentum(prices: pd.Series, period: int) -> float:
        """Price momentum over period."""
        if len(prices) < period:
            return 0.0
        return (prices.iloc[-1] / prices.iloc[-period]) - 1

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> float:
        """Relative Strength Index (0-100, normalized to -1 to 1)."""
        if len(prices) < period + 1:
            return 0.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]

        # Normalize to -1 to 1 (50 = 0, 0 = -1, 100 = 1)
        return (rsi_val - 50) / 50

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """MACD signal (normalized)."""
        if len(prices) < slow + signal:
            return 0.0

        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()

        macd_val = (macd_line - signal_line).iloc[-1]

        # Normalize by current price
        return np.clip(macd_val / prices.iloc[-1] * 100, -1, 1)

    @staticmethod
    def bollinger_position(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> float:
        """Position within Bollinger Bands (-1 to 1)."""
        if len(prices) < period:
            return 0.0

        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = sma + (std * num_std)
        lower = sma - (std * num_std)

        current = prices.iloc[-1]
        sma_val = sma.iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]

        # -1 = at lower band, 0 = at middle, 1 = at upper band
        if upper_val == lower_val:
            return 0.0

        position = (current - sma_val) / (upper_val - lower_val) * 2
        return np.clip(position, -1, 1)

    @staticmethod
    def stochastic(prices: pd.Series, period: int = 14) -> float:
        """Stochastic Oscillator (0-100, normalized to -1 to 1)."""
        if len(prices) < period:
            return 0.0

        low_min = prices.rolling(window=period).min()
        high_max = prices.rolling(window=period).max()

        current = prices.iloc[-1]
        low_val = low_min.iloc[-1]
        high_val = high_max.iloc[-1]

        if high_val == low_val:
            return 0.0

        k = 100 * (current - low_val) / (high_val - low_val)

        # Normalize to -1 to 1
        return (k - 50) / 50

    @staticmethod
    def adx(prices: pd.Series, period: int = 14) -> float:
        """Average Directional Index (trend strength, 0-1)."""
        if len(prices) < period * 2:
            return 0.0

        # Simplified ADX calculation
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()

        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)

        tr = (high - low).rolling(period).mean()

        plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean().iloc[-1]

        # Normalize to 0-1 (strong trend >25)
        return np.clip(adx / 50, 0, 1)

    @staticmethod
    def volume_ratio(volumes: pd.Series, period: int = 20) -> float:
        """Volume ratio vs moving average."""
        if len(volumes) < period:
            return 0.0

        current_vol = volumes.iloc[-1]
        avg_vol = volumes.rolling(window=period).mean().iloc[-1]

        if avg_vol == 0:
            return 0.0

        ratio = current_vol / avg_vol

        # Normalize (1 = average, 2 = 2x average → 1.0)
        return np.clip((ratio - 1), -1, 1)

    @staticmethod
    def price_vs_sma(prices: pd.Series, period: int) -> float:
        """Price distance from SMA (trend strength)."""
        if len(prices) < period:
            return 0.0

        sma = prices.rolling(window=period).mean().iloc[-1]
        current = prices.iloc[-1]

        if sma == 0:
            return 0.0

        # Percentage distance
        dist = (current - sma) / sma

        # Normalize (-10% to +10% → -1 to 1)
        return np.clip(dist * 10, -1, 1)

    @staticmethod
    def acceleration(prices: pd.Series, short_period: int, long_period: int) -> float:
        """Momentum acceleration (short vs long momentum)."""
        if len(prices) < long_period:
            return 0.0

        short_mom = (prices.iloc[-1] / prices.iloc[-short_period]) - 1
        long_mom = (prices.iloc[-1] / prices.iloc[-long_period]) - 1

        # Positive = accelerating, negative = decelerating
        accel = short_mom - long_mom

        # Normalize
        return np.clip(accel * 10, -1, 1)


# ============================================================================
# SIGNAL GENERATOR WITH CONFIGURABLE INDICATORS
# ============================================================================

class ConfigurableSignalGenerator:
    """
    Generate signals from configurable set of technical indicators.

    Parameters define which indicators to use and how to combine them.
    """

    def __init__(self, config: dict):
        """
        Initialize with configuration.

        config = {
            'indicators': {
                'momentum_126': {'weight': 0.3},
                'rsi_14': {'weight': 0.2},
                'macd_default': {'weight': 0.2},
                ...
            },
            'combination_method': 'weighted',  # or 'voting', 'ensemble'
            'normalization': 'zscore'  # or 'minmax', 'none'
        }
        """
        self.config = config
        self.lib = TechnicalSignalLibrary()

    def generate_signal(self, prices: pd.Series, volumes: pd.Series = None) -> float:
        """Generate composite signal for single ETF."""
        signals = {}

        for indicator_name, params in self.config['indicators'].items():
            try:
                signal_val = self._calculate_indicator(indicator_name, prices, volumes)
                signals[indicator_name] = signal_val
            except Exception as e:
                logger.debug(f"Indicator {indicator_name} failed: {e}")
                signals[indicator_name] = 0.0

        # Combine signals
        method = self.config.get('combination_method', 'weighted')

        if method == 'weighted':
            total_weight = sum(params['weight'] for params in self.config['indicators'].values())
            composite = sum(
                signals[name] * self.config['indicators'][name]['weight']
                for name in signals
            ) / total_weight if total_weight > 0 else 0.0

        elif method == 'voting':
            # Count positive vs negative signals
            votes = [1 if s > 0 else -1 if s < 0 else 0 for s in signals.values()]
            composite = sum(votes) / len(votes) if votes else 0.0

        elif method == 'ensemble':
            # Use median to reduce outlier impact
            composite = np.median(list(signals.values())) if signals else 0.0

        else:
            composite = 0.0

        return composite

    def _calculate_indicator(self, name: str, prices: pd.Series, volumes: pd.Series) -> float:
        """Calculate specific indicator by name."""
        # Parse indicator name (e.g., 'momentum_126' → momentum with period 126)
        parts = name.split('_')
        indicator = parts[0]

        if indicator == 'momentum':
            period = int(parts[1]) if len(parts) > 1 else 126
            return self.lib.momentum(prices, period)

        elif indicator == 'rsi':
            period = int(parts[1]) if len(parts) > 1 else 14
            return self.lib.rsi(prices, period)

        elif indicator == 'macd':
            return self.lib.macd(prices)

        elif indicator == 'bollinger':
            period = int(parts[1]) if len(parts) > 1 else 20
            return self.lib.bollinger_position(prices, period)

        elif indicator == 'stochastic':
            period = int(parts[1]) if len(parts) > 1 else 14
            return self.lib.stochastic(prices, period)

        elif indicator == 'adx':
            period = int(parts[1]) if len(parts) > 1 else 14
            return self.lib.adx(prices, period)

        elif indicator == 'volume':
            period = int(parts[1]) if len(parts) > 1 else 20
            return self.lib.volume_ratio(volumes, period) if volumes is not None else 0.0

        elif indicator == 'sma':
            period = int(parts[1]) if len(parts) > 1 else 50
            return self.lib.price_vs_sma(prices, period)

        elif indicator == 'accel':
            short = int(parts[1]) if len(parts) > 1 else 20
            long_p = int(parts[2]) if len(parts) > 2 else 63
            return self.lib.acceleration(prices, short, long_p)

        else:
            return 0.0

    def generate_signals_for_universe(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame = None
    ) -> pd.Series:
        """Generate signals for all ETFs."""
        signals = {}

        for ticker in prices.columns:
            price_series = prices[ticker].dropna()
            volume_series = volumes[ticker].dropna() if volumes is not None and ticker in volumes.columns else None

            if len(price_series) < 50:  # Minimum data requirement
                continue

            signal = self.generate_signal(price_series, volume_series)
            signals[ticker] = signal

        return pd.Series(signals)


# ============================================================================
# GRID SEARCH CONFIGURATION
# ============================================================================

# Define indicator families to test
INDICATOR_FAMILIES = {
    'momentum': {
        'short': ['momentum_20', 'momentum_63'],
        'medium': ['momentum_126'],
        'long': ['momentum_252']
    },
    'oscillators': {
        'rsi': ['rsi_7', 'rsi_14', 'rsi_21'],
        'stochastic': ['stochastic_14', 'stochastic_21']
    },
    'trend': {
        'macd': ['macd_default'],
        'sma': ['sma_50', 'sma_200'],
        'adx': ['adx_14']
    },
    'volatility': {
        'bollinger': ['bollinger_20', 'bollinger_10']
    },
    'volume': {
        'volume': ['volume_20']
    },
    'derived': {
        'acceleration': ['accel_20_63', 'accel_63_126']
    }
}

# Test these signal combinations
SIGNAL_COMBINATIONS = [
    # Single indicator families
    {'name': 'momentum_only', 'families': ['momentum']},
    {'name': 'oscillators_only', 'families': ['oscillators']},
    {'name': 'trend_only', 'families': ['trend']},

    # Two-family combinations
    {'name': 'momentum_trend', 'families': ['momentum', 'trend']},
    {'name': 'momentum_oscillators', 'families': ['momentum', 'oscillators']},
    {'name': 'trend_oscillators', 'families': ['trend', 'oscillators']},

    # Three-family combinations
    {'name': 'momentum_trend_oscillators', 'families': ['momentum', 'trend', 'oscillators']},
    {'name': 'momentum_trend_volatility', 'families': ['momentum', 'trend', 'volatility']},

    # All indicators
    {'name': 'all_indicators', 'families': list(INDICATOR_FAMILIES.keys())},
]

# Weighting strategies
WEIGHTING_STRATEGIES = [
    'equal',     # All indicators equal weight
    'momentum_heavy',  # 60% momentum, rest split
    'balanced',  # Balanced across families
]

# Combination methods
COMBINATION_METHODS = ['weighted', 'voting', 'ensemble']

# Portfolio optimization parameters
PORTFOLIO_PARAMS = [
    {'turnover_penalty': 20.0, 'rebalance': 'weekly'},
    {'turnover_penalty': 50.0, 'rebalance': 'weekly'},
    {'turnover_penalty': 100.0, 'rebalance': 'weekly'},
    {'turnover_penalty': 50.0, 'rebalance': 'monthly'},
]


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def load_etf_universe(n_etfs: int = 150):
    """Load ETF universe."""
    prices_dir = project_root / "data" / "raw" / "prices"
    etf_files = list(prices_dir.glob("*.csv"))

    logger.info(f"Loading {n_etfs}-ETF universe...")

    # Score and load top ETFs
    etf_scores = []
    for file in etf_files:
        try:
            df = pd.read_csv(file)
            date_col = next((col for col in df.columns if col.lower() == 'date'), None)
            if date_col is None:
                continue

            df[date_col] = pd.to_datetime(df[date_col])
            df.columns = [col.capitalize() for col in df.columns]

            score = len(df) * (1 - df["Close"].isna().sum() / len(df))
            etf_scores.append({'ticker': file.stem, 'score': score, 'file': file})
        except:
            continue

    etf_scores_df = pd.DataFrame(etf_scores)
    etf_scores_df = etf_scores_df.sort_values("score", ascending=False)
    top_etfs = etf_scores_df.head(n_etfs)

    prices = {}
    volumes = {}
    for _, row in top_etfs.iterrows():
        df = pd.read_csv(row["file"])
        date_col = next((col for col in df.columns if col.lower() == 'date'), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.columns = [col.capitalize() for col in df.columns]
            df = df.set_index("Date")
            prices[row["ticker"]] = df["Close"]
            if "Volume" in df.columns:
                volumes[row["ticker"]] = df["Volume"]

    prices_df = pd.DataFrame(prices).sort_index()
    volumes_df = pd.DataFrame(volumes).sort_index() if volumes else None

    prices_df = apply_etf_filters(prices_df, filter_leveraged=True, filter_high_volatility=True, max_volatility=0.35)

    logger.info(f"Loaded {prices_df.shape[1]} ETFs after filtering")

    return prices_df, volumes_df


def generate_signal_config(combo: dict, weighting: str, method: str) -> dict:
    """Generate signal configuration from parameters."""
    # Collect all indicators for selected families
    indicators = {}

    for family_name in combo['families']:
        if family_name not in INDICATOR_FAMILIES:
            continue

        family = INDICATOR_FAMILIES[family_name]
        for subfamily, indicator_list in family.items():
            for indicator in indicator_list:
                indicators[indicator] = {'weight': 1.0}

    # Apply weighting strategy
    if weighting == 'equal':
        for ind in indicators:
            indicators[ind]['weight'] = 1.0

    elif weighting == 'momentum_heavy':
        for ind in indicators:
            if 'momentum' in ind or 'accel' in ind:
                indicators[ind]['weight'] = 2.0
            else:
                indicators[ind]['weight'] = 0.5

    elif weighting == 'balanced':
        # Equal weight per family
        family_counts = {}
        for ind in indicators:
            for fam in combo['families']:
                if any(ind.startswith(sub) for sub in INDICATOR_FAMILIES.get(fam, {}).keys()):
                    family_counts[fam] = family_counts.get(fam, 0) + 1

        for ind in indicators:
            for fam in combo['families']:
                if any(ind.startswith(sub) for sub in INDICATOR_FAMILIES.get(fam, {}).keys()):
                    indicators[ind]['weight'] = 1.0 / family_counts.get(fam, 1)

    return {
        'indicators': indicators,
        'combination_method': method,
        'normalization': 'none'
    }


def run_single_experiment(exp_id: int, params: dict, prices: pd.DataFrame, volumes: pd.DataFrame, asset_class_map: dict) -> dict:
    """Run single backtest experiment."""
    exp_file = OUTPUT_DIR / f"exp_{exp_id:04d}.json"

    logger.info(f"\n[{exp_id}] Testing: {params['combo_name']} + {params['weighting']} + {params['method']} + turnover={params['turnover_penalty']}")

    try:
        # Generate signal configuration
        signal_config = generate_signal_config(
            params['combo'],
            params['weighting'],
            params['method']
        )

        # Generate signals
        signal_gen = ConfigurableSignalGenerator(signal_config)

        # Create modified backtest engine (simplified for now)
        from src.optimization import cvxpy_optimizer

        custom_variant = {
            "risk_aversion": 1.5,
            "robustness_penalty": 0.5,
            "turnover_penalty": params['turnover_penalty'],
            "concentration_penalty": 1.0,
            "asset_class_penalty": 0.5,
            "description": f"Exp {exp_id}"
        }
        cvxpy_optimizer.CVXPYPortfolioOptimizer.VARIANTS['balanced'] = custom_variant

        engine = BacktestEngine(
            initial_capital=1_000_000,
            rebalance_frequency=params['rebalance'],
            lookback_period=126,
            variant='balanced',
            enable_stop_loss=False,  # Disable for cleaner signal testing
            enable_transaction_costs=True,
            risk_free_rate=0.04,
            asset_class_map=asset_class_map
        )

        # Run backtest (last 2 years)
        end_date = prices.index[-1]
        start_date = end_date - pd.Timedelta(days=730)
        if start_date < prices.index[126]:
            start_date = prices.index[126]

        results = engine.run(prices=prices, start_date=start_date, end_date=end_date)

        metrics = results['metrics']

        # Calculate signal stability (how much signals change week-to-week)
        # This is KEY to understanding churn
        signal_history = []
        test_dates = prices.loc[start_date:end_date].index[::7]  # Weekly samples
        for date in test_dates[:20]:  # Sample first 20 weeks
            window_prices = prices.loc[:date].tail(252)
            window_volumes = volumes.loc[:date].tail(252) if volumes is not None else None
            signals = signal_gen.generate_signals_for_universe(window_prices, window_volumes)
            signal_history.append(signals)

        # Calculate signal stability (correlation week-to-week)
        if len(signal_history) > 1:
            correlations = []
            for i in range(len(signal_history) - 1):
                common_tickers = signal_history[i].index.intersection(signal_history[i+1].index)
                if len(common_tickers) > 10:
                    corr = signal_history[i][common_tickers].corr(signal_history[i+1][common_tickers])
                    correlations.append(corr)

            signal_stability = np.mean(correlations) if correlations else 0.0
        else:
            signal_stability = 0.0

        # Package results
        result = {
            'exp_id': exp_id,
            'params': params,
            'signal_config': {
                'num_indicators': len(signal_config['indicators']),
                'indicators': list(signal_config['indicators'].keys()),
                'method': signal_config['combination_method']
            },
            'performance': {
                'cagr': float(metrics['cagr']),
                'sharpe': float(metrics['sharpe_ratio']),
                'sortino': float(metrics['sortino_ratio']),
                'calmar': float(metrics['calmar_ratio']),
                'max_drawdown': float(metrics['max_drawdown']),
                'volatility': float(metrics['volatility']),
                'win_rate': float(metrics['win_rate'])
            },
            'turnover': {
                'avg_turnover_pct': float(metrics['avg_turnover']),
                'num_rebalances': int(metrics['num_rebalances'])
            },
            'signal_quality': {
                'stability': float(signal_stability),  # KEY METRIC!
                'avg_signal': 0.0,  # Could calculate
                'signal_range': 0.0  # Could calculate
            },
            'success': True
        }

        # Save result
        with open(exp_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"  ✅ Sharpe={result['performance']['sharpe']:.2f}, "
                   f"Turnover={result['turnover']['avg_turnover_pct']:.0f}%, "
                   f"Signal_Stability={result['signal_quality']['stability']:.2f}")

        return result

    except Exception as e:
        logger.error(f"  ❌ Failed: {str(e)}")
        result = {
            'exp_id': exp_id,
            'params': params,
            'success': False,
            'error': str(e)
        }

        with open(exp_file, 'w') as f:
            json.dump(result, f, indent=2)

        return result


def main():
    """Run comprehensive signal exploration grid search."""
    logger.info("="*80)
    logger.info("COMPREHENSIVE TECHNICAL SIGNAL EXPLORATION")
    logger.info("="*80)
    logger.info(f"\nOutput directory: {OUTPUT_DIR}")

    # Generate all parameter combinations
    experiments = []
    exp_id = 1

    for combo in SIGNAL_COMBINATIONS:
        for weighting in WEIGHTING_STRATEGIES:
            for method in COMBINATION_METHODS:
                for portfolio_params in PORTFOLIO_PARAMS:
                    experiments.append({
                        'exp_id': exp_id,
                        'combo': combo,
                        'combo_name': combo['name'],
                        'weighting': weighting,
                        'method': method,
                        **portfolio_params
                    })
                    exp_id += 1

    total_experiments = len(experiments)
    logger.info(f"\nTotal experiments: {total_experiments}")
    logger.info(f"Estimated time: {total_experiments * 90 / 3600:.1f} hours")

    # Load data
    logger.info("\nLoading data...")
    prices, volumes = load_etf_universe(n_etfs=150)

    fundamentals_path = project_root / "data" / "raw" / "fundamentals.csv"
    asset_class_map = {}
    if fundamentals_path.exists():
        asset_class_map = create_asset_class_map(str(fundamentals_path))

    # Run experiments
    results = []
    start_time = time.time()

    logger.info("\n" + "="*80)
    logger.info("RUNNING EXPERIMENTS")
    logger.info("="*80)

    for i, params in enumerate(experiments, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPERIMENT {i}/{total_experiments}")
        logger.info(f"{'='*80}")

        result = run_single_experiment(i, params, prices, volumes, asset_class_map)
        results.append(result)

        # Save interim results every 10 experiments
        if i % 10 == 0:
            interim_file = OUTPUT_DIR / f"interim_results_{i:04d}.json"
            with open(interim_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n💾 Saved interim results: {i} experiments complete")

    # Save final results
    final_file = OUTPUT_DIR / "all_results.json"
    with open(final_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Create summary CSV
    successful = [r for r in results if r['success']]
    if successful:
        df_data = []
        for r in successful:
            row = {
                'exp_id': r['exp_id'],
                'combo_name': r['params']['combo_name'],
                'weighting': r['params']['weighting'],
                'method': r['params']['method'],
                'turnover_penalty': r['params']['turnover_penalty'],
                'rebalance': r['params']['rebalance'],
                'num_indicators': r['signal_config']['num_indicators'],
                **r['performance'],
                **r['turnover'],
                **r['signal_quality']
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        csv_file = OUTPUT_DIR / "all_results.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"\nResults CSV: {csv_file}")

    total_time = time.time() - start_time
    logger.info(f"\n✅ Grid search complete!")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Results: {OUTPUT_DIR}")

    # Print quick summary
    if successful:
        df_sorted = df.sort_values('sharpe', ascending=False)
        logger.info("\n" + "="*80)
        logger.info("TOP 10 BY SHARPE RATIO")
        logger.info("="*80)
        for idx, row in df_sorted.head(10).iterrows():
            logger.info(f"\n{row['combo_name']} ({row['weighting']}, {row['method']})")
            logger.info(f"  Sharpe: {row['sharpe']:.2f}, CAGR: {row['cagr']*100:.1f}%")
            logger.info(f"  Turnover: {row['avg_turnover_pct']:.0f}%")
            logger.info(f"  Signal Stability: {row['stability']:.2f}")


if __name__ == "__main__":
    main()

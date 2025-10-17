"""Factor service - loads and serves factor scores."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import date

from app.core.config import settings


class FactorService:
    """Service for loading and serving factor scores from existing data."""

    def __init__(self):
        self._cache = {}

    @property
    def data_dir(self) -> Path:
        """Resolve data directory dynamically."""
        return Path(settings.DATA_DIR)

    @property
    def signals_dir(self) -> Path:
        """Resolve signals directory dynamically."""
        return self.data_dir / 'signals'

    def load_latest_scores(self) -> pd.DataFrame:
        """
        Load latest factor scores from Parquet files.

        Returns:
            DataFrame with columns: ticker, momentum, quality, value, volatility, composite
        """
        # Check cache
        if 'latest_scores' in self._cache:
            return self._cache['latest_scores']

        try:
            # Load integrated scores (pre-calculated composite)
            integrated_file = self.signals_dir / 'integrated_scores.parquet'

            print(f"DEBUG: integrated_file = {integrated_file}")
            print(f"DEBUG: integrated_file.exists() = {integrated_file.exists()}")

            if integrated_file.exists():
                # Load integrated scores and individual factors
                factor_files = {
                    'momentum': self.signals_dir / 'momentum_scores.parquet',
                    'quality': self.signals_dir / 'quality_scores.parquet',
                    'value': self.signals_dir / 'value_scores.parquet',
                    'volatility': self.signals_dir / 'volatility_scores.parquet',
                    'composite': integrated_file
                }

                scores = []
                for factor_name, filepath in factor_files.items():
                    print(f"DEBUG: Loading {factor_name} from {filepath}, exists={filepath.exists()}")
                    if filepath.exists():
                        df = pd.read_parquet(filepath)
                        print(f"DEBUG: Loaded {factor_name}: shape={df.shape}, columns={df.columns.tolist()}")
                        # Rename 'integrated' column to 'composite' if needed
                        if 'integrated' in df.columns:
                            df = df.rename(columns={'integrated': 'composite'})
                        scores.append(df)

                print(f"DEBUG: About to concat {len(scores)} DataFrames")
                # Combine into single DataFrame
                combined = pd.concat(scores, axis=1)
                print(f"DEBUG: Combined shape: {combined.shape}")

                # Cache
                self._cache['latest_scores'] = combined

                return combined
            else:
                print(f"DEBUG: integrated_file does not exist!")
                return pd.DataFrame()

        except Exception as e:
            import traceback
            error_msg = f"Error loading factor scores: {e}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return pd.DataFrame()

    def get_recommendations(
        self,
        num_positions: int = 20,
        optimizer_type: str = 'mvo'
    ) -> List[Dict]:
        """
        Get top ETF recommendations based on composite scores.

        Args:
            num_positions: Number of recommendations
            optimizer_type: Optimizer to use (affects weighting)

        Returns:
            List of {ticker, composite_score, target_weight, ...}
        """
        scores = self.load_latest_scores()

        if scores.empty:
            return []

        # Sort by composite score
        top_etfs = scores.nlargest(num_positions, 'composite')

        # Calculate weights (simple equal weight for now)
        weight = 1.0 / num_positions

        recommendations = []
        for ticker in top_etfs.index:
            row = top_etfs.loc[ticker]
            recommendations.append({
                'ticker': ticker,
                'composite_score': float(row['composite']),
                'momentum': float(row['momentum']),
                'quality': float(row['quality']),
                'value': float(row['value']),
                'volatility': float(row['volatility']),
                'target_weight': weight
            })

        return recommendations

    def get_ticker_history(
        self,
        ticker: str,
        days: int = 90
    ) -> List[Dict]:
        """Get factor score history for specific ticker."""
        try:
            factor_files = {
                'momentum': self.signals_dir / 'momentum_scores.parquet',
                'quality': self.signals_dir / 'quality_scores.parquet',
                'value': self.signals_dir / 'value_scores.parquet',
                'volatility': self.signals_dir / 'volatility_scores.parquet'
            }

            history = []
            for factor_name, filepath in factor_files.items():
                if filepath.exists():
                    df = pd.read_parquet(filepath)
                    if ticker in df.columns:
                        ticker_data = df[ticker].tail(days)
                        for date_val, score in ticker_data.items():
                            history.append({
                                'date': date_val.strftime('%Y-%m-%d'),
                                'factor': factor_name,
                                'score': float(score)
                            })

            return history

        except Exception as e:
            print(f"Error loading ticker history: {e}")
            return []


# Global instance
factor_service = FactorService()

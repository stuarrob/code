"""Data management API endpoints."""

from fastapi import APIRouter
from typing import Dict
from pathlib import Path
from datetime import datetime

from app.core.config import settings


router = APIRouter()


@router.get("/status")
async def get_data_status() -> Dict:
    """Check data freshness and availability."""
    data_dir = Path(settings.DATA_DIR)
    
    # Check for price data
    prices_file = data_dir / 'processed' / 'etf_prices_filtered.parquet'
    signals_dir = data_dir / 'signals'
    
    status = {
        "data_directory": str(data_dir),
        "prices_available": prices_file.exists(),
        "signals_available": signals_dir.exists()
    }
    
    if prices_file.exists():
        try:
            import pandas as pd
            prices = pd.read_parquet(prices_file)
            latest_date = prices.index.max()
            days_old = (datetime.now().date() - latest_date.date()).days
            
            status["latest_price_date"] = latest_date.strftime('%Y-%m-%d')
            status["days_old"] = days_old
            status["num_etfs"] = len(prices.columns)
        except Exception as e:
            status["error"] = str(e)
    
    if signals_dir.exists():
        factor_files = list(signals_dir.glob('*_scores.parquet'))
        status["num_factor_files"] = len(factor_files)
    
    return status


@router.post("/update")
async def trigger_data_update() -> Dict:
    """Trigger data collection (placeholder for async task)."""
    return {
        "status": "not_implemented",
        "message": "Data update requires manual script execution",
        "command": "python scripts/collect_etf_universe.py"
    }

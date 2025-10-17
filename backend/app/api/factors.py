"""Factor API endpoints."""

from fastapi import APIRouter
from typing import List, Dict

from app.services.factor_service import factor_service


router = APIRouter()


@router.get("/latest")
async def get_latest_scores() -> Dict:
    """Get latest factor scores for all ETFs."""
    import os
    import traceback
    from pathlib import Path
    cwd = Path.cwd()
    signals_dir = factor_service.signals_dir

    # Debug info
    debug_info = {
        "cwd": str(cwd),
        "data_dir": str(factor_service.data_dir),
        "signals_dir": str(signals_dir),
        "signals_dir_exists": signals_dir.exists(),
    }

    if signals_dir.exists():
        debug_info["files"] = [str(f) for f in signals_dir.iterdir()]

    try:
        # Clear cache to force fresh load
        factor_service._cache.clear()

        scores = factor_service.load_latest_scores()
        debug_info["scores_empty"] = scores.empty
        debug_info["scores_shape"] = str(scores.shape) if not scores.empty else "N/A"

        if scores.empty:
            return {"error": "No factor scores available", "debug": debug_info}
        return scores.to_dict('index')
    except Exception as e:
        debug_info["exception"] = str(e)
        debug_info["traceback"] = traceback.format_exc()
        return {"error": "Exception during loading", "debug": debug_info}


@router.get("/recommendations")
async def get_recommendations(
    num_positions: int = 20,
    optimizer: str = 'mvo'
) -> List[Dict]:
    """Get top ETF recommendations."""
    return factor_service.get_recommendations(num_positions, optimizer)


@router.get("/{ticker}/history")
async def get_ticker_history(
    ticker: str,
    days: int = 90
) -> List[Dict]:
    """Get factor score history for specific ticker."""
    return factor_service.get_ticker_history(ticker, days)

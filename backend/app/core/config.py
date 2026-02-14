"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "ETFTrader"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://etftrader:password@localhost:5432/etftrader"
    DATABASE_URL_SYNC: str = "postgresql://etftrader:password@localhost:5432/etftrader"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Data Paths (relative paths work in Docker /app)
    DATA_DIR: str = "data"
    RESULTS_DIR: str = str(Path.home() / "trading")

    # Paper Trading
    DEFAULT_COMMISSION: float = 1.0
    DEFAULT_SLIPPAGE_PCT: float = 0.0005

    # Stop-Loss
    DEFAULT_STOP_LOSS_PCT: float = 0.12
    USE_VIX_ADJUSTMENT: bool = True

    # Portfolio
    DEFAULT_NUM_POSITIONS: int = 20
    DEFAULT_DRIFT_THRESHOLD: float = 0.05
    DEFAULT_OPTIMIZER: str = "mvo"

    # Interactive Brokers Gateway
    IB_HOST: str = "127.0.0.1"
    IB_PORT: int = 4002  # 4002 = paper trading, 4001 = live
    IB_CLIENT_ID: int = 1
    IB_TIMEOUT: int = 10  # Connection timeout in seconds
    IB_READONLY: bool = True  # Read-only mode (no order execution)
    IB_ENABLED: bool = False  # Master switch: set True to enable IB connectivity
    IB_RECONNECT_INTERVAL: int = 30  # Seconds between reconnection attempts
    IB_ACCOUNT: str = ""  # IB account ID (blank = auto-detect first account)

    # Authentication (Phase 2)
    # SECRET_KEY: str = "your-secret-key-here"
    # ALGORITHM: str = "HS256"
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()

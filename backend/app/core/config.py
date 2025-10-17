"""Application configuration."""

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
    RESULTS_DIR: str = "results"

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

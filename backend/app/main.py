"""Main FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import init_db, close_db
from app.api import portfolios, positions, trades, factors, performance, risk, data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    Startup: Initialize database
    Shutdown: Close database connections
    """
    # Startup
    print("🚀 Starting ETFTrader API...")
    await init_db()
    print("✅ Database initialized")

    yield

    # Shutdown
    print("👋 Shutting down ETFTrader API...")
    await close_db()
    print("✅ Database connections closed")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AQR Multi-Factor ETF Investment Strategy - Backend API",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Include routers
app.include_router(portfolios.router, prefix="/api/portfolios", tags=["Portfolios"])
app.include_router(positions.router, prefix="/api/positions", tags=["Positions"])
app.include_router(trades.router, prefix="/api/trades", tags=["Trades"])
app.include_router(factors.router, prefix="/api/factors", tags=["Factors"])
app.include_router(performance.router, prefix="/api/performance", tags=["Performance"])
app.include_router(risk.router, prefix="/api/risk", tags=["Risk"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )

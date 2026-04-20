"""
ROX Proven Edge Engine v3.0 - Core Module
========================================
Core functionality including configuration, logging, and news.
"""

from .config import (
    # Enums
    MarketRegime,
    TradeDirection,
    ConvictionLevel,
    SentimentZone,
    # Config Classes
    NewsConfig,
    APIConfig,
    AgentConfig,
    RiskLimits,
    LoggingConfig,
    SystemConfig,
    # Functions
    load_env_config,
    get_system_config,
    # Constants
    DEFAULT_CONFIG,
    NIFTY_50_STOCKS,
    SECTOR_MAPPING,
    BASE_DIR,
    DATA_DIR,
    LOG_DIR,
)

from .logger import (
    setup_logging,
    get_logger,
    LoggerMixin,
    log_execution,
    log_async_execution,
    TradeLoggerAdapter,
    AuditLogger,
    init_logging,
)

from .news_fetcher import (
    NewsArticle,
    GeopoliticalEvent,
    StockPriceNews,
    NewsFetcher,
    fetch_news_sync,
)

__all__ = [
    # Enums
    "MarketRegime",
    "TradeDirection",
    "ConvictionLevel",
    "SentimentZone",
    # Config Classes
    "NewsConfig",
    "APIConfig",
    "AgentConfig",
    "RiskLimits",
    "LoggingConfig",
    "SystemConfig",
    # Config Functions
    "load_env_config",
    "get_system_config",
    # Constants
    "DEFAULT_CONFIG",
    "NIFTY_50_STOCKS",
    "SECTOR_MAPPING",
    "BASE_DIR",
    "DATA_DIR",
    "LOG_DIR",
    # Logging
    "setup_logging",
    "get_logger",
    "LoggerMixin",
    "log_execution",
    "log_async_execution",
    "TradeLoggerAdapter",
    "AuditLogger",
    "init_logging",
    # News
    "NewsArticle",
    "GeopoliticalEvent",
    "StockPriceNews",
    "NewsFetcher",
    "fetch_news_sync",
]

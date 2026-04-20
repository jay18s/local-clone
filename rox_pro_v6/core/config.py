"""
ROX Proven Edge Engine v3.0 - Production Configuration
=====================================================
Central configuration with environment variable support and Windows compatibility.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import logging
from datetime import time

# Windows-compatible path handling
def get_base_dir() -> Path:
    """Get base directory in a cross-platform way."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return Path(sys.executable).parent
    return Path(__file__).parent.parent.resolve()

BASE_DIR = get_base_dir()
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# Ensure directories exist
for directory in [DATA_DIR, LOG_DIR, CONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "BULL"
    BEAR = "BEAR"
    CONSOLIDATION = "CONSOLIDATION"
    CORRECTION = "CORRECTION"
    MILD_BULL = "MILD_BULL"
    MILD_BEAR = "MILD_BEAR"


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class ConvictionLevel(Enum):
    """Conviction levels"""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SKIP = "SKIP"


class SentimentZone(Enum):
    """Sentiment zones"""
    EUPHORIA = "EUPHORIA"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    PANIC = "PANIC"


@dataclass
class NewsConfig:
    """Configuration for news fetching"""
    # Geopolitical news sources
    geopolitical_sources: List[str] = field(default_factory=lambda: [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
        "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms"
    ])
    
    # Stock news sources
    stock_news_sources: List[str] = field(default_factory=lambda: [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "https://www.livemint.com/rss/markets"
    ])
    
    # API configurations
    news_api_key: str = ""
    news_api_enabled: bool = False
    cache_duration_minutes: int = 30
    max_articles_per_fetch: int = 50
    
    # Keywords for filtering
    geopolitical_keywords: List[str] = field(default_factory=lambda: [
        "war", "conflict", "sanctions", "trade", "tariff", "election",
        "geopolitical", "diplomatic", "treaty", "summit", "crisis",
        "oil", "opec", "china", "usa", "russia", "middle east", "asia"
    ])
    
    stock_keywords: List[str] = field(default_factory=lambda: [
        "nifty", "sensex", "stock", "share", "market", "nse", "bse",
        "rally", "crash", "earnings", "results", "ipo", "fpo",
        "dividend", "bonus", "split", "buyback"
    ])


@dataclass
class APIConfig:
    """Configuration for external APIs"""
    # FYERS API
    fyers_api_key: str = ""
    fyers_access_token: str = ""
    fyers_app_id: str = ""
    fyers_enabled: bool = False
    
    # Zerodha Kite
    zerodha_api_key: str = ""
    zerodha_access_token: str = ""
    zerodha_enabled: bool = False
    
    # NSE India
    nse_enabled: bool = True
    nse_request_delay: float = 1.0
    
    # Yahoo Finance
    yahoo_enabled: bool = True
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_enabled: bool = False


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    name: str
    domain: str
    baseline_weight: float
    current_weight: float = None
    
    def __post_init__(self):
        if self.current_weight is None:
            self.current_weight = self.baseline_weight


@dataclass
class RiskLimits:
    """Non-negotiable risk limits"""
    max_risk_per_trade: float = 0.02
    max_single_position: float = 0.15
    max_portfolio_heat: float = 0.08
    max_sector_exposure: float = 0.25
    min_cash_buffer: float = 0.05
    min_risk_reward_ratio: float = 1.5
    # Options-specific limits (added for OPTIMUS)
    max_options_exposure: float = float(os.getenv("MAX_OPTIONS_EXPOSURE", "0.10"))
    max_option_premium_per_trade: float = float(os.getenv("MAX_OPTION_PREMIUM", "0.02"))
    options_data_source: str = os.getenv("OPTIONS_DATA_SOURCE", "nse")


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_file_prefix: str = "rox_engine"
    max_log_size_mb: int = 10
    backup_count: int = 5
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class SystemConfig:
    """Main system configuration"""
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    news: NewsConfig = field(default_factory=NewsConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    target_win_rate: float = 0.65
    target_monthly_return: float = 0.10
    swing_trade_days: tuple = (2, 5)
    atr_stop_multiplier: float = 1.5
    min_data_completeness: int = 4
    default_portfolio_value: float = 1000000.0
    market_open_time: time = field(default_factory=lambda: time(9, 15))
    market_close_time: time = field(default_factory=lambda: time(15, 30))
    
    def __post_init__(self):
        if not self.agents:
            self.agents = {
                "ORION": AgentConfig("ORION", "Technical Analysis", 0.20),
                "VESPER": AgentConfig("VESPER", "Flow Analysis", 0.18),
                "KAIRO": AgentConfig("KAIRO", "Sentiment Analysis", 0.12),
                "SENTINEL": AgentConfig("SENTINEL", "Derivatives Analysis", 0.15),
                "NEXUS": AgentConfig("NEXUS", "Fundamental Analysis", 0.15),
                "PRUDENCE": AgentConfig("PRUDENCE", "Risk Management", 0.10),
                "CATALYST": AgentConfig("CATALYST", "Event Calendar", 0.10),
                "OPTIMUS": AgentConfig("OPTIMUS", "F&O Weekly Expiry", 0.15),
            }
    
    def normalize_weights(self):
        """Normalize all agent weights to sum to 1.0"""
        total = sum(a.current_weight for a in self.agents.values())
        if total > 0:
            for agent in self.agents.values():
                agent.current_weight = agent.current_weight / total


def load_env_config() -> Dict:
    """Load configuration from environment variables."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    return {
        "news": NewsConfig(
            news_api_key=os.getenv("NEWS_API_KEY", ""),
            news_api_enabled=os.getenv("NEWS_API_ENABLED", "false").lower() == "true",
            cache_duration_minutes=int(os.getenv("NEWS_CACHE_DURATION", "30")),
            max_articles_per_fetch=int(os.getenv("MAX_ARTICLES_PER_FETCH", "50")),
        ),
        "api": APIConfig(
            fyers_api_key=os.getenv("FYERS_API_KEY", ""),
            fyers_access_token=os.getenv("FYERS_ACCESS_TOKEN", ""),
            fyers_app_id=os.getenv("FYERS_APP_ID", ""),
            fyers_enabled=os.getenv("FYERS_ENABLED", "false").lower() == "true",
            zerodha_api_key=os.getenv("ZERODHA_API_KEY", ""),
            zerodha_access_token=os.getenv("ZERODHA_ACCESS_TOKEN", ""),
            zerodha_enabled=os.getenv("ZERODHA_ENABLED", "false").lower() == "true",
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_enabled=os.getenv("REDIS_ENABLED", "false").lower() == "true",
        ),
        "logging": LoggingConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_to_file=os.getenv("LOG_TO_FILE", "true").lower() == "true",
            log_to_console=os.getenv("LOG_TO_CONSOLE", "true").lower() == "true",
        ),
        "default_portfolio_value": float(os.getenv("PORTFOLIO_VALUE", "1000000")),
    }


def get_system_config() -> SystemConfig:
    """Get system configuration with environment variable overrides."""
    env_config = load_env_config()
    
    config = SystemConfig(
        news=env_config.get("news", NewsConfig()),
        api=env_config.get("api", APIConfig()),
        logging=env_config.get("logging", LoggingConfig()),
        default_portfolio_value=env_config.get("default_portfolio_value", 1000000.0),
    )
    
    return config


DEFAULT_CONFIG = get_system_config()

NIFTY_50_STOCKS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
    "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LTIM",
    "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO"
]

SECTOR_MAPPING = {
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE"],
    "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM"],
    "Energy": ["RELIANCE", "ONGC", "BPCL", "POWERGRID", "NTPC", "COALINDIA"],
    "Auto": ["MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
    "Metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
    "Infrastructure": ["LT", "ULTRACEMCO", "GRASIM"],
    "Telecom": ["BHARTIARTL"],
    "Others": ["ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "TITAN", "UPL", "ONGC"]
}

"""
ROX Proven Edge Engine v4.0 Unified - Configuration
====================================================
Merged from rox_v32_enhanced (v3.2) + rox_pro_v4 (v4.0).
"""

import os, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from datetime import time

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent.resolve()

BASE_DIR   = get_base_dir()
DATA_DIR   = BASE_DIR / "data"
LOG_DIR    = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"
for _d in [DATA_DIR, LOG_DIR, CONFIG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

class MarketRegime(Enum):
    BULL = "BULL"; MILD_BULL = "MILD_BULL"; CONSOLIDATION = "CONSOLIDATION"
    MILD_BEAR = "MILD_BEAR"; BEAR = "BEAR"; CORRECTION = "CORRECTION"

class TradeDirection(Enum):
    LONG = "LONG"; SHORT = "SHORT"; NEUTRAL = "NEUTRAL"

class ConvictionLevel(Enum):
    VERY_HIGH = "VERY_HIGH"; HIGH = "HIGH"; MEDIUM = "MEDIUM"
    LOW = "LOW"; SKIP = "SKIP"

class SentimentZone(Enum):
    EUPHORIA = "EUPHORIA"; BULLISH = "BULLISH"; NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"; PANIC = "PANIC"

class VIXRegime(Enum):
    LOW = "LOW"; MODERATE = "MODERATE"; HIGH = "HIGH"; EXTREME = "EXTREME"

class OptionType(Enum):
    CALL = "CE"; PUT = "PE"

class ProductType(Enum):
    NRML = "NRML"; MIS = "MIS"; CNC = "CNC"

@dataclass
class NewsConfig:
    geopolitical_sources: List[str] = field(default_factory=lambda: [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://feeds.reuters.com/reuters/topNews",
        "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
    ])
    stock_news_sources: List[str] = field(default_factory=lambda: [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "https://www.livemint.com/rss/markets",
    ])
    news_api_key: str = ""; news_api_enabled: bool = False
    cache_duration_minutes: int = 30; max_articles_per_fetch: int = 50
    geopolitical_keywords: List[str] = field(default_factory=lambda: [
        "war","conflict","sanctions","trade","tariff","election","geopolitical",
        "diplomatic","treaty","summit","crisis","oil","opec","china","usa","russia",
    ])
    stock_keywords: List[str] = field(default_factory=lambda: [
        "nifty","sensex","stock","share","market","nse","bse","rally","crash",
        "earnings","results","ipo","dividend","bonus","split","buyback",
    ])

@dataclass
class APIConfig:
    fyers_api_key: str = ""; fyers_access_token: str = ""; fyers_app_id: str = ""
    fyers_enabled: bool = False
    zerodha_api_key: str = ""; zerodha_access_token: str = ""; zerodha_enabled: bool = False
    nse_enabled: bool = True; nse_request_delay: float = 1.0; yahoo_enabled: bool = True
    redis_host: str = "localhost"; redis_port: int = 6379; redis_enabled: bool = False

@dataclass
class LoggingConfig:
    log_level: str = "INFO"; log_to_file: bool = True; log_to_console: bool = True
    log_file_prefix: str = "rox_engine"; max_log_size_mb: int = 10; backup_count: int = 5
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

# ═══════════════════════════════════════════════════════════════════════════
# LLM Configuration (v4.1 Enhanced)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    """Configuration for LLM-powered agents (OpenRouter API)."""
    enabled: bool = True
    api_key: str = ""  # OpenRouter API key (set via OPEN_ROUTER_API env var)
    model_name: str = ""  # Primary model (from OPEN_ROUTER_MODEL env)
    fallback_model: str = ""
    max_retries: int = 3
    timeout_seconds: int = 30
    cache_ttl_seconds: int = 300  # 5 minutes
    cache_enabled: bool = True
    temperature: float = 0.3  # Low temperature for analytical tasks
    max_output_tokens: int = 2048
    rate_limit_per_minute: int = 15
    log_prompts: bool = True
    log_responses: bool = True
    fallback_on_error: bool = True  # Graceful degradation

    # Feature flags for LLM enhancements
    regime_detection_enabled: bool = True  # P1.1
    cross_examination_enabled: bool = True  # P1.2
    news_impact_enabled: bool = True  # P2.1
    options_strategist_enabled: bool = True  # P2.2
    pattern_validation_enabled: bool = True  # P3.1
    meta_learning_enabled: bool = True  # P3.2
    calibration_enabled: bool = True  # F1
    quality_checklist_enabled: bool = True  # F2

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load LLM configuration from environment variables."""
        api_key = os.getenv("OPEN_ROUTER_API", "") or os.getenv("OPENROUTER_API_KEY", "")
        
        model_name = os.getenv("LLM_MODEL", "") or \
                     os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")
        
        return cls(
            enabled=os.getenv("LLM_ENABLED", "true").lower() == "true",
            api_key=api_key,
            model_name=model_name,
            cache_ttl_seconds=int(os.getenv("LLM_CACHE_TTL", "300")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            regime_detection_enabled=os.getenv("LLM_REGIME_DETECTION", "true").lower() == "true",
            cross_examination_enabled=os.getenv("LLM_CROSS_EXAMINATION", "true").lower() == "true",
            news_impact_enabled=os.getenv("LLM_NEWS_IMPACT", "true").lower() == "true",
            options_strategist_enabled=os.getenv("LLM_OPTIONS_STRATEGIST", "true").lower() == "true",
            pattern_validation_enabled=os.getenv("LLM_PATTERN_VALIDATION", "true").lower() == "true",
            meta_learning_enabled=os.getenv("LLM_META_LEARNING", "true").lower() == "true",
        )

@dataclass
class RiskLimits:
    max_risk_per_trade: float = 0.02; max_single_position: float = 0.15
    max_portfolio_heat: float = 0.08; max_sector_exposure: float = 0.25
    min_cash_buffer: float = 0.05; min_risk_reward_ratio: float = 1.5
    max_options_exposure: float = float(os.getenv("MAX_OPTIONS_EXPOSURE","0.10"))
    max_option_premium_per_trade: float = float(os.getenv("MAX_OPTION_PREMIUM","0.02"))
    options_data_source: str = os.getenv("OPTIONS_DATA_SOURCE","nse")

@dataclass
class FnoRiskLimits:
    max_portfolio_delta: float = 0.30; max_portfolio_gamma: float = 0.05
    max_portfolio_theta: float = -1000.0; max_portfolio_vega: float = 5000.0
    vix_threshold: float = 25.0; max_short_premium: float = 50000.0
    min_days_to_expiry: int = 7

@dataclass
class FnoConfig:
    enabled: bool = True; default_product_type: ProductType = ProductType.NRML
    auto_squareoff_time: str = "15:15"; max_expiry_days: int = 7; margin_buffer: float = 1.2
    physical_settlement_monitoring: bool = True; physical_settlement_days: int = 5
    physical_settlement_auto_exit: bool = True; mwpl_disclosure_threshold: float = 0.85
    ic_min_vix: float = 12.0; ic_max_vix: float = 22.0
    calendar_min_term_structure: float = 2.0; max_short_options_premium: float = 50000.0
    websocket_enabled: bool = True

@dataclass
class AgentConfig:
    name: str; domain: str; baseline_weight: float; current_weight: float = None
    def __post_init__(self):
        if self.current_weight is None:
            self.current_weight = self.baseline_weight

@dataclass
class SystemConfig:
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    risk_limits: RiskLimits      = field(default_factory=RiskLimits)
    fno_limits: FnoRiskLimits    = field(default_factory=FnoRiskLimits)
    fno_config: FnoConfig        = field(default_factory=FnoConfig)
    news: NewsConfig             = field(default_factory=NewsConfig)
    api: APIConfig               = field(default_factory=APIConfig)
    logging: LoggingConfig       = field(default_factory=LoggingConfig)
    llm: LLMConfig               = field(default_factory=LLMConfig)  # v4.1 Enhanced
    portfolio_value: float       = 1000000.0
    target_win_rate: float       = 0.65
    target_monthly_return: float = 0.10
    swing_trade_days: tuple      = (2, 5)
    atr_stop_multiplier: float   = 1.5
    min_data_completeness: int   = 4
    market_open_time: time       = field(default_factory=lambda: time(9, 15))
    market_close_time: time      = field(default_factory=lambda: time(15, 30))

    def __post_init__(self):
        if not self.agents:
            self.agents = {
                "ORION":    AgentConfig("ORION",    "Technical Analysis",    0.20),
                "VESPER":   AgentConfig("VESPER",   "Flow Analysis",         0.18),
                "KAIRO":    AgentConfig("KAIRO",    "Sentiment Analysis",    0.12),
                "SENTINEL": AgentConfig("SENTINEL", "Derivatives Analysis",  0.15),
                "NEXUS":    AgentConfig("NEXUS",    "Fundamental Analysis",  0.15),
                "PRUDENCE": AgentConfig("PRUDENCE", "Risk Management",       0.10),
                "CATALYST": AgentConfig("CATALYST", "Event Calendar",        0.10),
                "OPTIMUS":  AgentConfig("OPTIMUS",  "F&O Weekly Expiry",     0.15),
                "HERMES":   AgentConfig("HERMES",   "Execution",             0.08),
                "THETA":    AgentConfig("THETA",    "Greeks Management",     0.10),
                "DELTA":    AgentConfig("DELTA",    "Settlement/Compliance", 0.07),
            }

    def normalize_weights(self):
        total = sum(a.current_weight for a in self.agents.values())
        if total > 0:
            for a in self.agents.values():
                a.current_weight /= total

    @property
    def default_portfolio_value(self) -> float:
        return self.portfolio_value

# --- Market constants ---
NIFTY_50_STOCKS = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
    "BAJFINANCE","BAJAJFINSV","BPCL","BHARTIARTL","BRITANNIA","CIPLA","COALINDIA",
    "DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC",
    "JSWSTEEL","KOTAKBANK","LT","M&M","MARUTI","NESTLEIND","NTPC","ONGC",
    "POWERGRID","RELIANCE","SBILIFE","SBIN","SUNPHARMA","TATACONSUM","TATAMOTORS_CV","TATAMOTORS_PV",
    "TATASTEEL","TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO",
]

SECTOR_MAPPING = {
    "Banking":        ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK","BAJFINANCE","BAJAJFINSV","SBILIFE","HDFCLIFE"],
    "IT":             ["TCS","INFY","WIPRO","HCLTECH","TECHM"],
    "Energy":         ["RELIANCE","ONGC","BPCL","POWERGRID","NTPC","COALINDIA"],
    "Auto":           ["MARUTI","M&M","TATAMOTORS_CV","TATAMOTORS_PV","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT"],
    "Metals":         ["TATASTEEL","JSWSTEEL","HINDALCO"],
    "Pharma":         ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB"],
    "FMCG":           ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","TATACONSUM"],
    "Infrastructure": ["LT","ULTRACEMCO","GRASIM"],
    "Telecom":        ["BHARTIARTL"],
    "Others":         ["ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","TITAN","UPL"],
}

FNO_LOT_SIZES = {
    "NIFTY":75,"BANKNIFTY":15,"FINNIFTY":40,"MIDCPNIFTY":50,"SENSEX":10,"BANKEX":15,
    "RELIANCE":250,"TCS":175,"HDFCBANK":550,"ICICIBANK":700,"INFY":400,"ITC":1600,
    "SBIN":750,"BHARTIARTL":425,
}
DEFAULT_LOT_SIZE = 50

def get_lot_size(symbol: str) -> int:
    return FNO_LOT_SIZES.get(symbol.upper(), DEFAULT_LOT_SIZE)

def get_vix_regime(vix: float) -> VIXRegime:
    if vix < 12: return VIXRegime.LOW
    elif vix < 18: return VIXRegime.MODERATE
    elif vix < 25: return VIXRegime.HIGH
    return VIXRegime.EXTREME

def get_conviction_level(conviction: int) -> ConvictionLevel:
    if conviction >= 85: return ConvictionLevel.VERY_HIGH
    elif conviction >= 75: return ConvictionLevel.HIGH
    elif conviction >= 65: return ConvictionLevel.MEDIUM
    elif conviction >= 50: return ConvictionLevel.LOW
    return ConvictionLevel.SKIP

def load_env_config() -> dict:
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass
    return {
        "news": NewsConfig(
            news_api_key=os.getenv("NEWS_API_KEY",""),
            news_api_enabled=os.getenv("NEWS_API_ENABLED","false").lower()=="true",
        ),
        "api": APIConfig(
            fyers_api_key=os.getenv("FYERS_API_KEY",""),
            fyers_access_token=os.getenv("FYERS_ACCESS_TOKEN",""),
            fyers_app_id=os.getenv("FYERS_APP_ID",""),
            fyers_enabled=os.getenv("FYERS_ENABLED","false").lower()=="true",
            zerodha_api_key=os.getenv("ZERODHA_API_KEY",""),
            zerodha_access_token=os.getenv("ZERODHA_ACCESS_TOKEN",""),
            zerodha_enabled=os.getenv("ZERODHA_ENABLED","false").lower()=="true",
            redis_host=os.getenv("REDIS_HOST","localhost"),
            redis_port=int(os.getenv("REDIS_PORT","6379")),
            redis_enabled=os.getenv("REDIS_ENABLED","false").lower()=="true",
        ),
        "logging": LoggingConfig(log_level=os.getenv("LOG_LEVEL","INFO")),
        "portfolio_value": float(os.getenv("PORTFOLIO_VALUE","1000000")),
    }

def get_system_config() -> SystemConfig:
    env = load_env_config()
    cfg = SystemConfig(
        news=env.get("news", NewsConfig()),
        api=env.get("api", APIConfig()),
        logging=env.get("logging", LoggingConfig()),
        portfolio_value=env.get("portfolio_value", 1000000.0),
    )
    cfg.normalize_weights()
    return cfg

DEFAULT_CONFIG = get_system_config()


# ── FIX-STT-05: Securities Transaction Tax schedule ──────────────────────────
# STT on F&O is increasing from April 1, 2026. All strategy cost models
# must reference these constants rather than hardcoded values.
from datetime import date as _date

STT_HIKE_EFFECTIVE_DATE = _date(2026, 4, 1)

# Current rates (pre-hike, valid until March 31, 2026)
STT_FUT_CURRENT   = 0.0002   # 0.02% of futures turnover
STT_OPT_BUY_CURRENT = 0.001  # 0.10% of options premium (buy side)

# Post-hike rates (effective April 1, 2026)
STT_FUT_NEW       = 0.0005   # 0.05% of futures turnover  (+150% increase)
STT_OPT_BUY_NEW   = 0.0015   # 0.15% of options premium    (+50% increase)


def get_current_stt(instrument_type: str = "option") -> float:
    """
    Return the applicable STT rate for today's date.

    Args:
        instrument_type: "future" or "option"

    Returns:
        STT rate as a decimal (e.g. 0.0005 = 0.05%)
    """
    today = _date.today()
    if instrument_type == "future":
        return STT_FUT_NEW if today >= STT_HIKE_EFFECTIVE_DATE else STT_FUT_CURRENT
    else:
        return STT_OPT_BUY_NEW if today >= STT_HIKE_EFFECTIVE_DATE else STT_OPT_BUY_CURRENT


def days_to_stt_hike() -> int:
    """Return number of calendar days until STT hike. Negative means hike already active."""
    return (_date(2026, 4, 1) - _date.today()).days

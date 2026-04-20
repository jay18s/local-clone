"""
ROX PROVEN EDGE ENGINE v6.0 — Master Configuration
All settings centralized for easy tuning.
v5.1 — Added GEMINI_MODEL_ROUTING, response caching, RAM management.
v6.0 — ACTIVATED 2026-04-17 — Closed-loop learning mode.
       MetaLearner thresholds lowered: min_trades 50→20, min_win_rate 0.15→0.10.
"""

ROX_VERSION = "v6.0"

import os
import sys
import platform
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO & RISK
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PortfolioConfig:
    initial_capital: float = 10_00_000.0  # INR
    risk_per_trade_pct: float = 1.5        # % of portfolio
    max_portfolio_risk_pct: float = 3.0    # max total risk
    max_position_size_pct: float = 25.0    # single position cap
    max_sector_allocation_pct: float = 35.0  # sector concentration cap
    max_positions: int = 6


# ═══════════════════════════════════════════════════════════════════
# OPENROUTER MODEL ROUTING (centralized — edit this, not code)
# ═══════════════════════════════════════════════════════════════════
# Switch models without touching code.  FAST_MODEL is used for all
# non-critical calls (regime detection, pattern matching, news).
# SMART_MODEL is reserved for final trading decisions only.
# ═══════════════════════════════════════════════════════════════════

OPENROUTER_MODELS = {
    "planner": os.getenv("ROX_PLANNER_MODEL", os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")),
    "debate": os.getenv("ROX_DEBATE_MODEL", os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")),
    "analysis": os.getenv("ROX_ANALYSIS_MODEL", os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")),
    "swarm": os.getenv("ROX_SWARM_MODEL", os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")),
    "FAST_MODEL": os.getenv("ROX_FAST_MODEL", os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")),
    "SMART_MODEL": os.getenv("ROX_SMART_MODEL", os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")),
    "NEWS_MODEL": os.getenv("ROX_NEWS_MODEL", os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")),
    "CACHE_TTL_MINUTES": int(os.getenv("ROX_CACHE_TTL", "5")),
    "MAX_PARALLEL_LLM_CALLS": int(os.getenv("ROX_MAX_PARALLEL", "7")),
}

# Backward-compatible alias for code that still references GEMINI_MODEL_ROUTING
GEMINI_MODEL_ROUTING = OPENROUTER_MODELS


# ═══════════════════════════════════════════════════════════════════
# OPENROUTER LLM API
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("OPEN_ROUTER_API", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    
    # Model assignments — now centralized via OPENROUTER_MODELS
    model_pro: str = field(default_factory=lambda: OPENROUTER_MODELS["SMART_MODEL"])
    model_flash: str = field(default_factory=lambda: OPENROUTER_MODELS["FAST_MODEL"])
    model_news: str = field(default_factory=lambda: OPENROUTER_MODELS["NEWS_MODEL"])
    
    # Temperature
    temperature_cot: float = 0.3    # Low for analytical reasoning
    temperature_debate: float = 0.5  # Medium for diverse perspectives
    temperature_planning: float = 0.4
    temperature_news: float = 0.2   # Low for factual extraction
    
    # Rate limiting
    max_concurrent: int = field(default_factory=lambda: OPENROUTER_MODELS["MAX_PARALLEL_LLM_CALLS"])
    requests_per_minute: int = 15
    retry_max: int = 3
    retry_delay_sec: float = 2.0
    timeout_sec: float = 30.0
    
    # Response caching
    cache_ttl_seconds: int = field(default_factory=lambda: OPENROUTER_MODELS["CACHE_TTL_MINUTES"] * 60)
    cache_max_entries: int = 100
    
    # Token limits
    max_input_tokens: int = 8000
    max_output_tokens: int = 4096


# ═══════════════════════════════════════════════════════════════════
# REASONING UPGRADES (v5.0)
# ═══════════════════════════════════════════════════════════════════

class ComplexityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ReasoningConfig:
    # Chain-of-Thought
    cot_enabled: bool = True
    cot_steps_low: int = 3
    cot_steps_medium: int = 5
    cot_steps_high: int = 7
    
    # Debate Protocol
    debate_enabled: bool = True
    debate_rounds_low: int = 0      # Skip debate in calm markets
    debate_rounds_medium: int = 1    # Quick bull/bear
    debate_rounds_high: int = 2      # Full bull/bear/neutral
    debate_rounds_extreme: int = 3   # Maximum deliberation
    
    # Pattern Memory
    pattern_memory_enabled: bool = True
    pattern_db_path: str = "data/pattern_memory.duckdb"  # DuckDB (migrated from SQLite)
    pattern_match_count_low: int = 2
    pattern_match_count_medium: int = 3
    pattern_match_count_high: int = 5
    pattern_match_count_extreme: int = 5
    
    # Confidence Calibration
    calibration_enabled: bool = True
    confidence_threshold_low: float = 70.0    # Higher bar in calm markets
    confidence_threshold_medium: float = 60.0
    confidence_threshold_high: float = 55.0
    confidence_threshold_extreme: float = 50.0
    
    # Calibration signal weights (must sum to 1.0)
    weight_debate_agreement: float = 0.25
    weight_pattern_match: float = 0.20
    weight_technical_alignment: float = 0.20
    weight_volume_confirmation: float = 0.15
    weight_regime_consistency: float = 0.10
    weight_anti_consensus: float = 0.10
    
    # Adaptive Position Sizing
    max_position_pct_low: float = 1.0     # Full size
    max_position_pct_medium: float = 1.0
    max_position_pct_high: float = 0.75    # Reduce in volatile
    max_position_pct_extreme: float = 0.5  # Half size in crisis
    
    # Rule-Based Validator
    rule_validator_enabled: bool = True
    min_rr_ratio: float = 1.5
    rsi_long_min: float = 40.0
    rsi_short_max: float = 60.0
    rsi_overbought: float = 75.0
    rsi_oversold: float = 25.0
    volume_min_pct_of_avg: float = 0.8
    price_above_sma20_required: bool = True
    
    # Regime Cache
    regime_cache_enabled: bool = True
    regime_cache_ttl_high_conf: int = 900    # 15 min for confidence > 75%
    regime_cache_ttl_med_conf: int = 420     # 7 min for 50-75%
    regime_cache_min_confidence: float = 50.0
    regime_invalidate_vix_delta: float = 2.0
    regime_invalidate_dma_break: bool = True
    
    # MetaLearner Conditional
    meta_learner_enabled: bool = True
    meta_learner_min_trades: int = 20
    meta_learner_min_win_rate: float = 0.10
    
    # Self-Reflector
    self_reflector_enabled: bool = True


# ═══════════════════════════════════════════════════════════════════
# MARKET DATA
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MarketDataConfig:
    # Fyers API
    fyers_app_id: str = field(default_factory=lambda: os.getenv("FYERS_APP_ID", ""))
    fyers_app_secret: str = field(default_factory=lambda: os.getenv("FYERS_APP_SECRET", ""))
    fyers_access_token: str = field(default_factory=lambda: os.getenv("FYERS_ACCESS_TOKEN", ""))
    fyers_redirect_uri: str = "https://trade.fyers.in/"
    
    # Symbols
    symbols: list = field(default_factory=lambda: [
        "NSE:SBIN", "NSE:ICICIBANK", "NSE:HDFCBANK", "NSE:AXISBANK",
        "NSE:KOTAKBANK", "NSE:BAJFINANCE", "NSE:RELIANCE", "NSE:TCS",
        "NSE:INFY", "NSE:HCLTECH", "NSE:WIPRO", "NSE:TATASTEEL",
        "NSE:JSWSTEEL", "NSE:HINDALCO", "NSE:TATAMOTORS", "NSE:LT",
        "NSE:MARUTI", "NSE:ASIANPAINT", "NSE:SUNPHARMA", "NSE:CIPLA",
        "NSE:DRREDDY", "NSE:COALINDIA", "NSE:NTPC", "NSE:POWERGRID",
        "NSE:TITAN", "NSE:ULTRACEMCO", "NSE:GRASIM", "NSE:ITC",
        "NSE:HINDUNILVR", "NSE:BAJAJFINSV", "NSE:ADANIENT", "NSE:ADANIPORTS",
        "NSE:TITAN", "NSE:BPCL", "NSE:IOC", "NSE:ONGC",
        "NSE:HCLTECH", "NSE:TECHM", "NSE:DIVISLAB", "NSE:LT",
        "NSE:VEDL", "NSE:HINDALCO", "NSE:TATAPOWER", "NSE:DLF",
        "NSE:M_MFIN", "NSE:SBILIFE", "NSE:BRITANNIA",
    ])
    
    # Index symbols
    index_symbols: list = field(default_factory=lambda: [
        "NSE:NIFTY", "NSE:BANKNIFTY", "NSE:SENSEX", "NSE:FINNIFTY", "NSE:NIFTYBANK"
    ])
    
    # Timeframes
    timeframes_ohlcv: list = field(default_factory=lambda: ["5m", "15m", "1h", "1D"])
    candle_history_days: int = 5
    
    # Rate limiting
    fyers_rate_limit: int = 10  # requests per second


# ═══════════════════════════════════════════════════════════════════
# TRADING RULES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TradingRules:
    # Session
    market_open: str = "09:15"
    market_close: str = "15:30"
    
    # Entry rules
    min_signal_strength: str = "MEDIUM"  # LOW, MEDIUM, HIGH, STRONG
    
    # Stop loss
    default_sl_pct: float = 2.0       # % from entry
    max_sl_pct: float = 4.0
    trail_sl_after: float = 1.5       # % in profit before trailing
    
    # Target
    target_1_pct: float = 1.5
    target_2_pct: float = 3.0
    
    # Sector
    max_same_sector_trades: int = 2
    
    # Expiry rules
    no_new_positions_before_expiry_hours: int = 2
    
    # News
    respect_news_restrictions: bool = True


# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = "[%(asctime)s] %(levelname)-7s %(name)-25s %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    log_file: str = "logs/rox_engine.log"
    max_log_files: int = 7


# ═══════════════════════════════════════════════════════════════════
# SYSTEM / RAM MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    # RAM limits
    ram_warning_gb: float = 4.0       # Warn if < 4GB free at startup
    ram_critical_gb: float = 2.0     # Refuse to start if < 2GB free
    data_layer_ram_limit_gb: float = 2.0  # DuckDB + Polars layer cap
    
    # Platform
    is_windows: bool = field(default_factory=lambda: platform.system() == "Windows")
    
    # Event loop policy (Windows-specific)
    windows_selector_event_loop: bool = True  # Required for asyncio on Windows


def check_system_resources() -> dict:
    """Check available system resources at startup. Returns diagnostics dict."""
    import psutil
    
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    
    return {
        "total_ram_gb": round(total_gb, 1),
        "available_ram_gb": round(available_gb, 1),
        "used_ram_gb": round(mem.used / (1024 ** 3), 1),
        "ram_pct_used": mem.percent,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "warnings": [],
        "ready": True,
    }


def validate_startup() -> dict:
    """Validate system meets minimum requirements. Returns diagnostics dict."""
    diag = check_system_resources()
    sys_cfg = SystemConfig()
    
    if diag["available_ram_gb"] < sys_cfg.ram_critical_gb:
        diag["warnings"].append(
            f"CRITICAL: Only {diag['available_ram_gb']}GB RAM available. "
            f"Minimum {sys_cfg.ram_critical_gb}GB required. Engine may be unstable."
        )
        diag["ready"] = False
    elif diag["available_ram_gb"] < sys_cfg.ram_warning_gb:
        diag["warnings"].append(
            f"WARNING: Only {diag['available_ram_gb']}GB RAM available. "
            f"Recommended {sys_cfg.ram_warning_gb}GB+. Engine will proceed but may be slow."
        )
    
    if sys_cfg.is_windows and sys_cfg.windows_selector_event_loop:
        import asyncio
        if sys.version_info >= (3, 8):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            diag["warnings"].append(
                "INFO: Windows detected — set WindowsSelectorEventLoopPolicy "
                "to prevent RuntimeError in asyncio.gather()."
            )
    
    return diag


# ═══════════════════════════════════════════════════════════════════
# UNIFIED CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EngineConfig:
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    trading_rules: TradingRules = field(default_factory=TradingRules)
    logging: LogConfig = field(default_factory=LogConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    mode: str = "live"  # "live", "paper", "backtest"
    dry_run: bool = False   # If True, log trades but don't execute
    
    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Load config from environment variables."""
        cfg = cls()
        cfg.llm.api_key = os.getenv("OPEN_ROUTER_API", cfg.llm.api_key)
        cfg.llm.base_url = os.getenv("OPEN_ROUTER_BASE_URL", cfg.llm.base_url)
        cfg.market_data.fyers_app_id = os.getenv("FYERS_APP_ID", cfg.market_data.fyers_app_id)
        cfg.market_data.fyers_app_secret = os.getenv("FYERS_APP_SECRET", cfg.market_data.fyers_app_secret)
        cfg.market_data.fyers_access_token = os.getenv("FYERS_ACCESS_TOKEN", cfg.market_data.fyers_access_token)
        return cfg

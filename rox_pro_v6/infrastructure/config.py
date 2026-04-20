"""
ROX Proven Edge Engine v3.0 - Configuration Module
================================================
Central configuration for the trading system.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MARKET_TRENDS_DIR = os.path.join(DATA_DIR, "Market_Trends")
DAILY_REPORTS_DIR = os.path.join(DATA_DIR, "Daily_Reports")
HISTORICAL_PATTERNS_DIR = os.path.join(DATA_DIR, "Historical_Patterns")
AGENT_PERFORMANCE_DIR = os.path.join(DATA_DIR, "Agent_Performance")
WEEKLY_RECONCILIATION_DIR = os.path.join(DATA_DIR, "Weekly_Reconciliation")


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
    max_risk_per_trade: float = 0.02  # 2%
    max_single_position: float = 0.15  # 15%
    max_portfolio_heat: float = 0.08  # 8%
    max_sector_exposure: float = 0.25  # 25%
    min_cash_buffer: float = 0.05  # 5%
    min_risk_reward_ratio: float = 1.5  # 1.5:1


@dataclass
class SystemConfig:
    """Main system configuration"""
    # Agent configurations
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    
    # Risk limits
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    
    # Target performance metrics
    target_win_rate: float = 0.65
    target_monthly_return: float = 0.10
    
    # Trading parameters
    swing_trade_days: tuple = (2, 5)
    atr_stop_multiplier: float = 1.5
    
    # Data validation thresholds
    min_data_completeness: int = 4  # Out of 6 datasets
    
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
            }
    
    def normalize_weights(self):
        """Normalize all agent weights to sum to 1.0"""
        total = sum(a.current_weight for a in self.agents.values())
        if total > 0:
            for agent in self.agents.values():
                agent.current_weight = agent.current_weight / total


# Default system configuration
DEFAULT_CONFIG = SystemConfig()


# Nifty 50 stocks
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

# Nifty Midcap 50 high-quality swing candidates
NIFTY_MIDCAP_STOCKS = [
    "ABCAPITAL", "ABFRL", "AUROPHARMA", "BALKRISIND", "BANDHANBNK",
    "BIOCON", "COFORGE", "CROMPTON", "CUMMINSIND", "DALBHARAT",
    "FEDERALBNK", "GLENMARK", "GMRINFRA", "GODREJPROP", "GRANULES",
    "HINDPETRO", "IDFCFIRSTB", "INDHOTEL", "INDUSTOWER", "IRCTC",
    "JKCEMENT", "JUBLFOOD", "KANSAINER", "LICHSGFIN", "LUPIN",
    "MANAPPURAM", "MARICO", "MFSL", "MPHASIS", "MRF",
    "NATIONALUM", "NAVINFLUOR", "NMDC", "OBEROIRLTY", "OFSS",
    "PAGEIND", "PERSISTENT", "PFC", "PHOENIXLTD", "PIIND",
    "PNB", "POLYCAB", "PVRINOX", "RAMCOCEM", "RECLTD",
    "SAIL", "SCHAEFFLER", "SRF", "SYNGENE", "TRENT"
]

# Combined watchlist — full 100 stocks
WATCHLIST_ALL = NIFTY_50_STOCKS + NIFTY_MIDCAP_STOCKS

# Sector mappings (covers both lists)
SECTOR_MAPPING = {
    "Banking": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
        "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE",
        "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB", "LICHSGFIN",
        "MANAPPURAM", "ABCAPITAL", "MFSL", "PFC", "RECLTD"
    ],
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
        "MPHASIS", "COFORGE", "PERSISTENT", "OFSS"
    ],
    "Energy": [
        "RELIANCE", "ONGC", "BPCL", "POWERGRID", "NTPC", "COALINDIA",
        "HINDPETRO", "NMDC", "SAIL", "NATIONALUM"
    ],
    "Auto": [
        "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT",
        "BALKRISIND", "MRF", "SCHAEFFLER", "CROMPTON"
    ],
    "Metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO"],
    "Pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB",
        "AUROPHARMA", "BIOCON", "GLENMARK", "GRANULES", "LUPIN",
        "NAVINFLUOR", "PIIND", "SYNGENE"
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
        "MARICO", "JUBLFOOD", "PAGEIND"
    ],
    "Infrastructure": [
        "LT", "ULTRACEMCO", "GRASIM",
        "GMRINFRA", "INDUSTOWER", "POLYCAB", "JKCEMENT", "DALBHARAT",
        "RAMCOCEM", "KANSAINER", "CUMMINSIND"
    ],
    "Telecom": ["BHARTIARTL"],
    "Realty": ["GODREJPROP", "OBEROIRLTY", "PHOENIXLTD"],
    "Consumer": ["ABFRL", "PVRINOX", "INDHOTEL", "TRENT"],
    "Chemicals": ["SRF", "PIIND"],
    "Logistics": ["IRCTC"],
    "Others": [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "TITAN",
        "UPL", "ONGC"
    ]
}

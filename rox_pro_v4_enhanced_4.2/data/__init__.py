"""
ROX Proven Edge Engine v3.0 - Data Management Package
=====================================================
Data handling, CSV logging, and JSON database management.
"""

from .data_manager import DataManager, TradeRecord
from .trade_logger import TradeLogger
from .pattern_database import PatternDatabase
from .scorecard import AgentScorecard

# ── v6.0 Data Modules ───────────────────────────────────────────────────────
from .trade_outcome_logger import TradeOutcomeLogger

__all__ = [
    "DataManager", "TradeRecord", "TradeLogger", "PatternDatabase", "AgentScorecard",
    "TradeOutcomeLogger",
]

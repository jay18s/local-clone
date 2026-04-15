"""
ROX Engine v5.0 — Reasoning Package
Multi-agent debate, chain-of-thought, pattern memory, confidence calibration,
adaptive prompt selection, and rule-based validation.
"""

from reasoning.data_classes import (
    Signal, SignalDirection, SignalStrength,
    RegimeResult, NewsResult, TradePlan, PortfolioState, TradeRecord, MarketState,
)
from reasoning.cot_prompts import (
    build_regime_cot_prompt,
    build_news_prompt,
    build_cross_exam_prompt,
    build_final_arbiter_prompt,
    build_trading_planner_prompt,
    build_fno_brain_prompt,
    build_self_reflector_prompt,
)
from reasoning.debate_engine import DebateEngine, DebateResult
from reasoning.pattern_memory import PatternMemoryBank, PatternMatch, DailySnapshot
from reasoning.confidence_calibrator import ConfidenceCalibrator, CalibrationResult
from reasoning.rule_validator import RuleBasedValidator, ValidationResult, FailReason
from reasoning.adaptive_and_cache import (
    AdaptivePromptSelector, AdaptiveConfig, RegimeCache, CachedRegime,
)
from reasoning.config import ComplexityLevel

__all__ = [
    # Data classes
    "Signal", "SignalDirection", "SignalStrength",
    "RegimeResult", "NewsResult", "TradePlan", "PortfolioState", "TradeRecord",
    "MarketState",
    # ComplexityLevel
    "ComplexityLevel",
    # CoT prompts
    "build_regime_cot_prompt", "build_news_prompt", "build_cross_exam_prompt",
    "build_final_arbiter_prompt", "build_trading_planner_prompt",
    "build_fno_brain_prompt", "build_self_reflector_prompt",
    # Debate
    "DebateEngine", "DebateResult",
    # Pattern memory
    "PatternMemoryBank", "PatternMatch", "DailySnapshot",
    # Calibration
    "ConfidenceCalibrator", "CalibrationResult",
    # Rule validator
    "RuleBasedValidator", "ValidationResult", "FailReason",
    # Adaptive
    "AdaptivePromptSelector", "AdaptiveConfig", "RegimeCache", "CachedRegime",
]

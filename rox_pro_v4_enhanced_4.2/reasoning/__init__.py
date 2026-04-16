"""
ROX Engine v5.0 — Reasoning Module
Chain-of-thought, debate protocol, confidence calibration, pattern memory, rule-based validation.
"""

from reasoning_v5.data_classes import (
    Signal, SignalDirection, SignalStrength, ComplexityLevel,
    RegimeResult, NewsResult, TradePlan, PortfolioState,
    TradeRecord, MarketState,
)
from reasoning_v5.cot_prompts import (
    build_regime_cot_prompt,
    build_news_prompt,
    build_cross_exam_prompt,
    build_final_arbiter_prompt,
    build_trading_planner_prompt,
    build_fno_brain_prompt,
    build_self_reflector_prompt,
)
from reasoning_v5.debate_engine import DebateEngine, DebateResult
from reasoning_v5.pattern_memory import PatternMemoryBank, DailySnapshot, PatternMatch
from reasoning_v5.confidence_calibrator import ConfidenceCalibrator, CalibrationResult
from reasoning_v5.rule_validator import RuleBasedValidator, ValidationResult, FailReason
from reasoning_v5.adaptive_and_cache import (
    AdaptivePromptSelector, AdaptiveConfig, RegimeCache, CachedRegime,
)

__all__ = [
    # Data classes
    "Signal", "SignalDirection", "SignalStrength", "ComplexityLevel",
    "RegimeResult", "NewsResult", "TradePlan", "PortfolioState",
    "TradeRecord", "MarketState",
    # Prompt builders
    "build_regime_cot_prompt", "build_news_prompt",
    "build_cross_exam_prompt", "build_final_arbiter_prompt",
    "build_trading_planner_prompt", "build_fno_brain_prompt",
    "build_self_reflector_prompt",
    # Engines
    "DebateEngine", "DebateResult",
    "PatternMemoryBank", "DailySnapshot", "PatternMatch",
    "ConfidenceCalibrator", "CalibrationResult",
    "RuleBasedValidator", "ValidationResult", "FailReason",
    "AdaptivePromptSelector", "AdaptiveConfig", "RegimeCache", "CachedRegime",
]

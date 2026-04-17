"""
ROX Engine v5.0 — Reasoning Module
Chain-of-thought, debate protocol, confidence calibration, pattern memory, rule-based validation.
"""

from .data_classes import (
    Signal, SignalDirection, SignalStrength, ComplexityLevel,
    RegimeResult, NewsResult, TradePlan, PortfolioState,
    TradeRecord, MarketState,
)
from .cot_prompts import (
    build_regime_cot_prompt,
    build_news_prompt,
    build_cross_exam_prompt,
    build_final_arbiter_prompt,
    build_trading_planner_prompt,
    build_fno_brain_prompt,
    build_self_reflector_prompt,
)
from .debate_engine import DebateEngine, DebateResult
from .pattern_memory import PatternMemoryBank, DailySnapshot, PatternMatch
from .confidence_calibrator import ConfidenceCalibrator, CalibrationResult
from .rule_validator import RuleBasedValidator, ValidationResult, FailReason
from .adaptive_and_cache import (
    AdaptivePromptSelector, AdaptiveConfig, RegimeCache, CachedRegime,
)

# ── v6.0 Reasoning Modules ────────────────────────────────────────────────
from .rule_regime_classifier import RuleRegimeClassifier, RegimeClassification
from .regime_arbiter import RegimeArbiter, RegimeDecision
from .regime_transition_detector import RegimeTransitionDetector, TransitionEvent
from .regime_accuracy_tracker import RegimeAccuracyTracker
from .adaptive_calibrator import AdaptiveConfidenceCalibrator, CalibrationRecord

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
    # v6.0 Modules
    "RuleRegimeClassifier", "RegimeClassification",
    "RegimeArbiter", "RegimeDecision",
    "RegimeTransitionDetector", "TransitionEvent",
    "RegimeAccuracyTracker",
    "AdaptiveConfidenceCalibrator", "CalibrationRecord",
]

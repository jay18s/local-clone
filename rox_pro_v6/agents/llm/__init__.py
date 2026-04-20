"""
LLM-Powered Agent Modules for ROX Engine v5.0 Enhanced
========================================================

This module contains LLM-powered enhancements that integrate with the existing
multi-agent trading system. All components follow the principle of graceful
degradation - if LLM is unavailable, they fall back to rule-based logic.

Components:
- BaseLLMAgent: Base class for all LLM-powered agents
- LLMConfig: Configuration for LLM-powered agents
- LLMResponse: Standardized LLM response container
- LLMRegimeDetector: LLM-powered market regime detection (P1.1)
- LLMCrossExaminer: Consensus cross-examination (P1.2)
- LLMNewsImpactAnalyzer: Trading-focused news analysis (P2.1)
- LLMOptionsStrategist: Options strategy optimization (P2.2)
- LLMPatternValidator: Chart pattern validation (P3.1)
- LLMMetaLearner: Weekly performance analysis (P3.2)
"""

from .base_llm_agent import BaseLLMAgent, LLMConfig, LLMResponse
from .llm_regime_detector import LLMRegimeDetector, RegimeDetectionResult
from .llm_cross_examiner import LLMCrossExaminer, CrossExaminationResult
from .llm_news_analyzer import LLMNewsImpactAnalyzer, NewsImpactResult
from .llm_options_strategist import LLMOptionsStrategist, OptionsStrategyResult
from .llm_pattern_validator import LLMPatternValidator, PatternValidationResult
from .llm_meta_learner import LLMMetaLearner, MetaLearningResult
from .llm_history_analyzer import LLMHistoryAnalyzer, HistoryContext
from .llm_trading_planner import LLMTradingPlanner, TradingPlanOutput

__all__ = [
    'BaseLLMAgent',
    'LLMConfig',
    'LLMResponse',
    'LLMRegimeDetector',
    'RegimeDetectionResult',
    'LLMCrossExaminer',
    'CrossExaminationResult',
    'LLMNewsImpactAnalyzer',
    'NewsImpactResult',
    'LLMOptionsStrategist',
    'OptionsStrategyResult',
    'LLMPatternValidator',
    'PatternValidationResult',
    'LLMMetaLearner',
    'MetaLearningResult',
    'LLMHistoryAnalyzer',
    'HistoryContext',
    'LLMTradingPlanner',
    'TradingPlanOutput',
]

"""
LLM Pattern Validator - Chart pattern validation (Enhancement P3.1)
=====================================================================

Validates detected chart patterns with LLM analysis considering:
- Pattern validity in current market regime
- Technical context (RSI, volume, trends)
- News impact on the stock
- Historical performance data

Provides adjusted confidence and entry/exit recommendations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base_llm_agent import BaseLLMAgent, LLMConfig, LLMResponse

# Import from parent config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MarketRegime, TradeDirection


# Prompt Template
PATTERN_VALIDATION_PROMPT = """You are a technical analysis expert validating chart patterns for Indian equities.

DETECTED PATTERN:
- Type: {pattern_type}
- Direction: {pattern_direction}
- Stock: {stock}
- Entry Price: {entry_price}
- Stop Loss: {stop_loss}
- Target: {target}
- Pattern Confidence: {pattern_confidence}%

MARKET CONTEXT:
- Market Regime: {regime}
- Stock RSI: {rsi}
- Stock vs SMA20: {vs_sma20}%
- Stock vs SMA50: {vs_sma50}%
- Volume vs Average: {volume_ratio}x
- Recent 5-day Performance: {performance_5d}%
- Sector: {sector}
- Sector 5-day Performance: {sector_performance}%

RECENT NEWS FOR STOCK:
{stock_news}

HISTORICAL PATTERN PERFORMANCE:
- This pattern type in current regime: {regime_win_rate}% win rate ({regime_samples} samples)
- This stock swing trades: {stock_win_rate}% win rate ({stock_samples} samples)

Validate and adjust the pattern:

{{
    "pattern_valid": true|false,
    "adjusted_confidence": <integer 0-100>,
    "adjusted_entry": <float or null if unchanged>,
    "adjusted_stop_loss": <float or null if unchanged>,
    "adjusted_target": <float or null if unchanged>,
    "validation_notes": [
        "<positive factor 1>",
        "<positive factor 2>"
    ],
    "risk_notes": [
        "<negative factor 1>",
        "<negative factor 2>"
    ],
    "final_recommendation": "TAKE|AVOID|WAIT_FOR_CONFIRMATION",
    "reasoning": "<comprehensive explanation>"
}}

VALIDATION CRITERIA:
1. Pattern validity in current regime
2. RSI not overbought (>70) for LONG, not oversold (<30) for SHORT
3. Volume confirmation
4. Sector alignment
5. News impact
6. Historical win rate for similar setups

IMPORTANT:
- If pattern is in regime with <50% win rate, reduce confidence by 10-15
- If volume is below average, flag as concern
- If RSI is extreme, recommend WAIT_FOR_CONFIRMATION
"""


@dataclass
class PatternValidationResult:
    """Result of pattern validation."""
    pattern_name: str
    pattern_valid: bool
    adjusted_confidence: int  # 0-100
    adjusted_entry: Optional[float]
    adjusted_stop_loss: Optional[float]
    adjusted_target: Optional[float]
    validation_notes: List[str]  # Positive factors
    risk_notes: List[str]  # Negative factors
    final_recommendation: str  # "TAKE", "AVOID", "WAIT_FOR_CONFIRMATION"
    reasoning: str
    source: str = "LLM"  # "LLM" or "FALLBACK"
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[str] = None


class LLMPatternValidator(BaseLLMAgent):
    """
    LLM-powered pattern validation.

    Called after ORION detects patterns.
    Validates patterns with contextual analysis.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config, logger_name="LLMPatternValidator")
        self._validation_cache: Dict[str, PatternValidationResult] = {}

    def validate_pattern(
        self,
        pattern: Dict[str, Any],
        stock: str,
        market_context: Dict[str, Any],
        historical_performance: Dict[str, Any]
    ) -> PatternValidationResult:
        """
        Validate a detected pattern with LLM.

        Args:
            pattern: Detected pattern from ORION
            stock: Stock symbol
            market_context: Market regime, indicators, etc.
            historical_performance: Pattern performance history

        Returns:
            PatternValidationResult with adjusted confidence and recommendation
        """
        # Build prompt
        prompt = self._build_validation_prompt(
            pattern, stock, market_context, historical_performance
        )

        # Get LLM response (no fallback_handler - we handle fallback ourselves)
        response = self.generate(
            prompt=prompt,
            expect_json=True,
            fallback_handler=None
        )

        # Parse response - check if we got valid LLM response
        if response.source == "LLM" and response.parsed_json:
            result = self._parse_validation_response(
                response.parsed_json, pattern, response.content
            )
        else:
            # Use fallback when LLM unavailable
            result = self._fallback_validation(pattern, market_context, historical_performance)

        return result

    def validate_batch(
        self,
        patterns: List[Dict[str, Any]],
        stock: str,
        market_context: Dict[str, Any],
        historical_performance: Dict[str, Any] = None
    ) -> List[PatternValidationResult]:
        """
        Validate multiple patterns in single LLM call for efficiency.

        Args:
            patterns: List of detected patterns
            stock: Stock symbol
            market_context: Market regime, indicators, etc.
            historical_performance: Pattern performance history

        Returns:
            List of PatternValidationResults
        """
        results = []
        for pattern in patterns:
            result = self.validate_pattern(
                pattern, stock, market_context, historical_performance or {}
            )
            results.append(result)
        return results

    def _build_validation_prompt(
        self,
        pattern: Dict[str, Any],
        stock: str,
        market_context: Dict[str, Any],
        historical_performance: Dict[str, Any]
    ) -> str:
        """Construct validation prompt."""
        # Pattern details
        pattern_type = pattern.get('type', pattern.get('pattern_type', 'Unknown'))
        pattern_direction = pattern.get('direction', 'NEUTRAL')
        if hasattr(pattern_direction, 'value'):
            pattern_direction = pattern_direction.value
        entry_price = pattern.get('entry_price', pattern.get('entry', 0))
        stop_loss = pattern.get('stop_loss', 0)
        target = pattern.get('target', pattern.get('target_1', 0))
        pattern_confidence = pattern.get('confidence', pattern.get('conviction', 60))

        # Market context
        regime = market_context.get('regime', 'CONSOLIDATION')
        if hasattr(regime, 'value'):
            regime = regime.value

        indicators = market_context.get('indicators', {})
        rsi = indicators.get('rsi', 50)
        sma20 = indicators.get('sma20', entry_price)
        sma50 = indicators.get('sma50', entry_price)
        vs_sma20 = ((entry_price - sma20) / sma20 * 100) if sma20 > 0 else 0
        vs_sma50 = ((entry_price - sma50) / sma50 * 100) if sma50 > 0 else 0
        volume_ratio = indicators.get('vol_ratio', indicators.get('volume_ratio', 1.0))
        performance_5d = market_context.get('performance_5d', 0)

        # Sector info
        sector = market_context.get('sector', 'Others')
        sector_performance = market_context.get('sector_performance', 0)

        # News
        stock_news = market_context.get('news', [])
        if isinstance(stock_news, list):
            news_str = "\n".join(
                f"- {item.get('title', str(item))}" for item in stock_news[:3]
            )
        else:
            news_str = str(stock_news) if stock_news else "No recent news"

        # Historical performance
        regime_win_rate = historical_performance.get('regime_win_rate', 50)
        regime_samples = historical_performance.get('regime_samples', 0)
        stock_win_rate = historical_performance.get('stock_win_rate', 50)
        stock_samples = historical_performance.get('stock_samples', 0)

        return PATTERN_VALIDATION_PROMPT.format(
            pattern_type=pattern_type,
            pattern_direction=pattern_direction,
            stock=stock,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            pattern_confidence=pattern_confidence,
            regime=regime,
            rsi=rsi,
            vs_sma20=vs_sma20,
            vs_sma50=vs_sma50,
            volume_ratio=volume_ratio,
            performance_5d=performance_5d,
            sector=sector,
            sector_performance=sector_performance,
            stock_news=news_str,
            regime_win_rate=regime_win_rate,
            regime_samples=regime_samples,
            stock_win_rate=stock_win_rate,
            stock_samples=stock_samples
        )

    def _parse_validation_response(
        self,
        parsed: Dict[str, Any],
        pattern: Dict[str, Any],
        raw_response: str
    ) -> PatternValidationResult:
        """Parse JSON response."""
        try:
            pattern_type = pattern.get('type', pattern.get('pattern_type', 'Unknown'))

            recommendation = parsed.get("final_recommendation", "TAKE").upper()
            if recommendation not in ["TAKE", "AVOID", "WAIT_FOR_CONFIRMATION"]:
                recommendation = "TAKE"

            return PatternValidationResult(
                pattern_name=pattern_type,
                pattern_valid=bool(parsed.get("pattern_valid", True)),
                adjusted_confidence=int(parsed.get("adjusted_confidence", pattern.get('confidence', 60))),
                adjusted_entry=parsed.get("adjusted_entry"),
                adjusted_stop_loss=parsed.get("adjusted_stop_loss"),
                adjusted_target=parsed.get("adjusted_target"),
                validation_notes=parsed.get("validation_notes", []),
                risk_notes=parsed.get("risk_notes", []),
                final_recommendation=recommendation,
                reasoning=parsed.get("reasoning", ""),
                source="LLM",
                raw_response=raw_response
            )

        except Exception as e:
            self.logger.error(f"Failed to parse validation response: {e}")
            return self._fallback_validation(pattern, {}, {})

    def _fallback_validation(
        self,
        pattern: Dict[str, Any],
        market_context: Dict[str, Any],
        historical_performance: Dict[str, Any]
    ) -> PatternValidationResult:
        """Fallback validation when LLM unavailable."""
        self.logger.info("Using fallback pattern validation")

        pattern_type = pattern.get('type', pattern.get('pattern_type', 'Unknown'))
        base_confidence = pattern.get('confidence', pattern.get('conviction', 60))

        # Simple rule-based adjustments
        adjusted_confidence = base_confidence
        risk_notes = []
        validation_notes = []

        # Check RSI
        indicators = market_context.get('indicators', {})
        rsi = indicators.get('rsi', 50)
        direction = pattern.get('direction', 'NEUTRAL')
        if hasattr(direction, 'value'):
            direction = direction.value

        if direction == 'LONG' and rsi > 70:
            adjusted_confidence -= 10
            risk_notes.append("RSI overbought")
        elif direction == 'SHORT' and rsi < 30:
            adjusted_confidence -= 10
            risk_notes.append("RSI oversold")
        else:
            validation_notes.append("RSI in acceptable range")

        # Check volume
        volume_ratio = indicators.get('vol_ratio', indicators.get('volume_ratio', 1.0))
        if volume_ratio < 0.8:
            adjusted_confidence -= 5
            risk_notes.append("Below average volume")
        elif volume_ratio > 1.3:
            adjusted_confidence += 5
            validation_notes.append("Above average volume confirms pattern")

        # Check historical win rate
        regime_win_rate = historical_performance.get('regime_win_rate', 50)
        if regime_win_rate < 50:
            adjusted_confidence -= 10
            risk_notes.append(f"Low regime win rate: {regime_win_rate}%")

        # Determine recommendation
        if adjusted_confidence >= 65:
            recommendation = "TAKE"
        elif adjusted_confidence >= 50:
            recommendation = "WAIT_FOR_CONFIRMATION"
        else:
            recommendation = "AVOID"

        return PatternValidationResult(
            pattern_name=pattern_type,
            pattern_valid=adjusted_confidence >= 50,
            adjusted_confidence=max(0, min(100, adjusted_confidence)),
            adjusted_entry=None,
            adjusted_stop_loss=None,
            adjusted_target=None,
            validation_notes=validation_notes,
            risk_notes=risk_notes,
            final_recommendation=recommendation,
            reasoning="Rule-based validation (LLM unavailable)",
            source="FALLBACK"
        )

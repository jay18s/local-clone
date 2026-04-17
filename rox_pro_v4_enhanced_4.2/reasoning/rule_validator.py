"""
ROX Engine v5.0 — Rule-Based Pattern Validator
Deterministic signal validation replacing LLM calls.
Validates in <1ms with zero API cost.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from enum import Enum

logger = logging.getLogger("reasoning.rule_validator")


class FailReason(Enum):
    RSI_OVERSOLD_LONG = "RSI oversold for LONG entry"
    RSI_OVERBOUGHT_SHORT = "RSI overbought for SHORT entry"
    RISK_REWARD_TOO_LOW = f"Risk-Reward ratio below minimum"
    VOLUME_TOO_LOW = "Volume below 80% of 20D average"
    PRICE_BELOW_SMA20 = "Price below SMA-20 for LONG"
    PRICE_ABOVE_SMA20 = "Price above SMA-20 for SHORT (reversed)"
    NEWS_BLOCKED_LONG = "Sector/stock blocked for LONG positions"
    NEWS_BLOCKED_SHORT = "Sector/stock blocked for SHORT positions"
    COUNTER_TREND = "Signal direction opposes market regime"
    LOW_STRENGTH = "Signal strength below minimum threshold"
    SECTOR_OVERLAP = "Sector already has maximum positions"
    PASS = "All checks passed"


@dataclass
class ValidationResult:
    symbol: str
    passed: bool
    reason: str
    score: float = 0.0  # 0-100 composite score
    original_signal: dict = None  # Original signal data for downstream use


class RuleBasedValidator:
    """
    Deterministic signal validator that replaces LLM-based PatternValidator.
    Validates signals against technical rules in <1ms with zero API cost.
    """
    
    def __init__(
        self,
        min_rr_ratio: float = 1.5,
        rsi_long_min: float = 40.0,
        rsi_short_max: float = 60.0,
        rsi_overbought: float = 75.0,
        rsi_oversold: float = 25.0,
        volume_min_pct_of_avg: float = 0.8,
        min_signal_strength: str = "MEDIUM",
        require_price_above_sma20_long: bool = True,
    ):
        self.min_rr_ratio = min_rr_ratio
        self.rsi_long_min = rsi_long_min
        self.rsi_short_max = rsi_short_max
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_min_pct = volume_min_pct_of_avg
        self.min_signal_strength = min_signal_strength
        self.min_strength_order = ["LOW", "MEDIUM", "HIGH", "STRONG"]
        self.require_sma20 = require_price_above_sma20_long
        self._strength_order = {s: i for i, s in enumerate(self.min_strength_order)}
    
    def validate(
        self,
        signal: dict,
        market_data: dict = None,
        regime: dict = None,
        news_restrictions: dict = None,
        active_sectors: dict = None,
    ) -> ValidationResult:
        """
        Validate a single signal against all rules.
        
        Args:
            signal: {
                symbol, direction, strength, rr_ratio, rsi,
                volume, volume_avg_20d, price, sma_20,
                sector, agent
            }
            market_data: Current market context.
            regime: Regime detection result.
            news_restrictions: News engine restrictions.
            active_sectors: Dict of sector -> count of active positions.
        """
        market_data = market_data or {}
        symbol = signal.get("symbol", "UNKNOWN")
        direction = signal.get("direction", "").upper()
        strength = signal.get("strength", "LOW").upper()
        reasons_pass = []
        reasons_fail = []
        score = 50.0  # Base score
        
        # 1. Signal Strength Check
        if self._strength_order.get(strength, 0) < self._strength_order.get(self.min_signal_strength, 1):
            reasons_fail.append(
                f"Signal strength {strength} below minimum {self.min_signal_strength}"
            )
        else:
            score += 15
        
        # 2. Risk-Reward Ratio Check
        rr = signal.get("rr_ratio", 0)
        if rr < self.min_rr_ratio:
            reasons_fail.append(f"R:R {rr:.2f} below {self.min_rr_ratio} minimum")
        else:
            score += 20
        
        # 3. RSI Check
        rsi = signal.get("rsi")
        if rsi is not None:
            if direction == "LONG" and rsi < self.rsi_long_min:
                reasons_fail.append(f"RSI {rsi:.1f} below {self.rsi_long_min} for LONG")
            elif direction == "SHORT" and rsi > self.rsi_short_max:
                reasons_fail.append(f"RSI {rsi:.1f} above {self.rsi_short_max} for SHORT")
            elif direction == "LONG" and rsi > self.rsi_overbought:
                reasons_fail.append(f"RSI {rsi:.1f} overbought ({self.rsi_overbought}) for LONG")
                score -= 5
            elif direction == "SHORT" and rsi < self.rsi_oversold:
                reasons_fail.append(f"RSI {rsi:.1f} oversold ({self.rsi_oversold}) for SHORT")
                score -= 5
            else:
                score += 15
        else:
            reasons_fail.append("No RSI data available")
        
        # 4. Volume Check
        volume = signal.get("volume")
        volume_avg = signal.get("volume_avg_20d")
        if volume is not None and volume_avg is not None and volume_avg > 0:
            vol_ratio = volume / volume_avg
            if vol_ratio < self.volume_min_pct:
                reasons_fail.append(
                    f"Volume {volume:.0f} below {self.volume_min_pct*100:.0f}% of 20D avg ({volume_avg:.0f})"
                )
            else:
                score += 10
                if vol_ratio > 1.5:
                    score += 5  # Bonus for high volume
        
        # 5. SMA-20 Alignment
        price = signal.get("price")
        sma_20 = signal.get("sma_20")
        if price is not None and sma_20 is not None:
            if direction == "LONG" and self.require_sma20 and price < sma_20:
                reasons_fail.append(f"Price {price:.2f} below SMA-20 ({sma_20:.2f}) for LONG")
            elif direction == "SHORT" and price > sma_20:
                reasons_fail.append(f"Price {price:.2f} above SMA-20 ({sma_20:.2f}) for SHORT")
            else:
                score += 10
        else:
            reasons_fail.append("No price/SMA-20 data available")
        
        # 6. News Restrictions
        symbol_upper = symbol.upper().replace("NSE:", "")
        sector = signal.get("sector", "")
        
        if news_restrictions:
            if direction == "LONG":
                blocked_long = news_restrictions.get("block_long_sectors", [])
                blocked_symbols = news_restrictions.get("block_long_symbols", [])
                if sector and sector in blocked_long:
                    reasons_fail.append(f"News: {sector} blocked for LONG positions")
                    score -= 20
                if symbol_upper in blocked_symbols:
                    reasons_fail.append(f"News: {symbol_upper} blocked for LONG positions")
                    score -= 20
            elif direction == "SHORT":
                blocked_short = news_restrictions.get("block_short_sectors", [])
                if sector and sector in blocked_short:
                    reasons_fail.append(f"News: {sector} blocked for SHORT positions")
                    score -= 20
        
        # 7. Counter-Trend Check
        # FIX-REGIME-01: Actual regime values are BEAR, MILD_BEAR, CORRECTION, BULL, etc.
        # (not "BEARISH"/"BULLISH"). Original check never matched, so counter-trend
        # filtering was completely inert.
        if regime:
            regime_dir = regime.get("regime", "").upper()
            BEARISH_REGIMES = {"BEAR", "MILD_BEAR", "CORRECTION", "BEARISH", "WEAK BEARISH", "STRONGLY_BEARISH"}
            BULLISH_REGIMES = {"BULL", "MILD_BULL", "BULLISH", "WEAK BULLISH", "STRONGLY_BULLISH"}
            if regime_dir in BEARISH_REGIMES and direction == "LONG":
                reasons_fail.append(f"Counter-trend LONG in {regime_dir} regime")
                score -= 15
            elif regime_dir in BULLISH_REGIMES and direction == "SHORT":
                reasons_fail.append(f"Counter-trend SHORT in {regime_dir} regime")
                score -= 15
        
        # 8. Sector Overlap Check
        if active_sectors and sector:
            current_count = active_sectors.get(sector, 0)
            if current_count >= 2:
                reasons_fail.append(f"Sector overlap: {sector} already has {current_count} positions (max 2)")
                score -= 20
            else:
                score += 5
        
        # Cap score
        score = max(0, min(100, score))
        
        passed = len(reasons_fail) == 0
        
        return ValidationResult(
            symbol=symbol,
            passed=passed,
            reason=reasons_fail[0] if reasons_fail else "All checks passed",
            score=score,
            original_signal=signal,
        )
    
    def validate_batch(
        self,
        signals: list[dict],
        market_data: dict = None,
        regime: dict = None,
        news_restrictions: dict = None,
        active_sectors: dict = None,
    ) -> list[ValidationResult]:
        """
        Validate all signals in a single pass.
        Returns ALL results (passed and failed), sorted by score descending.
        """
        results = []
        passed_count = 0
        failed_count = 0
        
        for signal in signals:
            result = self.validate(
                signal=signal,
                market_data=market_data or {},
                regime=regime or {},
                news_restrictions=news_restrictions or {},
                active_sectors=active_sectors or {},
            )
            results.append(result)
            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        
        logger.info(
            f"Rule validation: {len(signals)} signals → "
            f"{passed_count} PASSED, {failed_count} FAILED "
            f"in <1ms (0 LLM calls)"
        )
        
        return results

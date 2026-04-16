"""
ROX Engine v5.0 — 6-Signal Confidence Calibrator
Combines multiple confidence signals into a calibrated, reliable prediction score.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("reasoning.calibrator")


@dataclass
class CalibrationResult:
    """Result of the confidence calibration process."""
    raw_confidence: float
    calibrated_confidence: float
    is_actionable: bool
    signal_breakdown: dict
    adjustments_applied: list[str]


class ConfidenceCalibrator:
    """
    Multi-signal confidence calibration model.
    
    Signals:
     1. Debate Agreement (25%) — Do all debate agents agree?
    2. Pattern Match Strength (20%) — How similar is history?
    3. Technical Alignment (20%) — Do indicators agree?
    4. Volume Confirmation (15%) — Is volume supporting?
    5. Regime Consistency (10%) — Does prediction match regime?
    6. Anti-Consensus Factor (10%) — Contrarian edge
    """
    
    # Default weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        "debate_agreement": 0.25,
        "pattern_match": 0.20,
        "technical_alignment": 0.20,
        "volume_confirmation": 0.15,
        "regime_consistency": 0.10,
        "anti_consensus": 0.10,
    }
    
    def __init__(self, weights: dict = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._validate_weights()
    
    def _validate_weights(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Calibration weights sum to {total:.2f}, normalizing...")
            for k in self.weights:
                self.weights[k] /= total
    
    def calibrate(
        self,
        debate_agreement: float,          # 0.0-1.0
        debate_confidence: float,          # 0-100
        pattern_match_score: float,        # 0.0-1.0 (best match similarity)
        pattern_match_count: int,          # Number of matches
        technical_aligned: float,           # 0.0-1.0 (fraction of aligned indicators)
        volume_confirms_price: bool,       # True/False
        volume_strength: float,            # 0-1.0
        regime_direction: str,              # BULLISH/BEARISH/NEUTRAL
        prediction_direction: str,           # BULLISH/BEARISH/NEUTRAL
        raw_confidence: float,              # 0-100 from LLM
        vix_level: float = 15.0,
        macro_event_count: int = 0,
        news_restrictions_count: int = 0,
        is_200dma_test: bool = False,
        is_ath_test: bool = False,
        position_overlaps: int = 0,
        market_positioning: str = "NEUTRAL",  # EXTREME_BULLISH/EXTREME_BEARISH/NEUTRAL
        pattern_bullish_count: int = 0,
        pattern_bearish_count: int = 0,
        pattern_total_count: int = 0,
    ) -> CalibrationResult:
        """
        Run the full calibration pipeline.
        
        Returns:
            CalibrationResult with calibrated confidence and actionability.
        """
        signals = {}
        adjustments = []
        
        # ───────────────────────────────────────────────────────
        # Signal 1: Debate Agreement (25%)
        # ───────────────────────────────────────────────────────
        if debate_agreement >= 0.99:
            # Unanimous agreement (all bull or all bear)
            # This is rare and very powerful
            signals["debate_agreement"] = debate_confidence * 1.2
            adjustments.append(f"Unanimous debate agreement: {debate_confidence:.0f}% x 1.2 multiplier")
        elif debate_agreement >= 0.66:
            # 2 of 3 agree
            signals["debate_agreement"] = debate_confidence * 0.9
        elif debate_agreement >= 0.50:
            signals["debate_agreement"] = debate_confidence * 0.7
        else:
            signals["debate_agreement"] = debate_confidence * 0.5
            adjustments.append("Low debate agreement penalty: disagreement among agents")
        
        # ───────────────────────────────────────────────────────
        # Signal 2: Pattern Match Strength (20%)
        # ───────────────────────────────────────────────────────
        if pattern_match_count > 0:
            if pattern_match_score >= 0.80:
                signals["pattern_match"] = 80
                adjustments.append(f"Strong historical support ({pattern_match_score:.0%} similarity)")
            elif pattern_match_score >= 0.65:
                signals["pattern_match"] = 65
            elif pattern_match_score >= 0.50:
                signals["pattern_match"] = 50
            else:
                signals["pattern_match"] = 30
            
            # Boost if historical outcomes agree with prediction
            if prediction_direction in ("BULLISH", "STRONGLY_BULLISH"):
                if pattern_bullish_count > pattern_bearish_count:
                    signals["pattern_match"] += 10
                    adjustments.append(f"Historical consensus supports bullish ({pattern_bullish_count}/{pattern_total_count} bullish)")
            elif prediction_direction in ("BEARISH", "STRONGLY_BEARISH"):
                if pattern_bearish_count > pattern_bullish_count:
                    signals["pattern_match"] += 10
                    adjustments.append(f"Historical consensus supports bearish ({pattern_bearish_count}/{pattern_total_count} bearish)")
        else:
            signals["pattern_match"] = 40  # No patterns = neutral
            adjustments.append("No historical patterns available")
        
        # ───────────────────────────────────────────────────────
        # Signal 3: Technical Alignment (20%)
        # ───────────────────────────────────────────────────────
        signals["technical_alignment"] = technical_aligned * 100
        
        if technical_aligned >= 0.75:
            adjustments.append("Strong technical alignment (>75% indicators)")
        elif technical_aligned >= 0.50:
            adjustments.append("Moderate technical alignment (50-75%)")
        else:
            adjustments.append("Weak technical alignment (<50% indicators)")
        
        # ───────────────────────────────────────────────────────
        # Signal 4: Volume Confirmation (15%)
        # ───────────────────────────────────────────────────────
        if volume_confirms_price:
            signals["volume_confirmation"] = 75 + (volume_strength * 25)
            adjustments.append("Volume confirms price move")
        else:
            signals["volume_confirmation"] = 25
            adjustments.append("Volume DIVERGENCE penalty: high volume against price")
        
        # ───────────────────────────────────────────────────────
        # Signal 5: Regime Consistency (10%)
        # ───────────────────────────────────────────────────────
        if regime_direction == prediction_direction:
            signals["regime_consistency"] = 85
        elif regime_direction == "NEUTRAL" or prediction_direction == "NEUTRAL":
            signals["regime_consistency"] = 60
        else:
            # Counter-regime trade
            signals["regime_consistency"] = 35
            adjustments.append("Counter-regime trade detected (higher risk)")
        
        # ───────────────────────────────────────────────────────
        # Signal 6: Anti-Consensus Factor (10%)
        # ───────────────────────────────────────────────────────
        if market_positioning == "EXTREME_BULLISH" and prediction_direction in ("BEARISH", "STRONGLY_BEARISH"):
            signals["anti_consensus"] = 80  # Crowd is wrong = opportunity
            adjustments.append("Contrarian edge: extreme bullish positioning, bearish prediction")
        elif market_positioning == "EXTREME_BEARISH" and prediction_direction in ("BULLISH", "STRONGLY_BULLISH"):
            signals["anti_consensus"] = 80
            adjustments.append("Contrarian edge: extreme bearish positioning, bullish prediction")
        elif market_positioning in ("EXTREME_BULLISH", "EXTREME_BEARISH"):
            signals["anti_consensus"] = 55
        else:
            signals["anti_consensus"] = 45
        
        # ───────────────────────────────────────────────────────
        # Weighted Combination
        # ───────────────────────────────────────────────────────
        weighted_sum = sum(
            signals.get(k, 0) * self.weights.get(k, 0)
            for k in self.weights
        )
        
        calibrated = weighted_sum
        
        # ───────────────────────────────────────────────────────
        # Dynamic Adjustments
        # ───────────────────────────────────────────────────────
        
        # VIX risk
        if vix_level > 25:
            calibrated -= 5
            adjustments.append(f"High VIX risk penalty: -5 (VIX={vix_level})")
        elif vix_level > 20:
            calibrated -= 3
            adjustments.append(f"Elevated VIX risk penalty: -3 (VIX={vix_level})")
        
        # Macro event uncertainty
        if macro_event_count >= 3:
            calibrated -= 4
            adjustments.append(f"Macro event uncertainty: -4 ({macro_event_count} events)")
        elif macro_event_count >= 2:
            calibrated -= 2
            adjustments.append(f"Macro event caution: -2 ({macro_event_count} events)")
        
        # News restrictions
        calibrated -= news_restrictions_count * 1.5
        if news_restrictions_count > 0:
            adjustments.append(f"News restrictions: -{news_restrictions_count * 1.5:.1f}")
        
        # Key level test risk
        if is_200dma_test:
            calibrated -= 3
            adjustments.append("200-DMA test risk: -3 (breakout uncertainty)")
        if is_ath_test:
            calibrated -= 2
            adjustments.append("ATH test risk: -2 (profit-taking potential)")
        
        # Position overlaps (correlated risk)
        if position_overlaps > 0:
            calibrated -= position_overlaps * 2
            adjustments.append(f"Position overlap penalty: -{position_overlaps * 2}")
        
        # Historical pattern bonus
        if pattern_match_count >= 3 and pattern_match_score >= 0.70:
            calibrated += 4
            adjustments.append("Historical pattern bonus: +4 (3+ strong matches)")
        elif pattern_match_count >= 2:
            calibrated += 2
        
        # Unanimous debate bonus
        if debate_agreement >= 0.99:
            calibrated += 5
            adjustments.append("Unanimous debate bonus: +5 (rare alignment)")
        
        # Cap between 25-90
        calibrated = max(25.0, min(90.0, calibrated))
        
        logger.info(
            f"Calibration: raw={raw_confidence:.1f}% → "
            f"calibrated={calibrated:.1f}% | "
            f"adjustments={len(adjustments)}"
        )
        
        return CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=round(calibrated, 1),
            is_actionable=calibrated >= 50,  # Actionable threshold
            signal_breakdown=signals,
            adjustments_applied=adjustments,
        )

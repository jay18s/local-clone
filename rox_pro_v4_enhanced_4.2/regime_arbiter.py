"""
Regime Arbiter — Resolves conflicts between Rule-Based and LLM regime outputs.
Determines which regime classification to trust based on accuracy history.

FIX-ARBITER-01: Added semantic proximity check — when both sources agree on
directional bias (both bullish or both bearish), use weighted merge instead
of falling to RANGE_BOUND default.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

logger = logging.getLogger("rox.reasoning.regime_arbiter")


# ── FIX-ARBITER-01: Regime spectrum mapping ──────────────────────────────
# Numeric scale: -2 (strongly bearish) to +2 (strongly bullish)
# Enables semantic proximity checks and weighted merging.
REGIME_SPECTRUM: Dict[str, int] = {
    "BEARISH":    -2,
    "BEAR":       -2,
    "MILD_BEAR":  -1,
    "CORRECTION": -1,
    "CAUTIOUS":   -1,
    "RANGE_BOUND": 0,
    "CONSOLIDATION": 0,
    "TRENDING":    1,
    "MILD_BULL":   1,
    "BULLISH":     2,
    "BULL":        2,
}

# Directional buckets for grouping
_BEARISH_REGIMES = {"BEARISH", "BEAR", "MILD_BEAR", "CORRECTION", "CAUTIOUS"}
_BULLISH_REGIMES = {"BULLISH", "BULL", "MILD_BULL", "TRENDING"}
_NEUTRAL_REGIMES = {"RANGE_BOUND", "CONSOLIDATION"}


@dataclass
class RegimeDecision:
    """Final regime decision after arbitration between rule-based and LLM sources."""
    regime: str
    confidence: float
    source: str
    rule_regime: str
    llm_regime: str
    rule_confidence: float
    llm_confidence: float


def _get_direction_bucket(regime: str) -> str:
    """Classify regime into bearish/neutral/bullish bucket."""
    if regime in _BEARISH_REGIMES:
        return "BEARISH"
    if regime in _BULLISH_REGIMES:
        return "BULLISH"
    return "NEUTRAL"


def _merge_regimes_weighted(
    rule_regime: str, rule_confidence: float,
    llm_regime: str, llm_confidence: float,
) -> Tuple[str, float]:
    """
    FIX-ARBITER-01: When both regimes share directional bias, merge them.

    Uses confidence-weighted average of the spectrum positions to produce
    a regime that reflects both signals' directional bias.

    Example: CAUTIOUS(-1, 62%) + CORRECTION(-1, 75%) → weighted position -1.0
             → maps to CAUTIOUS/CORRECTION (picks the higher-confidence label)
    """
    rule_pos = REGIME_SPECTRUM.get(rule_regime, 0)
    llm_pos = REGIME_SPECTRUM.get(llm_regime, 0)

    total_conf = rule_confidence + llm_confidence
    if total_conf == 0:
        return "RANGE_BOUND", 50.0

    weighted_pos = (rule_pos * rule_confidence + llm_pos * llm_confidence) / total_conf
    blended_conf = (rule_confidence + llm_confidence) / 2.0

    # Map weighted position back to regime label
    if weighted_pos <= -1.5:
        return "BEARISH", min(blended_conf, 90.0)
    elif weighted_pos <= -0.5:
        # Both bearish-leaning: pick the more specific label
        if llm_confidence >= rule_confidence:
            return llm_regime, blended_conf
        return rule_regime, blended_conf
    elif weighted_pos >= 1.5:
        return "BULLISH", min(blended_conf, 90.0)
    elif weighted_pos >= 0.5:
        if llm_confidence >= rule_confidence:
            return llm_regime, blended_conf
        return rule_regime, blended_conf
    else:
        return "RANGE_BOUND", blended_conf


class RegimeArbiter:
    """
    Resolves conflicts between rule-based and LLM regime classifications.

    FIX-ARBITER-01: New decision logic with semantic proximity:

    1. Both AGREE  → use shared regime, confidence = max
    2. LLM accuracy < 55% → always trust Rule-Based
    3. Rule conf > 70% AND LLM conf < 60% → trust Rule
    4. LLM conf > 80% AND Rule conf < 60% → trust LLM
    5. Same directional bucket → weighted merge (NEW)
       e.g., CAUTIOUS + CORRECTION → merge to bearish-leaning regime
    6. True cross-spectrum conflict → higher confidence wins
    7. Fallback → RANGE_BOUND (last resort only)
    """

    def resolve(
        self,
        rule_regime: str,
        rule_confidence: float,
        llm_regime: str,
        llm_confidence: float,
        llm_rolling_accuracy: float = 1.0,
    ) -> RegimeDecision:
        """
        Arbitrate between rule-based and LLM regime classifications.

        Args:
            rule_regime: Regime from rule-based classifier.
            rule_confidence: Rule classifier confidence (0-100).
            llm_regime: Regime from LLM classifier.
            llm_confidence: LLM confidence (0-100).
            llm_rolling_accuracy: Rolling accuracy of LLM regime (0.0-1.0).

        Returns:
            RegimeDecision with the chosen regime and arbitration source.
        """
        # Case 1: Agreement
        if rule_regime == llm_regime:
            decision = RegimeDecision(
                regime=rule_regime,
                confidence=max(rule_confidence, llm_confidence),
                source="BOTH_AGREE",
                rule_regime=rule_regime,
                llm_regime=llm_regime,
                rule_confidence=rule_confidence,
                llm_confidence=llm_confidence,
            )
            logger.info(f"Regime arbiter: BOTH_AGREE → {decision.regime}")
            return decision

        # Case 2: LLM degraded
        if llm_rolling_accuracy < 0.55:
            decision = RegimeDecision(
                regime=rule_regime,
                confidence=rule_confidence,
                source="RULE_OVERRIDE_LLM_DEGRADED",
                rule_regime=rule_regime,
                llm_regime=llm_regime,
                rule_confidence=rule_confidence,
                llm_confidence=llm_confidence,
            )
            logger.warning(
                f"Regime arbiter: LLM degraded (acc={llm_rolling_accuracy:.1%}) → "
                f"RULE_OVERRIDE to {rule_regime}"
            )
            return decision

        # Case 3: High-confidence rule
        if rule_confidence > 70 and llm_confidence < 60:
            decision = RegimeDecision(
                regime=rule_regime,
                confidence=rule_confidence,
                source="RULE_HIGH_CONF",
                rule_regime=rule_regime,
                llm_regime=llm_regime,
                rule_confidence=rule_confidence,
                llm_confidence=llm_confidence,
            )
            logger.info(f"Regime arbiter: RULE_HIGH_CONF → {rule_regime}")
            return decision

        # Case 4: High-confidence LLM
        if llm_confidence > 80 and rule_confidence < 60:
            decision = RegimeDecision(
                regime=llm_regime,
                confidence=llm_confidence,
                source="LLM_HIGH_CONF",
                rule_regime=rule_regime,
                llm_regime=llm_regime,
                rule_confidence=rule_confidence,
                llm_confidence=llm_confidence,
            )
            logger.info(f"Regime arbiter: LLM_HIGH_CONF → {llm_regime}")
            return decision

        # ── FIX-ARBITER-01: Semantic proximity check ──────────────────
        rule_bucket = _get_direction_bucket(rule_regime)
        llm_bucket = _get_direction_bucket(llm_regime)

        # Case 5: Same directional bucket → weighted merge
        if rule_bucket == llm_bucket:
            merged_regime, merged_conf = _merge_regimes_weighted(
                rule_regime, rule_confidence,
                llm_regime, llm_confidence,
            )
            decision = RegimeDecision(
                regime=merged_regime,
                confidence=round(merged_conf, 1),
                source="DIRECTIONAL_MERGE",
                rule_regime=rule_regime,
                llm_regime=llm_regime,
                rule_confidence=rule_confidence,
                llm_confidence=llm_confidence,
            )
            logger.info(
                f"Regime arbiter: DIRECTIONAL_MERGE (rule={rule_regime}/{rule_confidence:.0f}% "
                f"+ llm={llm_regime}/{llm_confidence:.0f}%, both {rule_bucket}) "
                f"→ {merged_regime} ({merged_conf:.0f}%)"
            )
            return decision

        # Case 6: Cross-spectrum conflict → higher confidence wins
        if rule_confidence >= llm_confidence + 10:
            decision = RegimeDecision(
                regime=rule_regime,
                confidence=rule_confidence,
                source="RULE_WINS_CROSS_SPECTRUM",
                rule_regime=rule_regime,
                llm_regime=llm_regime,
                rule_confidence=rule_confidence,
                llm_confidence=llm_confidence,
            )
            logger.info(
                f"Regime arbiter: RULE_WINS_CROSS_SPECTRUM "
                f"(rule={rule_regime}/{rule_confidence:.0f}% vs llm={llm_regime}/{llm_confidence:.0f}%)"
            )
            return decision
        elif llm_confidence >= rule_confidence + 10:
            decision = RegimeDecision(
                regime=llm_regime,
                confidence=llm_confidence,
                source="LLM_WINS_CROSS_SPECTRUM",
                rule_regime=rule_regime,
                llm_regime=llm_regime,
                rule_confidence=rule_confidence,
                llm_confidence=llm_confidence,
            )
            logger.info(
                f"Regime arbiter: LLM_WINS_CROSS_SPECTRUM "
                f"(llm={llm_regime}/{llm_confidence:.0f}% vs rule={rule_regime}/{rule_confidence:.0f}%)"
            )
            return decision

        # Case 7: Last resort — true conflict with no clear winner
        # Pick the more conservative (bearish) regime if one exists,
        # otherwise RANGE_BOUND
        if rule_bucket == "BEARISH" or llm_bucket == "BEARISH":
            bearish_regime = rule_regime if rule_bucket == "BEARISH" else llm_regime
            bearish_conf = max(rule_confidence, llm_confidence) * 0.8  # penalize conflict
            decision = RegimeDecision(
                regime=bearish_regime,
                confidence=round(bearish_conf, 1),
                source="CONFLICT_BEARISH_BIAS",
                rule_regime=rule_regime,
                llm_regime=llm_regime,
                rule_confidence=rule_confidence,
                llm_confidence=llm_confidence,
            )
            logger.warning(
                f"Regime arbiter: CONFLICT_BEARISH_BIAS "
                f"(rule={rule_regime}/{rule_confidence:.0f}% vs llm={llm_regime}/{llm_confidence:.0f}%) "
                f"→ {bearish_regime} (conservative bias)"
            )
            return decision

        decision = RegimeDecision(
            regime="RANGE_BOUND",
            confidence=50.0,
            source="CONFLICT_DEFAULT",
            rule_regime=rule_regime,
            llm_regime=llm_regime,
            rule_confidence=rule_confidence,
            llm_confidence=llm_confidence,
        )
        logger.warning(
            f"Regime arbiter: CONFLICT_DEFAULT (rule={rule_regime}/{rule_confidence:.0f}% "
            f"vs llm={llm_regime}/{llm_confidence:.0f}%) → RANGE_BOUND (last resort)"
        )
        return decision

"""
Regime Arbiter — Resolves conflicts between Rule-Based and LLM regime outputs.
Determines which regime classification to trust based on accuracy history.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("rox.reasoning.regime_arbiter")


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


class RegimeArbiter:
    """
    Resolves conflicts between rule-based and LLM regime classifications.

    Decision logic:
    1. Both AGREE  → use shared regime, confidence = max
    2. LLM accuracy < 55% → always trust Rule-Based
    3. Rule conf > 70% AND LLM conf < 60% → trust Rule
    4. LLM conf > 80% AND Rule conf < 60% → trust LLM
    5. Otherwise → RANGE_BOUND with 50% confidence (conservative default)
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

        # Case 5: Disagreement, no clear winner → conservative
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
            f"Regime arbiter: CONFLICT (rule={rule_regime}/{rule_confidence:.0f}% "
            f"vs llm={llm_regime}/{llm_confidence:.0f}%) → RANGE_BOUND default"
        )
        return decision

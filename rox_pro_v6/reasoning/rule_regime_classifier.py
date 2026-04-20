"""
Rule-Based Regime Classifier — PRIMARY regime source.
Uses leading indicators: VIX, Nifty DMA, FII flows, sector dispersion, momentum.
NO LLM calls. Pure Python deterministic logic.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("rox.reasoning.rule_regime")


@dataclass
class RegimeClassification:
    """Result from the rule-based regime classifier."""
    regime: str           # BULLISH, TRENDING, RANGE_BOUND, CAUTIOUS, BEARISH
    confidence: float     # 0-100
    source: str = "RULE_BASED"
    details: Optional[dict] = None


class RuleRegimeClassifier:
    """
    Deterministic regime classification using leading indicators.

    Input signals and weights:
      1. India VIX level + direction         — 30%
      2. Nifty position vs 20-DMA            — 25%
      3. FII net flow (cash market, Cr)      — 20%
      4. Sector dispersion (% green)         — 15%
      5. Nifty 5-day momentum slope          — 10%

    Output: RegimeClassification with label + confidence
    """

    def classify(
        self,
        vix: float,
        nifty_price: float,
        nifty_20dma: float,
        fii_net_flow: float = 0.0,
        sector_green_pct: float = 50.0,
        nifty_5d_slope: float = 0.0,
    ) -> RegimeClassification:
        """
        Classify the current market regime using deterministic rules.

        Args:
            vix: Current India VIX value.
            nifty_price: Current Nifty 50 price.
            nifty_20dma: Nifty 20-day moving average.
            fii_net_flow: FII net flow in Cr (positive = buying).
            sector_green_pct: Percentage of sectors in green (0-100).
            nifty_5d_slope: Nifty 5-day momentum slope.

        Returns:
            RegimeClassification with regime label, confidence, and signal details.
        """
        score = 0.0
        signals = {}

        # --- Signal 1: VIX (weight: 30) ---
        if vix > 22:
            vix_score = -30
        elif vix > 18:
            vix_score = -10
        elif vix < 13:
            vix_score = 20
        else:
            vix_score = 5
        score += vix_score
        signals["vix"] = {"value": vix, "score": vix_score}

        # --- Signal 2: Nifty vs 20-DMA (weight: 25) ---
        if nifty_20dma > 0:
            dma_pct = (nifty_price - nifty_20dma) / nifty_20dma * 100
        else:
            dma_pct = 0.0

        if dma_pct > 1.5:
            dma_score = 25
        elif dma_pct > 0:
            dma_score = 10
        elif dma_pct > -1.5:
            dma_score = -10
        else:
            dma_score = -25
        score += dma_score
        signals["dma"] = {"pct": round(dma_pct, 3), "score": dma_score}

        # --- Signal 3: FII flows (weight: 20) ---
        if fii_net_flow > 1000:
            fii_score = 20
        elif fii_net_flow > 0:
            fii_score = 5
        elif fii_net_flow > -1000:
            fii_score = -5
        else:
            fii_score = -20
        score += fii_score
        signals["fii"] = {"flow": fii_net_flow, "score": fii_score}

        # --- Signal 4: Sector dispersion (weight: 15) ---
        if sector_green_pct > 70:
            sector_score = 15
        elif sector_green_pct < 30:
            sector_score = -15
        else:
            sector_score = 0
        score += sector_score
        signals["sector"] = {"green_pct": sector_green_pct, "score": sector_score}

        # --- Signal 5: 5-day momentum (weight: 10) ---
        if nifty_5d_slope > 0.3:
            mom_score = 10
        elif nifty_5d_slope < -0.3:
            mom_score = -10
        else:
            mom_score = 0
        score += mom_score
        signals["momentum"] = {"slope": nifty_5d_slope, "score": mom_score}

        # --- Map composite score → regime ---
        if score > 40:
            regime = "BULLISH"
            confidence = min(95.0, 60 + score / 2)
        elif score > 15:
            regime = "TRENDING"
            confidence = min(85.0, 55 + score / 2)
        elif score > -15:
            regime = "RANGE_BOUND"
            # FIX-CONFIDENCE-01: RANGE_BOUND is fundamentally uncertain — the score is
            # near zero by definition.  Using abs(score) bonus gave up to 65% confidence
            # for a state that should be near 50%.  Fixed to 52 to signal mild certainty
            # of indirection without over-weighting the rule classifier in the arbiter.
            confidence = 52.0
        elif score > -40:
            regime = "CAUTIOUS"
            confidence = min(80.0, 55 + abs(score) / 2)
        else:
            regime = "BEARISH"
            confidence = min(95.0, 60 + abs(score) / 2)

        result = RegimeClassification(
            regime=regime,
            confidence=round(confidence, 1),
            source="RULE_BASED",
            details={"composite_score": round(score, 1), "signals": signals},
        )

        logger.info(
            f"Rule regime: {result.regime} ({result.confidence}%) "
            f"score={score:.1f} vix={vix} dma={dma_pct:.2f}% fii={fii_net_flow}"
        )
        return result

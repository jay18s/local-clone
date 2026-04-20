"""
Regime Transition Detector — Detects regime CHANGES and IMMINENT transitions.
Transitions are where edge lives. Catches leading indicator divergence
before the regime label actually flips.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("rox.reasoning.regime_transition")


@dataclass
class TransitionEvent:
    """Event describing a regime transition or imminent transition signal."""
    type: str                # "CONFIRMED", "IMMINENT", "NONE"
    from_regime: str = ""
    to_regime: str = ""
    signals: List[str] = field(default_factory=list)
    action: str = "NONE"     # What the system should do about it


class RegimeTransitionDetector:
    """
    Detects regime changes and imminent transitions using leading indicators.

    Triggers:
    1. VIX spike: >2 point increase from previous reading
    2. DMA break: Nifty crosses 20-DMA by >0.5%
    3. FII reversal: sign flip with >500 Cr delta

    CONFIRMED transition: regime label actually changed
    IMMINENT: regime hasn't changed yet but leading indicators firing
    """

    def __init__(self):
        self._previous_nifty_side = None  # "ABOVE" or "BELOW" DMA

    def detect(
        self,
        current_regime: str,
        previous_regime: str,
        vix_current: float,
        vix_previous: float,
        nifty_price: float,
        nifty_20dma: float,
        fii_current: float = 0.0,
        fii_previous: float = 0.0,
    ) -> TransitionEvent:
        """
        Detect regime transitions or imminent transition signals.

        Args:
            current_regime: Current regime label.
            previous_regime: Previous regime label.
            vix_current: Current VIX value.
            vix_previous: Previous VIX value.
            nifty_price: Current Nifty price.
            nifty_20dma: Nifty 20-DMA value.
            fii_current: Current FII net flow.
            fii_previous: Previous FII net flow.

        Returns:
            TransitionEvent describing what was detected.
        """
        # Confirmed transition
        is_transition = current_regime != previous_regime and previous_regime != ""

        if is_transition:
            event = TransitionEvent(
                type="CONFIRMED",
                from_regime=previous_regime,
                to_regime=current_regime,
                action="REDUCE_SIZE_AND_INCREASE_THRESHOLD",
            )
            logger.warning(
                f"REGIME TRANSITION CONFIRMED: {previous_regime} → {current_regime}"
            )
            return event

        # Check for IMMINENT transition signals
        signals = []

        # VIX spike
        vix_delta = vix_current - vix_previous
        if vix_delta > 2.0:
            signals.append("VIX_SPIKE")

        # DMA break
        if nifty_20dma > 0:
            current_side = "ABOVE" if nifty_price > nifty_20dma else "BELOW"
            dma_pct = abs(nifty_price - nifty_20dma) / nifty_20dma * 100
            if self._previous_nifty_side and current_side != self._previous_nifty_side and dma_pct > 0.5:
                signals.append("DMA_BREAK")
            self._previous_nifty_side = current_side

        # FII reversal
        if fii_previous != 0 and fii_current != 0:
            if (fii_current > 0) != (fii_previous > 0):
                if abs(fii_current - fii_previous) > 500:
                    signals.append("FII_REVERSAL")

        if signals:
            event = TransitionEvent(
                type="IMMINENT",
                signals=signals,
                action="INCREASE_CAUTION",
            )
            logger.info(f"REGIME TRANSITION IMMINENT: signals={signals}")
            return event

        return TransitionEvent(type="NONE")

    def reset(self):
        """Call at start of new trading day to reset DMA side tracking."""
        self._previous_nifty_side = None

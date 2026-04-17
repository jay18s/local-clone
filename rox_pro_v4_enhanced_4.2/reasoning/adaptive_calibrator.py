"""
Adaptive Confidence Calibrator — Learns which signals predict wins.
Replaces static weights with outcome-correlated Bayesian updating.
Uses EMA over a rolling window to continuously adjust signal weights
based on which signals actually correlated with winning trades.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("rox.reasoning.adaptive_calibrator")


@dataclass
class CalibrationRecord:
    """Record of a single calibration event for learning."""
    signal_scores: Dict[str, float]
    calibrated_score: float
    won: bool
    timestamp: str


class AdaptiveConfidenceCalibrator:
    """
    Adaptive confidence calibrator that learns from trade outcomes.

    Instead of static weights, this calibrator LEARNS which signals predict wins.

    Method:
    1. Start with initial weights as priors
    2. After each trade, compute: did each signal correlate with outcome?
    3. Update using EMA (alpha=0.1) over rolling 20-trade window
    4. Clamp: no weight < 0.05 or > 0.35, all sum to 1.0

    Default signal names (must match what callers pass):
    - debate_agreement, pattern_match, technical_alignment,
      volume_confirmation, regime_consistency, anti_consensus
    """

    DEFAULT_WEIGHTS = {
        "debate_agreement": 0.25,
        "pattern_match": 0.20,
        "technical_alignment": 0.20,
        "volume_confirmation": 0.15,
        "regime_consistency": 0.10,
        "anti_consensus": 0.10,
    }

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        alpha: float = 0.1,
        min_weight: float = 0.05,
        max_weight: float = 0.35,
        min_history: int = 5,
        window_size: int = 20,
    ):
        """
        Initialize the adaptive calibrator.

        Args:
            initial_weights: Starting weights for each signal. Defaults to DEFAULT_WEIGHTS.
            alpha: EMA smoothing factor for weight updates.
            min_weight: Minimum allowed weight for any signal.
            max_weight: Maximum allowed weight for any signal.
            min_history: Minimum trades before weight updates begin.
            window_size: Rolling window size for correlation calculation.
        """
        self.weights = dict(initial_weights or self.DEFAULT_WEIGHTS)
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_history = min_history
        self.window_size = window_size
        self.history: List[CalibrationRecord] = []

    def calibrate(self, signal_scores: Dict[str, float]) -> float:
        """
        Compute weighted confidence score from individual signal scores.

        Args:
            signal_scores: Dict mapping signal names to scores (0-100).

        Returns:
            Calibrated confidence score (0-100).
        """
        score = 0.0
        for key in self.weights:
            score += signal_scores.get(key, 50.0) * self.weights[key]
        return min(100.0, max(0.0, score))

    def update(self, signal_scores: Dict[str, float], won: bool,
               timestamp: str = "") -> None:
        """
        After each trade, update weights based on signal-outcome correlation.

        Only updates when enough history has been collected (min_history).
        Uses EMA smoothing to prevent wild swings from small samples.

        Args:
            signal_scores: Signal scores that were used for this trade.
            won: Whether the trade was a winner.
            timestamp: ISO timestamp for the record.
        """
        record = CalibrationRecord(
            signal_scores=dict(signal_scores),
            calibrated_score=self.calibrate(signal_scores),
            won=won,
            timestamp=timestamp,
        )
        self.history.append(record)

        if len(self.history) < self.min_history:
            logger.info(
                f"Calibrator: {len(self.history)}/{self.min_history} trades "
                f"collected, weight update deferred"
            )
            return

        # Rolling window
        recent = self.history[-self.window_size:]

        # Compute adjustments
        adjustments = {}
        for signal_name in self.weights:
            high_wins = sum(
                1 for r in recent
                if r.signal_scores.get(signal_name, 50) > 60 and r.won
            )
            high_losses = sum(
                1 for r in recent
                if r.signal_scores.get(signal_name, 50) > 60 and not r.won
            )
            total_high = high_wins + high_losses

            if total_high == 0:
                adjustments[signal_name] = 0.0
                continue

            win_rate_when_high = high_wins / total_high
            # Signal predicts wins when high → increase; predicts losses → decrease
            adjustments[signal_name] = (win_rate_when_high - 0.5) * self.alpha

        # Apply adjustments
        for signal_name, adj in adjustments.items():
            self.weights[signal_name] += adj

        # Normalize
        self._normalize()

        logger.info(
            f"Calibrator weights updated: "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.weights.items()))
        )

    def _normalize(self) -> None:
        """Normalize weights to sum to 1.0, clamp extremes, re-normalize."""
        # Clamp
        self.weights = {
            k: max(self.min_weight, min(self.max_weight, v))
            for k, v in self.weights.items()
        }
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def get_weights(self) -> Dict[str, float]:
        """
        Return a copy of the current signal weights.

        Returns:
            Dict mapping signal names to their current weights.
        """
        return dict(self.weights)

    def get_weight_evolution(self) -> List[Dict]:
        """
        Returns history of signal scores and outcomes for analysis.

        Returns:
            List of dicts with signal scores, calibrated score, won flag, and timestamp.
        """
        return [
            {
                "scores": r.signal_scores,
                "calibrated": r.calibrated_score,
                "won": r.won,
                "timestamp": r.timestamp,
            }
            for r in self.history
        ]

    def get_signal_correlations(self) -> Dict[str, float]:
        """
        Returns correlation of each signal with wins over history.

        A positive value means the signal tends to be high when trades win.
        A negative value means the signal tends to be high when trades lose.

        Returns:
            Dict mapping signal names to correlation values (-0.5 to 0.5).
        """
        if len(self.history) < self.min_history:
            return {k: 0.0 for k in self.weights}

        recent = self.history[-self.window_size:]
        correlations = {}
        for signal_name in self.weights:
            high_wins = sum(1 for r in recent if r.signal_scores.get(signal_name, 50) > 60 and r.won)
            high_losses = sum(1 for r in recent if r.signal_scores.get(signal_name, 50) > 60 and not r.won)
            total_high = high_wins + high_losses
            if total_high > 0:
                correlations[signal_name] = round((high_wins / total_high) - 0.5, 3)
            else:
                correlations[signal_name] = 0.0
        return correlations

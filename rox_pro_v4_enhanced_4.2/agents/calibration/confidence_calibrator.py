"""
Confidence Calibrator - Agent conviction calibration (Enhancement F1)
======================================================================

Adjusts agent conviction values based on historical accuracy:
- Tracks historical conviction vs outcome
- Builds calibration curves per agent and regime
- Applies post-processing to agent reports
- Ensures conviction reflects actual win rates
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import threading
import math

# Import from parent config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MarketRegime


@dataclass
class CalibratedResult:
    """Result of conviction calibration."""
    raw_conviction: int
    calibrated_conviction: int
    calibration_factor: float
    sample_size: int
    regime: MarketRegime
    source: str  # "CALIBRATED" or "RAW" (if insufficient data)


@dataclass
class CalibrationBucket:
    """Calibration data for a conviction range."""
    conviction_min: int
    conviction_max: int
    total_predictions: int
    wins: int
    actual_win_rate: float

    @property
    def expected_win_rate(self) -> float:
        """Expected win rate based on conviction."""
        return (self.conviction_min + self.conviction_max) / 2 / 100


@dataclass
class AgentCalibrationData:
    """Calibration data for a single agent."""
    agent_name: str
    buckets: Dict[str, List[CalibrationBucket]]  # regime -> buckets
    last_updated: datetime
    total_samples: int

    def get_bucket_for_conviction(
        self,
        conviction: int,
        regime: str
    ) -> Optional[CalibrationBucket]:
        """Get calibration bucket for a conviction value."""
        regime_buckets = self.buckets.get(regime, [])
        for bucket in regime_buckets:
            if bucket.conviction_min <= conviction <= bucket.conviction_max:
                return bucket
        return None


class ConfidenceCalibrator:
    """
    Calibrates agent conviction to reflect actual win rates.

    Tracks historical conviction vs outcome and builds calibration curves.
    Applied as post-processing to agent reports.
    """

    # Bucket ranges for calibration
    BUCKET_RANGES = [
        (0, 40),
        (40, 55),
        (55, 65),
        (65, 75),
        (75, 85),
        (85, 100)
    ]

    MIN_SAMPLES_FOR_CALIBRATION = 15
    CALIBRATION_SMOOTHING = 0.3  # Smooth calibration factor changes

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.calibration_file = self.data_dir / "calibration" / "agent_calibration.json"
        self.logger = logging.getLogger("ConfidenceCalibrator")
        self._lock = threading.Lock()

        # Calibration data: agent_name -> AgentCalibrationData
        self._calibration_data: Dict[str, AgentCalibrationData] = {}

        # In-memory prediction tracking for ongoing updates
        self._pending_outcomes: Dict[str, Dict] = {}

        self._load_calibration()

    def _load_calibration(self):
        """Load calibration data from disk."""
        try:
            if self.calibration_file.exists():
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)

                for agent_name, agent_data in data.items():
                    buckets = {}
                    for regime, regime_buckets in agent_data.get("buckets", {}).items():
                        buckets[regime] = [
                            CalibrationBucket(**b) for b in regime_buckets
                        ]

                    self._calibration_data[agent_name] = AgentCalibrationData(
                        agent_name=agent_name,
                        buckets=buckets,
                        last_updated=datetime.fromisoformat(agent_data.get("last_updated", datetime.now().isoformat())),
                        total_samples=agent_data.get("total_samples", 0)
                    )

                self.logger.info(f"Loaded calibration data for {len(self._calibration_data)} agents")

        except Exception as e:
            self.logger.warning(f"Could not load calibration data: {e}")

    def _save_calibration(self):
        """Save calibration data to disk."""
        try:
            self.calibration_file.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for agent_name, agent_cal in self._calibration_data.items():
                buckets = {}
                for regime, regime_buckets in agent_cal.buckets.items():
                    buckets[regime] = [
                        {
                            "conviction_min": b.conviction_min,
                            "conviction_max": b.conviction_max,
                            "total_predictions": b.total_predictions,
                            "wins": b.wins,
                            "actual_win_rate": b.actual_win_rate
                        }
                        for b in regime_buckets
                    ]

                data[agent_name] = {
                    "buckets": buckets,
                    "last_updated": agent_cal.last_updated.isoformat(),
                    "total_samples": agent_cal.total_samples
                }

            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Could not save calibration data: {e}")

    def calibrate(
        self,
        agent_name: str,
        raw_conviction: int,
        regime: MarketRegime
    ) -> CalibratedResult:
        """
        Calibrate an agent's conviction.

        Args:
            agent_name: Name of the agent
            raw_conviction: Original conviction value (0-100)
            regime: Current market regime

        Returns:
            CalibratedResult with adjusted conviction and calibration metadata
        """
        regime_str = regime.value if hasattr(regime, 'value') else str(regime)

        with self._lock:
            # Check if we have calibration data
            if agent_name not in self._calibration_data:
                return CalibratedResult(
                    raw_conviction=raw_conviction,
                    calibrated_conviction=raw_conviction,
                    calibration_factor=1.0,
                    sample_size=0,
                    regime=regime,
                    source="RAW"
                )

            agent_data = self._calibration_data[agent_name]
            bucket = agent_data.get_bucket_for_conviction(raw_conviction, regime_str)

            if bucket is None or bucket.total_predictions < self.MIN_SAMPLES_FOR_CALIBRATION:
                return CalibratedResult(
                    raw_conviction=raw_conviction,
                    calibrated_conviction=raw_conviction,
                    calibration_factor=1.0,
                    sample_size=bucket.total_predictions if bucket else 0,
                    regime=regime,
                    source="RAW"
                )

            # Calculate calibration factor
            expected_wr = bucket.expected_win_rate
            actual_wr = bucket.actual_win_rate

            if expected_wr > 0:
                calibration_factor = actual_wr / expected_wr
            else:
                calibration_factor = 1.0

            # Apply smoothing to avoid extreme adjustments
            calibration_factor = 1.0 + (calibration_factor - 1.0) * self.CALIBRATION_SMOOTHING

            # Clamp factor to reasonable range
            calibration_factor = max(0.7, min(1.3, calibration_factor))

            # Apply calibration
            calibrated_conviction = int(raw_conviction * calibration_factor)
            calibrated_conviction = max(0, min(100, calibrated_conviction))

            return CalibratedResult(
                raw_conviction=raw_conviction,
                calibrated_conviction=calibrated_conviction,
                calibration_factor=calibration_factor,
                sample_size=bucket.total_predictions,
                regime=regime,
                source="CALIBRATED"
            )

    def record_prediction(
        self,
        agent_name: str,
        raw_conviction: int,
        regime: MarketRegime,
        prediction_id: str
    ):
        """
        Record a prediction for later outcome tracking.

        Args:
            agent_name: Name of the agent
            raw_conviction: Original conviction value
            regime: Market regime at prediction time
            prediction_id: Unique identifier for this prediction
        """
        regime_str = regime.value if hasattr(regime, 'value') else str(regime)

        with self._lock:
            self._pending_outcomes[prediction_id] = {
                "agent_name": agent_name,
                "raw_conviction": raw_conviction,
                "regime": regime_str,
                "timestamp": datetime.now().isoformat()
            }

    def record_outcome(
        self,
        agent_name: str,
        raw_conviction: int,
        regime: MarketRegime,
        outcome: str,
        prediction_id: Optional[str] = None
    ):
        """
        Record prediction outcome for calibration updates.

        Args:
            agent_name: Name of the agent
            raw_conviction: Original conviction value
            regime: Market regime at prediction time
            outcome: "WIN" or "LOSS"
            prediction_id: Optional prediction ID for tracking
        """
        regime_str = regime.value if hasattr(regime, 'value') else str(regime)
        is_win = outcome.upper() == "WIN"

        with self._lock:
            # Initialize agent data if needed
            if agent_name not in self._calibration_data:
                self._calibration_data[agent_name] = AgentCalibrationData(
                    agent_name=agent_name,
                    buckets={},
                    last_updated=datetime.now(),
                    total_samples=0
                )

            agent_data = self._calibration_data[agent_name]

            # Initialize regime buckets if needed
            if regime_str not in agent_data.buckets:
                agent_data.buckets[regime_str] = [
                    CalibrationBucket(
                        conviction_min=min_v,
                        conviction_max=max_v,
                        total_predictions=0,
                        wins=0,
                        actual_win_rate=0.5
                    )
                    for min_v, max_v in self.BUCKET_RANGES
                ]

            # Find and update the appropriate bucket
            for bucket in agent_data.buckets[regime_str]:
                if bucket.conviction_min <= raw_conviction <= bucket.conviction_max:
                    bucket.total_predictions += 1
                    if is_win:
                        bucket.wins += 1
                    bucket.actual_win_rate = bucket.wins / bucket.total_predictions
                    break

            agent_data.total_samples += 1
            agent_data.last_updated = datetime.now()

            # Remove from pending if tracked
            if prediction_id and prediction_id in self._pending_outcomes:
                del self._pending_outcomes[prediction_id]

        # Periodically save
        if len(self._pending_outcomes) % 50 == 0:
            self._save_calibration()

    def rebuild_calibration(self, trade_history: List[Dict[str, Any]] = None):
        """
        Rebuild calibration curves from historical data.

        Args:
            trade_history: Optional list of historical trades
        """
        self.logger.info("Rebuilding calibration curves...")

        with self._lock:
            # Reset calibration data
            self._calibration_data = {}

            if trade_history:
                for trade in trade_history:
                    agent_name = trade.get("agent_name")
                    conviction = trade.get("conviction")
                    regime = trade.get("regime")
                    outcome = trade.get("outcome")

                    if all([agent_name, conviction, regime, outcome]):
                        self.record_outcome(
                            agent_name=agent_name,
                            raw_conviction=conviction,
                            regime=MarketRegime(regime) if isinstance(regime, str) else regime,
                            outcome=outcome
                        )

        self._save_calibration()
        self.logger.info("Calibration curves rebuilt")

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration data."""
        summary = {}
        for agent_name, agent_data in self._calibration_data.items():
            agent_summary = {
                "total_samples": agent_data.total_samples,
                "last_updated": agent_data.last_updated.isoformat(),
                "regimes": {}
            }

            for regime, buckets in agent_data.buckets.items():
                regime_summary = []
                for bucket in buckets:
                    if bucket.total_predictions > 0:
                        regime_summary.append({
                            "range": f"{bucket.conviction_min}-{bucket.conviction_max}",
                            "samples": bucket.total_predictions,
                            "actual_wr": f"{bucket.actual_win_rate*100:.1f}%",
                            "expected_wr": f"{bucket.expected_win_rate*100:.1f}%"
                        })
                agent_summary["regimes"][regime] = regime_summary

            summary[agent_name] = agent_summary

        return summary

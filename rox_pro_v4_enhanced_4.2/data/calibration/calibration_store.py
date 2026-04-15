"""
Calibration Store - Calibration data persistence
=================================================

Persists calibration data for agent conviction calibration:
- Historical calibration curves
- Performance tracking per conviction bucket
- Regime-specific calibration
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading


@dataclass
class CalibrationRecord:
    """A single calibration record."""
    agent_name: str
    regime: str
    conviction_range: str  # e.g., "65-75"
    total_predictions: int
    wins: int
    losses: int
    win_rate: float
    timestamp: str
    period_start: str
    period_end: str


@dataclass
class CalibrationSnapshot:
    """A snapshot of calibration state at a point in time."""
    timestamp: str
    agents: Dict[str, Dict[str, Any]]  # agent -> regime -> buckets
    total_samples: int
    version: str = "1.0"


class CalibrationStore:
    """
    Persistent storage for calibration data.
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.store_file = self.data_dir / "calibration" / "calibration_store.json"
        self.history_file = self.data_dir / "calibration" / "calibration_history.jsonl"
        self.logger = logging.getLogger("CalibrationStore")
        self._lock = threading.Lock()

        self._current: Dict[str, Dict[str, Any]] = {}
        self._history: List[CalibrationSnapshot] = []
        self._load()

    def _load(self):
        """Load calibration data from disk."""
        try:
            if self.store_file.exists():
                with open(self.store_file, 'r') as f:
                    self._current = json.load(f)

                self.logger.info("Loaded calibration store")

            # Load history
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self._history.append(CalibrationSnapshot(**data))

        except Exception as e:
            self.logger.warning(f"Could not load calibration store: {e}")

    def _save(self):
        """Save calibration data to disk."""
        try:
            self.store_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.store_file, 'w') as f:
                json.dump(self._current, f, indent=2)

        except Exception as e:
            self.logger.error(f"Could not save calibration store: {e}")

    def update(
        self,
        agent_name: str,
        regime: str,
        calibration_data: Dict[str, Any]
    ):
        """
        Update calibration data for an agent in a regime.

        Args:
            agent_name: Name of the agent
            regime: Market regime
            calibration_data: Calibration bucket data
        """
        with self._lock:
            if agent_name not in self._current:
                self._current[agent_name] = {}

            self._current[agent_name][regime] = calibration_data
            self._current[agent_name]["last_updated"] = datetime.now().isoformat()

            self._save()

    def get(
        self,
        agent_name: str,
        regime: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get calibration data for an agent.

        Args:
            agent_name: Name of the agent
            regime: Optional regime filter

        Returns:
            Calibration data or None
        """
        agent_data = self._current.get(agent_name)
        if not agent_data:
            return None

        if regime:
            return agent_data.get(regime)

        return agent_data

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all calibration data."""
        return self._current.copy()

    def create_snapshot(self):
        """Create a snapshot of current calibration state."""
        total_samples = sum(
            sum(
                bucket.get("total_predictions", 0)
                for regime_data in agent_data.values()
                if isinstance(regime_data, dict)
                for bucket in (regime_data if isinstance(regime_data, list) else [regime_data])
                if isinstance(bucket, dict)
            )
            for agent_data in self._current.values()
            if isinstance(agent_data, dict)
        )

        snapshot = CalibrationSnapshot(
            timestamp=datetime.now().isoformat(),
            agents={
                agent: data
                for agent, data in self._current.items()
                if agent != "last_updated"
            },
            total_samples=total_samples
        )

        with self._lock:
            self._history.append(snapshot)

            # Append to history file
            try:
                self.history_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.history_file, 'a') as f:
                    f.write(json.dumps(asdict(snapshot)) + "\n")
            except Exception as e:
                self.logger.error(f"Could not append to history: {e}")

        return snapshot

    def get_history(self, days: int = 90) -> List[CalibrationSnapshot]:
        """Get calibration history for the last N days."""
        cutoff = datetime.now() - timedelta(days=days)

        return [
            snap for snap in self._history
            if datetime.fromisoformat(snap.timestamp) > cutoff
        ]

    def get_trend(
        self,
        agent_name: str,
        regime: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get calibration trend for an agent in a regime.

        Args:
            agent_name: Name of the agent
            regime: Market regime
            days: Number of days to analyze

        Returns:
            Trend analysis
        """
        history = self.get_history(days)

        trend = {
            "agent": agent_name,
            "regime": regime,
            "snapshots": [],
            "win_rate_change": 0,
            "sample_change": 0
        }

        for snap in history:
            if agent_name in snap.agents:
                agent_data = snap.agents[agent_name]
                if regime in agent_data:
                    trend["snapshots"].append({
                        "timestamp": snap.timestamp,
                        "data": agent_data[regime]
                    })

        # Calculate changes if we have enough snapshots
        if len(trend["snapshots"]) >= 2:
            first = trend["snapshots"][0]
            last = trend["snapshots"][-1]

            if isinstance(first["data"], dict) and isinstance(last["data"], dict):
                trend["win_rate_change"] = (
                    last["data"].get("win_rate", 0) -
                    first["data"].get("win_rate", 0)
                )
                trend["sample_change"] = (
                    last["data"].get("total_predictions", 0) -
                    first["data"].get("total_predictions", 0)
                )

        return trend

    def clear_old_history(self, keep_days: int = 365):
        """Clear history older than keep_days."""
        cutoff = datetime.now() - timedelta(days=keep_days)

        with self._lock:
            self._history = [
                snap for snap in self._history
                if datetime.fromisoformat(snap.timestamp) > cutoff
            ]

        self.logger.info(f"Cleared calibration history older than {keep_days} days")

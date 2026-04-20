"""
Recommendations Store - Meta-learning data persistence
=======================================================

Stores and retrieves meta-learning recommendations:
- Historical recommendations
- Applied vs rejected recommendations
- Outcome tracking for recommendations
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading


@dataclass
class StoredRecommendation:
    """A stored recommendation from meta-learning."""
    id: str
    week_start: str
    week_end: str
    timestamp: str
    source: str  # "LLM" or "FALLBACK"
    agent_weight_adjustments: List[Dict]
    regime_specific_rules: List[Dict]
    pattern_adjustments: List[str]
    sector_insights: List[str]
    systemic_improvements: List[str]
    next_week_focus: str
    confidence: int
    status: str  # "PENDING", "APPROVED", "APPLIED", "REJECTED"
    outcome: Optional[str] = None
    notes: Optional[str] = None


class RecommendationsStore:
    """
    Persistent storage for meta-learning recommendations.
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self.store_file = self.data_dir / "meta_learning" / "recommendations.json"
        self.logger = logging.getLogger("RecommendationsStore")
        self._lock = threading.Lock()

        self._recommendations: Dict[str, StoredRecommendation] = {}
        self._load()

    def _load(self):
        """Load recommendations from disk."""
        try:
            if self.store_file.exists():
                with open(self.store_file, 'r') as f:
                    data = json.load(f)

                for rec_id, rec_data in data.items():
                    self._recommendations[rec_id] = StoredRecommendation(**rec_data)

                self.logger.info(f"Loaded {len(self._recommendations)} recommendations")

        except Exception as e:
            self.logger.warning(f"Could not load recommendations: {e}")

    def _save(self):
        """Save recommendations to disk."""
        try:
            self.store_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                rec_id: asdict(rec)
                for rec_id, rec in self._recommendations.items()
            }

            with open(self.store_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Could not save recommendations: {e}")

    def store(self, recommendation: Any) -> str:
        """
        Store a new recommendation.

        Args:
            recommendation: MetaLearningResult to store

        Returns:
            Unique recommendation ID
        """
        rec_id = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        stored = StoredRecommendation(
            id=rec_id,
            week_start=recommendation.week_start or "",
            week_end=recommendation.week_end or "",
            timestamp=datetime.now().isoformat(),
            source=recommendation.source,
            agent_weight_adjustments=[
                {
                    "agent_name": adj.agent_name,
                    "action": adj.action,
                    "amount": adj.amount,
                    "reason": adj.reason
                }
                for adj in recommendation.agent_weight_adjustments
            ],
            regime_specific_rules=[
                {
                    "regime": rule.regime,
                    "rule": rule.rule,
                    "reason": rule.reason
                }
                for rule in recommendation.regime_specific_rules
            ],
            pattern_adjustments=recommendation.pattern_adjustments,
            sector_insights=recommendation.sector_insights,
            systemic_improvements=recommendation.systemic_improvements,
            next_week_focus=recommendation.next_week_focus,
            confidence=recommendation.confidence_in_recommendations,
            status="PENDING"
        )

        with self._lock:
            self._recommendations[rec_id] = stored
            self._save()

        return rec_id

    def get(self, rec_id: str) -> Optional[StoredRecommendation]:
        """Get a recommendation by ID."""
        return self._recommendations.get(rec_id)

    def get_pending(self) -> List[StoredRecommendation]:
        """Get all pending recommendations."""
        return [
            rec for rec in self._recommendations.values()
            if rec.status == "PENDING"
        ]

    def get_recent(self, days: int = 30) -> List[StoredRecommendation]:
        """Get recommendations from the last N days."""
        cutoff = datetime.now() - timedelta(days=days)

        return [
            rec for rec in self._recommendations.values()
            if datetime.fromisoformat(rec.timestamp) > cutoff
        ]

    def update_status(
        self,
        rec_id: str,
        status: str,
        notes: Optional[str] = None,
        outcome: Optional[str] = None
    ):
        """
        Update the status of a recommendation.

        Args:
            rec_id: Recommendation ID
            status: New status
            notes: Optional notes
            outcome: Optional outcome description
        """
        with self._lock:
            if rec_id in self._recommendations:
                rec = self._recommendations[rec_id]
                rec.status = status
                if notes:
                    rec.notes = notes
                if outcome:
                    rec.outcome = outcome
                self._save()

    def mark_applied(self, rec_id: str, notes: str = None):
        """Mark a recommendation as applied."""
        self.update_status(rec_id, "APPLIED", notes)

    def mark_rejected(self, rec_id: str, reason: str):
        """Mark a recommendation as rejected."""
        self.update_status(rec_id, "REJECTED", reason)

    def get_effectiveness_report(self) -> Dict[str, Any]:
        """Generate a report on recommendation effectiveness."""
        applied = [
            rec for rec in self._recommendations.values()
            if rec.status == "APPLIED"
        ]

        report = {
            "total_recommendations": len(self._recommendations),
            "pending": len([r for r in self._recommendations.values() if r.status == "PENDING"]),
            "applied": len(applied),
            "rejected": len([r for r in self._recommendations.values() if r.status == "REJECTED"]),
            "recent_count": len(self.get_recent(30)),
            "avg_confidence": 0,
            "source_breakdown": {
                "LLM": 0,
                "FALLBACK": 0
            }
        }

        if self._recommendations:
            report["avg_confidence"] = sum(
                rec.confidence for rec in self._recommendations.values()
            ) / len(self._recommendations)

            for rec in self._recommendations.values():
                if rec.source in report["source_breakdown"]:
                    report["source_breakdown"][rec.source] += 1

        return report


# Import timedelta for the get_recent method
from datetime import timedelta

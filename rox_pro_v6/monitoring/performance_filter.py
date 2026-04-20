"""
Performance Filter - Performance monitoring module
===================================================

Filters and monitors agent performance:
- Tracks real-time performance metrics
- Identifies underperforming agents
- Provides performance-based filtering
- Generates performance reports
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent."""
    agent_name: str
    total_predictions: int
    wins: int
    losses: int
    win_rate: float
    avg_conviction: float
    avg_return: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    last_updated: datetime

    @property
    def is_reliable(self) -> bool:
        """Check if metrics are based on sufficient data."""
        return self.total_predictions >= 20


@dataclass
class PerformanceAlert:
    """Alert for performance issues."""
    agent_name: str
    alert_type: str  # "LOW_WIN_RATE", "HIGH_DRAWDOWN", "OVERCONFIDENT", etc.
    message: str
    severity: str  # "INFO", "WARNING", "CRITICAL"
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class PerformanceFilter:
    """
    Monitors and filters agent performance.

    Tracks real-time metrics and provides alerts for performance issues.
    """

    # Thresholds
    MIN_WIN_RATE = 0.45
    MIN_SAMPLES = 20
    MAX_DRAWDOWN = 0.15
    MAX_OVERCONFIDENCE_GAP = 0.20  # Max gap between conviction and actual win rate

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.logger = logging.getLogger("PerformanceFilter")

        self._lock = threading.Lock()
        self._predictions: Dict[str, List[Dict]] = defaultdict(list)
        self._metrics_cache: Dict[str, PerformanceMetrics] = {}
        self._alerts: List[PerformanceAlert] = []

    def record_prediction(
        self,
        agent_name: str,
        prediction: Dict[str, Any]
    ):
        """
        Record a prediction for tracking.

        Args:
            agent_name: Name of the agent
            prediction: Prediction details (direction, conviction, etc.)
        """
        with self._lock:
            prediction['timestamp'] = datetime.now()
            self._predictions[agent_name].append(prediction)

            # Clear old predictions
            cutoff = datetime.now() - timedelta(days=self.lookback_days)
            self._predictions[agent_name] = [
                p for p in self._predictions[agent_name]
                if p.get('timestamp', datetime.min) > cutoff
            ]

            # Invalidate cache
            if agent_name in self._metrics_cache:
                del self._metrics_cache[agent_name]

    def record_outcome(
        self,
        agent_name: str,
        prediction_id: str,
        outcome: str,
        return_pct: float = 0
    ):
        """
        Record the outcome of a prediction.

        Args:
            agent_name: Name of the agent
            prediction_id: ID of the prediction
            outcome: "WIN" or "LOSS"
            return_pct: Return percentage
        """
        with self._lock:
            for pred in self._predictions[agent_name]:
                if pred.get('id') == prediction_id:
                    pred['outcome'] = outcome
                    pred['return_pct'] = return_pct
                    pred['resolved_at'] = datetime.now()
                    break

            # Invalidate cache
            if agent_name in self._metrics_cache:
                del self._metrics_cache[agent_name]

        # Check for alerts
        self._check_performance_alerts(agent_name)

    def get_metrics(self, agent_name: str) -> Optional[PerformanceMetrics]:
        """
        Get performance metrics for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            PerformanceMetrics or None if no data
        """
        with self._lock:
            if agent_name in self._metrics_cache:
                return self._metrics_cache[agent_name]

            predictions = self._predictions.get(agent_name, [])
            resolved = [p for p in predictions if 'outcome' in p]

            if not resolved:
                return None

            wins = [p for p in resolved if p['outcome'] == 'WIN']
            losses = [p for p in resolved if p['outcome'] == 'LOSS']

            total = len(resolved)
            win_count = len(wins)
            win_rate = win_count / total if total > 0 else 0

            avg_conviction = sum(p.get('conviction', 50) for p in resolved) / total if total > 0 else 50
            avg_return = sum(p.get('return_pct', 0) for p in resolved) / total if total > 0 else 0

            # Profit factor
            total_wins = sum(p.get('return_pct', 0) for p in wins)
            total_losses = sum(abs(p.get('return_pct', 0)) for p in losses)
            profit_factor = total_wins / total_losses if total_losses > 0 else 0

            # Simple Sharpe approximation
            returns = [p.get('return_pct', 0) for p in resolved]
            if len(returns) > 1:
                avg_ret = sum(returns) / len(returns)
                variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
                std_dev = variance ** 0.5 if variance > 0 else 1
                sharpe = avg_ret / std_dev if std_dev > 0 else 0
            else:
                sharpe = 0

            # Max drawdown
            cumulative = 0
            peak = 0
            max_dd = 0
            for r in returns:
                cumulative += r
                if cumulative > peak:
                    peak = cumulative
                dd = (peak - cumulative) / 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            metrics = PerformanceMetrics(
                agent_name=agent_name,
                total_predictions=total,
                wins=win_count,
                losses=len(losses),
                win_rate=win_rate,
                avg_conviction=avg_conviction,
                avg_return=avg_return,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                last_updated=datetime.now()
            )

            self._metrics_cache[agent_name] = metrics
            return metrics

    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get metrics for all agents."""
        metrics = {}
        for agent_name in self._predictions.keys():
            m = self.get_metrics(agent_name)
            if m:
                metrics[agent_name] = m
        return metrics

    def _check_performance_alerts(self, agent_name: str):
        """Check for performance issues and generate alerts."""
        metrics = self.get_metrics(agent_name)
        if not metrics or not metrics.is_reliable:
            return

        # Check win rate
        if metrics.win_rate < self.MIN_WIN_RATE:
            self._alerts.append(PerformanceAlert(
                agent_name=agent_name,
                alert_type="LOW_WIN_RATE",
                message=f"{agent_name} win rate ({metrics.win_rate*100:.1f}%) below threshold",
                severity="WARNING" if metrics.win_rate > 0.35 else "CRITICAL",
                timestamp=datetime.now(),
                details={"win_rate": metrics.win_rate, "threshold": self.MIN_WIN_RATE}
            ))

        # Check drawdown
        if metrics.max_drawdown > self.MAX_DRAWDOWN:
            self._alerts.append(PerformanceAlert(
                agent_name=agent_name,
                alert_type="HIGH_DRAWDOWN",
                message=f"{agent_name} high drawdown ({metrics.max_drawdown*100:.1f}%)",
                severity="WARNING",
                timestamp=datetime.now(),
                details={"max_drawdown": metrics.max_drawdown, "threshold": self.MAX_DRAWDOWN}
            ))

        # Check overconfidence
        expected_wr = metrics.avg_conviction / 100
        gap = expected_wr - metrics.win_rate
        if gap > self.MAX_OVERCONFIDENCE_GAP:
            self._alerts.append(PerformanceAlert(
                agent_name=agent_name,
                alert_type="OVERCONFIDENT",
                message=f"{agent_name} conviction exceeds performance by {gap*100:.1f}%",
                severity="WARNING",
                timestamp=datetime.now(),
                details={"conviction": metrics.avg_conviction, "win_rate": metrics.win_rate}
            ))

    def get_alerts(
        self,
        agent_name: Optional[str] = None,
        severity: Optional[str] = None,
        hours: int = 24
    ) -> List[PerformanceAlert]:
        """
        Get performance alerts.

        Args:
            agent_name: Filter by agent (optional)
            severity: Filter by severity (optional)
            hours: Hours to look back

        Returns:
            List of matching alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        alerts = [
            a for a in self._alerts
            if a.timestamp > cutoff
        ]

        if agent_name:
            alerts = [a for a in alerts if a.agent_name == agent_name]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def filter_agents_by_performance(
        self,
        agents: List[str],
        min_win_rate: Optional[float] = None,
        min_samples: Optional[int] = None
    ) -> List[str]:
        """
        Filter agents by performance criteria.

        Args:
            agents: List of agent names
            min_win_rate: Minimum win rate (default: self.MIN_WIN_RATE)
            min_samples: Minimum samples (default: self.MIN_SAMPLES)

        Returns:
            List of agents meeting criteria
        """
        min_wr = min_win_rate or self.MIN_WIN_RATE
        min_samp = min_samples or self.MIN_SAMPLES

        filtered = []
        for agent in agents:
            metrics = self.get_metrics(agent)
            if metrics:
                if metrics.is_reliable and metrics.win_rate >= min_wr:
                    filtered.append(agent)
            else:
                # No data yet - include but flag
                filtered.append(agent)

        return filtered

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        all_metrics = self.get_all_metrics()

        report = {
            "generated_at": datetime.now().isoformat(),
            "lookback_days": self.lookback_days,
            "agents": {},
            "summary": {
                "total_agents": len(all_metrics),
                "reliable_agents": sum(1 for m in all_metrics.values() if m.is_reliable),
                "avg_win_rate": 0,
                "avg_profit_factor": 0,
            },
            "alerts": self.get_alerts(hours=168)  # Last week
        }

        if all_metrics:
            reliable = [m for m in all_metrics.values() if m.is_reliable]
            if reliable:
                report["summary"]["avg_win_rate"] = sum(m.win_rate for m in reliable) / len(reliable)
                report["summary"]["avg_profit_factor"] = sum(m.profit_factor for m in reliable) / len(reliable)

        for agent_name, metrics in all_metrics.items():
            report["agents"][agent_name] = {
                "total_predictions": metrics.total_predictions,
                "win_rate": f"{metrics.win_rate*100:.1f}%",
                "avg_conviction": f"{metrics.avg_conviction:.0f}",
                "profit_factor": f"{metrics.profit_factor:.2f}",
                "max_drawdown": f"{metrics.max_drawdown*100:.1f}%",
                "reliable": metrics.is_reliable
            }

        return report

"""
FIX 11 — Enhanced Rejection Logger
====================================
Structured rejection logging across all modules.
Every rejection point emits a standardised log line with:
  module, symbol, strategy, reason, threshold, actual_value, action

Also provides end-of-day summary aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger("rox.rejection")


@dataclass
class RejectionRecord:
    timestamp: datetime
    module: str        # "RULE_VALIDATOR" | "CROSS_EXAMINER" | "CHECKLIST" | "CORRELATION" | etc.
    symbol: str        # "NIFTY" | "ADANIPORTS" | etc.
    strategy: str      # "LONG_STRADDLE" | "BUY_CE" | etc.
    reason: str        # "R:R below threshold"
    threshold: str     # "R:R >= 1.5"
    actual_value: str  # "R:R = 0.63"
    action: str        # "REJECTED" | "WATCH_ONLY" | "REDUCED_SIZE"
    conviction: int = 0
    extra: Optional[dict] = None


class RejectionLogger:
    """
    Centralised rejection logging and aggregation.

    Usage:
        rejection_logger = RejectionLogger()
        rejection_logger.log(
            module="RULE_VALIDATOR",
            symbol="ADANIPORTS",
            strategy="BUY_CE",
            reason="R:R below 1.5:1",
            threshold=">=1.5",
            actual="0.63",
            action="REJECTED",
        )
    """

    def __init__(self):
        self._records: List[RejectionRecord] = []
        self._today = date.today()

    def log(
        self,
        module: str,
        symbol: str,
        strategy: str,
        reason: str,
        threshold: str,
        actual: str,
        action: str,
        conviction: int = 0,
        extra: Optional[dict] = None,
    ) -> None:
        """Log a rejection event with structured metadata."""
        record = RejectionRecord(
            timestamp=datetime.now(),
            module=module.upper(),
            symbol=symbol.upper(),
            strategy=strategy.upper(),
            reason=reason,
            threshold=threshold,
            actual_value=actual,
            action=action.upper(),
            conviction=conviction,
            extra=extra,
        )
        self._records.append(record)

        # Emit structured log line
        logger.info(
            f"[REJECT] module={record.module} | symbol={record.symbol} | "
            f"strategy={record.strategy} | reason=\"{record.reason}\" | "
            f"threshold=\"{record.threshold}\" | actual=\"{record.actual_value}\" | "
            f"action={record.action}"
        )

    def get_today_records(self) -> List[RejectionRecord]:
        """Return all records from today."""
        today = date.today()
        return [r for r in self._records if r.timestamp.date() == today]

    def get_daily_summary(self, target_date: Optional[date] = None) -> str:
        """
        Generate end-of-day rejection summary.

        Format:
        [DAILY-REJECT-SUMMARY] date=2026-04-17 | total_rejections=14
          By module: RULE_VALIDATOR=4 | CHECKLIST=3 | CROSS_EXAMINER=3
          By symbol: NIFTY=3 | ADANIPORTS=2 | AXISBANK=2
          By reason: R:R_fail=4 | max_loss_breach=3 | WAIT_hard=2
        """
        d = target_date or date.today()
        records = [r for r in self._records if r.timestamp.date() == d]

        if not records:
            return f"[DAILY-REJECT-SUMMARY] date={d} | total_rejections=0 — clean day!"

        # Aggregate by module
        by_module: Dict[str, int] = defaultdict(int)
        by_symbol: Dict[str, int] = defaultdict(int)
        by_reason: Dict[str, int] = defaultdict(int)

        for r in records:
            by_module[r.module] += 1
            by_symbol[r.symbol] += 1
            # Normalize reason for grouping
            reason_key = self._normalize_reason(r.reason)
            by_reason[reason_key] += 1

        # Format
        module_str = " | ".join(f"{k}={v}" for k, v in sorted(by_module.items(), key=lambda x: -x[1]))
        symbol_str = " | ".join(f"{k}={v}" for k, v in sorted(by_symbol.items(), key=lambda x: -x[1]))
        reason_str = " | ".join(f"{k}={v}" for k, v in sorted(by_reason.items(), key=lambda x: -x[1]))

        return (
            f"[DAILY-REJECT-SUMMARY] date={d} | total_rejections={len(records)}\n"
            f"  By module: {module_str}\n"
            f"  By symbol: {symbol_str}\n"
            f"  By reason: {reason_str}"
        )

    @staticmethod
    def _normalize_reason(reason: str) -> str:
        """Normalize reason strings for aggregation."""
        reason_lower = reason.lower()
        if "r:r" in reason_lower or "risk" in reason_lower and "reward" in reason_lower:
            return "R:R_fail"
        if "max_loss" in reason_lower or "max loss" in reason_lower:
            return "max_loss_breach"
        if "avoid" in reason_lower:
            return "AVOID"
        if "wait" in reason_lower and ("hard" in reason_lower or "block" in reason_lower):
            return "WAIT_hard"
        if "correlation" in reason_lower:
            return "correlation_block"
        if "theta" in reason_lower and "stop" in reason_lower:
            return "theta_time_stop"
        if "delta" in reason_lower:
            return "delta_fail"
        if "spread" in reason_lower or "liquidity" in reason_lower:
            return "liquidity_fail"
        if "rsi" in reason_lower:
            return "RSI_fail"
        if "volume" in reason_lower:
            return "volume_fail"
        # Fallback: truncate to first 30 chars
        return reason[:30].replace(" ", "_")

    def clear_today(self) -> None:
        """Clear today's records (for testing)."""
        today = date.today()
        self._records = [r for r in self._records if r.timestamp.date() != today]

    def total_today(self) -> int:
        """Count of rejections today."""
        return len(self.get_today_records())


# ── Module-level singleton ────────────────────────────────────────────────────
_global_rejection_logger: Optional[RejectionLogger] = None


def get_rejection_logger() -> RejectionLogger:
    """Get the global rejection logger instance."""
    global _global_rejection_logger
    if _global_rejection_logger is None:
        _global_rejection_logger = RejectionLogger()
    return _global_rejection_logger


def log_rejection(
    module: str,
    symbol: str,
    strategy: str,
    reason: str,
    threshold: str,
    actual: str,
    action: str,
    conviction: int = 0,
    extra: Optional[dict] = None,
) -> None:
    """Convenience function to log a rejection via the global logger."""
    get_rejection_logger().log(
        module=module,
        symbol=symbol,
        strategy=strategy,
        reason=reason,
        threshold=threshold,
        actual=actual,
        action=action,
        conviction=conviction,
        extra=extra,
    )

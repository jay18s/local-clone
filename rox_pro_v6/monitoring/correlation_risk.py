"""
FIX 8 — Correlation Risk Check
===============================
Checks for correlated index exposure when taking straddle/spread positions
on multiple indices. Prevents doubling risk on the same macro move.

Rules:
  - correlation > 0.90: BLOCK second position
  - correlation > 0.75: REDUCE_CAP (combined max_loss to 6%, not 9%)
  - otherwise: ALLOW
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("rox.correlation_risk")

# Hardcoded correlation matrix (updated quarterly)
# Symmetric — (A, B) and (B, A) are the same pair
INDEX_CORRELATIONS: Dict[Tuple[str, str], float] = {
    ("NIFTY", "BANKNIFTY"):    0.92,
    ("NIFTY", "SENSEX"):       0.95,
    ("NIFTY", "FINNIFTY"):     0.90,
    ("NIFTY", "BANKEX"):       0.88,
    ("BANKNIFTY", "SENSEX"):   0.85,
    ("BANKNIFTY", "BANKEX"):   0.93,
    ("SENSEX", "BANKEX"):      0.87,
    ("FINNIFTY", "BANKNIFTY"): 0.88,
}


def _get_correlation(idx_a: str, idx_b: str) -> Optional[float]:
    """Look up correlation between two indices (order-independent)."""
    if idx_a == idx_b:
        return 1.0
    pair = (idx_a, idx_b)
    reverse_pair = (idx_b, idx_a)
    return INDEX_CORRELATIONS.get(pair) or INDEX_CORRELATIONS.get(reverse_pair)


@dataclass
class CorrelationRiskResult:
    is_correlated: bool
    pair: Tuple[str, str]
    correlation: float
    action: str  # "ALLOW" | "REDUCE_CAP" | "BLOCK"
    combined_risk_cap: float  # adjusted cap if REDUCE_CAP
    message: str


class CorrelationRiskChecker:
    """
    Checks new F&O positions against existing positions for correlated exposure.
    """

    # Thresholds
    BLOCK_THRESHOLD = 0.90    # correlation > 0.90 → block
    REDUCE_THRESHOLD = 0.75   # correlation > 0.75 → reduce cap
    DEFAULT_RISK_CAP = 9.0    # default combined max loss (%)
    REDUCED_RISK_CAP = 6.0    # reduced cap for correlated pairs (%)

    def check(
        self,
        new_index: str,
        existing_indices: List[str],
        existing_risk_pcts: Optional[List[float]] = None,
    ) -> CorrelationRiskResult:
        """
        Check a new index position against existing positions.

        Args:
            new_index: Index name for the proposed position (e.g., "NIFTY")
            existing_indices: List of index names already in portfolio
            existing_risk_pcts: Optional list of risk % for each existing position

        Returns:
            CorrelationRiskResult with action and adjusted risk cap
        """
        # No existing positions — always allow
        if not existing_indices:
            return CorrelationRiskResult(
                is_correlated=False,
                pair=(new_index, ""),
                correlation=0.0,
                action="ALLOW",
                combined_risk_cap=self.DEFAULT_RISK_CAP,
                message="No existing positions",
            )

        # Find highest correlation with any existing position
        worst_pair = ("", "")
        worst_corr = 0.0
        for existing_idx in existing_indices:
            corr = _get_correlation(new_index, existing_idx)
            if corr is not None and corr > worst_corr:
                worst_corr = corr
                worst_pair = (new_index, existing_idx)

        # Decision
        if worst_corr >= self.BLOCK_THRESHOLD:
            action = "BLOCK"
            cap = self.DEFAULT_RISK_CAP
            msg = (
                f"{worst_pair[0]}/{worst_pair[1]} correlation={worst_corr:.2f} "
                f">= {self.BLOCK_THRESHOLD} — BLOCKED to prevent doubling exposure"
            )
            logger.warning(f"[CORRELATION] {msg}")

        elif worst_corr >= self.REDUCE_THRESHOLD:
            action = "REDUCE_CAP"
            cap = self.REDUCED_RISK_CAP
            msg = (
                f"{worst_pair[0]}/{worst_pair[1]} correlation={worst_corr:.2f} "
                f">= {self.REDUCE_THRESHOLD} — combined max_loss cap reduced to {cap}%"
            )
            logger.info(f"[CORRELATION] {msg}")

        else:
            action = "ALLOW"
            cap = self.DEFAULT_RISK_CAP
            if worst_corr > 0:
                msg = f"{worst_pair[0]}/{worst_pair[1]} correlation={worst_corr:.2f} — ALLOWED"
            else:
                msg = f"No correlation data for {new_index} — ALLOWED"

        return CorrelationRiskResult(
            is_correlated=(worst_corr >= self.REDUCE_THRESHOLD),
            pair=worst_pair if worst_corr > 0 else (new_index, ""),
            correlation=worst_corr,
            action=action,
            combined_risk_cap=cap,
            message=msg,
        )

    def check_batch(
        self,
        proposed_indices: List[str],
    ) -> Dict[str, CorrelationRiskResult]:
        """
        Check a batch of proposed indices, building up the portfolio incrementally.

        Returns:
            Dict mapping each index name to its CorrelationRiskResult
        """
        results: Dict[str, CorrelationRiskResult] = {}
        accepted: List[str] = []

        for idx in proposed_indices:
            result = self.check(idx, accepted)
            results[idx] = result
            if result.action != "BLOCK":
                accepted.append(idx)

        return results

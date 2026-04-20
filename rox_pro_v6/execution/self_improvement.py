"""
ROX Self-Improvement Engine — Adaptive Parameter Calibrator
============================================================
Reads virtual trade history and automatically tunes engine parameters:
  - Conviction thresholds per strategy
  - Regime-specific position sizing
  - SL/Target ratios per IV regime
  - Agent weight adjustments
  - Max hold days per DTE bucket

Runs after every N closed trades (configurable).
Writes calibration state to: data/virtual_trades/calibration_state.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("rox.self_improvement")


class SelfImprovementEngine:
    """
    Analyses virtual trade outcomes and updates engine calibration.

    Triggered:
      - After every EVAL_EVERY closed trades
      - Manually via evaluate_now()

    Produces calibration_state.json with adjusted parameters that
    LeadCoordinator reads at the start of each cycle.
    """

    CALIBRATION_FILE = "data/virtual_trades/calibration_state.json"
    EVAL_EVERY = 5  # Re-calibrate after every N closed trades

    # Default baseline parameters
    DEFAULT_PARAMS = {
        "min_conviction_by_strategy": {
            "BUY_CE": 65,
            "BUY_PE": 65,
            "LONG_STRADDLE": 60,
            "LONG_STRANGLE": 62,
            "BULL_SPREAD": 65,
            "BEAR_SPREAD": 65,
            "IRON_CONDOR": 58,
        },
        "position_size_by_regime": {
            "BULL": 1.0,
            "MILD_BULL": 0.9,
            "CONSOLIDATION": 0.7,
            "MILD_BEAR": 0.6,
            "BEAR": 0.5,
            "CORRECTION": 0.4,
            "CAUTIOUS": 0.6,
            "RANGE_BOUND": 0.7,
        },
        "max_hold_days_by_dte": {
            "0-3": 1,
            "4-7": 3,
            "8-14": 5,
            "15+": 7,
        },
        "sl_pct_by_iv_regime": {
            "LOW": 0.40,     # 40% of premium as SL
            "MODERATE": 0.45,
            "HIGH": 0.50,
            "EXTREME": 0.55,
        },
        "target_pct_by_iv_regime": {
            "LOW": 0.80,
            "MODERATE": 0.75,
            "HIGH": 0.70,
            "EXTREME": 0.65,
        },
        "agent_adjustments": {},    # {agent_name: multiplier}
        "strategy_enabled": {       # Can disable underperforming strategies
            "BUY_CE": True,
            "BUY_PE": True,
            "LONG_STRADDLE": True,
            "LONG_STRANGLE": True,
            "BULL_SPREAD": True,
            "BEAR_SPREAD": True,
            "IRON_CONDOR": True,
        },
        "version": 1,
        "last_calibrated": None,
        "trades_at_last_calibration": 0,
        "calibration_notes": [],
    }

    def __init__(self, data_dir: str = "data"):
        self._state_path = Path(self.CALIBRATION_FILE)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()
        logger.info("[SELF-IMPROVE] SelfImprovementEngine ready")

    # ------------------------------------------------------------------ #
    #  Main Entry Point                                                    #
    # ------------------------------------------------------------------ #

    def maybe_calibrate(self, closed_trade_count: int, perf_file: str) -> Optional[Dict]:
        """
        Run calibration if enough new trades have accumulated.
        Returns updated params dict or None if not triggered.
        """
        last_count = self._state.get("trades_at_last_calibration", 0)
        if closed_trade_count - last_count < self.EVAL_EVERY:
            return None

        logger.info(
            f"[SELF-IMPROVE] Triggering calibration | "
            f"closed_trades={closed_trade_count} | "
            f"since_last={closed_trade_count - last_count}"
        )
        return self.evaluate_now(perf_file, closed_trade_count)

    def evaluate_now(self, perf_file: str, closed_count: int = 0) -> Dict:
        """
        Run full calibration cycle. Returns updated params.
        """
        try:
            with open(perf_file) as f:
                perf = json.load(f)
        except Exception as e:
            logger.warning(f"[SELF-IMPROVE] Cannot read perf file: {e}")
            return self._state

        notes = []
        params = dict(self._state)

        # --- 1. Adjust conviction thresholds per strategy ---
        by_strategy = perf.get("by_strategy", {})
        for strategy, stats in by_strategy.items():
            if stats["trades"] < 3:
                continue
            wr = stats["wins"] / stats["trades"] * 100
            current_min = params["min_conviction_by_strategy"].get(strategy, 65)

            if wr < 35:
                # Significantly underperforming — raise threshold
                new_min = min(current_min + 5, 85)
                params["min_conviction_by_strategy"][strategy] = new_min
                notes.append(
                    f"[{strategy}] Win rate {wr:.0f}% → raised min conviction "
                    f"{current_min}→{new_min}"
                )
                if wr < 25:
                    params["strategy_enabled"][strategy] = False
                    notes.append(f"[{strategy}] DISABLED — win rate below 25%")

            elif wr > 65 and current_min > 55:
                # Outperforming — can lower threshold slightly
                new_min = max(current_min - 3, 55)
                params["min_conviction_by_strategy"][strategy] = new_min
                notes.append(
                    f"[{strategy}] Win rate {wr:.0f}% → lowered min conviction "
                    f"{current_min}→{new_min}"
                )

            elif wr >= 35 and not params["strategy_enabled"].get(strategy, True):
                # Re-enable if recovering
                params["strategy_enabled"][strategy] = True
                notes.append(f"[{strategy}] RE-ENABLED — win rate recovered to {wr:.0f}%")

        # --- 2. Adjust position sizing by regime ---
        by_regime = perf.get("by_regime", {})
        for regime, stats in by_regime.items():
            if stats["trades"] < 3:
                continue
            wr = stats["wins"] / stats["trades"] * 100
            current_size = params["position_size_by_regime"].get(regime, 0.7)

            if wr < 35:
                new_size = round(max(current_size * 0.75, 0.25), 2)
                params["position_size_by_regime"][regime] = new_size
                notes.append(
                    f"[REGIME:{regime}] Win rate {wr:.0f}% → size {current_size}→{new_size}"
                )
            elif wr > 65:
                new_size = round(min(current_size * 1.15, 1.5), 2)
                params["position_size_by_regime"][regime] = new_size
                notes.append(
                    f"[REGIME:{regime}] Win rate {wr:.0f}% → size {current_size}→{new_size}"
                )

        # --- 3. SL/Target calibration from exit analysis ---
        improvement_signals = perf.get("improvement_signals", [])
        for sig in improvement_signals:
            sig_type = sig.get("type")

            if sig_type == "SL_HIT_RATE_HIGH":
                # SL too tight — widen
                for iv in params["sl_pct_by_iv_regime"]:
                    old = params["sl_pct_by_iv_regime"][iv]
                    params["sl_pct_by_iv_regime"][iv] = round(min(old + 0.05, 0.65), 2)
                notes.append("SL widened across all IV regimes (+5%)")

            elif sig_type == "THETA_DECAY_PROBLEM":
                # Reduce max hold days
                for dte_bucket in params["max_hold_days_by_dte"]:
                    old = params["max_hold_days_by_dte"][dte_bucket]
                    params["max_hold_days_by_dte"][dte_bucket] = max(old - 1, 1)
                notes.append("Max hold days reduced by 1 day (theta decay issue)")

        # --- 4. Meta: win rate trend ---
        overall_wr = perf.get("win_rate_pct", 0)
        if overall_wr < 40:
            notes.append(
                f"⚠ Overall win rate {overall_wr:.0f}% below 40% — engine in recalibration mode. "
                f"Conviction thresholds raised, position sizes reduced."
            )
        elif overall_wr > 60:
            notes.append(
                f"✅ Overall win rate {overall_wr:.0f}% — engine performing well."
            )

        # Update metadata
        params["version"] = self._state.get("version", 1) + 1
        params["last_calibrated"] = datetime.now().isoformat()
        params["trades_at_last_calibration"] = closed_count
        params["calibration_notes"] = notes[-20:]  # Keep last 20 notes

        self._state = params
        self._save_state()

        logger.info(
            f"[SELF-IMPROVE] Calibration complete | "
            f"v{params['version']} | {len(notes)} adjustments | "
            f"overall_wr={overall_wr:.0f}%"
        )
        for note in notes:
            logger.info(f"  [SELF-IMPROVE] {note}")

        return params

    # ------------------------------------------------------------------ #
    #  Parameter Access                                                    #
    # ------------------------------------------------------------------ #

    def get_min_conviction(self, strategy: str) -> int:
        return self._state.get("min_conviction_by_strategy", {}).get(
            strategy, self.DEFAULT_PARAMS["min_conviction_by_strategy"].get(strategy, 65)
        )

    def get_position_size_multiplier(self, regime: str) -> float:
        return self._state.get("position_size_by_regime", {}).get(
            regime, self.DEFAULT_PARAMS["position_size_by_regime"].get(regime, 0.7)
        )

    def get_sl_pct(self, iv_regime: str = "MODERATE") -> float:
        return self._state.get("sl_pct_by_iv_regime", {}).get(
            iv_regime, 0.40
        )

    def get_target_pct(self, iv_regime: str = "MODERATE") -> float:
        return self._state.get("target_pct_by_iv_regime", {}).get(
            iv_regime, 0.80
        )

    def get_max_hold_days(self, dte: int) -> int:
        buckets = self._state.get("max_hold_days_by_dte", self.DEFAULT_PARAMS["max_hold_days_by_dte"])
        if dte <= 3: return buckets.get("0-3", 1)
        if dte <= 7: return buckets.get("4-7", 3)
        if dte <= 14: return buckets.get("8-14", 5)
        return buckets.get("15+", 7)

    def is_strategy_enabled(self, strategy: str) -> bool:
        return self._state.get("strategy_enabled", {}).get(strategy, True)

    def get_calibration_summary(self) -> str:
        """Return human-readable calibration summary for logs."""
        notes = self._state.get("calibration_notes", [])
        version = self._state.get("version", 1)
        last = self._state.get("last_calibrated", "never")
        lines = [
            f"[SELF-IMPROVE] Calibration v{version} | last={last}",
            f"  Conviction thresholds: {self._state.get('min_conviction_by_strategy', {})}",
            f"  Recent notes: {notes[-5:] if notes else 'none'}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _load_state(self) -> Dict:
        if not self._state_path.exists():
            return dict(self.DEFAULT_PARAMS)
        try:
            with open(self._state_path) as f:
                loaded = json.load(f)
            # Merge with defaults (add any new keys from defaults)
            merged = dict(self.DEFAULT_PARAMS)
            merged.update(loaded)
            return merged
        except Exception as e:
            logger.warning(f"[SELF-IMPROVE] Could not load calibration state: {e}")
            return dict(self.DEFAULT_PARAMS)

    def _save_state(self):
        try:
            with open(self._state_path, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[SELF-IMPROVE] Failed to save calibration state: {e}")


# Singleton
_sie_instance: Optional[SelfImprovementEngine] = None

def get_self_improvement_engine() -> SelfImprovementEngine:
    global _sie_instance
    if _sie_instance is None:
        _sie_instance = SelfImprovementEngine()
    return _sie_instance

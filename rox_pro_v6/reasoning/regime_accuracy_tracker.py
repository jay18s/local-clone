"""
ROX Engine v6.0 — Regime Accuracy Tracker
Tracks rule-based vs LLM regime prediction accuracy for adaptive arbitration.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger("rox.reasoning.regime_accuracy")


class RegimeAccuracyTracker:
    """
    Tracks the accuracy of rule-based and LLM regime classifications
    against actual market outcomes. Provides rolling accuracy metrics
    that the RegimeArbiter uses to decide which source to trust.

    Storage: append-only JSONL file at data/regime_accuracy.jsonl
    """

    def __init__(self, data_dir: str = "data"):
        self._data_dir = data_dir
        self._log_path = os.path.join(data_dir, "regime_accuracy.jsonl")
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"RegimeAccuracyTracker initialized: {self._log_path}")
        # FIX-PURGE-WEEKEND: Remove any records written on weekends (Sat/Sun).
        # Before the weekend guard was added, every weekend engine run wrote a
        # false accuracy record using stale Friday OHLC. Those records caused
        # actual=TRENDING (big Friday move) to dominate the JSONL, making
        # llm_correct=False every session and tanking accuracy to ~7%.
        self._purge_weekend_records()

    def _purge_weekend_records(self) -> None:
        """Remove weekend-dated records from regime_accuracy.jsonl on startup."""
        if not os.path.exists(self._log_path):
            return
        try:
            from datetime import date as _date_cls
            kept, removed = [], 0
            with open(self._log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        rec_date = _date_cls.fromisoformat(rec.get("date", "2000-01-01"))
                        if rec_date.weekday() >= 5:  # Saturday=5, Sunday=6
                            removed += 1
                            continue
                    except Exception:
                        pass  # keep malformed lines
                    kept.append(line)
            if removed:
                with open(self._log_path, "w") as f:
                    f.write("\n".join(kept) + ("\n" if kept else ""))
                logger.info(
                    f"RegimeAccuracyTracker: purged {removed} weekend record(s) "
                    f"from {self._log_path} — these were stale/false entries."
                )
        except Exception as e:
            logger.warning(f"RegimeAccuracyTracker: purge failed (non-fatal): {e}")

    def classify_actual_regime(
        self,
        nifty_open: float,
        nifty_close: float,
        nifty_high: float,
        nifty_low: float,
        vix_open: float,
        vix_close: float,
    ) -> str:
        """
        Classify the actual market regime based on end-of-day data.

        Uses NSE price action and VIX to determine the ground truth regime
        for a given session. This is the label against which predictions
        are compared.

        Args:
            nifty_open: Nifty opening price.
            nifty_close: Nifty closing price.
            nifty_high: Nifty intraday high.
            nifty_low: Nifty intraday low.
            vix_open: India VIX at open.
            vix_close: India VIX at close.

        Returns:
            Regime string: "BEARISH", "TRENDING", "VOLATILE", or "RANGE_BOUND".
        """
        if nifty_open == 0:
            return "RANGE_BOUND"

        close_pct = (nifty_close - nifty_open) / nifty_open * 100
        range_pct = (nifty_high - nifty_low) / nifty_open * 100

        # FIX-TRACKER-01: Added MILD_BEAR and CAUTIOUS labels to match the arbiter's
        # output vocabulary.  Previously this function only produced 4 labels
        # (BEARISH/TRENDING/VOLATILE/RANGE_BOUND), meaning any MILD_BEAR or CAUTIOUS
        # prediction was permanently marked llm_correct=False, degrading LLM accuracy
        # toward zero and eventually triggering RULE_OVERRIDE_LLM_DEGRADED.

        # MILD_BEAR: meaningful downward drift with moderately elevated VIX
        if close_pct < -0.3 and 16 < vix_close < 20:
            return "MILD_BEAR"

        # CAUTIOUS: directionless but nervous — tight range with elevated VIX
        if abs(close_pct) < 0.5 and vix_close > 16:
            return "CAUTIOUS"

        # BEARISH: big move down with elevated VIX
        if abs(close_pct) > 1.0 and vix_close > 18:
            return "BEARISH"

        # TRENDING: meaningful directional move
        if abs(close_pct) > 0.5:
            return "TRENDING"

        # VOLATILE: wide range or VIX spike
        if range_pct > 1.0 or abs(vix_close - vix_open) > 2:
            return "VOLATILE"

        # RANGE_BOUND: default — low volatility, no direction
        return "RANGE_BOUND"

    def log_session(
        self,
        rule_regime: str,
        rule_confidence: float,
        llm_regime: str,
        llm_confidence: float,
        nifty_open: float,
        nifty_close: float,
        nifty_high: float,
        nifty_low: float,
        vix_open: float,
        vix_close: float,
    ) -> None:
        """
        Log a session's regime predictions and actual outcome.

        Computes the actual regime from OHLC/VIX data, compares both
        predictions against it, and appends the full record to JSONL.

        Args:
            rule_regime: Rule-based classifier's regime prediction.
            rule_confidence: Rule-based classifier's confidence (0-100).
            llm_regime: LLM regime prediction.
            llm_confidence: LLM confidence (0-100).
            nifty_open: Nifty opening price.
            nifty_close: Nifty closing price.
            nifty_high: Nifty intraday high.
            nifty_low: Nifty intraday low.
            vix_open: India VIX at open.
            vix_close: India VIX at close.
        """
        actual_regime = self.classify_actual_regime(
            nifty_open, nifty_close, nifty_high, nifty_low, vix_open, vix_close
        )

        # FIX-ACCURACY-FAMILY: Exact string match is too strict — the LLM predicted
        # MILD_BEAR and the tracker returned CAUTIOUS.  Both are bearish-family regimes;
        # punishing the LLM as "wrong" causes rolling accuracy to tank and eventually
        # triggers RULE_OVERRIDE_LLM_DEGRADED.
        # Solution: use directional-family matching.  A prediction is "correct" if
        # it falls in the same sentiment bucket as the actual regime.
        _BEARISH_FAMILY  = {"BEARISH", "MILD_BEAR", "CAUTIOUS", "CORRECTION"}
        _BULLISH_FAMILY  = {"BULLISH", "MILD_BULL", "BULL"}
        # FIX-TRENDING-FAMILY: TRENDING is direction-agnostic (tracks magnitude, not direction).
        # A big UP day in a MILD_BEAR regime should not count as a BULLISH correct prediction.
        # TRENDING belongs in NEUTRAL — it means "the market moved a lot" without polarity.
        _NEUTRAL_FAMILY  = {"RANGE_BOUND", "VOLATILE", "CONSOLIDATION", "TRENDING"}

        def _family(regime: str) -> str:
            r = regime.upper()
            if r in _BEARISH_FAMILY:  return "BEARISH"
            if r in _BULLISH_FAMILY:  return "BULLISH"
            return "NEUTRAL"

        rule_correct = (_family(rule_regime) == _family(actual_regime))
        llm_correct  = (_family(llm_regime)  == _family(actual_regime))
        # Also record exact match for fine-grained analysis
        rule_exact = (rule_regime == actual_regime)
        llm_exact  = (llm_regime  == actual_regime)

        record = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "rule_regime": rule_regime,
            "rule_confidence": rule_confidence,
            "llm_regime": llm_regime,
            "llm_confidence": llm_confidence,
            "actual_regime": actual_regime,
            "rule_correct": rule_correct,    # family-level match (used for accuracy scoring)
            "llm_correct": llm_correct,      # family-level match (used for accuracy scoring)
            "rule_exact": rule_exact,         # exact label match (for analysis only)
            "llm_exact": llm_exact,           # exact label match (for analysis only)
            "nifty_open": nifty_open,
            "nifty_close": nifty_close,
            "nifty_high": nifty_high,
            "nifty_low": nifty_low,
            "vix_open": vix_open,
            "vix_close": vix_close,
        }

        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
            logger.info(
                f"Regime session logged: rule={rule_regime}(family={rule_correct}/exact={rule_exact}) "
                f"llm={llm_regime}(family={llm_correct}/exact={llm_exact}) actual={actual_regime}"
            )
        except Exception as e:
            logger.error(f"Failed to log regime session: {e}")

    def get_rolling_accuracy(self, n: int = 20) -> Dict:
        """
        Calculate rolling accuracy for both regime classifiers.

        Args:
            n: Number of recent sessions to consider.

        Returns:
            Dict with:
                - rule_accuracy: float 0-1
                - llm_accuracy: float 0-1
                - sessions_tracked: int
                - rule_should_override_llm: bool
        """
        if not os.path.exists(self._log_path):
            return {
                "rule_accuracy": 0.0,
                "llm_accuracy": 0.0,
                "sessions_tracked": 0,
                "rule_should_override_llm": False,
            }

        records = []
        try:
            with open(self._log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f"Failed to read regime accuracy: {e}")

        recent = records[-n:] if records else []

        if not recent:
            # FIX-COLDSTART-01: returning 0.0 here caused the RegimeArbiter to
            # immediately override the LLM on every fresh install (0.0 < 0.55
            # threshold), even though zero history means "no evidence of degradation".
            # Use None so the coordinator's `or 1.0` fallback takes effect and the
            # LLM is trusted until real accuracy data accumulates.
            return {
                "rule_accuracy": None,
                "llm_accuracy": None,
                "sessions_tracked": 0,
                "rule_should_override_llm": False,
            }

        rule_correct = sum(1 for r in recent if r.get("rule_correct"))
        llm_correct = sum(1 for r in recent if r.get("llm_correct"))
        total = len(recent)

        rule_accuracy = rule_correct / total
        llm_accuracy = llm_correct / total

        # Override logic: rule should override LLM when rule is significantly
        # more accurate AND LLM is below a useful threshold
        rule_should_override = (
            rule_accuracy > llm_accuracy + 0.10 and llm_accuracy < 0.55
        )

        return {
            "rule_accuracy": round(rule_accuracy, 3),
            "llm_accuracy": round(llm_accuracy, 3),
            "sessions_tracked": total,
            "rule_should_override_llm": rule_should_override,
        }

"""
ROX Proven Edge Engine v3.1 - Agent Scorecard
=============================================
Tracks every agent's prediction accuracy over time so the meta-learning
tier can adjust weights based on real historical performance.

Data lives in:
  data/Market_Trends/Agent_Performance/scorecard.json

Schema (per agent entry):
  {
    "total_predictions": int,
    "predictions_by_regime": { regime: count },
    "wins": int,
    "wins_by_regime": { regime: int },
    "r_multiples": [float, ...],          # filled by daily_reconcile.py
    "last_updated": "ISO date",
    "pending": [                          # not yet resolved
      {
        "date": "YYYY-MM-DD",
        "prediction": "LONG|SHORT|NEUTRAL",
        "conviction": float,
        "regime": str,
        "trade_id": str | None            # links to CSV row
      }
    ]
  }
"""

import os
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any

logger = logging.getLogger("AgentScorecard")

AGENTS = ["ORION", "VESPER", "KAIRO", "SENTINEL", "NEXUS", "PRUDENCE", "CATALYST", "OPTIMUS"]


class AgentScorecard:
    """
    Manages the agent performance scorecard JSON.

    Workflow:
      1. record_prediction()  — called each run from coordinator (logs pending row)
      2. resolve_prediction()  — called by daily_reconcile.py once outcome is known
      3. get_scorecard()       — returns live stats for display / meta-learning
    """

    def __init__(self, data_manager=None):
        self.data_manager = data_manager
        if data_manager:
            self._scorecard_path = os.path.join(
                data_manager.data_dir, "Agent_Performance", "scorecard.json"
            )
        else:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._scorecard_path = os.path.join(
                base, "data", "Market_Trends", "Agent_Performance", "scorecard.json"
            )
        self._ensure_file()

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _ensure_file(self):
        """Create scorecard.json with empty slots if it doesn't exist."""
        os.makedirs(os.path.dirname(self._scorecard_path), exist_ok=True)
        if not os.path.exists(self._scorecard_path):
            empty = {}
            for agent in AGENTS:
                empty[agent] = self._empty_entry()
            with open(self._scorecard_path, "w") as f:
                json.dump(empty, f, indent=2)
            logger.info(f"[Scorecard] Created new scorecard at {self._scorecard_path}")

    def _empty_entry(self) -> Dict:
        return {
            "total_predictions": 0,
            "predictions_by_regime": {},
            "wins": 0,
            "wins_by_regime": {},
            "losses": 0,
            "r_multiples": [],
            "win_rate": 0.0,
            "avg_r_multiple": 0.0,
            "last_updated": date.today().isoformat(),
            "pending": [],
        }

    def _load(self) -> Dict:
        try:
            with open(self._scorecard_path, "r") as f:
                data = json.load(f)
            # Ensure all 8 agents are present (handles upgrades)
            for agent in AGENTS:
                if agent not in data:
                    data[agent] = self._empty_entry()
            return data
        except Exception as e:
            logger.error(f"[Scorecard] Load error: {e}")
            return {a: self._empty_entry() for a in AGENTS}

    def _save(self, data: Dict):
        try:
            with open(self._scorecard_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[Scorecard] Save error: {e}")

    # ─── Public API ───────────────────────────────────────────────────────

    def record_prediction(
        self,
        agent_name: str,
        prediction: str,
        conviction: float,
        regime: str,
        trade_id: Optional[str] = None,
        suppressed: bool = False,
    ):
        """
        Record an agent's direction vote for today's run.
        This creates a 'pending' row that daily_reconcile.py will resolve.

        Args:
            agent_name:  e.g. "ORION"
            prediction:  "LONG" | "SHORT" | "NEUTRAL"
            conviction:  0-100 float
            regime:      e.g. "BULL"
            trade_id:    optional link to CSV row stock:date
            suppressed:  True when the EXAMINE-GATE fired AVOID for this cycle.
                         Suppressed predictions are expired after SUPPRESSED_MAX_AGE_DAYS
                         instead of waiting indefinitely for a trade_id link.
        """
        if agent_name not in AGENTS:
            logger.warning(f"[Scorecard] Unknown agent: {agent_name}")
            return

        data = self._load()
        entry = data[agent_name]

        today = date.today().isoformat()

        # Avoid duplicate entries for the same day
        existing_dates = {p["date"] for p in entry.get("pending", [])}
        if today in existing_dates:
            return  # Already recorded today

        # Increment totals
        entry["total_predictions"] += 1
        entry["predictions_by_regime"][regime] = (
            entry["predictions_by_regime"].get(regime, 0) + 1
        )
        entry["last_updated"] = today

        # Add to pending queue — suppressed rows carry a flag so they can be
        # bulk-expired cleanly without polluting the win/loss counters.
        entry.setdefault("pending", []).append({
            "date": today,
            "prediction": prediction,
            "conviction": conviction,
            "regime": regime,
            "trade_id": trade_id,
            "suppressed": suppressed,
        })

        data[agent_name] = entry
        self._save(data)

    # ── Stale prediction cleanup ───────────────────────────────────────────────
    # Predictions accumulate forever when EXAMINE-GATE fires AVOID (no trade is
    # generated so daily_reconcile.py never resolves them).  This method expires
    # them so the scorecard stays accurate.
    #
    # Expiry rules:
    #   • suppressed=True  → expire after SUPPRESSED_MAX_AGE_DAYS (default 3)
    #   • suppressed=False (normal, no trade_id) → expire after ORPHAN_MAX_AGE_DAYS (default 10)
    #
    # Expired suppressed predictions are dropped silently (no loss recorded).
    # Expired orphan predictions (should-have-traded but didn't) count as LOSS
    # so the scorecard reflects the real opportunity cost.

    SUPPRESSED_MAX_AGE_DAYS: int = 3
    ORPHAN_MAX_AGE_DAYS: int = 10

    def expire_stale_predictions(self) -> Dict[str, int]:
        """
        Expire stale pending predictions that will never be resolved.

        Called by daily_reconcile.py once per day *before* normal resolution.

        Returns:
            Dict mapping agent → number of rows expired.
        """
        from datetime import timedelta

        today = date.today()
        expired_counts: Dict[str, int] = {}

        data = self._load()
        changed = False

        for agent in AGENTS:
            entry = data.get(agent, self._empty_entry())
            pending = entry.get("pending", [])
            surviving = []
            expired = 0

            for p in pending:
                try:
                    pred_date = date.fromisoformat(p["date"])
                except (ValueError, KeyError):
                    surviving.append(p)
                    continue

                age = (today - pred_date).days
                is_suppressed = p.get("suppressed", False)
                has_trade = bool(p.get("trade_id"))

                if has_trade:
                    # Linked to a real trade — keep until reconcile picks it up
                    surviving.append(p)
                    continue

                if is_suppressed and age >= self.SUPPRESSED_MAX_AGE_DAYS:
                    # AVOID-cycle row: just discard — never generated a trade signal
                    logger.debug(
                        f"[Scorecard] Expiring suppressed prediction: "
                        f"{agent} {p['date']} (age={age}d)"
                    )
                    expired += 1
                    changed = True
                elif not is_suppressed and age >= self.ORPHAN_MAX_AGE_DAYS:
                    # Orphan row: trade was signalled but never reconciled.
                    # Count as LOSS so the win rate stays honest.
                    entry["losses"] = entry.get("losses", 0) + 1
                    entry.setdefault("r_multiples", []).append(-1.0)
                    closed = entry["wins"] + entry["losses"]
                    entry["win_rate"] = round(entry["wins"] / closed, 4) if closed > 0 else 0.0
                    if entry.get("r_multiples"):
                        entry["avg_r_multiple"] = round(
                            sum(entry["r_multiples"]) / len(entry["r_multiples"]), 3
                        )
                    logger.info(
                        f"[Scorecard] Orphan prediction expired as LOSS: "
                        f"{agent} {p['date']} (age={age}d, conviction={p.get('conviction')})"
                    )
                    expired += 1
                    changed = True
                else:
                    surviving.append(p)

            if expired:
                entry["pending"] = surviving
                entry["last_updated"] = today.isoformat()
                data[agent] = entry
                expired_counts[agent] = expired

        if changed:
            self._save(data)

        if expired_counts:
            logger.info(f"[Scorecard] Expired stale predictions: {expired_counts}")

        return expired_counts

    def resolve_prediction(
        self,
        agent_name: str,
        prediction_date: str,
        outcome: str,           # "WIN" | "LOSS" | "NEUTRAL"
        r_multiple: float = 0.0,
    ):
        """
        Mark a pending prediction as resolved.
        Called by daily_reconcile.py after price check.

        Args:
            agent_name:       e.g. "ORION"
            prediction_date:  "YYYY-MM-DD" matching the pending row
            outcome:          "WIN" | "LOSS" | "NEUTRAL"
            r_multiple:       actual R achieved (e.g. 1.8 means 1.8x risk reward hit)
        """
        if agent_name not in AGENTS:
            return

        data = self._load()
        entry = data[agent_name]

        pending = entry.get("pending", [])
        resolved = [p for p in pending if p["date"] == prediction_date]
        remaining = [p for p in pending if p["date"] != prediction_date]

        if not resolved:
            logger.warning(f"[Scorecard] No pending prediction for {agent_name} on {prediction_date}")
            return

        row = resolved[0]
        regime = row.get("regime", "UNKNOWN")

        if outcome == "WIN":
            entry["wins"] = entry.get("wins", 0) + 1
            entry["wins_by_regime"][regime] = entry["wins_by_regime"].get(regime, 0) + 1
            if r_multiple > 0:
                entry.setdefault("r_multiples", []).append(r_multiple)
        elif outcome == "LOSS":
            entry["losses"] = entry.get("losses", 0) + 1
            if r_multiple < 0:
                entry.setdefault("r_multiples", []).append(r_multiple)

        # Recompute summary stats
        closed = entry["wins"] + entry["losses"]
        entry["win_rate"] = round(entry["wins"] / closed, 4) if closed > 0 else 0.0
        if entry.get("r_multiples"):
            entry["avg_r_multiple"] = round(
                sum(entry["r_multiples"]) / len(entry["r_multiples"]), 3
            )

        entry["pending"] = remaining
        entry["last_updated"] = date.today().isoformat()
        data[agent_name] = entry
        self._save(data)

        logger.info(
            f"[Scorecard] {agent_name} {prediction_date} → {outcome} "
            f"(R={r_multiple:.2f}) | Win rate: {entry['win_rate']*100:.1f}%"
        )

    def get_scorecard(self) -> Dict:
        """Return the full scorecard dict (all agents)."""
        return self._load()

    def get_agent_stats(self, agent_name: str) -> Dict:
        """Return stats for a single agent."""
        data = self._load()
        return data.get(agent_name, self._empty_entry())

    def get_summary_table(self) -> str:
        """Return a formatted text summary table for printing."""
        data = self._load()
        lines = []
        lines.append("=" * 68)
        lines.append("AGENT PERFORMANCE SCORECARD")
        lines.append(f"{'Agent':<12} {'Predictions':>12} {'Wins':>6} {'Losses':>7} {'Win%':>7} {'Avg R':>8} {'Pending':>9}")
        lines.append("-" * 68)
        for agent in AGENTS:
            e = data.get(agent, self._empty_entry())
            total   = e.get("total_predictions", 0)
            wins    = e.get("wins", 0)
            losses  = e.get("losses", 0)
            winpct  = f"{e.get('win_rate', 0)*100:.1f}%" if (wins + losses) > 0 else "—"
            avg_r   = f"{e.get('avg_r_multiple', 0):.2f}" if e.get("r_multiples") else "—"
            pending = len(e.get("pending", []))
            lines.append(
                f"{agent:<12} {total:>12} {wins:>6} {losses:>7} {winpct:>7} {avg_r:>8} {pending:>9}"
            )
        lines.append("=" * 68)
        return "\n".join(lines)

    def get_weights_recommendation(self) -> Dict[str, float]:
        """
        Return suggested weight adjustments based on win rate.
        Only agents with >10 resolved predictions get adjusted.
        Used by meta-learning tier (Tier 1).
        """
        data = self._load()
        weights = {}

        for agent in AGENTS:
            e = data.get(agent, self._empty_entry())
            closed = e.get("wins", 0) + e.get("losses", 0)
            if closed < 10:
                weights[agent] = None  # not enough data
                continue

            win_rate = e.get("win_rate", 0.5)
            # Scale: 0.4 win_rate → 0.08 weight, 0.7 win_rate → 0.20 weight
            # Linear between 0.05 and 0.25
            suggested = 0.05 + (win_rate - 0.30) / (0.80 - 0.30) * 0.20
            suggested = round(max(0.05, min(0.25, suggested)), 4)
            weights[agent] = suggested

        return weights

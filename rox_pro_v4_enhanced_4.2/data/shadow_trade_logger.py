"""
Shadow Trade Logger — Records setups that were suppressed by WAIT/AVOID
so the engine can learn from its own conservatism.

When cross-examiner says WAIT, setups are still computed but not acted on.
This logger records them with entry/SL/TGT so post-market analysis can
compare against actual price movement and calibrate WAIT thresholds.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("ShadowTradeLogger")

SHADOW_DIR = Path("data") / "shadow_trades"
SHADOW_FILE = SHADOW_DIR / "shadow_trades.jsonl"


class ShadowTradeLogger:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else SHADOW_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / "shadow_trades.jsonl"

    def log_suppressed_setup(
        self,
        stock: str,
        direction: str,
        conviction: int,
        entry_price: float,
        stop_loss: float,
        target_1: float,
        target_2: float,
        risk_reward: float,
        regime: str,
        examiner_recommendation: str,  # WAIT or AVOID
        examiner_reasoning: str,
        agent_votes: Dict[str, any],
        timestamp: Optional[datetime] = None,
    ):
        """Log a setup that was suppressed by cross-examiner."""
        ts = timestamp or datetime.now()
        record = {
            "timestamp": ts.isoformat(),
            "date": ts.strftime("%Y-%m-%d"),
            "stock": stock,
            "direction": direction,
            "conviction": conviction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "risk_reward": risk_reward,
            "regime": regime,
            "examiner_recommendation": examiner_recommendation,
            "examiner_reasoning": examiner_reasoning[:200],
            "agent_votes": {
                k: {"direction": v.direction.value if hasattr(v.direction, 'value') else str(v.direction),
                    "conviction": v.conviction}
                for k, v in agent_votes.items()
            } if agent_votes else {},
            "outcome": None,  # Filled post-market
            "actual_move_pct": None,
            "would_have_hit_target": None,
            "would_have_hit_stop": None,
        }

        try:
            with open(self.filepath, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.debug(
                f"[SHADOW] Logged suppressed {direction} {stock} "
                f"(examiner: {examiner_recommendation})"
            )
        except Exception as e:
            logger.warning(f"[SHADOW] Failed to log: {e}")

    def log_suppressed_setups_batch(
        self,
        setups: list,
        regime: str,
        examiner_recommendation: str,
        examiner_reasoning: str,
    ):
        """Log multiple suppressed setups at once."""
        for s in setups:
            if not hasattr(s, 'stock'):
                continue
            self.log_suppressed_setup(
                stock=s.stock,
                direction=s.direction.value if hasattr(s.direction, 'value') else str(s.direction),
                conviction=getattr(s, 'conviction', 0),
                entry_price=getattr(s, 'entry_price', 0),
                stop_loss=getattr(s, 'stop_loss', 0),
                target_1=getattr(s, 'target_1', 0),
                target_2=getattr(s, 'target_2', 0),
                risk_reward=getattr(s, 'risk_reward', 0),
                regime=regime,
                examiner_recommendation=examiner_recommendation,
                examiner_reasoning=examiner_reasoning,
                agent_votes=getattr(s, 'agent_votes', {}),
            )

    def get_pending_outcomes(self, date: Optional[str] = None) -> List[Dict]:
        """Get shadow trades that haven't been resolved yet."""
        target_date = date or datetime.now().strftime("%Y-%m-%d")
        pending = []
        try:
            if not self.filepath.exists():
                return pending
            with open(self.filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("date") == target_date and record.get("outcome") is None:
                        pending.append(record)
        except Exception as e:
            logger.warning(f"[SHADOW] Failed to read pending: {e}")
        return pending


# Singleton
_shadow_logger = None

def get_shadow_logger() -> ShadowTradeLogger:
    global _shadow_logger
    if _shadow_logger is None:
        _shadow_logger = ShadowTradeLogger()
    return _shadow_logger

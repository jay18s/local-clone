"""
ROX Engine v6.0 — Trade Outcome Logger
Appends trade lifecycle data to JSONL for closed-loop learning.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict

logger = logging.getLogger("rox.data.trade_outcome")


class TradeOutcomeLogger:
    """
    Logs trade entries, exits, and metadata to a JSONL file.
    Enables closed-loop learning by capturing the full context
    of each trade for later analysis and calibration updates.
    """

    def __init__(self, log_dir: str = "data"):
        self._log_dir = log_dir
        self._log_path = os.path.join(log_dir, "trade_outcomes.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"TradeOutcomeLogger initialized: {self._log_path}")

    def log_trade(
        self,
        timestamp_entry: str,
        timestamp_exit: Optional[str],
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: Optional[float],
        pnl: Optional[float],
        regime_at_entry: str,
        regime_confidence: float,
        debate_agreement_score: float,
        calibration_score: float,
        agent_verdicts: List[Dict],
        signals_passed: List[str],
        signals_failed: List[Dict],
        news_sentiment: str,
        pattern_match_ids: List[str],
        cycle_number: int,
    ) -> None:
        """
        Append a trade record as a single JSON line.

        Args:
            timestamp_entry: ISO timestamp of trade entry.
            timestamp_exit: ISO timestamp of trade exit (nullable).
            symbol: Trading symbol (e.g. "NSE:SBIN").
            direction: "LONG" or "SHORT".
            entry_price: Entry price of the trade.
            exit_price: Exit price (nullable if still open).
            pnl: Profit and loss (nullable if still open).
            regime_at_entry: Market regime at entry time.
            regime_confidence: Regime confidence (0-100).
            debate_agreement_score: Debate agreement (0-100).
            calibration_score: Calibration score (0-100).
            agent_verdicts: List of dicts with agent, direction, conviction, weighted_vote.
            signals_passed: List of symbol+direction strings that passed validation.
            signals_failed: List of dicts with symbol, direction, reason for failures.
            news_sentiment: News sentiment label at entry.
            pattern_match_ids: List of pattern match IDs from pattern memory.
            cycle_number: Engine cycle number when trade was entered.
        """
        record = {
            "timestamp_entry": timestamp_entry,
            "timestamp_exit": timestamp_exit,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "regime_at_entry": regime_at_entry,
            "regime_confidence": regime_confidence,
            "debate_agreement_score": debate_agreement_score,
            "calibration_score": calibration_score,
            "agent_verdicts": agent_verdicts,
            "signals_passed": signals_passed,
            "signals_failed": signals_failed,
            "news_sentiment": news_sentiment,
            "pattern_match_ids": pattern_match_ids,
            "cycle_number": cycle_number,
        }

        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
            logger.info(
                f"Trade logged: {symbol} {direction} entry={entry_price} "
                f"regime={regime_at_entry} cycle={cycle_number}"
            )
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def update_trade(
        self,
        symbol: str,
        timestamp_entry: Optional[str],
        exit_price: float,
        pnl: float,
    ) -> None:
        """
        Find and update an existing trade with exit data.

        Matches by symbol and timestamp_entry (or the last open trade
        for the symbol if timestamp_entry is None).

        Args:
            symbol: Trading symbol to match.
            timestamp_entry: ISO timestamp of entry (nullable — finds last open).
            exit_price: Exit price of the trade.
            pnl: Profit and loss of the trade.
        """
        try:
            trades = self.get_all_trades()
            updated = False

            for i, trade in enumerate(trades):
                match = (
                    trade.get("symbol") == symbol
                    and trade.get("exit_price") is None
                    and (
                        timestamp_entry is None
                        or trade.get("timestamp_entry") == timestamp_entry
                    )
                )
                if match:
                    trade["exit_price"] = exit_price
                    trade["pnl"] = pnl
                    trade["timestamp_exit"] = datetime.now().isoformat()
                    trades[i] = trade
                    updated = True
                    logger.info(
                        f"Trade updated: {symbol} exit={exit_price} pnl={pnl:.2f}"
                    )
                    break

            if not updated:
                logger.warning(
                    f"No open trade found to update for {symbol}"
                )
                return

            # Rewrite the file with updated trades
            with open(self._log_path, "w") as f:
                for trade in trades:
                    f.write(json.dumps(trade, default=str) + "\n")

        except Exception as e:
            logger.error(f"Failed to update trade: {e}")

    def get_recent_trades(self, n: int = 20) -> List[Dict]:
        """
        Return the last N trades as a list of dicts.

        Args:
            n: Number of recent trades to return.

        Returns:
            List of trade record dicts, most recent last.
        """
        trades = self.get_all_trades()
        return trades[-n:]

    def get_all_trades(self) -> List[Dict]:
        """
        Return all trades as a list of dicts.

        Returns:
            List of all trade record dicts in file order.
        """
        if not os.path.exists(self._log_path):
            return []

        trades = []
        try:
            with open(self._log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping malformed line in trade log")
        except Exception as e:
            logger.error(f"Failed to read trades: {e}")

        return trades

    def get_trade_count(self) -> int:
        """
        Return the total number of logged trades.

        Returns:
            Integer count of all trades.
        """
        return len(self.get_all_trades())

    def get_win_rate(self, last_n: Optional[int] = None) -> float:
        """
        Calculate the win rate from closed trades.

        A win is defined as a trade where pnl > 0.

        Args:
            last_n: If provided, only consider the last N closed trades.

        Returns:
            Win rate as a float between 0.0 and 1.0.
        """
        closed_trades = [
            t for t in self.get_all_trades()
            if t.get("pnl") is not None
        ]

        if last_n is not None and last_n > 0:
            closed_trades = closed_trades[-last_n:]

        if not closed_trades:
            return 0.0

        wins = sum(1 for t in closed_trades if t.get("pnl", 0) > 0)
        return wins / len(closed_trades)

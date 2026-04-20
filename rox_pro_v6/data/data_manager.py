"""
ROX Proven Edge Engine v3.0 - Data Manager
=========================================
Central data management for market data and trading logs.
"""

import os
import json
import csv
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from dataclasses import dataclass, asdict
import logging

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "Market_Trends")


@dataclass
class TradeRecord:
    """Trade record for CSV logging"""
    date_recommended: str
    stock: str
    direction: str
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float
    recommending_agents: str
    regime_at_entry: str
    conviction_confidence: int
    date_closed: str = ""
    exit_price: float = 0.0
    pnl_pct: float = 0.0


class DataManager:
    """
    Central data management for the ROX Edge Engine.
    
    Handles:
    - Trade logging to CSV
    - Pattern database management
    - Agent performance tracking
    - Daily report generation
    """
    
    CSV_HEADERS = [
        "date_recommended", "stock", "direction", "entry_price", "stop_loss",
        "target_price", "risk_reward_ratio", "recommending_agents",
        "regime_at_entry", "conviction_confidence", "date_closed",
        "exit_price", "pnl_pct"
    ]
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        self._ensure_directories()
        self.logger = logging.getLogger("DataManager")
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.data_dir,
            os.path.join(self.data_dir, "Daily_Reports"),
            os.path.join(self.data_dir, "Historical_Patterns"),
            os.path.join(self.data_dir, "Agent_Performance"),
            os.path.join(self.data_dir, "Weekly_Reconciliation")
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    # =====================
    # Trade Logging Methods
    # =====================
    
    def log_trade(self, trade: TradeRecord) -> bool:
        """
        Log a trade recommendation to CSV.
        Deduplicates on (date_recommended, stock, direction) — persistent across
        restarts, so re-running main.py never doubles up entries.
        """
        csv_path = os.path.join(self.data_dir, "Recommendation_Accuracy_Log.csv")

        try:
            # Persistent dedup: scan existing CSV for same (date, stock, direction)
            key = (trade.date_recommended, trade.stock, trade.direction)
            if os.path.exists(csv_path):
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if (row.get('date_recommended') == key[0] and
                                row.get('stock') == key[1] and
                                row.get('direction') == key[2]):
                            return False   # already logged today — skip silently

            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(asdict(trade))

            self.logger.info(f"Logged trade: {trade.stock} {trade.direction}")
            return True

        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
            return False
    
    def update_trade_outcome(self, stock: str, date_recommended: str,
                            exit_price: float, pnl_pct: float) -> bool:
        """
        Update a trade's outcome in the log.
        
        Args:
            stock: Stock symbol
            date_recommended: Original recommendation date
            exit_price: Exit price
            pnl_pct: P&L percentage
            
        Returns:
            True if successful, False otherwise
        """
        csv_path = os.path.join(self.data_dir, "Recommendation_Accuracy_Log.csv")
        
        if not os.path.exists(csv_path):
            return False
        
        try:
            # Read all records
            records = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
            
            # Find and update matching record
            updated = False
            for record in records:
                if (record['stock'] == stock and 
                    record['date_recommended'] == date_recommended):
                    record['date_closed'] = date.today().isoformat()
                    record['exit_price'] = exit_price
                    record['pnl_pct'] = pnl_pct
                    updated = True
                    break
            
            if updated:
                # Write back
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
                    writer.writeheader()
                    writer.writerows(records)
                
                self.logger.info(f"Updated trade outcome: {stock}")
            
            return updated
            
        except Exception as e:
            self.logger.error(f"Error updating trade: {e}")
            return False

    def auto_resolve_open_trades(self, live_prices: Dict[str, float]) -> int:
        """
        Auto-resolve open trades that have hit their stop loss or target.

        Called once per 60-second cycle with the current LTP dict.
        Updates pnl_pct and date_closed so MetaLearner has WIN/LOSS data.

        Returns the number of trades resolved.
        """
        csv_path = os.path.join(self.data_dir, "Recommendation_Accuracy_Log.csv")
        if not os.path.exists(csv_path):
            return 0
        try:
            records = []
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)

            resolved = 0
            today = date.today().isoformat()
            for record in records:
                if record.get("date_closed"):
                    continue   # already resolved
                stock = record.get("stock", "")
                ltp = live_prices.get(stock, 0.0)
                if not ltp:
                    continue
                entry  = float(record.get("entry_price", 0) or 0)
                sl     = float(record.get("stop_loss", 0) or 0)
                target = float(record.get("target_price", 0) or 0)
                dirn   = record.get("direction", "LONG")
                if not entry:
                    continue

                hit_target = hit_sl = False
                if dirn == "LONG":
                    hit_target = target > 0 and ltp >= target
                    hit_sl     = sl > 0    and ltp <= sl
                elif dirn == "SHORT":
                    hit_target = target > 0 and ltp <= target
                    hit_sl     = sl > 0    and ltp >= sl

                if hit_target or hit_sl:
                    exit_price = target if hit_target else sl
                    pnl_pct = ((exit_price - entry) / entry * 100) if dirn == "LONG" \
                               else ((entry - exit_price) / entry * 100)
                    record["date_closed"] = today
                    record["exit_price"]  = round(exit_price, 2)
                    record["pnl_pct"]     = round(pnl_pct, 2)
                    outcome_label = "WIN" if pnl_pct > 0 else "LOSS"
                    resolved += 1
                    self.logger.info(
                        f"[AUTO-RESOLVE] {stock} {dirn} → {outcome_label} | "
                        f"entry={entry:.2f} exit={exit_price:.2f} pnl={pnl_pct:+.1f}%"
                    )

            if resolved:
                fieldnames = records[0].keys() if records else self.CSV_HEADERS
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(records)
            return resolved

        except Exception as e:
            self.logger.error(f"auto_resolve_open_trades error: {e}")
            return 0

    # FIX 4.3: F&O Trade Logging — Feed MetaLearner
    FNO_CSV_HEADERS = [
        "date_recommended", "index", "strategy", "direction",
        "strike", "expiry", "entry_premium", "stop_loss_premium",
        "target_premium", "iv_rank", "conviction", "regime",
        "real_ltp_used",  # True when entry_premium came from real Fyers LTP
        "date_closed", "exit_premium", "pnl_pct", "outcome",
    ]

    def log_fno_trade(self, suggestion) -> bool:
        """
        FIX 4.3: Log an F&O option suggestion to fno_paper_trades.csv.
        Called immediately after DirectionalOptionAdvisor generates a PROCEED suggestion.
        Feeds MetaLearner with labelled F&O outcome data.

        Parameters
        ----------
        suggestion : OptionSuggestion  (from directional_option_advisor)
        """
        fno_csv = os.path.join(self.data_dir, "..", "fno_paper_trades.csv")
        fno_csv = os.path.normpath(fno_csv)
        try:
            row = {
                "date_recommended": datetime.now().strftime("%Y-%m-%d"),
                "index":            getattr(suggestion, "index", ""),
                "strategy":         getattr(suggestion, "strategy", ""),
                "direction":        getattr(suggestion, "option_type", ""),
                "strike":           getattr(suggestion, "strike", 0),
                "expiry":           str(getattr(suggestion, "expiry", "")),
                "entry_premium":    getattr(suggestion, "estimated_premium", 0),
                "stop_loss_premium": getattr(suggestion, "sl_premium", 0),
                "target_premium":   getattr(suggestion, "target_premium", 0),
                "iv_rank":          getattr(suggestion, "iv_rank", 0),
                "conviction":       getattr(suggestion, "confidence", 0),
                "regime":           getattr(suggestion, "regime", ""),
                # True when entry_premium came from real Fyers chain LTP (Fix 1)
                "real_ltp_used":    getattr(suggestion, "_real_ltp_used", False),
                "date_closed":      "",
                "exit_premium":     0.0,
                "pnl_pct":          0.0,
                "outcome":          "",
            }
            file_exists = os.path.exists(fno_csv)
            with open(fno_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FNO_CSV_HEADERS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            self.logger.info(
                f"[FNO-LOG] {row['strategy']} {row['index']} "
                f"strike={row['strike']} prem={row['entry_premium']}"
            )
            return True
        except Exception as e:
            self.logger.error(f"log_fno_trade error: {e}")
            return False

    def get_trades_for_period(self, week_start, week_end) -> List[Dict]:
        """
        Return closed trades whose date_recommended falls within [week_start, week_end].

        Accepted types for week_start / week_end: datetime.date, datetime.datetime,
        or ISO-8601 string (YYYY-MM-DD).

        Returns a list of dicts with at minimum:
            stock, direction, date_recommended, date_closed,
            pnl_pct, outcome  ('WIN' | 'LOSS' | '')
        """
        from datetime import date as _date, datetime as _dt

        def _to_date(val):
            if isinstance(val, _dt):
                return val.date()
            if isinstance(val, _date):
                return val
            return _date.fromisoformat(str(val)[:10])

        start = _to_date(week_start)
        end   = _to_date(week_end)

        all_trades = self.get_trade_history(limit=1000)
        result = []
        for t in all_trades:
            try:
                rec_date = _to_date(t.get('date_recommended', ''))
                if not (start <= rec_date <= end):
                    continue
                # Enrich with normalised outcome field
                pnl = float(t.get('pnl_pct') or 0)
                outcome = t.get('outcome', '')
                if not outcome:
                    if t.get('date_closed'):
                        outcome = 'WIN' if pnl > 0 else 'LOSS'
                result.append({**t, 'outcome': outcome, 'return_pct': pnl})
            except Exception:
                continue
        return result

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history"""
        csv_path = os.path.join(self.data_dir, "Recommendation_Accuracy_Log.csv")
        
        if not os.path.exists(csv_path):
            return []
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                records = list(reader)
                return records[-limit:]
        except Exception as e:
            self.logger.error(f"Error reading trade history: {e}")
            return []
    
    # =========================
    # Agent Performance Methods
    # =========================
    
    def log_agent_performance(self, agent_name: str, prediction: str,
                             outcome: str, correct: bool, regime: str):
        """Log agent prediction for accuracy tracking"""
        agent_file = os.path.join(
            self.data_dir, "Agent_Performance", f"{agent_name.lower()}_accuracy.csv"
        )
        
        headers = ["date", "prediction", "outcome", "correct", "regime"]
        
        try:
            file_exists = os.path.exists(agent_file)
            
            with open(agent_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    "date": date.today().isoformat(),
                    "prediction": prediction,
                    "outcome": outcome,
                    "correct": 1 if correct else 0,
                    "regime": regime
                })
                
        except Exception as e:
            self.logger.error(f"Error logging agent performance: {e}")
    
    def get_agent_accuracy(self, agent_name: str, 
                          regime: str = None) -> Dict:
        """Calculate agent accuracy statistics"""
        agent_file = os.path.join(
            self.data_dir, "Agent_Performance", f"{agent_name.lower()}_accuracy.csv"
        )
        
        if not os.path.exists(agent_file):
            return {"accuracy": 0.5, "total": 0, "correct": 0}
        
        try:
            correct = 0
            total = 0
            regime_stats = {}
            
            with open(agent_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    if row['correct'] == '1':
                        correct += 1
                    
                    r = row.get('regime', 'UNKNOWN')
                    if r not in regime_stats:
                        regime_stats[r] = {"correct": 0, "total": 0}
                    regime_stats[r]["total"] += 1
                    if row['correct'] == '1':
                        regime_stats[r]["correct"] += 1
            
            result = {
                "accuracy": correct / total if total > 0 else 0.5,
                "total": total,
                "correct": correct,
                "by_regime": {}
            }
            
            for r, stats in regime_stats.items():
                result["by_regime"][r] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.5
            
            if regime:
                return {
                    "accuracy": result["by_regime"].get(regime, 0.5),
                    "total": total,
                    "correct": correct
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating agent accuracy: {e}")
            return {"accuracy": 0.5, "total": 0, "correct": 0}
    
    # ===================
    # Report Methods
    # ===================
    
    def save_daily_report(self, report_content: str, report_date: date = None):
        """Save daily trading report"""
        report_date = report_date or date.today()
        report_path = os.path.join(
            self.data_dir, "Daily_Reports", f"{report_date.isoformat()}.md"
        )
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"Saved daily report: {report_path}")
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
    
    def save_weekly_reconciliation(self, week_num: int, content: str):
        """Save weekly reconciliation report"""
        report_path = os.path.join(
            self.data_dir, "Weekly_Reconciliation", f"week_{week_num:02d}_summary.md"
        )
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"Saved weekly reconciliation: {report_path}")
        except Exception as e:
            self.logger.error(f"Error saving reconciliation: {e}")
    
    # ===================
    # Pattern Methods
    # ===================
    
    def save_pattern(self, pattern: Dict):
        """Save a new pattern to the database"""
        patterns_file = os.path.join(
            self.data_dir, "Historical_Patterns", "patterns_database.json"
        )
        
        try:
            # Load existing patterns
            patterns = []
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    patterns = json.load(f)
            
            # Add new pattern
            pattern['trade_id'] = f"T{len(patterns) + 1:04d}"
            pattern['date_added'] = datetime.now().isoformat()
            patterns.append(pattern)
            
            # Save back
            with open(patterns_file, 'w') as f:
                json.dump(patterns, f, indent=2)
            
            self.logger.info(f"Saved pattern: {pattern.get('setup_type', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Error saving pattern: {e}")
    
    def search_patterns(self, criteria: Dict, limit: int = 10) -> List[Dict]:
        """Search patterns matching criteria"""
        patterns_file = os.path.join(
            self.data_dir, "Historical_Patterns", "patterns_database.json"
        )
        
        if not os.path.exists(patterns_file):
            return []
        
        try:
            with open(patterns_file, 'r') as f:
                patterns = json.load(f)
            
            matches = []
            for pattern in patterns:
                score = self._calculate_similarity(pattern, criteria)
                if score > 0.5:
                    matches.append({**pattern, "similarity_score": score})
            
            # Sort by similarity
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            return matches[:limit]
            
        except Exception as e:
            self.logger.error(f"Error searching patterns: {e}")
            return []
    
    def _calculate_similarity(self, pattern: Dict, criteria: Dict) -> float:
        """Calculate similarity score between pattern and criteria"""
        score = 0.0
        weights = {
            'setup_type': 0.40,
            'regime': 0.20,
            'direction': 0.15,
            'flow_direction': 0.15,
            'sentiment_zone': 0.10
        }
        
        for key, weight in weights.items():
            if key in pattern and key in criteria:
                if pattern[key] == criteria[key]:
                    score += weight
        
        return score
    
    def update_regime_history(self, regime: str, date_val: date = None):
        """Update regime history — writes at most once per calendar date per regime."""
        date_val = date_val or date.today()
        history_file = os.path.join(
            self.data_dir, "Historical_Patterns", "regime_history.json"
        )
        
        try:
            history = []
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            
            # IMPROVEMENT 5: Guard against duplicate entries (one per calendar day).
            # Previously appended once per 60-second live cycle → thousands of entries.
            today_str = date_val.isoformat()
            if history and history[-1].get("date") == today_str and history[-1].get("regime") == regime:
                return  # already recorded this regime for today

            history.append({
                "date": today_str,
                "regime": regime
            })
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error updating regime history: {e}")
    
    # ===================
    # Statistics Methods
    # ===================
    
    def get_trading_statistics(self, days: int = 30) -> Dict:
        """Get trading statistics for the last N days"""
        trades = self.get_trade_history(limit=500)
        
        # Filter by date
        cutoff = date.today()
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=days)
        
        recent_trades = [
            t for t in trades 
            if date.fromisoformat(t['date_recommended']) >= cutoff
        ]
        
        if not recent_trades:
            return {"total": 0, "win_rate": 0, "avg_return": 0}
        
        closed_trades = [t for t in recent_trades if t['date_closed']]
        
        wins = sum(1 for t in closed_trades if float(t['pnl_pct']) > 0)
        total_return = sum(float(t['pnl_pct']) for t in closed_trades)
        
        return {
            "total": len(recent_trades),
            "closed": len(closed_trades),
            "wins": wins,
            "losses": len(closed_trades) - wins,
            "win_rate": wins / len(closed_trades) if closed_trades else 0,
            "avg_return": total_return / len(closed_trades) if closed_trades else 0,
            "total_return": total_return
        }

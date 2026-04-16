"""
ROX Engine v5.0 — Historical Pattern Memory Bank
DuckDB-based storage of historical market conditions and outcomes.
Provides few-shot learning via similarity search.
"""

import json
import duckdb
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("reasoning.pattern_memory")


@dataclass
class PatternMatch:
    """A matched historical pattern with similarity score."""
    date: str
    similarity: float
    conditions: dict
    outcome: str
    optimal_strategy: str
    lesson: str


@dataclass
class DailySnapshot:
    """Daily market condition snapshot."""
    date: str
    nifty_close: float
    nifty_vs_200dma_pct: float
    nifty_vs_50dma_pct: float
    nifty_vs_20dma_pct: float
    vix_close: float
    vix_vs_30d_avg: float
    usd_inr_close: float
    usd_inr_trend_5d: float  # positive = weakening INR
    crude_close: float
    fii_net_buy_sell_cr: float
    fii_trend_days: int
    dii_net_buy_sell_cr: float
    nifty_intraday_range_pct: float
    nifty_close_vs_open: float
    expiry_days_remaining: int
    pcr: float
    max_pain_level: float
    put_call_oi_ratio: float
    global_cues_summary: str
    sector_rotation_pattern: str
    outcome: Optional[str] = None  # Filled after 2+ days
    optimal_strategy: Optional[str] = None
    nifty_next_day_change: Optional[float] = None
    nifty_next_day_range: Optional[tuple] = None

    def to_dict(self) -> dict:
        d = {
            "date": self.date,
            "nifty_close": self.nifty_close,
            "nifty_vs_200dma_pct": self.nifty_vs_200dma_pct,
            "nifty_vs_50dma_pct": self.nifty_vs_50dma_pct,
            "nifty_vs_20dma_pct": self.nifty_vs_20dma_pct,
            "vix_close": self.vix_close,
            "vix_vs_30d_avg": self.vix_vs_30d_avg,
            "usd_inr_close": self.usd_inr_close,
            "usd_inr_trend_5d": self.usd_inr_trend_5d,
            "crude_close": self.crude_close,
            "fii_net_buy_sell_cr": self.fii_net_buy_sell_cr,
            "fii_trend_days": self.fii_trend_days,
            "dii_net_buy_sell_cr": self.dii_net_buy_sell_cr,
            "nifty_intraday_range_pct": self.nifty_intraday_range_pct,
            "nifty_close_vs_open": self.nifty_close_vs_open,
            "expiry_days_remaining": self.expiry_days_remaining,
            "pcr": self.pcr,
            "max_pain_level": self.max_pain_level,
            "put_call_oi_ratio": self.put_call_oi_ratio,
            "global_cues_summary": self.global_cues_summary,
            "sector_rotation_pattern": self.sector_rotation_pattern,
        }
        if self.nifty_next_day_range is not None:
            d["nifty_next_day_range"] = list(self.nifty_next_day_range)
        if self.outcome:
            d["outcome"] = self.outcome
        if self.optimal_strategy:
            d["optimal_strategy"] = self.optimal_strategy
        if self.nifty_next_day_change is not None:
            d["nifty_next_day_change"] = self.nifty_next_day_change
        return d


class PatternMemoryBank:
    """
    DuckDB-backed historical pattern storage and similarity search.
    """
    
    # Feature weights for similarity calculation
    FEATURE_WEIGHTS = {
        "nifty_vs_200dma_pct": 0.20,
        "fii_net_buy_sell_cr": 0.15,
        "vix_close": 0.12,
        "usd_inr_close": 0.12,
        "fii_trend_days": 0.10,
        "nifty_intraday_range_pct": 0.08,
        "usd_inr_trend_5d": 0.08,
        "put_call_oi_ratio": 0.08,
        "global_cues_summary": 0.07,
    }
    
    # Binning thresholds for categorical features
    VIX_BINS = [(-999, 12, "very_low"), (12, 15, "low"), (15, 18, "medium"), 
                (18, 25, "high"), (25, 999, "very_high")]
    
    def __init__(self, db_path: str = "data/pattern_memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create database tables if they don't exist."""
        conn = duckdb.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                date TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_created 
            ON snapshots(created_at)
        """)
        
        conn.commit()
        conn.close()
    
    def save_snapshot(self, snapshot: DailySnapshot):
        """Save or update a daily snapshot."""
        conn = duckdb.connect(self.db_path)
        cursor = conn.cursor()
        
        data_json = json.dumps(snapshot.to_dict())
        
        cursor.execute("""
            INSERT INTO snapshots (date, data)
            VALUES (?, ?)
            ON CONFLICT (date) DO UPDATE SET data = excluded.data
        """, (snapshot.date, data_json))
        
        conn.commit()
        conn.close()
        logger.debug(f"Saved snapshot for {snapshot.date}")
    
    def load_snapshot(self, date: str) -> Optional[DailySnapshot]:
        """Load a snapshot by date."""
        conn = duckdb.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT data FROM snapshots WHERE date = ?", (date,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = json.loads(row[0])
            return DailySnapshot(**data)
        return None
    
    def update_outcome(self, date: str, outcome: str, optimal_strategy: str,
                        next_day_change: float = None):
        """Update the outcome for a historical snapshot."""
        conn = duckdb.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT data FROM snapshots WHERE date = ?", (date,))
        row = cursor.fetchone()
        
        if row:
            data = json.loads(row[0])
            data["outcome"] = outcome
            data["optimal_strategy"] = optimal_strategy
            if next_day_change is not None:
                data["nifty_next_day_change"] = next_day_change
            
            cursor.execute("""
                UPDATE snapshots SET data = ? WHERE date = ?
            """, (json.dumps(data), date))
            conn.commit()
        
        conn.close()
    
    def get_pattern_count(self) -> int:
        """Get total number of stored patterns with outcomes."""
        conn = duckdb.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM snapshots 
            WHERE json_extract(data, '$.outcome') IS NOT NULL
        """)
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def find_similar(self, current: DailySnapshot, top_k: int = 5) -> list[PatternMatch]:
        """
        Find the K most similar historical patterns.
        Uses weighted feature matching.
        """
        conn = duckdb.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, data FROM snapshots 
            WHERE json_extract(data, '$.outcome') IS NOT NULL
            ORDER BY date DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        scored = []
        for date_str, data_json in rows:
            data = json.loads(data_json)
            score = self._compute_similarity(current, data)
            if score > 0:  # Only include if there's some similarity
                scored.append(PatternMatch(
                    date=date_str,
                    similarity=score,
                    conditions=data,
                    outcome=data.get("outcome", "Unknown"),
                    optimal_strategy=data.get("optimal_strategy", "Unknown"),
                    lesson=data.get("lesson", ""),
                ))
        
        scored.sort(key=lambda x: x.similarity, reverse=True)
        return scored[:top_k]
    
    def _compute_similarity(self, current: DailySnapshot, historical: dict) -> float:
        """
        Weighted multi-feature similarity score.
        Returns 0.0-1.0.
        """
        total_score = 0.0
        
        for feature, weight in self.FEATURE_WEIGHTS.items():
            current_val = getattr(current, feature, None)
            hist_val = historical.get(feature)
            
            if current_val is None or hist_val is None:
                continue
            
            score = self._feature_similarity(feature, current_val, hist_val)
            total_score += score * weight
        
        return min(total_score, 1.0)
    
    def _feature_similarity(self, feature: str, current, historical) -> float:
        """Compute similarity for a single feature (0.0-1.0)."""
        
        # Handle categorical features
        if feature == "global_cues_summary":
            return 1.0 if str(current).lower() == str(historical).lower() else 0.3
        if feature == "sector_rotation_pattern":
            return 1.0 if str(current).lower() == str(historical).lower() else 0.3
        if feature == "fii_trend_days":
            # Continuous but directional
            if current > 0 and historical > 0:
                return min(1.0, min(current, historical) / max(current, historical))
            elif current < 0 and historical < 0:
                return min(1.0, min(abs(current), abs(historical)) / max(abs(current), abs(historical)))
            else:
                return 0.0
        
        # Handle continuous features with normalization
        try:
            c_val = float(current)
            h_val = float(historical)
            
            # Normalize: if both values are small relative to the larger one,
            # they are similar
            if abs(h_val) < 0.001 and abs(c_val) < 0.001:
                return 1.0
            
            # Percentage-based similarity for ratio-type features
            if feature in ("usd_inr_trend_5d", "usd_inr_close", "nifty_vs_200dma_pct"):
                return 1.0 - min(1.0, abs(c_val - h_val) / max(abs(h_val), 0.01))
            
            # Absolute difference with scaling
            if feature == "vix_close":
                return 1.0 - min(1.0, abs(c_val - h_val) / 10.0)
            if feature == "pcr":
                return 1.0 - min(1.0, abs(c_val - h_val) / 1.0)
            
            # Default: normalized absolute difference
            max_val = max(abs(c_val), abs(h_val), 0.001)
            return 1.0 - min(1.0, abs(c_val - h_val) / max_val)
            
        except (ValueError, TypeError):
            return 0.0
    
    def format_as_few_shot(self, matches: list[PatternMatch]) -> str:
        """Format matches as few-shot examples for LLM prompts."""
        if not matches:
            return ""
        
        lines = ["## HISTORICAL PRECEDENTS (similar market conditions)\n"]
        for i, m in enumerate(matches, 1):
            lines.append(f"\n### Historical Case {i} (Similarity: {m.similarity:.0%})")
            lines.append(f"- Date: {m.date}")
            lines.append(f"- Conditions: {json.dumps(m.conditions, indent=2, default=str)}")
            lines.append(f"- What actually happened: {m.outcome}")
            lines.append(f"- Best strategy was: {m.optimal_strategy}")
            if m.lesson:
                lines.append(f"- Key lesson: {m.lesson}")
        
        if matches:
            bull_outcomes = sum(1 for m in matches if "BULL" in m.outcome.upper() or "RALLY" in m.outcome.upper() or "UP" in m.outcome.upper())
            bear_outcomes = sum(1 for m in matches if "BEAR" in m.outcome.upper() or "FELL" in m.outcome.upper() or "DOWN" in m.outcome.upper())
            
            lines.append(f"\nHISTORICAL CONSENSUS:")
            lines.append(f"  → {bull_outcomes} of {len(matches)} cases: BULLISH/UP outcome")
            lines.append(f"  → {bear_outcomes} of {len(matches)} cases: BEARISH/DOWN outcome")
            lines.append(f"  → TODAY's conditions match most closely to Case #1 (highest similarity)")
        
        return "\n".join(lines)

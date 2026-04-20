"""
LLM History Analyzer
====================
Reads Recommendation_Accuracy_Log.csv BEFORE the daily plan is generated
and builds a compact historical performance context that gets injected into:

  1. LLMCrossExaminer  — so it knows what has worked/failed in this regime before
  2. LLMPatternValidator — so per-stock validation has stock-specific win rate context

This replaces the XGBoost route for now because:
  - Works with even 5–10 resolved trades (LLM reasons, doesn't overfit)
  - Understands context (regime + direction + conviction combinations)
  - Explains its findings in natural language
  - Zero new infrastructure — plugs into existing LLM layer

Flow:
    coordinator.py
        └── LeadCoordinator.__init__
                └── LLMHistoryAnalyzer(csv_path)      ← new, runs once at startup
        └── generate_trading_plan()
                └── history_analyzer.build_context()  ← runs before cross-examiner
                └── injects HistoryContext into:
                        - llm_examiner.examine_consensus(market_context=...)
                        - llm_validator.validate_pattern(market_context=...)
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeStats:
    regime: str
    total: int = 0
    wins: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    @property
    def win_rate(self) -> float:
        return round(self.wins / self.total * 100, 1) if self.total > 0 else 0.0

    def summary(self) -> str:
        if self.total < 3:
            return f"{self.regime}: only {self.total} resolved trade(s) — insufficient data"
        return (
            f"{self.regime}: {self.win_rate}% WR ({self.wins}W/{self.total - self.wins}L "
            f"from {self.total} trades) | avg_win=+{self.avg_win_pct:.1f}% "
            f"avg_loss=-{self.avg_loss_pct:.1f}%"
        )


@dataclass
class StockStats:
    stock: str
    total: int = 0
    wins: int = 0
    avg_pnl: float = 0.0
    recent_outcome: str = ""   # last exit_reason
    regimes_seen: List[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return round(self.wins / self.total * 100, 1) if self.total > 0 else 0.0

    def summary(self) -> str:
        if self.total == 0:
            return f"{self.stock}: no history"
        if self.total < 2:
            return f"{self.stock}: 1 resolved trade → {self.recent_outcome}"
        return (
            f"{self.stock}: {self.win_rate}% WR over {self.total} trades "
            f"| avg_pnl={self.avg_pnl:+.1f}% | last={self.recent_outcome}"
        )


@dataclass
class HistoryContext:
    """
    Compact historical context injected into LLM calls each cycle.
    Designed to fit in ~400 tokens so it doesn't bloat prompts.
    """
    total_resolved: int
    total_open: int
    overall_win_rate: float
    regime_stats: Dict[str, RegimeStats]         # keyed by regime name
    stock_stats: Dict[str, StockStats]           # keyed by stock symbol
    conviction_insight: str                      # e.g. "conviction>65 wins 71% vs 45% below"
    recent_trend: str                            # e.g. "last 5 trades: 4 losses — bearish run"
    direction_stats: Dict[str, float]            # {"LONG": 55.0, "SHORT": 40.0}
    generated_at: datetime = field(default_factory=datetime.now)

    def for_cross_examiner(self) -> str:
        """
        Returns a compact multi-line string for injection into the
        cross-examiner's market_context dict.
        Covers regime-level and system-level patterns.
        """
        if self.total_resolved == 0:
            return "No resolved trade history yet — first few days of operation."

        lines = [
            f"=== HISTORICAL PERFORMANCE ({self.total_resolved} resolved trades) ===",
            f"Overall win rate : {self.overall_win_rate:.1f}%",
            f"Conviction insight: {self.conviction_insight}",
            f"Recent trend     : {self.recent_trend}",
            "",
            "PERFORMANCE BY REGIME:",
        ]
        for rs in sorted(self.regime_stats.values(), key=lambda x: -x.total):
            lines.append(f"  {rs.summary()}")

        lines.append("")
        lines.append("PERFORMANCE BY DIRECTION:")
        for dirn, wr in self.direction_stats.items():
            lines.append(f"  {dirn}: {wr:.1f}% win rate")

        return "\n".join(lines)

    def for_stock(self, symbol: str) -> str:
        """
        Returns a 1-2 line stock-specific history string for injection
        into the pattern validator's market_context dict.
        """
        if symbol in self.stock_stats:
            return self.stock_stats[symbol].summary()
        return f"{symbol}: no prior trade history in this engine"


# ─────────────────────────────────────────────────────────────────────────────
# Main analyzer class
# ─────────────────────────────────────────────────────────────────────────────

class LLMHistoryAnalyzer:
    """
    Reads Recommendation_Accuracy_Log.csv and builds HistoryContext.

    Runs once at engine startup (cheap — pure CSV parsing, no API calls).
    Can also be refreshed mid-session by calling build_context() again.
    """

    DEFAULT_CSV = (
        Path(__file__).resolve()
        .parent.parent.parent           # project root
        / "data" / "Market_Trends" / "Recommendation_Accuracy_Log.csv"
    )

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path      = Path(csv_path) if csv_path else self.DEFAULT_CSV
        self.logger        = logging.getLogger("LLMHistoryAnalyzer")
        self._context: Optional[HistoryContext] = None
        self._cached_mtime: float = 0.0  # mtime of CSV when context was last built

    # ── Public API ────────────────────────────────────────────────────────────

    def _csv_mtime(self) -> float:
        """Return CSV last-modified timestamp, or 0 if file missing."""
        try:
            return self.csv_path.stat().st_mtime
        except OSError:
            return 0.0

    def build_context(self) -> HistoryContext:
        """
        Parse the CSV and return a HistoryContext.
        Re-parses ONLY when the CSV file has changed since last call.
        Safe to call on every engine run — typically returns cached result in <1ms.
        """
        current_mtime = self._csv_mtime()
        if self._context is not None and current_mtime == self._cached_mtime:
            self.logger.debug("[HISTORY] CSV unchanged — returning cached context")
            return self._context

        trades = self._load_csv()
        ctx    = self._compute_context(trades)
        self._context      = ctx
        self._cached_mtime = current_mtime

        resolved = ctx.total_resolved
        if resolved == 0:
            self.logger.info(
                "[HISTORY] No resolved trades yet — context will be empty. "
                "Run daily_reconcile.py after market close to build history."
            )
        else:
            self.logger.info(
                f"[HISTORY] Context rebuilt | resolved={resolved} | open={ctx.total_open} "
                f"| win_rate={ctx.overall_win_rate:.1f}% | {ctx.recent_trend}"
            )
        return ctx

    def get_context(self) -> HistoryContext:
        """Return context, rebuilding only if CSV has changed since last call."""
        return self.build_context()

    def get_for_cross_examiner(self) -> str:
        return self.get_context().for_cross_examiner()

    def get_for_stock(self, symbol: str) -> str:
        return self.get_context().for_stock(symbol)

    # ── CSV loading ───────────────────────────────────────────────────────────

    def _load_csv(self) -> List[Dict]:
        if not self.csv_path.exists():
            self.logger.warning(f"[HISTORY] CSV not found: {self.csv_path}")
            return []
        try:
            with open(self.csv_path, "r", newline="") as f:
                return [dict(row) for row in csv.DictReader(f)]
        except Exception as e:
            self.logger.error(f"[HISTORY] CSV read failed: {e}")
            return []

    # ── Context computation ───────────────────────────────────────────────────

    def _compute_context(self, trades: List[Dict]) -> HistoryContext:
        resolved    = [t for t in trades if self._is_resolved(t)]
        open_trades = [t for t in trades if not self._is_resolved(t)]

        if not resolved:
            return HistoryContext(
                total_resolved=0,
                total_open=len(open_trades),
                overall_win_rate=0.0,
                regime_stats={},
                stock_stats={},
                conviction_insight="No resolved trades yet.",
                recent_trend="No history yet.",
                direction_stats={},
            )

        # ── Overall win rate ──────────────────────────────────────────────────
        wins   = [t for t in resolved if self._exit_reason(t) == "TARGET_HIT"]
        losses = [t for t in resolved if self._exit_reason(t) != "TARGET_HIT"]
        overall_wr = len(wins) / len(resolved) * 100

        # ── Per-regime stats ──────────────────────────────────────────────────
        regime_buckets: Dict[str, List[Dict]] = defaultdict(list)
        for t in resolved:
            regime_buckets[t.get("regime_at_entry", "UNKNOWN")].append(t)

        regime_stats: Dict[str, RegimeStats] = {}
        for regime, bucket in regime_buckets.items():
            rs       = RegimeStats(regime=regime)
            rs.total = len(bucket)
            rs.wins  = sum(1 for t in bucket if self._exit_reason(t) == "TARGET_HIT")
            win_pnls  = [float(t.get("pnl_pct", 0) or 0) for t in bucket
                         if self._exit_reason(t) == "TARGET_HIT"]
            loss_pnls = [abs(float(t.get("pnl_pct", 0) or 0)) for t in bucket
                         if self._exit_reason(t) != "TARGET_HIT"]
            rs.avg_win_pct  = round(sum(win_pnls)  / len(win_pnls),  2) if win_pnls  else 0.0
            rs.avg_loss_pct = round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0
            regime_stats[regime] = rs

        # ── Per-stock stats ───────────────────────────────────────────────────
        stock_buckets: Dict[str, List[Dict]] = defaultdict(list)
        for t in resolved:
            stock_buckets[t.get("stock", "UNKNOWN")].append(t)

        stock_stats: Dict[str, StockStats] = {}
        for sym, bucket in stock_buckets.items():
            bucket_sorted = sorted(bucket, key=lambda x: x.get("date_closed", ""))
            ss = StockStats(stock=sym)
            ss.total   = len(bucket_sorted)
            ss.wins    = sum(1 for t in bucket_sorted if self._exit_reason(t) == "TARGET_HIT")
            pnls       = [float(t.get("pnl_pct", 0) or 0) for t in bucket_sorted]
            ss.avg_pnl = round(sum(pnls) / len(pnls), 2) if pnls else 0.0
            ss.recent_outcome = self._exit_reason(bucket_sorted[-1])
            ss.regimes_seen   = list({t.get("regime_at_entry", "") for t in bucket_sorted})
            stock_stats[sym]  = ss

        # ── Conviction insight ────────────────────────────────────────────────
        conviction_insight = self._conviction_insight(resolved)

        # ── Recent trend (last 5 resolved trades) ─────────────────────────────
        recent       = sorted(resolved, key=lambda x: x.get("date_closed", ""))[-5:]
        recent_wins  = sum(1 for t in recent if self._exit_reason(t) == "TARGET_HIT")
        recent_losses = len(recent) - recent_wins
        if len(recent) < 3:
            recent_trend = f"Only {len(recent)} resolved trade(s) so far"
        elif recent_wins >= 4:
            recent_trend = f"last {len(recent)} trades: {recent_wins}W/{recent_losses}L — strong run"
        elif recent_losses >= 4:
            recent_trend = f"last {len(recent)} trades: {recent_wins}W/{recent_losses}L — losing run, caution advised"
        else:
            recent_trend = f"last {len(recent)} trades: {recent_wins}W/{recent_losses}L — mixed"

        # ── Direction stats ───────────────────────────────────────────────────
        direction_buckets: Dict[str, List[Dict]] = defaultdict(list)
        for t in resolved:
            direction_buckets[t.get("direction", "LONG").upper()].append(t)

        direction_stats: Dict[str, float] = {}
        for dirn, bucket in direction_buckets.items():
            w = sum(1 for t in bucket if self._exit_reason(t) == "TARGET_HIT")
            direction_stats[dirn] = round(w / len(bucket) * 100, 1) if bucket else 0.0

        return HistoryContext(
            total_resolved  = len(resolved),
            total_open      = len(open_trades),
            overall_win_rate= round(overall_wr, 1),
            regime_stats    = regime_stats,
            stock_stats     = stock_stats,
            conviction_insight = conviction_insight,
            recent_trend    = recent_trend,
            direction_stats = direction_stats,
        )

    @staticmethod
    def _is_resolved(t: Dict) -> bool:
        """A trade is resolved if exit_reason is set OR date_closed is present."""
        exit_reason = (t.get("exit_reason") or "").strip()
        if exit_reason in ("TARGET_HIT", "STOP_HIT", "TIMED_OUT"):
            return True
        # Fallback: date_closed filled but exit_reason missing (older reconciler runs)
        if (t.get("date_closed") or "").strip():
            return True
        return False

    @staticmethod
    def _exit_reason(t: Dict) -> str:
        """Derive exit reason even when the column is blank."""
        explicit = (t.get("exit_reason") or "").strip()
        if explicit in ("TARGET_HIT", "STOP_HIT", "TIMED_OUT"):
            return explicit
        # Infer from pnl_pct sign when column is missing
        try:
            pnl = float(t.get("pnl_pct") or 0)
            return "TARGET_HIT" if pnl > 0 else "STOP_HIT"
        except (ValueError, TypeError):
            return "STOP_HIT"

    def _conviction_insight(self, resolved: List[Dict]) -> str:
        """
        Checks whether higher conviction scores actually correlate with wins
        in YOUR engine's history. Returns a 1-line insight string.
        """
        if len(resolved) < 5:
            return "Insufficient data for conviction analysis yet."

        threshold = 65
        high = [t for t in resolved if float(t.get("conviction_confidence", 0) or 0) >= threshold]
        low  = [t for t in resolved if float(t.get("conviction_confidence", 0) or 0) <  threshold]

        high_wr = (sum(1 for t in high if self._exit_reason(t) == "TARGET_HIT") / len(high) * 100) if high else 0.0
        low_wr  = (sum(1 for t in low  if self._exit_reason(t) == "TARGET_HIT") / len(low)  * 100) if low  else 0.0

        if not high or not low:
            return f"All {len(resolved)} resolved trades in one conviction band — more data needed."

        gap = high_wr - low_wr
        if abs(gap) < 5:
            return (
                f"Conviction threshold ({threshold}) shows NO meaningful edge: "
                f"high={high_wr:.0f}% vs low={low_wr:.0f}% WR — conviction scores may need recalibration."
            )
        elif gap > 0:
            return (
                f"Conviction ≥{threshold} wins {high_wr:.0f}% vs {low_wr:.0f}% below "
                f"({len(high)} vs {len(low)} trades) — high conviction is working."
            )
        else:
            return (
                f"⚠ Conviction ≥{threshold} actually UNDERPERFORMS: {high_wr:.0f}% vs {low_wr:.0f}% WR "
                f"— high-conviction calls may be over-confident in current conditions."
            )

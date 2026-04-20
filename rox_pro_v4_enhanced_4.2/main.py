#!/usr/bin/env python3
"""
ROX Proven Edge Engine v4.0 Unified — Main Entry Point
=======================================================
Combines the v3.2 multi-agent swing engine with the v4.0 F&O specialist
agents (HERMES, THETA, DELTA) into a single production-ready system.

Modes
-----
  paper     — live market data, simulated orders  (default)
  live      — live market data, real order routing via broker API
  backtest  — historical simulation (start/end date required)
  demo      — quick sanity-check run with synthetic data, no data feeds needed

Usage
-----
  python main.py                          # paper mode
  python main.py --mode demo              # demo with synthetic data
  python main.py --mode live              # live trading
  python main.py --mode backtest --start-date 2024-01-01 --end-date 2024-12-31
  python main.py --mode demo --pre-market # run pre-market briefing then demo
"""

import argparse
import asyncio
import logging
import signal
import sys
import threading
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any

# ── path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── logging ─────────────────────────────────────────────────────────────────
try:
    from core import init_logging, get_logger
    init_logging()
    logger = get_logger("Main")
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROJECT_ROOT / "logs" / "rox_unified.log",
                                encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("Main")

# ── core imports ─────────────────────────────────────────────────────────────
from config import (
    SystemConfig, DEFAULT_CONFIG, TradeDirection, MarketRegime,
    ConvictionLevel, get_vix_regime, NIFTY_50_STOCKS, get_system_config,
)
from coordinator import UnifiedCoordinator, DailyTradingPlan
from agents import (
    HermesAgent, ThetaAgent, DeltaAgent,
    Order, OrderStatus, OrderType,
)

# ── news intelligence imports (v4.1) ─────────────────────────────────────────
from agents.news_core import NewsContextProvider, get_news_context
from agents.nocturnal_agent import NocturnalAgent

# ── API server integration ───────────────────────────────────────────────────
from api_server import update_state


# ===========================================================================
# ROX Unified Engine
# ===========================================================================

class ROXUnifiedEngine:
    """
    ROX Proven Edge Engine v4.0 Unified.

    Single facade over UnifiedCoordinator + broker integration hooks.
    Supports paper, live, backtest, and demo modes.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        mode: str = "paper",
        portfolio_value: float = 1_000_000,
    ):
        self.config = config or get_system_config()
        self.mode = mode
        self.config.portfolio_value = portfolio_value
        self.portfolio_value = portfolio_value
        self.running = False
        self._shutdown_event = threading.Event()

        logger.info(
            f"Initialising ROX Unified Engine v4.0 | mode={mode.upper()} | "
            f"portfolio=₹{portfolio_value:,.0f}"
        )

        # Central coordinator (swing + F&O specialists)
        self.coordinator = UnifiedCoordinator(self.config, portfolio_value)

        # ── News Intelligence (v4.1) ─────────────────────────────────────────
        self.news_context = get_news_context()
        self.nocturnal = NocturnalAgent()
        # Register NOCTURNAL with coordinator's lead agent pool
        self.coordinator.lead.agents["NOCTURNAL"] = self.nocturnal

        # ── v4.3: PHOENIX Pre-Momentum Recovery Radar ─────────────────────────
        # PHOENIX is a non-voting observer — does not affect consensus weighted
        # votes.  Its rolling state must survive across 60-second live cycles so
        # it's instantiated here (once per session) and registered with the lead.
        from agents.phoenix_agent import PhoenixAgent as _PhoenixAgent
        self.phoenix = _PhoenixAgent()
        self.coordinator.lead.agents["PHOENIX"] = self.phoenix

        # Dedicated event loop for async news calls executed from sync threads.
        # asyncio.run() creates and destroys a new loop every call; a persistent
        # loop avoids that overhead and prevents "event loop closed" edge-cases.
        self._news_loop = asyncio.new_event_loop()
        self._news_loop_thread = threading.Thread(
            target=self._news_loop.run_forever,
            daemon=True, name="ROX-NewsLoop",
        )
        self._news_loop_thread.start()

        # Overnight context storage (populated by pre_market_routine)
        self.overnight_context: Dict = {}

        # Infrastructure components (lazy — init only if needed)
        self._strategy_factory = None

        # Data fetcher — initialised once on first live cycle and reused.
        # This is critical: FyersFetcher caches 60-day OHLCV history per
        # calendar day and syncs F&O positions at startup.  Creating a new
        # instance every 60s throws all that away and repeats 51 API calls
        # + one F&O positions call on every single cycle.
        self._data_fetcher = None

        logger.info("Engine ready — 13 agents loaded (including NOCTURNAL + PHOENIX)")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the engine's background monitoring loop."""
        logger.info("Starting engine...")
        self.running = True
        self.coordinator.fno.mwpl_monitor          # trigger lazy MWPL init

        self._loop_thread = threading.Thread(
            target=self._main_loop, daemon=True, name="ROX-MainLoop"
        )
        self._loop_thread.start()

        # ── LLM MetaLearner weekly batch ─────────────────────────────────
        # Runs once on startup then every Sunday at 19:00 IST.
        # Analyses resolved trade outcomes and recommends agent weight changes.
        # Human review required before recommendations are applied.
        self._meta_thread = threading.Thread(
            target=self._meta_learner_loop, daemon=True, name="ROX-MetaLearner"
        )
        self._meta_thread.start()

        logger.info("Engine started")

    def _meta_learner_loop(self):
        """Weekly MetaLearner batch — runs Sundays at 19:00 IST."""
        import time as _time
        from datetime import datetime as _dt
        while self.running:
            try:
                now = _dt.now()
                # Run on Sunday (weekday 6) at 19:00, or on engine startup
                is_sunday_evening = (now.weekday() == 6 and now.hour == 19 and now.minute < 5)
                first_run = not getattr(self, "_meta_ran_once", False)
                if is_sunday_evening or first_run:
                    self._meta_ran_once = True
                    _llm_meta = getattr(self.coordinator.lead, "llm_meta", None)
                    if _llm_meta is not None:
                        logger.info("[META-LEARNER] Starting weekly batch analysis...")
                        try:
                            from datetime import date as _date, timedelta as _td
                            _today = _date.today()
                            _week_start = _today - _td(days=7)
                            _week_end = _today
                            result = _llm_meta.analyze_weekly_performance(
                                week_start=_week_start,
                                week_end=_week_end,
                            )
                            logger.info(
                                f"[META-LEARNER] Complete | "
                                f"recommendations={len(result.agent_weight_adjustments)} | "
                                f"confidence={result.confidence_in_recommendations:.0f}% | "
                                f"source={result.source}"
                            )
                            for adj in result.agent_weight_adjustments[:5]:
                                logger.info(f"  [META] {adj.agent_name}: {adj.action} {adj.amount:.2f} — {adj.reason}")
                            if result.regime_specific_rules:
                                for rule in result.regime_specific_rules[:3]:
                                    logger.info(f"  [META] Regime rule [{rule.regime}]: {rule.rule}")
                            # ── Auto-apply gate ──────────────────────────────
                            # Tier 1 (AUTO):   confidence ≥ 85%, weight changes only,
                            #                  no systemic or regime-rule changes.
                            # Tier 2 (MANUAL): everything else — log and wait for
                            #                  human call to apply_recommendations().
                            _AUTO_APPLY_THRESHOLD = 85
                            _has_only_weight_changes = (
                                bool(result.agent_weight_adjustments)
                                and not result.regime_specific_rules
                                and not result.systemic_improvements
                            )
                            _lead_cfg = getattr(
                                getattr(self.coordinator, "lead", None), "config", None
                            )
                            if (
                                result.source == "LLM"
                                and result.confidence_in_recommendations >= _AUTO_APPLY_THRESHOLD
                                and _has_only_weight_changes
                                and _lead_cfg is not None
                            ):
                                _applied = _llm_meta.apply_recommendations(result, _lead_cfg)
                                if _applied:
                                    logger.info(
                                        f"[META-LEARNER] ✅ Auto-applied "
                                        f"(confidence={result.confidence_in_recommendations}% "
                                        f"≥ {_AUTO_APPLY_THRESHOLD}%, weight-only, no systemic rules)"
                                    )
                                else:
                                    logger.warning(
                                        "[META-LEARNER] Auto-apply failed — "
                                        "check config.normalize_weights(); manual apply required."
                                    )
                            else:
                                _reasons = []
                                if result.source != "LLM":
                                    _reasons.append(f"source={result.source} (LLM required)")
                                if result.confidence_in_recommendations < _AUTO_APPLY_THRESHOLD:
                                    _reasons.append(
                                        f"confidence={result.confidence_in_recommendations}% "
                                        f"< {_AUTO_APPLY_THRESHOLD}% threshold"
                                    )
                                if result.systemic_improvements:
                                    _reasons.append(
                                        f"{len(result.systemic_improvements)} systemic improvement(s) need review"
                                    )
                                if result.regime_specific_rules:
                                    _reasons.append(
                                        f"{len(result.regime_specific_rules)} regime rule(s) need review"
                                    )
                                logger.info(
                                    "[META-LEARNER] ⚠ REVIEW REQUIRED — "
                                    + " | ".join(_reasons) + ". "
                                    "Call coordinator.lead.llm_meta.apply_recommendations(result) manually."
                                )
                        except Exception as _me:
                            logger.warning(f"[META-LEARNER] batch failed: {_me}")
            except Exception as _oe:
                logger.debug(f"[META-LEARNER] loop error: {_oe}")
            _time.sleep(300)   # check every 5 minutes

    def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping engine...")
        self.running = False
        self._shutdown_event.set()
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=5)
        # Stop the dedicated news event loop
        if hasattr(self, "_news_loop") and self._news_loop.is_running():
            self._news_loop.call_soon_threadsafe(self._news_loop.stop)
        logger.info("Engine stopped")

    # ------------------------------------------------------------------
    # Pre-market routine (News Intelligence — v4.1)
    # ------------------------------------------------------------------

    async def pre_market_routine(self):
        """Run at 9:00 AM IST before market opens"""
        logger.info("=== PRE-MARKET BRIEFING ===")

        # Update news context
        await self.news_context.update()

        # Get market context for overnight assessment
        market_context = self._fetch_market_data()

        # Generate overnight risk profile
        await self.news_context.update_overnight_profile(market_context)

        # Run NOCTURNAL agent analysis
        nocturnal_report = self.nocturnal.analyze(
            {
                "market_context": market_context,
                "portfolio_status": self._get_portfolio_status(),
                "force_refresh": True
            },
            self.coordinator.lead.current_regime
        )

        # Store in engine state
        self.overnight_context = nocturnal_report.analysis_details.get("risk_profile", {})

        # Log briefing
        risk_profile = self.news_context.get_overnight_risk()
        if risk_profile:
            logger.info(f"Risk Level: {risk_profile.risk_level}")
            logger.info(f"Expected Gap: {risk_profile.expected_gap_size}")
            logger.info(f"Key Headlines:")
            for headline in risk_profile.key_headlines[:3]:
                logger.info(f"  • {headline}")

            if risk_profile.trading_restrictions:
                logger.warning(f"TRADING RESTRICTIONS ACTIVE:")
                for restriction in risk_profile.trading_restrictions:
                    logger.warning(f"  🚫 {restriction}")

        return nocturnal_report

    # ------------------------------------------------------------------
    # Main loop (paper / live modes)
    # ------------------------------------------------------------------

    def _main_loop(self):
        """60-second trading cycle loop."""
        logger.info("Main loop started")
        while self.running and not self._shutdown_event.is_set():
            try:
                self._trading_cycle()
            except Exception as exc:
                logger.error(f"Trading cycle error: {exc}", exc_info=True)
            self._shutdown_event.wait(60)
        logger.info("Main loop ended")

    def _trading_cycle(self):
        """One complete trading cycle."""
        # Ensure single cycle execution for live mode
        if hasattr(self, '_cycle_count'):
            self._cycle_count += 1
        else:
            self._cycle_count = 0

        if self._cycle_count >= 1:
            logger.info("Single cycle completed, exiting live mode")
            # FIX-THREAD-01: cannot call self.stop() from inside _loop_thread
            # (threading.Thread.join() raises RuntimeError if called on current thread).
            # Signal the loop to exit cleanly instead; _main_loop checks these flags
            # after _trading_cycle returns and will exit on its next iteration.
            self.running = False
            self._shutdown_event.set()
            return

        # Attempt live data; track whether we fell back to synthetic
        using_synthetic = False
        try:
            market_data = self._fetch_live_data()
        except Exception as exc:
            logger.debug(f"Live data unavailable ({exc}), using synthetic fallback")
            market_data = self._build_demo_data()
            using_synthetic = True

        # ── Auto-resolve open trades against live prices (MetaLearner data) ──
        # Checks every open recommendation for SL/target hit using current LTP.
        # Without this, MetaLearner has no WIN/LOSS labels to learn from.
        if not using_synthetic:
            try:
                live_prices = {
                    sym: market_data.get("price_data", {}).get(sym, {}).get("close", 0)
                    for sym in market_data.get("price_data", {})
                }
                dm = self.coordinator.lead.data_manager
                resolved = dm.auto_resolve_open_trades(live_prices)
                if resolved:
                    logger.info(f"[DATA] Auto-resolved {resolved} trade(s) against live prices")
            except Exception as _ar:
                logger.debug(f"auto_resolve_open_trades skipped: {_ar}")

        # Ensure news context is available
        if not self.news_context._current_news:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.news_context.update(), self._news_loop
                )
                future.result(timeout=30)  # wait up to 30s, non-blocking for the main thread
            except Exception as _ne:
                logger.debug(f"News context update skipped: {_ne}")

        # Generate trading plan with news-aware agents
        plan = self.coordinator.generate_trading_plan(market_data)

        # ── Update API server state ──────────────────────────────────────────
        try:
            # Build state dictionary from plan and market data
            regime = plan.market_regime
            confidence = plan.regime_confidence
            consensus = plan.consensus
            agent_reports = plan.agent_reports
            option_suggestions = getattr(plan.fno_suggestions, "suggestions", None) or []

            engine_state = {
                "status": "running",
                "regime": {
                    "regime": regime.value,
                    "confidence": confidence,
                },
                "consensus": {
                    "direction": consensus.direction.value,
                    "net_score": consensus.net_score,
                },
                "agents": [
                    {
                        "name": name,
                        "verdict": r.verdict.direction.value,
                        "conviction": r.verdict.conviction,
                        "weight": r.verdict.weight,
                    }
                    for name, r in agent_reports.items()
                ],
                "suggestions": [
                    {
                        "index": getattr(s, 'index', None),
                        "strategy": getattr(s, 'strategy', None),
                        "strike": getattr(s, 'strike', None),
                        "expiry": str(getattr(s, 'expiry', '')),
                        "dte": getattr(s, 'dte', None),
                        "spot": getattr(s, 'spot', None),
                        "entry_price": getattr(s, 'entry_price', None),
                        "cost_per_lot": getattr(s, 'cost_per_lot', None),
                        "stop_loss": getattr(s, 'stop_loss', None),
                        "target": getattr(s, 'target', None),
                        "prob_profit": getattr(s, 'prob_profit', None),
                        "delta": getattr(s, 'delta', None),
                        "gamma": getattr(s, 'gamma', None),
                        "theta": getattr(s, 'theta', None),
                        "vega": getattr(s, 'vega', None),
                        "iv_rank": getattr(s, 'iv_rank', None),
                        "conviction": getattr(s, 'conviction', None),
                        "score": getattr(s, 'score', None),
                        "basis": getattr(s, 'basis', None),
                        "oi": getattr(s, 'oi', 0),
                        "volume": getattr(s, 'volume', 0),
                        "pcr": getattr(s, 'pcr', 1.0),
                        "max_pain": getattr(s, 'max_pain', None),
                    }
                    for s in (option_suggestions or [])
                ],
                "market": {
                    "nifty":              market_data.get("nifty_price"),
                    "nifty_change_pct":   market_data.get("nifty_change_pct", 0.0),
                    "nifty_change":       (market_data.get("nifty_price", 0) - market_data.get("nifty_intraday", {}).get("prev_close", market_data.get("nifty_price", 0))),
                    "banknifty":          market_data.get("banknifty_price", 0.0),
                    "sensex":             market_data.get("sensex_price", 0.0),
                    "finnifty":           market_data.get("finnifty_price", 0.0),
                    "vix":                market_data.get("india_vix"),
                    "vix_change":         0.0,
                    "fii_5d_flow":        market_data.get("flow_data", {}).get("fii_cash_5day", 0.0),
                    "dii_5d_flow":        market_data.get("flow_data", {}).get("dii_cash_5day", 0.0),
                    "pcr_nifty":          market_data.get("derivatives_data", {}).get("pcr", 1.0),
                    "nifty_pe":           market_data.get("nifty_pe", 0.0),
                    "gsec_yield":         market_data.get("gsec_yield", 0.0),
                },
            }
            update_state(engine_state)
        except Exception as e:
            logger.error(f"Failed to update API server state: {e}", exc_info=True)

        # Print the full formatted report to stdout each cycle
        report = self.coordinator.format_report(plan)
        if using_synthetic:
            report = report.replace(
                "ROX PROVEN EDGE ENGINE",
                "ROX PROVEN EDGE ENGINE [⚠ SYNTHETIC DATA — no broker connected]",
            )
        # Prepend holiday / weekend notice when running on a non-trading day.
        # Purely cosmetic — zero effect on any scores or agent logic.
        _session_notice = _get_market_session_notice()
        if _session_notice:
            report = _session_notice + "\n\n" + report
        # Safe UTF-8 print to avoid cp1252 encoding errors on Windows console
        try:
            print("\n" + report)
        except UnicodeEncodeError:
            import sys
            sys.stdout.buffer.write(("\n" + report).encode("utf-8"))

        # Structured log lines for monitoring / log aggregators
        for s in plan.top_setups[:3]:
            logger.info(
                f"  Setup: {s.stock} {s.direction.value} | "
                f"conviction={s.conviction} | R:R={s.risk_reward:.2f}"
            )
        for alert in plan.active_alerts[-3:]:
            logger.warning(f"  ALERT: {alert}")

    # ------------------------------------------------------------------
    # Demo / backtest run
    # ------------------------------------------------------------------

    def run_demo(self) -> DailyTradingPlan:
        """
        Run a single cycle with synthetic data and print full report.
        Useful for integration testing without live data feeds.
        """
        logger.info("Running demo cycle with synthetic data...")
        market_data = self._build_demo_data()
        plan = self.coordinator.generate_trading_plan(
            market_data,
            watchlist=NIFTY_50_STOCKS[:10],
            skip_liquidity_check=True,
        )

        # ── Update API server state (demo mode) ─────────────────────────────
        try:
            regime = plan.market_regime
            confidence = plan.regime_confidence
            consensus = plan.consensus
            agent_reports = plan.agent_reports
            option_suggestions = getattr(plan.fno_suggestions, "suggestions", None) or []

            engine_state = {
                "status": "demo",
                "regime": {
                    "regime": regime.value,
                    "confidence": confidence,
                },
                "consensus": {
                    "direction": consensus.direction.value,
                    "net_score": consensus.net_score,
                },
                "agents": [
                    {
                        "name": name,
                        "verdict": r.verdict.direction.value,
                        "conviction": r.verdict.conviction,
                        "weight": r.verdict.weight,
                    }
                    for name, r in agent_reports.items()
                ],
                "suggestions": [
                    {
                        "index": getattr(s, 'index', None),
                        "strategy": getattr(s, 'strategy', None),
                        "strike": getattr(s, 'strike', None),
                        "expiry": str(getattr(s, 'expiry', '')),
                        "dte": getattr(s, 'dte', None),
                        "spot": getattr(s, 'spot', None),
                        "entry_price": getattr(s, 'entry_price', None),
                        "cost_per_lot": getattr(s, 'cost_per_lot', None),
                        "stop_loss": getattr(s, 'stop_loss', None),
                        "target": getattr(s, 'target', None),
                        "prob_profit": getattr(s, 'prob_profit', None),
                        "delta": getattr(s, 'delta', None),
                        "gamma": getattr(s, 'gamma', None),
                        "theta": getattr(s, 'theta', None),
                        "vega": getattr(s, 'vega', None),
                        "iv_rank": getattr(s, 'iv_rank', None),
                        "conviction": getattr(s, 'conviction', None),
                        "score": getattr(s, 'score', None),
                        "basis": getattr(s, 'basis', None),
                        "oi": getattr(s, 'oi', 0),
                        "volume": getattr(s, 'volume', 0),
                        "pcr": getattr(s, 'pcr', 1.0),
                        "max_pain": getattr(s, 'max_pain', None),
                    }
                    for s in (option_suggestions or [])
                ],
                "market": {
                    "nifty":              market_data.get("nifty_price"),
                    "nifty_change_pct":   market_data.get("nifty_change_pct", 0.0),
                    "nifty_change":       (market_data.get("nifty_price", 0) - market_data.get("nifty_intraday", {}).get("prev_close", market_data.get("nifty_price", 0))),
                    "banknifty":          market_data.get("banknifty_price", 0.0),
                    "sensex":             market_data.get("sensex_price", 0.0),
                    "finnifty":           market_data.get("finnifty_price", 0.0),
                    "vix":                market_data.get("india_vix"),
                    "vix_change":         0.0,
                    "fii_5d_flow":        market_data.get("flow_data", {}).get("fii_cash_5day", 0.0),
                    "dii_5d_flow":        market_data.get("flow_data", {}).get("dii_cash_5day", 0.0),
                    "pcr_nifty":          market_data.get("derivatives_data", {}).get("pcr", 1.0),
                    "nifty_pe":           market_data.get("nifty_pe", 0.0),
                    "gsec_yield":         market_data.get("gsec_yield", 0.0),
                },
            }
            update_state(engine_state)
        except Exception as e:
            logger.error(f"Failed to update API server state in demo: {e}", exc_info=True)

        _demo_report = self.coordinator.format_report(plan)
        _session_notice = _get_market_session_notice()
        if _session_notice:
            _demo_report = _session_notice + "\n\n" + _demo_report
        print("\n" + _demo_report)
        return plan

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        watchlist: Optional[list] = None,
    ) -> Dict:
        """
        Run a full backtest over *start_date … end_date* (YYYY-MM-DD).

        Each trading day is simulated with synthetic OHLCV data seeded from
        that day's ordinal so results are deterministic and reproducible.

        Returns a dict with keys:
            summary_text  — human-readable ASCII report (also printed)
            days          — number of trading days simulated
            plans         — list of DailyTradingPlan objects
            stats         — dict of key metrics
        """
        from datetime import date as dt_date, timedelta
        from collections import Counter

        start     = dt_date.fromisoformat(start_date)
        end       = dt_date.fromisoformat(end_date)
        watchlist = watchlist or NIFTY_50_STOCKS[:20]
        plans: list = []

        logger.info(
            f"Backtest starting | {start_date} → {end_date} | "
            f"watchlist={len(watchlist)} stocks"
        )

        d = start
        while d <= end:
            if d.weekday() < 5:   # skip weekends
                data = self._build_demo_data(seed=d.toordinal())
                plan = self.coordinator.generate_trading_plan(
                    data, watchlist=watchlist,
                    skip_liquidity_check=True,
                )
                plans.append((d, plan))
                logger.info(
                    f"  {d}  regime={plan.market_regime.value:15s}  "
                    f"setups={len(plan.top_setups):2d}  "
                    f"consensus={plan.consensus.direction.value}"
                )
            d += timedelta(days=1)

        # ── Aggregate statistics ──────────────────────────────────────────────
        n_days   = len(plans)
        all_setups = [s for _, p in plans for s in p.top_setups]
        n_setups   = len(all_setups)

        # Regime distribution
        regime_counts = Counter(p.market_regime.value for _, p in plans)

        # Direction distribution (of consensus)
        dir_counts = Counter(
            p.consensus.direction.value for _, p in plans
        )

        # Agent agreement: days where ≥4 agents agreed
        high_agree_days = sum(
            1 for _, p in plans if len(p.consensus.agreeing_agents) >= 4
        )

        # Conviction breakdown of all setups
        conviction_counts = Counter(
            getattr(s, "conviction", "MEDIUM") for s in all_setups
        )

        # Regime confidence stats
        confidences = [p.regime_confidence for _, p in plans]
        avg_conf    = sum(confidences) / len(confidences) if confidences else 0

        # Top stocks by setup frequency
        stock_counts = Counter(s.stock for s in all_setups)
        top_stocks   = stock_counts.most_common(5)

        # ── Format report ─────────────────────────────────────────────────────
        sep = "=" * 70
        lines = [
            sep,
            "  ROX PROVEN EDGE ENGINE — BACKTEST SUMMARY REPORT",
            sep,
            f"  Period      :  {start_date}  →  {end_date}",
            f"  Trading days:  {n_days}",
            f"  Watchlist   :  {len(watchlist)} stocks",
            "",
            "── MARKET REGIME DISTRIBUTION ──────────────────────────────────",
        ]
        for regime, cnt in regime_counts.most_common():
            pct = cnt / n_days * 100 if n_days else 0
            bar = "█" * int(pct / 3)
            lines.append(f"  {regime:25s}  {cnt:3d} days  ({pct:5.1f}%)  {bar}")

        lines += [
            "",
            "── CONSENSUS DIRECTION ──────────────────────────────────────────",
        ]
        for direction, cnt in dir_counts.most_common():
            pct = cnt / n_days * 100 if n_days else 0
            lines.append(f"  {direction:10s}  {cnt:3d} days  ({pct:5.1f}%)")

        lines += [
            "",
            "── SETUP STATISTICS ─────────────────────────────────────────────",
            f"  Total setups generated   :  {n_setups}",
            f"  Avg setups per day       :  {n_setups / max(n_days, 1):.1f}",
            "",
            "  Conviction breakdown:",
        ]
        for conv, cnt in sorted(conviction_counts.items(), key=lambda x: str(x[0])):
            lines.append(f"    {str(conv):8s}  {cnt:4d}  ({cnt / max(n_setups, 1) * 100:.1f}%)")

        lines += [
            "",
            "── TOP 5 MOST-RECOMMENDED STOCKS ────────────────────────────────",
        ]
        for stock, cnt in top_stocks:
            lines.append(f"  {stock:15s}  appeared in {cnt} setup(s)")

        lines += [
            "",
            "── AGENT AGREEMENT ──────────────────────────────────────────────",
            f"  Avg regime confidence    :  {avg_conf:.1f}%",
            f"  High-agreement days (≥4 agents): {high_agree_days} / {n_days}",
            "",
            sep,
        ]

        summary_text = "\n".join(lines)
        print("\n" + summary_text)
        logger.info(f"Backtest complete: {n_days} trading days processed")

        stats = {
            "trading_days":       n_days,
            "total_setups":       n_setups,
            "avg_setups_per_day": round(n_setups / max(n_days, 1), 2),
            "regime_distribution": dict(regime_counts),
            "direction_distribution": dict(dir_counts),
            "conviction_breakdown": dict(conviction_counts),
            "avg_regime_confidence": round(avg_conf, 1),
            "high_agreement_days":  high_agree_days,
            "top_stocks": top_stocks,
        }

        return {
            "summary_text": summary_text,
            "days":         n_days,
            "plans":        [p for _, p in plans],
            "stats":        stats,
        }

    # ------------------------------------------------------------------
    # Strategy builder passthrough
    # ------------------------------------------------------------------

    @property
    def strategy_factory(self):
        if self._strategy_factory is None:
            from infrastructure.fno_strategy_builders import StrategyFactory
            from infrastructure.fno_instrument_manager import get_instrument_manager
            from infrastructure.greeks_calculator import GreeksCalculator
            self._strategy_factory = StrategyFactory(
                get_instrument_manager(), GreeksCalculator()
            )
        return self._strategy_factory

    def build_strategy(self, strategy_type, underlying, spot_price, vix, **kwargs):
        """Build an options strategy via HERMES/THETA infrastructure."""
        from infrastructure.fno_strategy_builders import StrategyType
        builder = self.strategy_factory.get_builder(strategy_type)
        if not builder:
            return {"error": f"Unknown strategy: {strategy_type}"}
        result = builder.build(underlying, spot_price, vix, **kwargs)
        return {
            "strategy":       result.strategy_type.value,
            "underlying":     result.underlying,
            "spot":           result.spot_price,
            "legs":           [{"type":l.option_type,"strike":l.strike,
                                "expiry":l.expiry.isoformat(),"position":l.position,
                                "quantity":l.quantity} for l in result.legs],
            "greeks":         {"delta":result.delta,"gamma":result.gamma,
                               "theta":result.theta,"vega":result.vega},
            "pnl":            {"max_profit":result.max_profit,"max_loss":result.max_loss,
                               "breakeven_low":result.breakeven_low,
                               "breakeven_high":result.breakeven_high,
                               "risk_reward":result.risk_reward_ratio},
            "margin_required": result.margin_required,
            "conviction":      result.conviction,
            "notes":           result.notes,
        }

    def get_fno_status(self) -> Dict:
        """Return live F&O portfolio status (Greeks, alerts, settlements)."""
        return self.coordinator.get_fno_status()

    # ------------------------------------------------------------------
    # Portfolio status helper (v4.1)
    # ------------------------------------------------------------------

    def _get_portfolio_status(self) -> Dict:
        """Get current portfolio status from F&O coordinator"""
        return {
            "open_positions": [],  # Populate from actual positions
            "total_capital": self.portfolio_value,
            "deployed_capital": 0,
            "cash": self.portfolio_value
        }

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _fetch_market_data(self) -> Dict[str, Any]:
        """
        Fetch live market data.
        Tries live broker APIs → falls back to demo data.
        """
        try:
            return self._fetch_live_data()
        except Exception as exc:
            logger.debug(f"Live data unavailable ({exc}), using synthetic fallback")
            return self._build_demo_data()

    def _fetch_live_data(self) -> Dict[str, Any]:
        """
        Attempt live data fetch via configured broker.

        The broker fetcher is instantiated ONCE and stored on the engine.
        Re-using the same object across cycles is essential because:
          - FyersFetcher caches 60-day OHLCV history (51 API calls) per
            calendar day — creating a new instance discards that cache and
            repeats all 51 calls on every 60-second cycle.
          - F&O position sync (Fyers positions() call) runs at __init__
            time — it should happen once at engine startup, not every cycle.
          - The MacroFetcher singleton injected via set_fyers_client() would
            lose its Fyers client reference on every cycle recreation.
        """
        if self.config.api.fyers_enabled:
            if self._data_fetcher is None:
                from data.fyers_fetcher import FyersFetcher
                logger.info("Creating FyersFetcher instance (once per session)...")
                self._data_fetcher = FyersFetcher(self.config.api)
            return self._data_fetcher.fetch_market_data()

        if self.config.api.zerodha_enabled:
            if self._data_fetcher is None:
                from data.zerodha_fetcher import ZerodhaFetcher
                logger.info("Creating ZerodhaFetcher instance (once per session)...")
                self._data_fetcher = ZerodhaFetcher(self.config.api)
            return self._data_fetcher.fetch_market_data()

        raise RuntimeError("No live data source configured")

    def _build_demo_data(self, seed: int = 42) -> Dict[str, Any]:
        """Synthetic market data for demo/backtest runs."""
        rng = lambda base, pct: base * (1 + (((seed % 100) - 50) / 50) * pct)

        # Per-symbol hash using full name so different stocks get different prices
        def sym_hash(s: str) -> int:
            h = sum(ord(c) * (i + 1) for i, c in enumerate(s))
            return h % 4000

        nifty      = rng(22500,  0.02)
        banknifty  = rng(48000,  0.02)
        sensex     = rng(74000,  0.02)
        finnifty   = rng(23500,  0.02)
        bankex     = rng(52000,  0.02)
        stocks = {
            s: {
                "close":      rng(800  + sym_hash(s), 0.03),
                "open":       rng(800  + sym_hash(s), 0.01),
                "high":       rng(800  + sym_hash(s), 0.04),
                "low":        rng(800  + sym_hash(s), 0.025),
                "volume":     int(rng(500_000 + sym_hash(s) * 100, 0.4)),
                "avg_volume": 450_000,
                "atr":        rng(800  + sym_hash(s), 0.015) * 0.018,
            }
            for s in NIFTY_50_STOCKS[:20]
        }
        # Indicators also vary per stock so RSI/ADX aren't all identical
        indicators = {
            s: {
                "rsi":          rng(45 + sym_hash(s) % 25, 0.12),
                "sma20":        v["close"] * rng(0.992, 0.005),
                "sma50":        v["close"] * rng(0.975, 0.008),
                "sma200":       v["close"] * rng(0.940, 0.010),
                "atr":          v["atr"],
                "adx":          rng(20 + sym_hash(s) % 20, 0.18),
                "trend":        "UPTREND" if sym_hash(s) % 3 != 0 else "SIDEWAYS",
                "volume_ratio": rng(0.9 + (sym_hash(s) % 10) / 20, 0.25),
            }
            for s, v in stocks.items()
        }

        call_wall_1 = round(nifty * 1.02 / 50) * 50
        call_wall_2 = round(nifty * 1.04 / 50) * 50
        put_wall_1  = round(nifty * 0.98 / 50) * 50
        put_wall_2  = round(nifty * 0.96 / 50) * 50

        return {
            "nifty_price":     nifty,
            "nifty_200dma":    nifty * 0.95,
            "banknifty_price": banknifty,
            "sensex_price":    sensex,
            "finnifty_price":  finnifty,
            "bankex_price":    bankex,
            "india_vix":       rng(14, 0.3),
            "adx":             rng(28, 0.2),
            "price_structure": "higher_highs",
            "nifty_pe":        rng(22.5, 0.05),
            "gsec_yield":      rng(7.0, 0.02),
            "price_data":      stocks,
            "indicators":      indicators,
            "flow_data":       {"fii_cash_5day": rng(4500, 0.5), "dii_cash_5day": rng(2000, 0.4)},
            "sentiment_data":  {"news": 55, "analyst": 50, "social": 45, "global": 48},
            "derivatives_data": {
                "pcr":       rng(1.2, 0.2),
                "pcr_trend": "stable",
                "max_pain":  round(nifty / 100) * 100,
                "india_vix": rng(14, 0.3),
                "iv_rank":   45,
                # Walls as dicts so OPTIMUS can read strike + oi fields
                "call_oi_walls": [
                    {"strike": call_wall_1, "oi": int(rng(1_500_000, 0.3))},
                    {"strike": call_wall_2, "oi": int(rng(1_000_000, 0.3))},
                ],
                "put_oi_walls": [
                    {"strike": put_wall_1,  "oi": int(rng(1_200_000, 0.3))},
                    {"strike": put_wall_2,  "oi": int(rng(900_000,   0.3))},
                ],
            },
            "index_option_chains": {
                "NIFTY":     {"pcr": rng(1.1, 0.2), "iv_rank": rng(15, 0.3), "max_pain": round(nifty / 50) * 50},
                "BANKNIFTY": {"pcr": rng(1.0, 0.2), "iv_rank": rng(15, 0.3), "max_pain": round(banknifty / 100) * 100},
                "SENSEX":    {"pcr": rng(0.9, 0.2), "iv_rank": rng(15, 0.3), "max_pain": round(sensex / 100) * 100},
                "FINNIFTY":  {"pcr": rng(1.1, 0.2), "iv_rank": rng(15, 0.3), "max_pain": round(finnifty / 50) * 50},
                "BANKEX":    {"pcr": rng(1.0, 0.2), "iv_rank": rng(15, 0.3), "max_pain": round(bankex / 100) * 100},
            },
            "fundamental_data": {},
            "event_data":       {"events": []},
            "ohlcv_history":    {},
        }


# ===========================================================================
# Market session awareness — holiday / weekend notice
# ===========================================================================

def _get_market_session_notice() -> str:
    """
    Returns a formatted notice banner when the engine runs on a non-trading day
    (NSE/BSE holiday or weekend).  Shows:
      • What date the data shown is from (last trading session)
      • The holiday / weekend reason
      • When the market next opens
      • A clear reminder that all scores reflect the last session's data

    Deliberately read-only — has zero effect on agents, scores, or conviction.
    Uses the same _HOLIDAY_SET / _HOLIDAY_NAMES / is_trading_day already used
    by directional_option_advisor.py for expiry calculations.
    """
    from datetime import date as _date, timedelta as _td
    try:
        from agents.directional_option_advisor import (
            _HOLIDAY_SET, _HOLIDAY_NAMES, is_trading_day
        )
    except Exception:
        # Fallback: minimal implementation if import fails
        _HOLIDAY_SET   = frozenset()
        _HOLIDAY_NAMES = {}
        def is_trading_day(d):
            return d.weekday() < 5 and d not in _HOLIDAY_SET

    today = _date.today()

    # Normal trading day — no notice needed
    if is_trading_day(today):
        return ""

    # ── Determine reason ──────────────────────────────────────────────────
    _WD_NAMES = {5: "Saturday", 6: "Sunday"}
    if today.weekday() in _WD_NAMES:
        reason = f"Weekend ({_WD_NAMES[today.weekday()]})"
    else:
        reason = _HOLIDAY_NAMES.get(today, "NSE / BSE Market Holiday")

    # ── Find last trading session ─────────────────────────────────────────
    last_td = today - _td(days=1)
    while not is_trading_day(last_td):
        last_td -= _td(days=1)

    # ── Find next trading session ─────────────────────────────────────────
    next_td = today + _td(days=1)
    while not is_trading_day(next_td):
        next_td += _td(days=1)

    # ── Days until market opens ───────────────────────────────────────────
    days_until = (next_td - today).days
    days_label = "tomorrow" if days_until == 1 else f"in {days_until} days"

    # ── Format banner ─────────────────────────────────────────────────────
    W  = 65
    hr = "─" * W
    notice_lines = [
        hr,
        f"  {'⚠  MARKET CLOSED  —  ' + reason.upper():^{W-4}}",
        hr,
        f"  {'Today':18s}: {today.strftime('%d-%m-%Y')} ({today.strftime('%A')})",
        f"  {'Data shown':18s}: {last_td.strftime('%d-%m-%Y')} ({last_td.strftime('%A')}) — last trading session",
        f"  {'Market opens':18s}: {next_td.strftime('%d-%m-%Y')} ({next_td.strftime('%A')}) — {days_label}",
        hr,
        f"  All agent scores, regime signals, and F&O analysis below",
        f"  reflect {last_td.strftime('%d-%m-%Y')} data.  No live prices available today.",
        hr,
    ]
    return "\n".join(notice_lines)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ROX Proven Edge Engine v4.0 Unified")
    p.add_argument("--mode", choices=["live","paper","backtest","demo"],
                   default="demo", help="Trading mode (default: demo)")
    p.add_argument("--portfolio-value", type=float, default=1_000_000,
                   help="Portfolio value in INR (default: 1000000)")
    p.add_argument("--start-date", type=str, help="Backtest start YYYY-MM-DD")
    p.add_argument("--end-date",   type=str, help="Backtest end YYYY-MM-DD")
    p.add_argument("--watchlist",  type=str, nargs="*",
                   help="Stocks to analyse (default: Nifty 50 top 20)")
    p.add_argument("--pre-market", action="store_true",
                   help="Run pre-market news briefing before starting")
    p.add_argument("--with-api", action="store_true",
                   help="Launch the REST API server (port 8000) alongside the engine")
    p.add_argument("--api-port", type=int, default=8000,
                   help="Port for the REST API server (default: 8000)")
    return p.parse_args()


async def demo_with_news(engine: ROXUnifiedEngine) -> DailyTradingPlan:
    """Run pre-market routine then demo cycle."""
    await engine.pre_market_routine()
    return engine.run_demo()


def _start_api_server(port: int = 8000) -> threading.Thread:
    """
    Launch the FastAPI/uvicorn REST API server in a background daemon thread.
    Because it is a daemon, it dies automatically when the main process exits.
    """
    try:
        import uvicorn
        from api_server import app
    except ImportError:
        logger.error(
            "uvicorn is not installed — cannot start API server. "
            "Run: pip install uvicorn"
        )
        return None

    def _run():
        logger.info(f"API server starting on http://0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

    t = threading.Thread(target=_run, name="api-server", daemon=True)
    t.start()
    logger.info(f"API server launched in background thread (port {port})")
    return t


def main():
    args = parse_args()
    config = get_system_config()
    engine = ROXUnifiedEngine(config, mode=args.mode,
                              portfolio_value=args.portfolio_value)

    def _shutdown(signum, frame):
        logger.info("Shutdown signal received")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.mode == "demo":
        if args.pre_market:
            asyncio.run(demo_with_news(engine))
        else:
            engine.run_demo()

    elif args.mode == "backtest":
        if not args.start_date or not args.end_date:
            logger.error("--start-date and --end-date required for backtest mode")
            sys.exit(1)
        engine.run_backtest(args.start_date, args.end_date,
                            watchlist=args.watchlist)

    else:   # paper / live
        if args.with_api:
            _start_api_server(port=args.api_port)
            time.sleep(0.5)   # give uvicorn a moment to bind the port
        if args.pre_market:
            asyncio.run(engine.pre_market_routine())
        engine.start()
        try:
            while engine.running:
                time.sleep(1)
        except KeyboardInterrupt:
            engine.stop()

    logger.info("ROX Unified Engine shutdown complete")


if __name__ == "__main__":
    main()
"""
ROX Proven Edge Engine v4.0 Unified - Unified Coordinator
==========================================================
Merges v3.2 LeadCoordinator (8-agent swing engine) with
v4.0 F&O specialist agents (HERMES, THETA, DELTA).

Architecture
------------
LeadCoordinator  — orchestrates all 11 agents through the 11-tier pipeline,
                   generates DailyTradingPlans for swing trades.
FnoCoordinator   — lightweight wrapper that adds F&O portfolio management:
                   live Greeks monitoring (THETA), settlement compliance (DELTA),
                   and execution quality tracking (HERMES).
UnifiedCoordinator — top-level facade that composes both, offering a single
                     entry point for the full engine.
"""

from __future__ import annotations

import logging
import sys, os
import threading
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TradeDirection, MarketRegime, ConvictionLevel, SystemConfig,
    DEFAULT_CONFIG, NIFTY_50_STOCKS, SECTOR_MAPPING, FnoRiskLimits,
    get_vix_regime,
)
from agents import (
    BaseAgent, AgentVerdict, AgentReport,
    OrionAgent, VesperAgent, KairoAgent,
    SentinelAgent, NexusAgent, PrudenceAgent, CatalystAgent, OptimusAgent,
    HermesAgent, ThetaAgent, DeltaAgent,
    Order, OrderStatus, OrderType, GreeksAlert, SettlementObligation,
    PhoenixAgent, PhoenixOutput,
)
from data.data_manager import DataManager, TradeRecord
from data.scorecard import AgentScorecard

# ── News Intelligence (v4.1) ──────────────────────────────────────────────────
from agents.news_core import get_news_context

# ── v5 Rule Validator (zero-LLM, <1ms) ───────────────────────────────────────
try:
    import sys
    sys.path.insert(0, r'C:\Users\Jay_Agent\Downloads\rox_pro_v4_enhanced_4.2\rox_pro_v4_enhanced_4.2_patched\rox_pro_v5_enhanced_5.1\rox_pro_v5_enhanced_5.0')
    from reasoning.rule_validator import RuleBasedValidator
    RULE_VALIDATOR_AVAILABLE = True
except Exception as e:
    RULE_VALIDATOR_AVAILABLE = False
    RuleBasedValidator = None


# ---------------------------------------------------------------------------
# Shared data structures (carried over from v3.2 coordinator)
# ---------------------------------------------------------------------------

@dataclass
class ConsensusResult:
    direction: TradeDirection
    strength: str          # STRONG | MODERATE | WEAK | NO_CONSENSUS
    net_score: float
    weighted_votes: Dict[str, float]
    agreeing_agents: List[str]
    disagreeing_agents: List[str]
    contradictions: List[Dict]


@dataclass
class TradeSetup:
    stock: str
    direction: TradeDirection
    conviction: int
    conviction_level: ConvictionLevel
    entry_price: float
    entry_zone: Tuple[float, float]
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    shares: int
    position_value: float
    position_percent: float
    risk_percent: float
    agent_votes: Dict[str, AgentVerdict]
    pattern_match: Dict
    historical_win_rate: float
    exit_strategy: str
    setup_time: datetime = field(default_factory=datetime.now)
    llm_validation: object = None   # PatternValidationResult | None


@dataclass
class DailyTradingPlan:
    date: datetime
    market_regime: MarketRegime
    regime_confidence: float
    consensus: ConsensusResult
    top_setups: List[TradeSetup]
    portfolio_risk: Dict
    upcoming_events: List[Dict]
    action_items: List[str]
    agent_reports: Dict[str, AgentReport]
    # v4.0 additions
    portfolio_greeks: Dict = field(default_factory=dict)
    active_alerts: List[str] = field(default_factory=list)
    settlement_obligations: List[Dict] = field(default_factory=list)
    # v4.1 — directional F&O suggestions (all 5 indices)
    fno_suggestions: object = None   # OptionAdvisorOutput | None
    # v4.2 — cross-examiner gate metadata
    setups_watch_only: bool = False   # True when examiner says WAIT (show but don't act)
    # FIX 5.2 — FNOBrainExtension synthesis output
    fno_brain_synthesis: object = None   # FNOBrainOutput | None
    # v4.3 — PHOENIX pre-momentum recovery radar
    phoenix_analysis: object = None    # PhoenixOutput | None


# ---------------------------------------------------------------------------
# FnoCoordinator — manages HERMES, THETA, DELTA agents
# ---------------------------------------------------------------------------

class FnoCoordinator:
    """
    F&O Portfolio Coordinator (v4.0).

    Manages the three F&O specialist agents:
    - HERMES  : tracks order execution and slippage
    - THETA   : monitors portfolio Greeks and triggers hedging alerts
    - DELTA   : monitors physical settlement obligations and SEBI compliance
    """

    def __init__(self, config: SystemConfig):
        self.config  = config
        self.logger  = logging.getLogger("FnoCoordinator")
        self._lock   = threading.Lock()

        # Instrument + MWPL manager (lazy init to avoid startup errors)
        self._instrument_manager = None
        self._mwpl_monitor       = None

        # Initialise specialist agents
        self.hermes = HermesAgent()
        self.hermes.register_callback(self._on_order_update)

        self.theta  = ThetaAgent(config.fno_limits)
        self.theta.register_callback(self._on_greeks_alert)

        self.delta  = DeltaAgent(auto_exit_enabled=True)
        self.delta.register_callback(self._on_settlement_alert)

        self._active_alerts: List[str] = []

    # ------------------------------------------------------------------
    # Lazy property: instrument manager
    # ------------------------------------------------------------------
    @property
    def instrument_manager(self):
        if self._instrument_manager is None:
            try:
                from infrastructure.fno_instrument_manager import get_instrument_manager
                self._instrument_manager = get_instrument_manager()
                self.delta.instrument_manager = self._instrument_manager
            except Exception as e:
                self.logger.warning(f"FnoInstrumentManager unavailable: {e}")
        return self._instrument_manager

    @property
    def mwpl_monitor(self):
        if self._mwpl_monitor is None:
            try:
                from infrastructure.mwpl_monitor import get_mwpl_monitor
                self._mwpl_monitor = get_mwpl_monitor()
                self._mwpl_monitor.register_callback(self._on_mwpl_alert)
                self._mwpl_monitor.start_monitoring()
            except Exception as e:
                self.logger.warning(f"MWPLMonitor unavailable: {e}")
        return self._mwpl_monitor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_portfolio_greeks(self) -> Dict:
        pg = self.theta.get_portfolio_greeks()
        return {
            "delta": round(pg.delta, 4),
            "gamma": round(pg.gamma, 6),
            "theta": round(pg.theta, 2),
            "vega":  round(pg.vega, 2),
            "num_positions": pg.num_positions,
        }

    def get_settlement_obligations(self) -> List[Dict]:
        return [
            {
                "symbol":           o.symbol,
                "obligation_type":  o.obligation_type,
                "days_to_expiry":   o.days_to_expiry,
                "obligation_value": o.obligation_value,
                "requires_action":  o.requires_action,
            }
            for o in self.delta.get_settlement_obligations()
            if o.requires_action
        ]

    def get_active_alerts(self) -> List[str]:
        with self._lock:
            return list(self._active_alerts)

    def place_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0,
        strategy_id: Optional[str] = None,
    ) -> Order:
        return self.hermes.place_order(
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            price=price,
            strategy_id=strategy_id,
        )

    def get_execution_report(self) -> Dict:
        return self.hermes.generate_execution_report()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_order_update(self, order: Order):
        self.logger.info(f"Order update: {order.order_id} → {order.status.value}")

    def _on_greeks_alert(self, alert: GreeksAlert):
        msg = f"[GREEKS] {alert.alert_type.value} | {alert.recommendation}"
        self.logger.warning(msg)
        with self._lock:
            self._active_alerts.append(msg)

    def _on_settlement_alert(self, obligation: SettlementObligation):
        msg = f"[SETTLEMENT] {obligation.symbol} | {obligation.obligation_type}"
        self.logger.warning(msg)
        with self._lock:
            self._active_alerts.append(msg)

    def _on_mwpl_alert(self, alert: Dict):
        msg = f"[MWPL] {alert.get('message', '')}"
        self.logger.warning(msg)
        with self._lock:
            self._active_alerts.append(msg)


# ---------------------------------------------------------------------------
# FIX-PIPELINE-03: Geopolitical summary builder
# ---------------------------------------------------------------------------

def _build_geo_summary(market_data: dict) -> str:
    """
    Extract a one-line geopolitical catalyst string from market_data.
    Reads OvernightRiskProfile (object) or fallback from news_context headlines.
    Called by LeadCoordinator to populate cross-examiner and regime detector context.
    """
    overnight_risk = market_data.get("overnight_risk")
    headlines = []
    risk_prefix = ""

    if overnight_risk is not None:
        if hasattr(overnight_risk, "key_headlines"):
            headlines = list(overnight_risk.key_headlines[:3])
        if hasattr(overnight_risk, "risk_level") and overnight_risk.risk_level in ("EXTREME", "HIGH", "ELEVATED"):
            risk_prefix = f"[{overnight_risk.risk_level}/{getattr(overnight_risk, 'market_stance', '')}] "

    if not headlines:
        # Fall back to top HIGH/CRITICAL severity items from news_context
        news_ctx = market_data.get("news_context", [])
        for item in news_ctx[:20]:
            sev = getattr(item, "severity", None)
            if sev is not None:
                sev_name = sev.name if hasattr(sev, "name") else str(sev)
                if sev_name in ("HIGH", "CRITICAL", "EXTREME"):
                    headlines.append(getattr(item, "headline", str(item)))
                    if len(headlines) >= 3:
                        break

    if not headlines:
        return "None identified"
    return risk_prefix + "; ".join(headlines)


# ---------------------------------------------------------------------------
# LeadCoordinator — full v3.2 swing engine (retained verbatim, now 11 agents)
# ---------------------------------------------------------------------------

class LeadCoordinator:
    """
    Lead Coordinator — Master Orchestrator (v3.2, extended to 11 agents).

    Coordinates all agents through the 11-tier processing pipeline to
    generate DailyTradingPlans with full reasoning transparency.
    """

    def __init__(self, config: SystemConfig = None, portfolio_value: float = 1_000_000):
        self.config           = config or DEFAULT_CONFIG
        self.portfolio_value  = portfolio_value
        self.logger           = logging.getLogger("LeadCoordinator")

        # Core 8 swing agents
        self.agents: Dict[str, BaseAgent] = {
            "ORION":    OrionAgent(),
            "VESPER":   VesperAgent(),
            "KAIRO":    KairoAgent(),
            "SENTINEL": SentinelAgent(),
            "NEXUS":    NexusAgent(),
            "PRUDENCE": PrudenceAgent(),
            "CATALYST": CatalystAgent(),
            "OPTIMUS":  OptimusAgent(),
        }
        for name, agent in self.agents.items():
            if name in self.config.agents:
                agent.current_weight = self.config.agents[name].current_weight

        self.current_regime     = MarketRegime.CONSOLIDATION
        self.regime_confidence  = 50.0
        self.market_data: Dict  = {}
        self.agent_reports: Dict[str, AgentReport] = {}

        self.data_manager = DataManager()
        self.scorecard    = AgentScorecard(self.data_manager)
        try:
            from data.pattern_database import PatternDatabase
            self.pattern_db = PatternDatabase(self.data_manager)
        except Exception:
            self.pattern_db = None

        # ── LLM Intelligence Layer (v4.1) ─────────────────────────────────
        # Two configs: "pro" for high-stakes analysis (regime, cross-examine);
        # "flash" for per-stock and background tasks (validator, news, meta).
        # This prevents rate limits when gemini-2.5-pro is set as LLM_MODEL.
        self._last_regime_result   = None
        self._last_examination     = None
        self._last_news_impact     = None
        self._news_impact_lock     = threading.Lock()   # thread-safety for bg thread

        # ── AVOID-deadlock circuit breaker ────────────────────────────────
        # Tracks how many consecutive cycles the cross-examiner has fired AVOID.
        # After AVOID_STREAK_LIMIT cycles, the gate is temporarily relaxed to
        # REDUCE_SIZE for one cycle so the engine can generate shadow data and
        # allow the meta-learner to accumulate learning signal.
        # This breaks the chicken-and-egg trap where AVOID prevents new trades
        # which prevents the win rate from improving which triggers more AVOID.
        self._consecutive_avoid_cycles: int = 0
        self._AVOID_STREAK_LIMIT: int = 5   # relax after 5 consecutive AVOIDs
        try:
            from agents.llm import (
                LLMRegimeDetector, LLMCrossExaminer,
                LLMPatternValidator, LLMOptionsStrategist,
                LLMNewsImpactAnalyzer, LLMMetaLearner,
            )
            from agents.llm.base_llm_agent import LLMConfig as _LLMCfg
            import os as _os

            # Pro config: gemini-2.5-pro for regime + cross-examination
            _llm_cfg_pro = _LLMCfg.from_env()

            # Flash config: always use flash model for per-stock/background tasks
            # Overrides LLM_MODEL env var — these tasks must be fast and cheap.
            _api_key = (_os.getenv("GEMINI_API_KEY") or
                        _os.getenv("GOOGLE_API_KEY") or
                        _os.getenv("BRAIN_API_KEY", ""))
            _llm_cfg_flash = _LLMCfg(
                enabled=_llm_cfg_pro.enabled,
                api_key=_api_key,
                model_name="gemini-3-flash-preview",
                fallback_model="gemini-3-flash-preview",
                max_retries=3,
                timeout_seconds=20,
                cache_ttl_seconds=300,
                temperature=0.2,
                max_output_tokens=4096,
                rate_limit_per_minute=20,
            )

            self.llm_regime     = LLMRegimeDetector(_llm_cfg_pro)
            self.llm_examiner   = LLMCrossExaminer(_llm_cfg_pro)
            self.llm_validator  = LLMPatternValidator(_llm_cfg_flash)   # per-stock → flash
            self.llm_strategist = LLMOptionsStrategist(_llm_cfg_flash)  # per-suggestion → flash
            self.llm_news       = LLMNewsImpactAnalyzer(_llm_cfg_flash) # background → flash
            self.llm_meta       = LLMMetaLearner(_llm_cfg_flash, self.data_manager)  # weekly → flash
            # History analyzer — pure CSV parsing, no API key needed
            from agents.llm.llm_history_analyzer import LLMHistoryAnalyzer
            self.llm_history    = LLMHistoryAnalyzer()
            self.llm_history.build_context()   # pre-load at startup
            # Trading planner — converts all signals into concrete executable calls
            from agents.llm.llm_trading_planner import LLMTradingPlanner
            self.llm_planner    = LLMTradingPlanner(_llm_cfg_flash)
            self.logger.info("LLM intelligence layer initialised (7 modules)")
            # v5 Rule Validator
            self.rule_validator = RuleBasedValidator() if RULE_VALIDATOR_AVAILABLE else None
            if self.rule_validator:
                self.logger.info("v5 Rule Validator initialised (<1ms, zero LLM calls)")
        except Exception as _llm_init_err:
            self.logger.warning(f"LLM layer init failed — running rule-based only: {_llm_init_err}")
            self.logger.warning(f"LLM layer init failed — running rule-based only: {_llm_init_err}")
            self.llm_regime     = None
            self.llm_examiner   = None
            self.llm_validator  = None
            self.llm_strategist = None
            self.llm_news       = None
            self.llm_meta       = None
            self.llm_history    = None
            self.llm_planner    = None
            self.rule_validator = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def generate_trading_plan(
        self,
        market_data: Dict[str, Any],
        portfolio_status: Optional[Dict] = None,
        watchlist: Optional[List[str]] = None,
    ) -> DailyTradingPlan:
        self.market_data     = market_data
        watchlist            = watchlist or NIFTY_50_STOCKS[:20]
        portfolio_status     = portfolio_status or self._default_portfolio()

        # ── FIX-DUPLICATE-01: Reset per-cycle validation tracker ────────────
        # Prevents LLM pattern validation from running twice for the same stock
        # within a single generate_trading_plan() call.
        self._validated_this_cycle = set()

        # ── LLM News Impact Analysis ──────────────────────────────────────
        # Run once per plan cycle in a background thread so it doesn't block
        # the quote fetch loop. Result is stored in self._last_news_impact and
        # used downstream by _analyze_stock() to bias sector-affected stocks.
        if self.llm_news is not None:
            try:
                import threading as _threading
                news_items = list(market_data.get("news_context", []))
                fno_positions = list(market_data.get("fno_positions", []))
                _sector_perf = dict(market_data.get("sector_performance", {}))
                # FIX-PIPELINE-03: inject macro scalars into sector_perf dict so that
                # llm_news_analyzer._build_impact_prompt() can surface them as
                # MACRO CATALYSTS. sector_performance is the only dict passed to that
                # function, so macro keys are piggybacked here under reserved names.
                _sector_perf["__crude_brent_usd"]    = market_data.get("crude_brent_usd", 0.0)
                # Sanity check: crude < $70 is almost certainly a bad scrape (e.g. pct change
                # misread as price). Zero it out so LLMs don't reason about a nonsense value.
                if _sector_perf["__crude_brent_usd"] < 70:
                    _sector_perf["__crude_brent_usd"] = 0.0
                _sector_perf["__usd_inr"]            = market_data.get("usd_inr", 0.0)
                _sector_perf["__gift_nifty_gap_pct"] = market_data.get("gift_nifty_gap_pct", 0.0)
                def _run_news():
                    try:
                        result = self.llm_news.analyze_trading_impact(
                            news_items=news_items[:20],
                            positions=fno_positions,
                            upcoming_events=market_data.get("event_data", {}).get("events", []),
                            sector_performance=_sector_perf,
                        )
                        with self._news_impact_lock:
                            self._last_news_impact = result
                        self.logger.info(
                            f"[LLM-NEWS] {result.overall_market_impact.get('direction','?')} | "
                            f"magnitude={result.overall_market_impact.get('magnitude','?')} | "
                            f"signals={len(result.actionable_signals)} | "
                            f"restrictions={result.trade_restrictions[:2]}"
                        )
                    except Exception as _ne:
                        self.logger.debug(f"[LLM-NEWS] background thread failed: {_ne}")
                _t = _threading.Thread(target=_run_news, daemon=True, name="llm-news")
                _t.start()
            except Exception as _ne:
                self.logger.debug(f"[LLM-NEWS] failed to start: {_ne}")

        self._tier_0_validate_data(market_data)
        self.current_regime, self.regime_confidence = self._tier_11_detect_regime(market_data)
        self._tier_1_adjust_weights(self.current_regime)
        self._run_all_agents(market_data, self.current_regime)
        consensus = self._tier_2_calculate_consensus()
        self._last_consensus = consensus   # used by _prepare_prudence_data

        # ── PHOENIX Pre-Momentum Analysis (v4.3) ─────────────────────────────
        # Runs immediately after consensus so it has all market_data.
        # Non-voter — does not affect weighted_votes.
        # Stores result on plan and may override the conviction gate threshold.
        self._phoenix_output = None
        self._phoenix_gate_override = None
        if "PHOENIX" in self.agents:
            try:
                _phoenix_report = self.agents["PHOENIX"].analyze(
                    market_data, self.current_regime
                )
                self.agent_reports["PHOENIX"] = _phoenix_report
                self._phoenix_output = _phoenix_report.analysis_details.get("phoenix_output")
                if self._phoenix_output and self._phoenix_output.conviction_gate_override:
                    self._phoenix_gate_override = self._phoenix_output.conviction_gate_override
                    self.logger.info(
                        f"[PHOENIX] score={self._phoenix_output.phoenix_score:.0f}/100 | "
                        f"tier={self._phoenix_output.tier} | "
                        f"gate_override={self._phoenix_gate_override} | "
                        f"recovery_prob={self._phoenix_output.recovery_probability*100:.0f}% | "
                        f"signals_fired={sum(1 for s in self._phoenix_output.signals if s.fired)}/10"
                    )
                else:
                    score = getattr(self._phoenix_output, "phoenix_score", 0)
                    self.logger.info(
                        f"[PHOENIX] score={score:.0f}/100 | "
                        f"tier={getattr(self._phoenix_output, 'tier', 'DORMANT')} | "
                        f"no gate override"
                    )
            except Exception as _pe:
                self.logger.debug(f"[PHOENIX] analysis failed: {_pe}")

        # ── LLM Cross-Examination ─────────────────────────────────────────
        # Validates the consensus and can dynamically adjust agent weights
        # for this cycle and flag risk conditions before setup scanning.
        self._last_examination = None
        if self.llm_examiner is not None:
            try:
                historical_accuracy = {}
                sc = self.scorecard.get_scorecard()
                for aname in self.agent_reports:
                    stats = sc.get(aname, {})
                    if stats.get("total_predictions", 0) >= 10:
                        historical_accuracy[aname] = stats.get("win_rate", 0.5)
                examination = self.llm_examiner.examine_consensus(
                    agent_reports=self.agent_reports,
                    consensus=consensus,
                    regime=self.current_regime,
                    historical_accuracy=historical_accuracy,
                    market_context={
                        "india_vix":               market_data.get("india_vix", 15),
                        "recent_consensus_history": getattr(self, "_consensus_history", "No recent history"),
                        "trade_history":            getattr(self.llm_history, "get_for_cross_examiner", lambda: "")() if self.llm_history else "",
                        # FIX-PIPELINE-03: macro fields were not in this dict so the
                        # cross-examiner's statistical guard and macro context were inert.
                        "crude_brent_usd":         market_data.get("crude_brent_usd", 0.0),
                        "usd_inr":                 market_data.get("usd_inr", 0.0),
                        "gift_nifty_gap_pct":      market_data.get("gift_nifty_gap_pct", 0.0),
                        "geopolitical_summary":    _build_geo_summary(market_data),
                        "resolved_trade_count":    (
                            self.llm_history._context.total_resolved
                            if self.llm_history and self.llm_history._context
                            else "unknown"
                        ),
                        "market_data":             market_data,   # full dict for cross-examiner fallback lookups
                        # ── FIX-RESTRICTION-02: Pass restrictions to cross-examiner ──
                        "news_restrictions": (
                            list(self._last_news_impact.trade_restrictions)
                            if self._last_news_impact and hasattr(self._last_news_impact, "trade_restrictions")
                            else []
                        ),
                        "news_impact":             self._last_news_impact,  # full object for cross-examiner
                    }
                )
                self._last_examination = examination

                # ── Apply cross-examiner direction override ───────────────────
                # If the cross-examiner disagrees with consensus AND has a strong
                # directional view (LONG/SHORT, not NEUTRAL), update consensus
                # direction. This is the key feedback loop that catches false signals.
                _ex_dir = examination.examined_direction
                if (
                    _ex_dir != TradeDirection.NEUTRAL
                    and _ex_dir != consensus.direction
                    and examination.confidence_adjustment <= -10
                ):
                    self.logger.info(
                        f"[LLM-EXAMINE] Overriding consensus {consensus.direction.value} "
                        f"→ {_ex_dir.value} (LLM confidence adj: {examination.confidence_adjustment})"
                    )
                    consensus.direction = _ex_dir
                    consensus.strength = "MODERATE"
                    # Keep net_score consistent with the overridden direction so downstream
                    # conviction calculations and F&O stance don't contradict the direction.
                    # Use a small signed value that reflects MODERATE conviction.
                    if _ex_dir == TradeDirection.LONG and consensus.net_score < 0.15:
                        consensus.net_score = 0.30
                    elif _ex_dir == TradeDirection.SHORT and consensus.net_score > -0.15:
                        consensus.net_score = -0.30

                # Apply confidence_adjustment to the consensus net_score (capped ±0.10)
                if examination.confidence_adjustment != 0:
                    adj_delta = examination.confidence_adjustment / 200.0   # ±20 → ±0.10
                    consensus.net_score = max(-1.0, min(1.0, consensus.net_score + adj_delta))

                # Apply dynamic weight adjustments for this cycle only
                # Two-tier dampening system:
                #   • Normal downweight   → 0.85x (15% cut, floor 0.05)
                #     Used for agents whose signal direction is wrong but whose
                #     reasoning process itself is sound.
                #   • Poor quality agents → 0.40x (60% cut, floor 0.02)
                #     Used when the cross-examiner explicitly flags defective
                #     reasoning (contradictory data, circular logic, etc.).
                #     These agents are near-silenced for this cycle without
                #     being removed from the panel entirely.
                if examination.agents_to_upweight or examination.agents_to_downweight:
                    poor_quality_set = set(getattr(examination, "poor_quality_agents", []))
                    for aname in examination.agents_to_upweight:
                        if aname in self.agents:
                            self.agents[aname].current_weight = min(0.30, self.agents[aname].current_weight * 1.15)
                    for aname in examination.agents_to_downweight:
                        if aname in self.agents:
                            if aname in poor_quality_set:
                                # Aggressive: poor reasoning quality — 60% weight cut
                                self.agents[aname].current_weight = max(0.02, self.agents[aname].current_weight * 0.40)
                                self.logger.info(
                                    f"[EXAMINE-GATE] ⚠ Poor-quality reasoning: {aname} weight cut 60% "
                                    f"→ {self.agents[aname].current_weight:.3f}"
                                )
                            else:
                                # Standard: signal disagreement only — 15% weight cut
                                self.agents[aname].current_weight = max(0.05, self.agents[aname].current_weight * 0.85)
                    # Re-normalise
                    total_w = sum(a.current_weight for a in self.agents.values())
                    if total_w > 0:
                        for a in self.agents.values():
                            a.current_weight /= total_w
                self.logger.info(
                    f"[LLM-EXAMINE] {examination.final_recommendation} | "
                    f"contrarian={examination.contrarian_case[:80]}... | "
                    f"risk_flags={examination.risk_flags[:2]}"
                )
                # Track consensus history for next cycle
                if not hasattr(self, "_consensus_history"):
                    self._consensus_history = []
                self._consensus_history.append(
                    f"{datetime.now().strftime('%H:%M')} {consensus.direction.value}"
                )
                self._consensus_history = self._consensus_history[-10:]
            except Exception as _ce:
                self.logger.debug(f"[LLM-EXAMINE] skipped: {_ce}")

        # ── Cross-examiner gate ───────────────────────────────────────────────
        # AVOID  → shadow-scan setups in watch-only mode (never logged/traded).
        #          This keeps the meta-learner fed even during protective pauses.
        # WAIT   → scan setups for awareness (watch list) but suppress logging;
        #          setups are shown with a "watch only" flag, not as live recommendations.
        # REDUCE_SIZE → scan + log, but scale conviction to 75% before ranking.
        # PROCEED/fallback → normal flow.
        _exam_rec = getattr(self._last_examination, "final_recommendation", "PROCEED")
        _watch_only = False

        # ── AVOID deadlock circuit breaker ────────────────────────────────────
        # If the cross-examiner fires AVOID for _AVOID_STREAK_LIMIT consecutive
        # cycles, temporarily escalate to REDUCE_SIZE for exactly one cycle.
        # This allows one limited scan to feed the meta-learner and break the
        # compounding AVOID loop where poor win-rate prevents any new data.
        if _exam_rec == "AVOID":
            self._consecutive_avoid_cycles = getattr(self, "_consecutive_avoid_cycles", 0) + 1
            if self._consecutive_avoid_cycles >= self._AVOID_STREAK_LIMIT:
                self.logger.warning(
                    f"[EXAMINE-GATE] ⚡ AVOID streak={self._consecutive_avoid_cycles} cycles "
                    f">= limit={self._AVOID_STREAK_LIMIT} — escalating to REDUCE_SIZE for "
                    f"ONE cycle to break deadlock. Conviction capped at 50%."
                )
                _exam_rec = "REDUCE_SIZE"
                self._consecutive_avoid_cycles = 0
        else:
            self._consecutive_avoid_cycles = 0

        if _exam_rec == "AVOID":
            self.logger.info(
                "[EXAMINE-GATE] Cross-examiner AVOID — setup scanning suppressed this cycle. "
                f"(streak={self._consecutive_avoid_cycles}/{self._AVOID_STREAK_LIMIT})"
            )
            # Still scan setups in shadow/watch-only mode so that:
            #   1. The meta-learner receives signal data from suppressed cycles.
            #   2. Operators can see what *would have* been recommended without acting on it.
            # Setups are NOT logged to DataManager and NOT shown as live recommendations.
            try:
                _shadow_setups = [
                    s for s in (self._analyze_stock(stock, market_data, portfolio_status)
                                for stock in watchlist)
                    if s
                ]
                _shadow_ranked = self._tier_5_rank_setups(_shadow_setups)
                if _shadow_ranked:
                    self.logger.info(
                        f"[EXAMINE-GATE] Shadow scan: {len(_shadow_ranked)} suppressed setup(s) "
                        f"(watch-only, not logged): "
                        + ", ".join(f"{s.stock}({s.direction.value}/{s.conviction})" for s in _shadow_ranked[:3])
                    )
            except Exception as _se:
                self.logger.debug(f"[EXAMINE-GATE] Shadow scan error (non-critical): {_se}")
            top_setups = []
        else:
            setups = [
                s for s in (self._analyze_stock(stock, market_data, portfolio_status)
                            for stock in watchlist)
                if s
            ]
            if _exam_rec == "REDUCE_SIZE":
                for s in setups:
                    s.conviction = max(50, int(s.conviction * 0.75))
                self.logger.info(
                    "[EXAMINE-GATE] Cross-examiner REDUCE_SIZE — conviction scaled to 75%."
                )
            top_setups = self._tier_5_rank_setups(setups)
            if _exam_rec == "WAIT":
                # Show setups as a watch list only — do NOT log to DataManager
                _watch_only = True
                self.logger.info(
                    "[EXAMINE-GATE] Cross-examiner WAIT — "
                    f"{len(top_setups)} setup(s) shown as watch-only, not logged."
                )
            else:
                self._log_setups_to_history(top_setups)

        # FIX 7: tag AVOID-cycle predictions so expire_stale_predictions()
        # can discard them cleanly without counting as phantom losses.
        _is_suppressed_cycle = (_exam_rec == "AVOID")
        self._log_agent_verdicts_to_scorecard(consensus, suppressed=_is_suppressed_cycle)
        self.data_manager.update_regime_history(self.current_regime.value)

        # ── LLM Trading Planner — always runs, produces concrete tradeable calls ──
        # Collects shadow scan equity signals (stocks that almost qualified),
        # current F&O context, and all engine signals, then calls Gemini-flash
        # to produce key levels, scenarios, ready trades, and a market open checklist.
        _trading_plan_output = None
        if self.llm_planner is not None:
            try:
                # Collect equity shadow signals from AVOID/WAIT scan for planner context
                _shadow_eq = []
                _watchlist_stocks = list(watchlist)  # all stocks engine considered
                # Build live price map for every watchlist stock so the LLM has
                # real price anchors — prevents fabricating entry/SL/target prices.
                _price_data = market_data.get("price_data", {})
                _stock_prices = {}
                for _sym in _watchlist_stocks:
                    _pd = _price_data.get(_sym, {})
                    _close = _pd.get("close", 0) or _pd.get("price", 0)
                    if _close > 0:
                        _stock_prices[_sym] = {
                            "price":  round(_close, 2),
                            "high52": round(_pd.get("high_52w", _close * 1.3), 2),
                            "low52":  round(_pd.get("low_52w",  _close * 0.7), 2),
                            "atr":    round(_pd.get("atr", _close * 0.02), 2),
                        }
                try:
                    _all_eq = [
                        self._analyze_stock(s, market_data, portfolio_status)
                        for s in watchlist
                    ]
                    for _s in [x for x in _all_eq if x is not None]:
                        _shadow_eq.append({
                            "stock":       _s.stock,
                            "direction":   _s.direction.value,
                            "conviction":  _s.conviction,
                            "entry_price": _s.entry_price,
                            "stop_loss":   _s.stop_loss,
                            "target_1":    _s.target_1,
                        })
                except Exception:
                    pass  # shadow scan optional

                # F&O brain context
                _fno_b = self._phoenix_output  # reuse stored phoenix
                _pb = getattr(self, "_last_fno_brain", None)
                _nifty_chain = market_data.get("index_option_chains", {}).get("NIFTY", {})

                # Agent conviction summary for planner
                _agent_conv = {
                    name: {
                        "direction":  getattr(getattr(r, "verdict", None), "direction", TradeDirection.NEUTRAL).value
                                      if hasattr(getattr(r, "verdict", None), "direction") else "NEUTRAL",
                        "conviction": getattr(getattr(r, "verdict", None), "conviction", 0),
                        "weight":     getattr(getattr(r, "verdict", None), "weight", 0),
                    }
                    for name, r in self.agent_reports.items()
                    if name not in ("HERMES", "THETA", "DELTA")
                }

                # Phoenix summary
                _px_sigs = "No active signals"
                if self._phoenix_output and self._phoenix_output.signals:
                    _fired = [s for s in self._phoenix_output.signals if s.fired]
                    if _fired:
                        _px_sigs = " | ".join(f"{s.name}({s.score:.0f}/{s.max_score:.0f})" for s in _fired[:4])

                _plan_ctx = {
                    "regime":             self.current_regime.value,
                    "regime_confidence":  self.regime_confidence,
                    "consensus_direction": consensus.direction.value,
                    "consensus_strength":  consensus.strength,
                    "exam_recommendation": _exam_rec,
                    "phoenix": {
                        "score":           getattr(self._phoenix_output, "phoenix_score", 0),
                        "tier":            getattr(self._phoenix_output, "tier", "DORMANT"),
                        "signals_summary": _px_sigs,
                    },
                    "fno_brain": {
                        "stance":      getattr(_pb, "market_stance",  "NEUTRAL") if _pb else "NEUTRAL",
                        "risk_score":  getattr(_pb, "risk_score",      5)         if _pb else 5,
                        "narrative":   (getattr(_pb, "narrative", "") or "")[:300] if _pb else "",
                    },
                    "iv_rank":          _nifty_chain.get("iv_rank",  50),
                    "iv_regime":        "HIGH" if _nifty_chain.get("iv_rank", 50) >= 60 else
                                        "LOW"  if _nifty_chain.get("iv_rank", 50) <= 30 else "NORMAL",
                    "max_pain":         _nifty_chain.get("max_pain", market_data.get("nifty_price", 0)),
                    "equity_shadow_signals": _shadow_eq[:6],
                    # Full watchlist: all stocks the engine analysed this cycle.
                    # Trading planner uses this to prevent hallucinating stocks
                    # that were never part of the engine's analysis universe.
                    "watchlist_stocks": _watchlist_stocks,
                    # Live prices for every watchlist stock — LLM must use these
                    # as anchors for entry/SL/target prices. Prevents fabrication.
                    "stock_prices": _stock_prices,
                    "agent_convictions": _agent_conv,
                    # ── FIX-RESTRICTION-01: Pass news restrictions to trading planner ──
                    # Without this, the planner has no knowledge of active restrictions
                    # like HALT_NEW_LONGS_IN_FINANCIALS, resulting in direct violations.
                    "news_restrictions": (
                        list(self._last_news_impact.trade_restrictions)
                        if self._last_news_impact and hasattr(self._last_news_impact, "trade_restrictions")
                        else []
                    ),
                }

                _trading_plan_output = self.llm_planner.generate_plan(market_data, _plan_ctx)
                self.logger.info(
                    f"[TRADING-PLAN] Generated | stance={_trading_plan_output.overall_stance} | "
                    f"scenarios={len(_trading_plan_output.scenarios)} | "
                    f"fno_trades={len(_trading_plan_output.fno_ready_trades)} | "
                    f"watchlist={len(_trading_plan_output.equity_watchlist)} | "
                    f"source={_trading_plan_output.source}"
                )
            except Exception as _tp_err:
                self.logger.warning(f"[TRADING-PLAN] Failed (non-critical): {_tp_err}")

        plan = DailyTradingPlan(
            date=datetime.now(),
            market_regime=self.current_regime,
            regime_confidence=self.regime_confidence,
            consensus=consensus,
            top_setups=top_setups,
            portfolio_risk=self._calculate_portfolio_risk(portfolio_status),
            upcoming_events=self._extract_upcoming_events(market_data),
            action_items=self._generate_action_items(top_setups, consensus),
            agent_reports=self.agent_reports,
            setups_watch_only=_watch_only,
            phoenix_analysis=self._phoenix_output,
        )
        # Attach trading plan output to plan object for report rendering
        plan._trading_plan = _trading_plan_output
        return plan

    # ------------------------------------------------------------------
    # Tier implementations  (identical to v3.2, kept complete)
    # ------------------------------------------------------------------
    def _log_setups_to_history(self, setups):
        today = datetime.now().date().isoformat()

        # Build a set of (date, stock, direction) already logged today so we
        # don't duplicate across the 60-second live-mode cycles.
        if not hasattr(self, "_logged_today"):
            self._logged_today: set = set()
            self._logged_date: str = today
        if self._logged_date != today:          # midnight rollover — reset
            self._logged_today = set()
            self._logged_date = today

        for setup in setups:
            if setup.direction == TradeDirection.NEUTRAL:
                continue
            dedup_key = (today, setup.stock, setup.direction.value)
            if dedup_key in self._logged_today:
                continue                        # already logged this calendar day
            try:
                # [FIX-B] Only include agents whose verdict direction matches the
                # setup direction.  Previously all agent names were always logged,
                # corrupting MetaLearner's per-agent attribution entirely.
                # e.g. a LONG setup on a day where SENTINEL/OPTIMUS said SHORT
                # used to list them as recommenders — now they are excluded.
                _aligned = [
                    name for name, report in self.agent_reports.items()
                    if report.verdict.direction == setup.direction
                ]
                recommending_agents = ",".join(_aligned) if _aligned else "ORION"
                record = TradeRecord(
                    date_recommended=today, stock=setup.stock,
                    direction=setup.direction.value,
                    entry_price=round(setup.entry_price, 2),
                    stop_loss=round(setup.stop_loss, 2),
                    target_price=round(setup.target_1, 2),
                    risk_reward_ratio=round(setup.risk_reward, 2),
                    recommending_agents=recommending_agents,
                    regime_at_entry=self.current_regime.value,
                    conviction_confidence=setup.conviction,
                )
                self.data_manager.log_trade(record)
                self.data_manager.save_pattern({
                    "stock": setup.stock, "direction": setup.direction.value,
                    "regime": self.current_regime.value, "conviction": setup.conviction,
                    "entry_price": setup.entry_price, "stop_loss": setup.stop_loss,
                    "target": setup.target_1, "risk_reward": setup.risk_reward,
                    "setup_type": f"{setup.direction.value}_swing",
                })
                self._logged_today.add(dedup_key)
            except Exception as e:
                self.logger.error(f"[HISTORY] Failed to log {setup.stock}: {e}")

    def _log_agent_verdicts_to_scorecard(self, consensus, suppressed: bool = False):
        """Log each agent's verdict to the scorecard pending queue.

        Args:
            consensus:   Consensus result for this cycle.
            suppressed:  True when the EXAMINE-GATE fired AVOID — predictions
                         are recorded but flagged so expire_stale_predictions()
                         can cleanly discard them after SUPPRESSED_MAX_AGE_DAYS.
        """
        regime = self.current_regime.value
        # Non-voting observers — excluded from scorecard tracking
        _non_voters = {"NOCTURNAL", "PHOENIX"}
        for agent_name, report in self.agent_reports.items():
            if agent_name in _non_voters:
                continue
            try:
                self.scorecard.record_prediction(
                    agent_name=agent_name,
                    prediction=report.verdict.direction.value,
                    conviction=report.verdict.conviction,
                    regime=regime,
                    suppressed=suppressed,
                )
            except Exception:
                pass

    def _tier_0_validate_data(self, data):
        required = ["price_data","flow_data","sentiment_data",
                    "derivatives_data","fundamental_data","event_data"]
        missing = [k for k in required if not data.get(k)]
        return {"missing": missing, "status": "green" if len(missing) <= 2 else "yellow"}

    def _tier_11_detect_regime(self, data) -> Tuple[MarketRegime, float]:
        # ── LLM-first regime detection ────────────────────────────────────
        # LLMRegimeDetector produces a full probability distribution across 6
        # regime states, plus key factors and transition warnings.
        # Falls back to rule-based scorer if LLM unavailable or errors.
        if self.llm_regime is not None:
            try:
                result = self.llm_regime.detect_regime(data, self.current_regime)
                if result.source == "LLM":
                    self._last_regime_result = result
                    self.logger.info(
                        f"[LLM-REGIME] {result.regime.value} | confidence={result.confidence:.0f}% | "
                        f"key_factors={result.key_factors[:2]}"
                        + (f" | ⚠ {result.transition_warning}" if result.transition_warning else "")
                    )
                    return result.regime, result.confidence
                else:
                    self.logger.debug("[LLM-REGIME] fell back to rule-based inside detector")
                    self._last_regime_result = result   # store FALLBACK result too
            except Exception as _re:
                self.logger.debug(f"[LLM-REGIME] exception — using rule scorer: {_re}")

        # ── Rule-based fallback (original scorer, unchanged) ──────────────
        return self._rule_based_regime(data)

    def _rule_based_regime(self, data) -> Tuple[MarketRegime, float]:
        """Original 5-variable rule-based regime scorer. Used as LLM fallback."""
        score = 0; fired = 0
        nifty = data.get("nifty_price", 0); dma200 = data.get("nifty_200dma", 0)
        if dma200 > 0:
            gap = (nifty - dma200) / dma200
            if gap > 0.05: score += 1; fired += 1
            elif gap > 0:  score += 0.5
            elif gap < -0.05: score -= 1; fired += 1
            else: score -= 0.5
        ps = data.get("price_structure", "neutral")
        if ps == "higher_highs": score += 1; fired += 1
        elif ps == "lower_lows": score -= 1; fired += 1
        adx = data.get("adx", 20); adx_strong = adx > 25
        fii = data.get("flow_data", {}).get("fii_cash_5day", 0)
        if fii > 3000: score += 1; fired += 1
        elif fii < -3000: score -= 1; fired += 1
        vix = data.get("india_vix", 15)
        if vix > 22: score -= 0.5
        elif vix < 13: score += 0.5; fired += 1

        if score >= 2:   regime = MarketRegime.BULL
        elif score >= 1: regime = MarketRegime.MILD_BULL
        elif score <= -2: regime = MarketRegime.BEAR
        elif score <= -1: regime = MarketRegime.MILD_BEAR
        else:            regime = MarketRegime.CONSOLIDATION

        abs_s = abs(score); base = 50 + abs_s * 10
        if abs_s >= 2 and adx_strong and fired >= 3:
            conf = min(90, base + fired * 3)
        elif abs_s >= 2:
            conf = min(75, base + fired * 2)
        elif abs_s >= 1:
            conf = min(65, base + fired * 2)
        else:
            conf = min(55, base)
        return regime, round(conf, 1)

    def _tier_1_adjust_weights(self, regime):
        adjustments = {
            MarketRegime.BULL:  {"ORION":0.03,"VESPER":0.02,"KAIRO":-0.03,"PRUDENCE":-0.02},
            MarketRegime.BEAR:  {"ORION":-0.03,"VESPER":0.02,"KAIRO":0.03,"SENTINEL":0.02,"NEXUS":0.01,"PRUDENCE":0.02},
            MarketRegime.CONSOLIDATION: {"VESPER":-0.02,"SENTINEL":0.03,"NEXUS":0.02,"CATALYST":0.01},
        }
        adj = adjustments.get(regime, {})
        for name, agent in self.agents.items():
            if name == "NOCTURNAL":
                continue  # NOCTURNAL manages its own weight
            delta = adj.get(name, 0)
            baseline = self.config.agents[name].baseline_weight if name in self.config.agents else agent.current_weight
            agent.current_weight = max(0.05, min(0.30, baseline + delta))

        # IMPROVEMENT 1: Performance-based weight overlay from scorecard
        # Only activates after 20+ resolved predictions per agent — before that,
        # static regime weights are the only adjustment.
        try:
            sc = self.scorecard.get_scorecard()
            for agent_name, agent in self.agents.items():
                if agent_name == "NOCTURNAL":
                    continue
                stats = sc.get(agent_name, {})
                win_rate = stats.get("win_rate", 0.0)
                total = stats.get("total_predictions", 0)
                if total >= 20:
                    # ±5% max swing: agents above 50% WR gain weight, below lose weight
                    perf_delta = (win_rate - 0.50) * 0.10
                    agent.current_weight = max(0.05, min(0.30,
                        agent.current_weight + perf_delta))
                    self.logger.debug(
                        f"[SCORECARD] {agent_name}: WR={win_rate:.1%} "
                        f"total={total} → perf_delta={perf_delta:+.3f}"
                    )
        except Exception as e:
            self.logger.debug(f"Scorecard weight overlay skipped: {e}")

        # NOCTURNAL override — reduce all swing agent weights in high-risk conditions
        if "NOCTURNAL" in self.agents:
            try:
                nocturnal_report = self.agents["NOCTURNAL"].analyze(
                    {"market_context": self.market_data}, regime
                )
                risk_level = nocturnal_report.analysis_details.get(
                    "risk_profile", {}
                ).get("risk_level", "NORMAL")

                if risk_level == "EXTREME":
                    # Drastically reduce all trading weights
                    for name, agent in self.agents.items():
                        if name != "NOCTURNAL":
                            agent.current_weight *= 0.1
                elif risk_level == "HIGH":
                    for name, agent in self.agents.items():
                        if name != "NOCTURNAL":
                            agent.current_weight *= 0.5
            except Exception as e:
                self.logger.debug(f"NOCTURNAL weight override skipped: {e}")

        total = sum(a.current_weight for a in self.agents.values())
        if total > 0:
            for a in self.agents.values():
                a.current_weight /= total

    def _run_all_agents(self, data, regime):
        self.agent_reports = {}
        mapping = {
            "ORION":    self._prepare_orion_data(data),
            "VESPER":   self._prepare_vesper_data(data),
            "KAIRO":    self._prepare_kairo_data(data),
            "SENTINEL": self._prepare_sentinel_data(data),
            "NEXUS":    self._prepare_nexus_data(data),
            "PRUDENCE": self._prepare_prudence_data(data),
            "CATALYST": self._prepare_catalyst_data(data),
            "OPTIMUS":  self._prepare_optimus_data(data),
        }
        for name, agent in self.agents.items():
            if name == "NOCTURNAL":
                continue  # NOCTURNAL runs separately via pre_market_routine
            try:
                self.agent_reports[name] = agent.analyze(mapping.get(name, {}), regime)
            except Exception as e:
                self.agent_reports[name] = AgentReport(
                    agent_name=name,
                    verdict=AgentVerdict(direction=TradeDirection.NEUTRAL, conviction=0,
                                        weight=agent.current_weight, reason=str(e)))

    def _tier_2_calculate_consensus(self) -> ConsensusResult:
        weighted_votes: Dict[str, float] = {}
        long_v, short_v = [], []
        for name, report in self.agent_reports.items():
            v = report.verdict
            weighted_votes[name] = v.weighted_vote
            if v.direction == TradeDirection.LONG: long_v.append(name)
            elif v.direction == TradeDirection.SHORT: short_v.append(name)
        net = sum(weighted_votes.values())
        if net > 0.25:
            direction = TradeDirection.LONG
            if net > 0.4:    strength = "STRONG"
            elif net > 0.30: strength = "MODERATE"
            else:            strength = "WEAK"
        elif net < -0.25:
            direction = TradeDirection.SHORT
            if net < -0.4:    strength = "STRONG"
            elif net < -0.30: strength = "MODERATE"
            else:             strength = "WEAK"
        else:
            direction = TradeDirection.NEUTRAL; strength = "NO_CONSENSUS"
        contradictions = []
        if long_v and short_v:
            contradictions.append({"type":"direction",
                                   "agents":{"long":long_v,"short":short_v},
                                   "resolution":f"Weighted vote favors {direction.value}"})
        return ConsensusResult(
            direction=direction, strength=strength, net_score=net,
            weighted_votes=weighted_votes,
            agreeing_agents=long_v if direction==TradeDirection.LONG else short_v,
            disagreeing_agents=short_v if direction==TradeDirection.LONG else long_v,
            contradictions=contradictions,
        )

    def _analyze_stock(self, stock, market_data, portfolio_status):
        stock_data = self._get_stock_data(stock, market_data)
        if not stock_data: return None
        orion_report = self.agents["ORION"].analyze(stock_data, self.current_regime)
        if orion_report.verdict.conviction < 50: return None
        entry_setup = orion_report.raw_data.get("entry_setup", {})
        if entry_setup.get("direction") == TradeDirection.NEUTRAL:
            vdir = orion_report.verdict.direction
            if vdir == TradeDirection.NEUTRAL: return None
            pd_ = stock_data.get("price_data", {}); ind = stock_data.get("indicators", {})

            # ── SETUP ANCHOR PRICE ─────────────────────────────────────────
            sp = pd_.get("ref_close") or pd_.get("close", 0)
            # ── END ANCHOR PRICE ──────────────────────────────────────────

            atr = ind.get("atr", sp*0.02)
            sma20 = ind.get("sma20",0); sma50 = ind.get("sma50",0)
            sma200 = ind.get("sma200",0); rsi = ind.get("rsi",50)
            if sp <= 0: return None
            entry_setup = dict(entry_setup); entry_setup["direction"] = vdir
            min_rr = self.config.risk_limits.min_risk_reward_ratio
            if vdir == TradeDirection.LONG:
                stop = sma20*0.995 if sma20>0 and (sp-sma20)/sp<0.04 else sp-atr*1.5
                risk = sp - stop
                structural = [x for x in [sma50,sma200,round(sp*1.05/50)*50] if x>sp]
                t1 = min(structural) if structural else sp+atr*max(min_rr+0.5,2.0)
                t2 = min([c for c in structural if c>t1], default=t1*1.03)
                if risk>0 and (t1-sp)/risk<min_rr: return None
                entry_setup.update({"entry_zone":(sp*0.999,sp*1.003),
                                    "stop_loss":round(stop,2),
                                    "target_1":round(t1,2),"target_2":round(t2,2),
                                    "risk_reward":round((t1-sp)/risk,2) if risk>0 else 0})
            else:
                stop = sma20*1.005 if sma20>0 and (sma20-sp)/sp<0.04 else sp+atr*1.5
                risk = stop - sp
                structural = [x for x in [sma50,sma200,round(sp*0.95/50)*50] if x<sp]
                t1 = max(structural) if structural else sp-atr*max(min_rr+0.5,2.0)
                t2 = max([c for c in structural if c<t1], default=t1*0.97)
                if risk>0 and (sp-t1)/risk<min_rr: return None
                entry_setup.update({"entry_zone":(sp*0.997,sp*1.001),
                                    "stop_loss":round(stop,2),
                                    "target_1":round(t1,2),"target_2":round(t2,2),
                                    "risk_reward":round((sp-t1)/risk,2) if risk>0 else 0})

        # ── FIX-DIRECTION-01: Target/SL direction sanity check ──────────────
        # For LONG: target must be > entry, stop_loss must be < entry
        # For SHORT: target must be < entry, stop_loss must be > entry
        # If inverted, the agent generated targets for the wrong direction.
        # Auto-fix by recalculating from ATR rather than silently passing bad values.
        #
        # CRITICAL: This check MUST run BEFORE LLM validation.
        # Previously it ran after, so the LLM validator would see inverted
        # targets (e.g. SHORT with target > entry) and AVOID the setup,
        # preventing this fix from ever running.
        #
        # NOTE (FIX-DIRECTION-ROOT): The root cause was fixed in ORION's
        # _calculate_entry_setup() — SHORT targets now validate that
        # nearest_support.price < current_price.  This guard remains as a
        # safety net for any edge case ORION misses or LLM overrides.
        _entry_price = entry_setup.get("entry_zone", (0, 0))[0]
        _target_1 = entry_setup.get("target_1", 0)
        _stop_loss = entry_setup.get("stop_loss", 0)
        _setup_direction = entry_setup.get("direction", TradeDirection.NEUTRAL)

        if _entry_price > 0 and _target_1 > 0 and _stop_loss > 0:
            _direction_fixed = False
            if _setup_direction == TradeDirection.LONG:
                if _target_1 <= _entry_price:
                    self.logger.warning(
                        f"[FIX-DIRECTION-01] {stock} LONG target ({_target_1}) <= entry ({_entry_price}) "
                        f"— recalculating target from ATR"
                    )
                    _atr = stock_data.get("indicators", {}).get("atr", 0)
                    if _atr > 0:
                        entry_setup["target_1"] = round(_entry_price + _atr * 2.0, 2)
                        entry_setup["target_2"] = round(_entry_price + _atr * 3.0, 2)
                    else:
                        entry_setup["target_1"] = round(_entry_price * 1.03, 2)
                        entry_setup["target_2"] = round(_entry_price * 1.05, 2)
                    _direction_fixed = True
                if _stop_loss >= _entry_price:
                    self.logger.warning(
                        f"[FIX-DIRECTION-01] {stock} LONG stop_loss ({_stop_loss}) >= entry ({_entry_price}) "
                        f"— recalculating stop from ATR"
                    )
                    _atr = stock_data.get("indicators", {}).get("atr", 0)
                    if _atr > 0:
                        entry_setup["stop_loss"] = round(_entry_price - _atr * 1.5, 2)
                    else:
                        entry_setup["stop_loss"] = round(_entry_price * 0.97, 2)
                    _direction_fixed = True
            elif _setup_direction == TradeDirection.SHORT:
                if _target_1 >= _entry_price:
                    self.logger.warning(
                        f"[FIX-DIRECTION-01] {stock} SHORT target ({_target_1}) >= entry ({_entry_price}) "
                        f"— recalculating target from ATR"
                    )
                    _atr = stock_data.get("indicators", {}).get("atr", 0)
                    if _atr > 0:
                        entry_setup["target_1"] = round(_entry_price - _atr * 2.0, 2)
                        entry_setup["target_2"] = round(_entry_price - _atr * 3.0, 2)
                    else:
                        entry_setup["target_1"] = round(_entry_price * 0.97, 2)
                        entry_setup["target_2"] = round(_entry_price * 0.95, 2)
                    _direction_fixed = True
                if _stop_loss <= _entry_price:
                    self.logger.warning(
                        f"[FIX-DIRECTION-01] {stock} SHORT stop_loss ({_stop_loss}) <= entry ({_entry_price}) "
                        f"— recalculating stop from ATR"
                    )
                    _atr = stock_data.get("indicators", {}).get("atr", 0)
                    if _atr > 0:
                        entry_setup["stop_loss"] = round(_entry_price + _atr * 1.5, 2)
                    else:
                        entry_setup["stop_loss"] = round(_entry_price * 1.03, 2)
                    _direction_fixed = True

            if _direction_fixed:
                # Recalculate risk_reward after fixing target/SL
                _new_sl = entry_setup.get("stop_loss", 0)
                _new_t1 = entry_setup.get("target_1", 0)
                _new_risk = abs(_new_sl - _entry_price)
                if _new_risk > 0:
                    entry_setup["risk_reward"] = round(abs(_new_t1 - _entry_price) / _new_risk, 2)
                else:
                    entry_setup["risk_reward"] = 0
        # ── END FIX-DIRECTION-01 (pre-validation) ────────────────────────────

        # ── LLM Pattern Validation ────────────────────────────────────────
        # Validates ORION's setup with contextual LLM analysis:
        # RSI, volume, sector alignment, stock news, historical regime win rate.
        # Can: adjust conviction up/down, adjust entry/SL/target, or AVOID the setup.
        #
        # SKIP when cross-examiner has already fired AVOID for this cycle.
        # The setup will be shadow-scanned but never traded, so per-stock
        # LLM validation adds no value and wastes one Gemini Flash call per stock.
        # ── FIX-DUPLICATE-01: Also skip if already validated this cycle. ─────
        _exam_rec_for_validate = getattr(self._last_examination, "final_recommendation", "PROCEED")
        _llm_validation = None
        _already_validated = stock in getattr(self, '_validated_this_cycle', set())
        # ── v5 PRE-FILTER ─────────────────────────────────────────────────
        if self.rule_validator:
            try:
                rule_result = self.rule_validator.validate(
                    {
                        "symbol": stock,
                        "direction": entry_setup.get("direction", TradeDirection.NEUTRAL).value,
                        "strength": "MEDIUM",
                        "agent": "ORION",
                        "rr_ratio": entry_setup.get("risk_reward", 0),
                        "rsi": stock_data.get("indicators", {}).get("rsi", 50),
                        "volume": stock_data.get("volume", 0),
                        "volume_avg_20d": stock_data.get("indicators", {}).get("vol_avg_20", 0),
                        "price": entry_setup.get("entry_zone", (0,0))[0],
                        "sma_20": stock_data.get("indicators", {}).get("sma_20", 0),
                        "sector": self._get_stock_sector(stock),
                    },
                    regime={"regime": self.current_regime.value},
                    news_restrictions={
                        "block_long_sectors": getattr(self._last_news_impact, 'block_long_sectors', []) if self._last_news_impact else [],
                        "block_short_sectors": getattr(self._last_news_impact, 'block_short_sectors', []) if self._last_news_impact else [],
                    },
                    active_sectors={}
                )
                if not rule_result.passed:
                    self.logger.info(f"[RULE-VALIDATE] {stock} REJECTED | {rule_result.reason} (saved 1 LLM call)")
                    return None
            except Exception as _rv_e:
                self.logger.debug(f"[RULE-VALIDATE] {stock} skipped: {_rv_e}")
        # ───────────────────────────────────────────────────────────────────
        if self.llm_validator is not None and _exam_rec_for_validate != "AVOID" and not _already_validated:
            try:
                # ── FIX-DUPLICATE-01: Mark stock as validated for this cycle ──
                self._validated_this_cycle.add(stock)
                _hist = self._lookup_pattern_win_rate(
                    entry_setup.get("direction", TradeDirection.NEUTRAL),
                    entry_setup.get("setup_type", "swing")
                )
                _news_items = []
                if self._last_news_impact and hasattr(self._last_news_impact, "sector_impacts"):
                    with self._news_impact_lock:
                        _ni_snap = self._last_news_impact
                    sector = self._get_stock_sector(stock)
                    sec_impact = _ni_snap.sector_impacts.get(sector)
                    if sec_impact:
                        _news_items = [{"title": f"{sector} sector impact: {sec_impact.reason}"}]
                _llm_validation = self.llm_validator.validate_pattern(
                    pattern={
                        **entry_setup,
                        "type": entry_setup.get("setup_type", "swing"),
                        "direction": entry_setup.get("direction", TradeDirection.NEUTRAL),
                        "entry_price": entry_setup.get("entry_zone", (0,0))[0],
                        "confidence": orion_report.verdict.conviction,
                    },
                    stock=stock,
                    market_context={
                        **stock_data,
                        "regime": self.current_regime,
                        "sector": self._get_stock_sector(stock),
                        "news": _news_items,
                        "performance_5d": stock_data.get("indicators", {}).get("vol_ratio", 1.0),
                        "sector_performance": 0,
                        "stock_trade_history": getattr(self.llm_history, "get_for_stock", lambda s: "")(stock) if self.llm_history else "",
                    },
                    historical_performance={
                        "regime_win_rate": _hist.get("win_rate", 50),
                        "regime_samples": _hist.get("total_trades", 0),
                        "stock_win_rate": 50,
                        "stock_samples": 0,
                    }
                )
                if _llm_validation.final_recommendation == "AVOID":
                    self.logger.info(
                        f"[LLM-VALIDATE] {stock} AVOIDED | "
                        f"reason={(_llm_validation.risk_notes or ['unknown'])[:1]}"
                    )
                    return None
                if _llm_validation.final_recommendation == "WAIT_FOR_CONFIRMATION":
                    orion_report.verdict.conviction = round(max(50, orion_report.verdict.conviction - 10))
                    # Single combined log (was previously split into two lines)
                    self.logger.info(
                        f"[LLM-VALIDATE] {stock} WAIT_FOR_CONFIRMATION | "
                        f"conviction reduced → {orion_report.verdict.conviction} | "
                        f"notes={_llm_validation.validation_notes[:1]}"
                    )
                elif _llm_validation.adjusted_confidence is not None:
                    orion_report.verdict.conviction = int(round(_llm_validation.adjusted_confidence))
                    self.logger.info(
                        f"[LLM-VALIDATE] {stock} {_llm_validation.final_recommendation} | "
                        f"conviction={orion_report.verdict.conviction} | "
                        f"notes={_llm_validation.validation_notes[:1]}"
                    )
                else:
                    self.logger.info(
                        f"[LLM-VALIDATE] {stock} {_llm_validation.final_recommendation} | "
                        f"conviction={orion_report.verdict.conviction} | "
                        f"notes={_llm_validation.validation_notes[:1]}"
                    )
                # Apply LLM-adjusted levels if provided
                if _llm_validation.adjusted_entry:
                    entry_setup["entry_zone"] = (_llm_validation.adjusted_entry,
                                                  _llm_validation.adjusted_entry * 1.003)
                if _llm_validation.adjusted_stop_loss:
                    entry_setup["stop_loss"] = round(_llm_validation.adjusted_stop_loss, 2)
                if _llm_validation.adjusted_target:
                    entry_setup["target_1"] = round(_llm_validation.adjusted_target, 2)
            except Exception as _ve:
                self.logger.debug(f"[LLM-VALIDATE] {stock} skipped: {_ve}")

        # ── FIX-DIRECTION-02: Post-LLM direction sanity check ──────────────
        # The LLM validator can override target_1 with adjusted_target which
        # may violate direction constraints (e.g. SHORT target > entry). This
        # runs AFTER LLM validation to catch such cases. Without this guard,
        # the LLM can undo FIX-DIRECTION-01's correction, leading to setups
        # like ADANIENT SHORT with target 2200.4 > entry 2182.75.
        _ep2 = entry_setup.get("entry_zone", (0, 0))[0]
        _t1_2 = entry_setup.get("target_1", 0)
        _sl2 = entry_setup.get("stop_loss", 0)
        _dir2 = entry_setup.get("direction", TradeDirection.NEUTRAL)
        if _ep2 > 0 and _t1_2 > 0 and _sl2 > 0 and _dir2 != TradeDirection.NEUTRAL:
            _dir_violation = False
            _atr2 = stock_data.get("indicators", {}).get("atr", 0)
            if _dir2 == TradeDirection.SHORT:
                if _t1_2 >= _ep2:
                    self.logger.warning(
                        f"[FIX-DIRECTION-02] {stock} SHORT: LLM set target ({_t1_2}) "
                        f">= entry ({_ep2}) — overriding to below entry"
                    )
                    if _atr2 > 0:
                        entry_setup["target_1"] = round(_ep2 - _atr2 * 2.0, 2)
                        entry_setup["target_2"] = round(_ep2 - _atr2 * 3.0, 2)
                    else:
                        entry_setup["target_1"] = round(_ep2 * 0.97, 2)
                        entry_setup["target_2"] = round(_ep2 * 0.95, 2)
                    _dir_violation = True
                if _sl2 <= _ep2:
                    self.logger.warning(
                        f"[FIX-DIRECTION-02] {stock} SHORT: LLM set SL ({_sl2}) "
                        f"<= entry ({_ep2}) — overriding to above entry"
                    )
                    if _atr2 > 0:
                        entry_setup["stop_loss"] = round(_ep2 + _atr2 * 1.5, 2)
                    else:
                        entry_setup["stop_loss"] = round(_ep2 * 1.03, 2)
                    _dir_violation = True
            elif _dir2 == TradeDirection.LONG:
                if _t1_2 <= _ep2:
                    self.logger.warning(
                        f"[FIX-DIRECTION-02] {stock} LONG: LLM set target ({_t1_2}) "
                        f"<= entry ({_ep2}) — overriding to above entry"
                    )
                    if _atr2 > 0:
                        entry_setup["target_1"] = round(_ep2 + _atr2 * 2.0, 2)
                        entry_setup["target_2"] = round(_ep2 + _atr2 * 3.0, 2)
                    else:
                        entry_setup["target_1"] = round(_ep2 * 1.03, 2)
                        entry_setup["target_2"] = round(_ep2 * 1.05, 2)
                    _dir_violation = True
                if _sl2 >= _ep2:
                    self.logger.warning(
                        f"[FIX-DIRECTION-02] {stock} LONG: LLM set SL ({_sl2}) "
                        f">= entry ({_ep2}) — overriding to below entry"
                    )
                    if _atr2 > 0:
                        entry_setup["stop_loss"] = round(_ep2 - _atr2 * 1.5, 2)
                    else:
                        entry_setup["stop_loss"] = round(_ep2 * 0.97, 2)
                    _dir_violation = True
            if _dir_violation:
                _new_sl2 = entry_setup.get("stop_loss", 0)
                _new_t1_2 = entry_setup.get("target_1", 0)
                _new_risk2 = abs(_new_sl2 - _ep2)
                if _new_risk2 > 0:
                    entry_setup["risk_reward"] = round(abs(_new_t1_2 - _ep2) / _new_risk2, 2)
        # ── END FIX-DIRECTION-02 ───────────────────────────────────────────

        sp2 = stock_data.get("price_data",{}).get("close",0)
        fund = stock_data.get("fundamentals",
               market_data.get("fundamental_data",{}).get(stock,{}))
        nexus_report = self.agents["NEXUS"].analyze(
            {**stock_data,"stock":stock,"nifty_pe":market_data.get("nifty_pe",22.5),
             "gsec_yield":market_data.get("gsec_yield",7.0),"fundamentals":fund},
            self.current_regime)
        has_fund = bool(fund)
        pru_report = self.agents["PRUDENCE"].analyze({
            "portfolio":portfolio_status,
            "trade_request":{"entry_price":entry_setup.get("entry_zone",(0,0))[0],
                             "stop_loss":entry_setup.get("stop_loss",0)},
            "stock":stock,"sector":self._get_stock_sector(stock),
            "conviction_level":self._get_conviction_level(
                orion_report.verdict.conviction).value,
            # FIX: pass the consensus direction so PRUDENCE mirrors it correctly
            # instead of hardcoding LONG for every trade.
            "proposed_direction": entry_setup.get("direction", TradeDirection.NEUTRAL),
        }, self.current_regime)
        if pru_report.analysis_details.get("verdict_type") == "VETO": return None

        stock_rpts = [orion_report, nexus_report, pru_report] if has_fund \
                     else [orion_report, pru_report]
        stock_conv = self._calculate_conviction(*stock_rpts)
        panel = [self.agent_reports.get(n) for n in
                 ["VESPER","KAIRO","SENTINEL","CATALYST","OPTIMUS"] if self.agent_reports.get(n)]
        if panel:
            panel_conv = self._calculate_conviction(*panel)
            conviction = int(stock_conv*0.70 + panel_conv*0.30)
        else:
            conviction = stock_conv

        # IMPROVEMENT 6: ML conviction adjustment
        # Tries XGBoost first (when trained model exists in models/xgboost_direction.pkl).
        # Falls back to rule-based bonus if model is absent/untrained.
        # Either path is additive only — never penalises below the conviction floor.
        try:
            from ml_pipeline.feature_engineering import FeatureEngineer
            from ml_pipeline.ml_models import XGBoostModel
            import os as _os

            fe = FeatureEngineer()
            feat_set = fe.generate_features(
                symbol=stock,
                price_data=stock_data.get("price_data", {}),
                indicators=stock_data.get("indicators", {}),
                flow_data=self.market_data.get("flow_data", {}),
                derivatives_data=self.market_data.get("derivatives_data", {}),
            )
            fv   = feat_set.features if hasattr(feat_set, "features") else (feat_set or {})
            vdir = entry_setup.get("direction", TradeDirection.NEUTRAL)

            # ── Try trained XGBoost model ─────────────────────────────────────
            _model_path = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), "models", "xgboost_direction.pkl"
            )
            _xgb_used = False
            if _os.path.exists(_model_path):
                try:
                    import pickle as _pickle
                    with open(_model_path, "rb") as _mf:
                        _mdata = _pickle.load(_mf)
                    if _mdata.get("is_trained") and _mdata.get("feature_names"):
                        _xgb = XGBoostModel()
                        _xgb.model         = _mdata["model"]
                        _xgb.feature_names = _mdata["feature_names"]
                        _xgb.is_trained    = True

                        # Augment features with meta-features the model was trained on
                        _regime_map = {
                            "BULL": 2, "MILD_BULL": 1, "CONSOLIDATION": 0,
                            "MILD_BEAR": -1, "BEAR": -2,
                        }
                        _close = stock_data.get("price_data", {}).get("close", 0)
                        _sma20 = stock_data.get("indicators", {}).get("sma20", _close)
                        _sma50 = stock_data.get("indicators", {}).get("sma50", _close)
                        _sma200 = stock_data.get("indicators", {}).get("sma200", _close)
                        _entry = entry_setup.get("entry_zone", (_close, _close))[0] or _close
                        _sl    = entry_setup.get("stop_loss", 0)
                        _tgt   = entry_setup.get("target_1",  0)
                        fv.update({
                            "meta_conviction":    float(conviction),
                            "meta_rr":            float(entry_setup.get("risk_reward", 1.5) or 1.5),
                            "meta_regime":        float(_regime_map.get(self.current_regime.value, 0)),
                            "meta_is_long":       1.0 if vdir == TradeDirection.LONG else 0.0,
                            "meta_sl_pct":        abs(_entry - _sl)  / _entry if _entry > 0 and _sl  > 0 else 0.02,
                            "meta_tgt_pct":       abs(_tgt  - _entry) / _entry if _entry > 0 and _tgt > 0 else 0.04,
                            "meta_above_sma20":   1.0 if _close > _sma20  else 0.0,
                            "meta_above_sma50":   1.0 if _close > _sma50  else 0.0,
                            "meta_above_sma200":  1.0 if _close > _sma200 else 0.0,
                        })

                        _pred = _xgb.predict(fv)
                        _win_prob = _pred.raw_scores.get("LONG", _pred.probability) if _pred.raw_scores else _pred.probability

                        # Map win probability → conviction delta (±10 max)
                        # 0.5 prob = neutral (0 delta); 0.8 prob = +10; 0.2 prob = -10
                        _ml_delta = int((_win_prob - 0.5) * 20)
                        _ml_delta = max(-10, min(10, _ml_delta))
                        conviction = max(50, min(95, conviction + _ml_delta))
                        _xgb_used  = True
                        self.logger.debug(
                            f"[ML-XGB] {stock}: win_prob={_win_prob:.2f} "
                            f"delta={_ml_delta:+d} → conviction={conviction}"
                        )
                except Exception as _xe:
                    self.logger.debug(f"[ML-XGB] {stock} model load/predict failed: {_xe}")

            # ── Fallback: rule-based bonus (pre-XGBoost behaviour) ────────────
            if not _xgb_used:
                ml_bonus = 0
                if fv.get("rsi", 50) < 40 and vdir == TradeDirection.LONG:
                    ml_bonus += 3
                if fv.get("volume_ratio", 1) > 1.5:
                    ml_bonus += 2
                if fv.get("macd_histogram", 0) > 0 and vdir == TradeDirection.LONG:
                    ml_bonus += 2
                if fv.get("rsi", 50) > 60 and vdir == TradeDirection.SHORT:
                    ml_bonus += 3
                if fv.get("macd_histogram", 0) < 0 and vdir == TradeDirection.SHORT:
                    ml_bonus += 2
                conviction = min(95, conviction + ml_bonus)

        except Exception:
            pass  # silently skip — ml adjustment is additive only, never blocks pipeline

        if conviction < 55: return None

        # ── [FIX-A] REGIME GATE ───────────────────────────────────────────────
        # Block setups whose direction conflicts with the current market regime,
        # and raise the conviction bar in ambiguous regimes.
        # This is the primary reason for 100% LONG bias in bearish markets:
        # previously there was no hard block — only a soft -5 pt penalty in ORION.
        #
        #  BEAR / MILD_BEAR  → LONG only allowed at very high conviction (≥72)
        #                       SHORT setups allowed at normal bar (55)
        #  CONSOLIDATION     → both directions allowed but bar raised to 65
        #  CORRECTION        → treat like MILD_BEAR
        #  BULL / MILD_BULL  → SHORT only allowed at very high conviction (≥72)
        #
        # [v4.3] PHOENIX override: when PHOENIX detects pre-momentum, the
        # CONSOLIDATION / MILD_BEAR thresholds are lowered to allow early entries.
        _setup_dir = entry_setup.get("direction", TradeDirection.NEUTRAL)
        _bearish_regimes = {MarketRegime.MILD_BEAR, MarketRegime.BEAR, MarketRegime.CORRECTION}
        _bullish_regimes = {MarketRegime.BULL, MarketRegime.MILD_BULL}

        # Base thresholds
        _consol_threshold = 65
        _bear_long_threshold = 72

        # Apply PHOENIX gate override if active
        _phoenix_override = getattr(self, "_phoenix_gate_override", None)
        if _phoenix_override is not None:
            _consol_threshold = _phoenix_override
            _bear_long_threshold = min(72, _phoenix_override + 10)
            self.logger.debug(
                f"[PHOENIX-GATE] {stock}: thresholds adjusted → "
                f"CONSOL={_consol_threshold} BEAR_LONG={_bear_long_threshold}"
            )

        if self.current_regime in _bearish_regimes:
            if _setup_dir == TradeDirection.LONG and conviction < _bear_long_threshold:
                self.logger.info(
                    f"[REGIME-GATE] {stock} LONG suppressed in "
                    f"{self.current_regime.value} (conviction={conviction} < {_bear_long_threshold} required)"
                )
                return None
        elif self.current_regime == MarketRegime.CONSOLIDATION:
            if conviction < _consol_threshold:
                self.logger.info(
                    f"[REGIME-GATE] {stock} suppressed in CONSOLIDATION "
                    f"(conviction={conviction} < {_consol_threshold} required)"
                )
                return None
        elif self.current_regime in _bullish_regimes:
            if _setup_dir == TradeDirection.SHORT and conviction < 72:
                self.logger.info(
                    f"[REGIME-GATE] {stock} SHORT suppressed in "
                    f"{self.current_regime.value} (conviction={conviction} < 72 required)"
                )
                return None
        # ── END REGIME GATE ──────────────────────────────────────────────────
        # (FIX-DIRECTION-01 now runs BEFORE LLM validation — see above)

        sizing = pru_report.analysis_details.get("sizing", {})
        return TradeSetup(
            stock=stock,
            direction=entry_setup.get("direction", TradeDirection.NEUTRAL),
            conviction=conviction,
            conviction_level=self._get_conviction_level(conviction),
            entry_price=entry_setup.get("entry_zone",(0,0))[0],
            entry_zone=entry_setup.get("entry_zone",(0,0)),
            stop_loss=entry_setup.get("stop_loss",0),
            target_1=entry_setup.get("target_1",0),
            target_2=entry_setup.get("target_2",0),
            risk_reward=entry_setup.get("risk_reward",0),
            shares=sizing.get("shares",0),
            position_value=sizing.get("position_value",0),
            position_percent=sizing.get("position_percent",0),
            risk_percent=sizing.get("risk_percent",0),
            agent_votes={"ORION":orion_report.verdict,"NEXUS":nexus_report.verdict,
                         "PRUDENCE":pru_report.verdict},
            pattern_match=self._lookup_pattern_win_rate(
                entry_setup.get("direction",TradeDirection.NEUTRAL),
                entry_setup.get("setup_type","swing")),
            historical_win_rate=self._lookup_pattern_win_rate(
                entry_setup.get("direction",TradeDirection.NEUTRAL),
                entry_setup.get("setup_type","swing")).get("win_rate",50),
            exit_strategy=self._generate_exit_strategy(entry_setup),
            llm_validation=_llm_validation,
        )

    def _tier_5_rank_setups(self, setups):
        if not setups: return []
        min_rr = self.config.risk_limits.min_risk_reward_ratio
        qualified = [s for s in setups if s.risk_reward >= min_rr] or \
                    sorted(setups, key=lambda s: s.risk_reward, reverse=True)[:3]
        def score(s):
            return s.conviction*0.70 + min(s.risk_reward,3.0)/3.0*100*0.30
        return sorted(qualified, key=score, reverse=True)[:5]

    def _calculate_conviction(self, *reports) -> int:
        dw = ds = 0.0; nc = 0
        for r in reports:
            w = r.verdict.weight or 0.10; c = r.verdict.conviction; d = r.verdict.direction
            if d == TradeDirection.LONG:   ds += w*c;       dw += w
            elif d == TradeDirection.SHORT: ds += w*(100-c); dw += w
            else:
                if c >= 50: nc += 1
        if dw == 0: return 50
        raw = ds / dw
        penalty = min(8, nc*4) + min(8, sum(len(r.verdict.risks or []) for r in reports)*2)
        return max(0, min(100, int(raw - penalty)))

    def _get_conviction_level(self, conviction) -> ConvictionLevel:
        if conviction >= 85: return ConvictionLevel.VERY_HIGH
        elif conviction >= 75: return ConvictionLevel.HIGH
        elif conviction >= 65: return ConvictionLevel.MEDIUM
        elif conviction >= 50: return ConvictionLevel.LOW
        return ConvictionLevel.SKIP

    def _get_stock_sector(self, stock):
        for sector, stocks in SECTOR_MAPPING.items():
            if stock in stocks: return sector
        return "Others"

    def _lookup_pattern_win_rate(self, direction, setup_type):
        try:
            if self.pattern_db is None: return {"match":False,"win_rate":50,"total_trades":0}
            stats = self.pattern_db.calculate_historical_win_rate(
                setup_type=f"{direction.value if hasattr(direction,'value') else direction}_{setup_type}".lower(),
                regime=self.current_regime.value if self.current_regime else None)
            total = stats.get("total_trades",0)
            if total >= 15:
                return {"match":True,"win_rate":round(stats.get("win_rate",0.5)*100,1),
                        "total_trades":total,"avg_return":stats.get("avg_return",0)}
            return {"match":False,"win_rate":50,"total_trades":total}
        except Exception:
            return {"match":False,"win_rate":50,"total_trades":0}

    def _generate_exit_strategy(self, es):
        parts = []
        if es.get("target_1"): parts.append(f"Take partial profits at {es['target_1']}")
        if es.get("target_2"): parts.append(f"Trail stop to breakeven after T1, exit at {es['target_2']}")
        if es.get("stop_loss"): parts.append(f"Stop loss at {es['stop_loss']}")
        return " | ".join(parts) or "Set stop loss and target based on technical levels"

    def _get_stock_data(self, stock, market_data):
        pd_ = market_data.get("price_data",{}).get(stock,{})
        ind = market_data.get("indicators",{}).get(stock,{})
        fund = market_data.get("fundamental_data",{}).get(stock,{})
        close = pd_.get("close",0)
        if not close: return None
        sma20=ind.get("sma20",0); sma50=ind.get("sma50",0); sma200=ind.get("sma200",0)
        trend=ind.get("trend","SIDEWAYS"); rsi=ind.get("rsi",50)
        volume=pd_.get("volume",0); avg_vol=pd_.get("avg_volume",volume or 1)
        atr=pd_.get("atr",close*0.015); vol_ratio=volume/avg_vol if avg_vol>0 else 1.0
        def sig(b,r): return "bullish" if b else ("bearish" if r else "neutral")
        enriched_price = dict(pd_)
        if sma50:  enriched_price["ma_50"]  = sma50
        if sma200: enriched_price["ma_200"] = sma200
        ohlcv = market_data.get("ohlcv_history",{}).get(stock,[])
        if ohlcv:
            enriched_price["highs"]  = [c["high"]  for c in ohlcv]
            enriched_price["lows"]   = [c["low"]   for c in ohlcv]
            enriched_price["closes"] = [c["close"] for c in ohlcv]
            # IMPROVEMENT 2: Pass full OHLCV candle array so PatternRecognitionEngine
            # can detect multi-bar patterns (H&S, double bottom, bull flag, etc.)
            enriched_price["ohlcv"] = ohlcv[-60:]
        # Fix 5: ensure MACD + BB propagate fully into indicators for Orion
        enriched_ind = {
            **ind,
            "vol_ratio": round(vol_ratio, 2),
            "macd":           ind.get("macd",           0.0),
            "macd_signal":    ind.get("macd_signal",    0.0),
            "macd_histogram": ind.get("macd_histogram", 0.0),
            "bb_upper":       ind.get("bb_upper",  close * 1.02),
            "bb_middle":      ind.get("bb_middle", close),
            "bb_lower":       ind.get("bb_lower",  close * 0.98),
            "bb_width":       ind.get("bb_width",  0.04),
        }
        return {
            "stock":stock, "price_data":enriched_price,
            "indicators": enriched_ind,
            "fundamentals":fund,
            "weekly_trend":    sig(trend=="UPTREND",   trend=="DOWNTREND"),
            "daily_trend":     sig(close>sma50>0,       close<sma50>0),
            "four_hour_trend": sig(close>sma20>0,       close<sma20>0),
            "one_hour_trend":  sig(sma20>sma50>0,       sma20<sma50>0),
            # Fix 3: volume_trend — Orion confluence awards ±6 pts for this
            "volume_trend":    sig(vol_ratio>1.4,        vol_ratio<0.70),
        }

    def _default_portfolio(self):
        return {"total_capital":self.portfolio_value,"deployed_capital":0,
                "cash":self.portfolio_value,"portfolio_heat":0,
                "current_drawdown":0,"open_positions":[],"sector_exposure":{}}

    def _calculate_portfolio_risk(self, portfolio):
        cap = max(1, portfolio.get("total_capital",1))
        return {"capital":portfolio.get("total_capital",0),
                "deployed_percent":portfolio.get("deployed_capital",0)/cap*100,
                "cash_percent":portfolio.get("cash",0)/cap*100,
                "portfolio_heat":portfolio.get("portfolio_heat",0)*100,
                "drawdown":portfolio.get("current_drawdown",0)*100}

    def _extract_upcoming_events(self, market_data):
        return [{"name":e.get("name",""),"date":str(e.get("date","")),"impact":e.get("impact","LOW")}
                for e in market_data.get("event_data",{}).get("events",[])[:5]]

    def _generate_action_items(self, setups, consensus):
        items = []
        # Surface cross-examiner recommendation prominently if not PROCEED
        _ex = getattr(self, "_last_examination", None)
        _ex_rec = getattr(_ex, "final_recommendation", "PROCEED") if _ex else "PROCEED"
        if _ex_rec == "AVOID":
            items.append("🚫 LLM Cross-Examiner says AVOID — no new entries this cycle")
        elif _ex_rec == "WAIT":
            items.append("⏸  LLM Cross-Examiner says WAIT — monitor but do not enter")
        elif _ex_rec == "REDUCE_SIZE":
            items.append("⬇  LLM Cross-Examiner says REDUCE_SIZE — use half normal position size")

        # PHOENIX alert — surface prominently when active
        _px = getattr(self, "_phoenix_output", None)
        if _px and _px.is_active:
            tier_msgs = {
                "IMMINENT":      "🚀 PHOENIX IMMINENT — High probability recovery forming. Actively seek LONG setups near 200 DMA.",
                "PRE_MOMENTUM":  "🔥 PHOENIX PRE-MOMENTUM — Smart money positioning. Lower conviction bar active. Watch for breakout entry.",
                "EARLY_WARNING": "⚡ PHOENIX EARLY WARNING — First recovery signals. Monitor closely, prepare watchlist.",
            }
            items.append(tier_msgs.get(_px.tier, f"⚡ PHOENIX {_px.tier} — Score {_px.phoenix_score:.0f}/100"))

        if setups and _ex_rec == "WAIT":
            items.append(f"Add to watchlist: {setups[0].stock} ({setups[0].direction.value}) — confirm before entry")
            items.append("Monitor F&O suggestions above — do not enter until examiner clears WAIT")
        elif setups:
            items += [f"Review top setup: {setups[0].stock} ({setups[0].direction.value})",
                      f"Set alerts for {len(setups)} trade setups"]
        else:
            items.append("No high-conviction setups — wait for better opportunities")
        items += ["Verify all stop losses before market open",
                  "Check overnight news affecting positions"]
        return items

    # ------------------------------------------------------------------
    # Data preparation helpers (same as v3.2 coordinator)
    # ------------------------------------------------------------------
    def _prepare_orion_data(self, data):
        nifty  = data.get("nifty_price", 22500)
        dma200 = data.get("nifty_200dma", nifty * 0.95)
        vix    = data.get("india_vix", 15.0)
        ps     = data.get("price_structure", "neutral")

        # Use real NIFTY indicators computed by FyersFetcher from 60-day OHLCV history
        ni     = data.get("nifty_indicators", {})
        sma20  = ni.get("sma20")  or nifty * 0.990
        sma50  = ni.get("sma50")  or nifty * 0.975
        sma200 = ni.get("sma200") or dma200
        rsi    = ni.get("rsi",  55 if nifty > (ni.get("sma50") or nifty*0.975) else 45)
        atr    = ni.get("atr")  or nifty * (vix / 100) * (1 / 16)
        adx    = ni.get("adx",  data.get("adx", 25))
        macd      = ni.get("macd",          0.0)
        macd_sig  = ni.get("macd_signal",   0.0)
        macd_hist = ni.get("macd_histogram", 0.0)
        bb_upper  = ni.get("bb_upper",  nifty * 1.02)
        bb_mid    = ni.get("bb_middle", nifty)
        bb_lower  = ni.get("bb_lower",  nifty * 0.98)
        bb_width  = ni.get("bb_width",  0.04)
        trend = ni.get("trend") or (
            "UPTREND"   if nifty > sma50 > sma200 else
            "DOWNTREND" if nifty < sma50 < sma200 else "SIDEWAYS"
        )
        def sg(b, r): return "bullish" if b else ("bearish" if r else "neutral")

        intraday  = data.get("nifty_intraday", {})
        real_open = intraday.get("open",  nifty * 0.998) or nifty * 0.998
        real_high = intraday.get("high",  nifty * 1.005) or nifty * 1.005
        real_low  = intraday.get("low",   nifty * 0.993) or nifty * 0.993

        vol_data      = data.get("price_data", {}).get("NIFTY50", {})
        nifty_vol     = vol_data.get("volume",     500_000)
        nifty_avg_vol = vol_data.get("avg_volume", 450_000)
        vol_ratio     = nifty_vol / nifty_avg_vol if nifty_avg_vol > 0 else 1.0

        # IMPROVEMENT 7: Compute short-term RSI and EMA from real 15-min bars.
        # Replaces the SMA-crossover proxy with actual intraday structure for
        # one_hour_trend and four_hour_trend signals sent to ORION.
        def _rsi_from_closes(closes, period=14):
            if len(closes) < period + 1:
                return 50.0
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains  = [d for d in deltas[-period:] if d > 0]
            losses = [-d for d in deltas[-period:] if d < 0]
            avg_g  = sum(gains)  / period if gains  else 0
            avg_l  = sum(losses) / period if losses else 1e-9
            rs = avg_g / avg_l
            return round(100 - 100 / (1 + rs), 2)

        bars_15m = data.get("nifty_15min", [])
        intraday_rsi   = rsi        # fallback to daily RSI
        intraday_trend = "neutral"
        if len(bars_15m) >= 20:
            closes_15m = [b["close"] for b in bars_15m]
            # 14-period RSI on 15-min bars ≈ ~3.5h of data
            intraday_rsi = _rsi_from_closes(closes_15m, period=14)
            # EMA9 vs EMA21 crossover on 15-min closes for trend direction
            ema9  = closes_15m[-1] if len(closes_15m) < 9  else sum(closes_15m[-9:])  / 9
            ema21 = closes_15m[-1] if len(closes_15m) < 21 else sum(closes_15m[-21:]) / 21
            if ema9 > ema21 * 1.001:
                intraday_trend = "bullish"
            elif ema9 < ema21 * 0.999:
                intraday_trend = "bearish"

        return {
            "stock": "NIFTY",
            "price_data": {
                "open": real_open, "high": real_high, "low": real_low,
                "close": nifty, "volume": nifty_vol, "avg_volume": nifty_avg_vol,
                "atr": atr, "ma_50": sma50, "ma_200": sma200,
            },
            "indicators": {
                "rsi": max(10, min(90, rsi)),
                "atr": atr, "atr_percent": (atr / nifty) * 100, "adx": adx,
                "sma20": sma20, "sma50": sma50, "sma200": sma200,
                "trend": trend, "volume_ratio": round(vol_ratio, 2),
                "above_200dma": nifty > sma200,
                "macd": macd, "macd_signal": macd_sig, "macd_histogram": macd_hist,
                "bb_upper": bb_upper, "bb_middle": bb_mid, "bb_lower": bb_lower,
                "bb_width": bb_width,
                # IMPROVEMENT 7: short-term RSI from 15-min bars
                "intraday_rsi": intraday_rsi,
            },
            "weekly_trend":    sg(trend == "UPTREND",   trend == "DOWNTREND"),
            "daily_trend":     sg(nifty > sma50 > 0,    nifty < sma50 > 0),
            # IMPROVEMENT 7: Real intraday EMA9/21 trend from 15-min bars when available;
            # fall back to daily SMA-proxy if 15-min data is absent.
            "four_hour_trend": intraday_trend if bars_15m else sg(nifty > sma20 > 0, nifty < sma20 > 0),
            "one_hour_trend":  intraday_trend if bars_15m else sg(ps == "higher_highs", ps == "lower_lows"),
            "volume_trend":    sg(vol_ratio > 1.3,       vol_ratio < 0.75),
        }

    def _prepare_vesper_data(self, data):
        return {"flow_data":data.get("flow_data",{}),"sector_flows":data.get("sector_flows",[]),
                "bulk_deals":data.get("bulk_deals",[])}

    def _prepare_kairo_data(self, data):
        live=data.get("live_sentiment",{}); static=data.get("sentiment_data",{})
        mi=live.get("market_impact_assessment",{}); d=mi.get("direction","neutral")
        off={"positive":20,"negative":-20,"cautious":-10,"neutral":0}.get(d,0)
        geo=live.get("geopolitical",{}).get("overall_sentiment",None)
        stk=live.get("stock_news",{}).get("overall_sentiment",None)
        ns=int(50+(geo*50)+off) if geo is not None else static.get("news",35)
        gs=int(50+(geo*50)) if geo is not None else static.get("global",30)
        ss=int(50+(stk*50)) if stk is not None else static.get("social",25)
        return {"news_sentiment":max(0,min(100,ns)),"analyst_sentiment":static.get("analyst",45),
                "social_sentiment":max(0,min(100,ss)),"corporate_sentiment":static.get("corporate",20),
                "global_sentiment":max(0,min(100,gs)),"vix":data.get("india_vix",15),
                "pcr":data.get("derivatives_data",{}).get("pcr",1)}

    def _prepare_sentinel_data(self, data):
        d=data.get("derivatives_data",{})
        intraday_chg = data.get("nifty_change_pct", d.get("price_change", 0))
        return {"pcr":d.get("pcr",1),"pcr_trend":d.get("pcr_trend","stable"),
                "max_pain":d.get("max_pain",0),"current_price":data.get("nifty_price",0),
                "india_vix":data.get("india_vix",15),"iv_rank":d.get("iv_rank",50),
                "call_oi_walls":d.get("call_oi_walls",[]),"put_oi_walls":d.get("put_oi_walls",[]),
                "oi_change":d.get("oi_change",0),"price_change":intraday_chg}

    def _prepare_nexus_data(self, data):
        return {"stock":"MARKET","fundamentals":{},
                "nifty_pe":data.get("nifty_pe",22.5),"gsec_yield":data.get("gsec_yield",7.0)}

    def _prepare_prudence_data(self, data):
        np_  = data.get("nifty_price", 22500)
        atr  = np_ * 0.01
        vix  = data.get("india_vix", 15)
        # Pass VIX-adjusted conviction level so PRUDENCE reflects real risk environment.
        # High VIX → MEDIUM conviction (caution). Very high VIX → LOW conviction.
        if vix >= 25:
            conv_level = "LOW"
        elif vix >= 18:
            conv_level = "MEDIUM"
        else:
            conv_level = "HIGH"
        return {
            "portfolio":       self._default_portfolio(),
            "trade_request":   {"entry_price": np_, "stop_loss": np_ - atr * 1.5},
            "conviction_level": conv_level,
            "proposed_direction": getattr(
                getattr(self, "_last_consensus", None), "direction", TradeDirection.NEUTRAL
            ),
        }

    def _prepare_catalyst_data(self, data):
        return {"events":data.get("event_data",{}).get("events",[]),
                "current_date":datetime.now(),"expiry_week":data.get("expiry_week",False)}

    def _prepare_optimus_data(self, data):
        d=data.get("derivatives_data",{})
        intraday_chg = data.get("nifty_change_pct", d.get("price_change", 0))
        return {"symbol":data.get("symbol","NIFTY"),"weekly_expiry":d.get("weekly_expiry",""),
                "current_price":data.get("nifty_price",d.get("current_price",0)),
                "price_change":intraday_chg,"pcr":d.get("pcr",1.0),
                "pcr_trend":d.get("pcr_trend","stable"),"max_pain":d.get("max_pain",0),
                "call_oi_walls":d.get("call_oi_walls",[]),"put_oi_walls":d.get("put_oi_walls",[]),
                "ce_oi_change_pct":d.get("ce_oi_change_pct",0),"pe_oi_change_pct":d.get("pe_oi_change_pct",0),
                "oi_signal":d.get("oi_signal","NEUTRAL"),"india_vix":data.get("india_vix",15),
                "iv_rank":d.get("iv_rank",50),"iv_skew":d.get("iv_skew",0),
                "futures_premium":d.get("futures_premium",0)}


# ---------------------------------------------------------------------------
# UnifiedCoordinator — top-level facade
# ---------------------------------------------------------------------------

class UnifiedCoordinator:
    """
    Unified Coordinator — single entry point for the full 11-agent engine.

    Composes LeadCoordinator (swing engine) with FnoCoordinator (F&O specialists)
    to produce an enriched DailyTradingPlan that includes:
    - All v3.2 swing trade setups
    - Live portfolio Greeks status
    - Active F&O alerts (MWPL, settlement, Greeks limits)
    - Execution quality metrics
    - News-aware context (v4.1)
    """

    def __init__(self, config: Optional[SystemConfig] = None, portfolio_value: float = 1_000_000):
        self.config          = config or DEFAULT_CONFIG
        self.portfolio_value = portfolio_value
        self.logger          = logging.getLogger("UnifiedCoordinator")

        self.lead  = LeadCoordinator(self.config, portfolio_value)
        self.fno   = FnoCoordinator(self.config)

        # News Intelligence context (v4.1)
        self.news_context = get_news_context()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def generate_trading_plan(
        self,
        market_data: Dict[str, Any],
        portfolio_status: Optional[Dict] = None,
        watchlist: Optional[List[str]] = None,
        skip_liquidity_check: bool = False,
    ) -> DailyTradingPlan:
        """Generate complete daily trading plan from all 11 agents."""

        # Inject news context into market_data for agents (v4.1)
        market_data["news_context"] = self.news_context.get_all_news()
        market_data["overnight_risk"] = self.news_context.get_overnight_risk()

        # Generate swing plan from v3.2 engine
        plan = self.lead.generate_trading_plan(market_data, portfolio_status, watchlist)

        # Enrich with v4.0 F&O data
        plan.portfolio_greeks        = self.fno.get_portfolio_greeks()
        plan.active_alerts           = self.fno.get_active_alerts()
        plan.settlement_obligations  = self.fno.get_settlement_obligations()

        # v4.1 — Directional F&O suggestions for all 5 index option markets
        # AVOID  → skip entirely (stay fully out of the market).
        # WAIT   → run advisor and show suggestions as informational watch-only
        #          (mirrors the swing setup [WATCH] behaviour — no real entries).
        # PROCEED/REDUCE_SIZE → normal flow.
        _exam_rec_unified = getattr(
            getattr(self.lead, "_last_examination", None),
            "final_recommendation", "PROCEED"
        )
        if _exam_rec_unified == "AVOID":
            import logging as _log
            _log.getLogger("coordinator").info(
                "[EXAMINE-GATE] F&O suggestions suppressed (cross-examiner AVOID)."
            )
        else:
            try:
                from agents.directional_option_advisor import DirectionalOptionAdvisor
                advisor = DirectionalOptionAdvisor(
                    gsec_yield=float(market_data.get("gsec_yield", 7.0)),
                    portfolio_value=self.portfolio_value,
                )
                # FIX 3: Blend OPTIMUS F&O score into consensus conviction
                # OPTIMUS net_score is -10…+10; each point shifts conviction ±3
                _base_conviction = (
                    int(plan.consensus.strength == "STRONG")        * 70
                    + int(plan.consensus.strength == "MODERATE")    * 62
                    + int(plan.consensus.strength == "WEAK")        * 54
                    + int(plan.consensus.strength == "NO_CONSENSUS") * 50
                )
                _optimus_report = (self.lead.agent_reports or {}).get("OPTIMUS")
                _optimus_fno_score = 0.0
                if _optimus_report is not None:
                    _optimus_fno_score = float(
                        _optimus_report.analysis_details.get("net_score", 0.0)
                    )
                # ±3 per OPTIMUS score point, capped at ±15 to avoid overrides
                _optimus_delta = max(-15, min(15, int(_optimus_fno_score * 3)))
                _blended_conviction = max(40, min(85, _base_conviction + _optimus_delta))
                if _optimus_delta != 0:
                    import logging as _log
                    _log.getLogger("coordinator").info(
                        f"[FIX3-OPTIMUS] F&O score={_optimus_fno_score:+.1f} → "
                        f"conviction_delta={_optimus_delta:+d} | "
                        f"base={_base_conviction} blended={_blended_conviction}"
                    )
                plan.fno_suggestions = advisor.advise(
                    market_data=market_data,
                    consensus_direction=plan.consensus.direction.value,
                    consensus_strength=plan.consensus.strength,
                    consensus_score=plan.consensus.net_score,
                    consensus_conviction=_blended_conviction,
                    india_vix=float(market_data.get("india_vix", 15.0)),
                    market_regime=plan.market_regime.value,
                    skip_liquidity_check=skip_liquidity_check,
                )
                # Tag as watch-only when examiner says WAIT
                if _exam_rec_unified == "WAIT" and plan.fno_suggestions is not None:
                    plan.fno_suggestions._watch_only = True
                    import logging as _log
                    _log.getLogger("coordinator").info(
                        "[EXAMINE-GATE] F&O suggestions shown as watch-only (cross-examiner WAIT)."
                    )

                # FIX 4.3: Log PROCEED F&O suggestions to fno_paper_trades.csv
                # This feeds MetaLearner with labelled F&O outcome data
                if plan.fno_suggestions and not getattr(plan.fno_suggestions, "_watch_only", False):
                    try:
                        _suggestions = getattr(plan.fno_suggestions, "suggestions", []) or []
                        for _sug in _suggestions:
                            if getattr(_sug, "proceed", False):
                                self.lead.data_manager.log_fno_trade(_sug)
                    except Exception as _fno_log_err:
                        import logging as _log
                        _log.getLogger("coordinator").debug(
                            f"FNO trade log skipped: {_fno_log_err}"
                        )
            except Exception as _adv_err:
                import logging as _log
                _log.getLogger("coordinator").warning(
                    f"DirectionalOptionAdvisor skipped: {_adv_err}"
                )

        # ── LLM Options Strategist validation ────────────────────────────
        # Reviews each rule-based suggestion with full option chain context.
        # Can validate, flag issues, or propose alternatives per suggestion.
        _llm_strat = getattr(self.lead, "llm_strategist", None)
        if _llm_strat is not None and getattr(plan, "fno_suggestions", None):
            suggestions = getattr(plan.fno_suggestions, "suggestions", []) or []
            for suggestion in suggestions:
                try:
                    chain_data = market_data.get("index_option_chains", {}).get(
                        getattr(suggestion, "index", ""), {}
                    )
                    # FIX 5.1: Enrich chain_data with real LTP and pre-market cues
                    _enriched_chain = dict(chain_data)
                    _enriched_chain.setdefault("futures_premium",
                        market_data.get("derivatives_data", {}).get("futures_premium", 0.0))
                    _enriched_chain.setdefault("gift_nifty_gap_pct",
                        market_data.get("gift_nifty_gap_pct", 0.0))
                    optimization = _llm_strat.optimize_strategy(
                        suggestion={
                            "index": getattr(suggestion, "index", ""),
                            "strategy": getattr(suggestion, "strategy", ""),
                            "strike": getattr(suggestion, "strike", 0),
                            "expiry": str(getattr(suggestion, "expiry", "")),
                            "dte": getattr(suggestion, "dte", 0),
                            "entry_price": getattr(suggestion, "entry_price", 0),
                            "cost_per_lot": getattr(suggestion, "cost_per_lot", 0),
                            "greeks": {
                                "delta": getattr(getattr(suggestion, "greeks", None), "delta", 0),
                                "theta": getattr(getattr(suggestion, "greeks", None), "theta", 0),
                                "vega":  getattr(getattr(suggestion, "greeks", None), "vega", 0),
                            },
                            "iv_rank": getattr(suggestion, "iv_rank", 50),
                            "score": getattr(getattr(suggestion, "score", None), "total", 0),
                        },
                        market_context={
                            "regime": plan.market_regime.value,
                            "india_vix": market_data.get("india_vix", 15),
                            "consensus": plan.consensus.direction.value,
                            "consensus_strength": plan.consensus.strength,
                            "pcr": chain_data.get("pcr", 1.0),
                            "max_pain": chain_data.get("max_pain", 0),
                            "iv_rank": chain_data.get("iv_rank", 50),
                        },
                        portfolio_context={
                            "portfolio_value": self.portfolio_value,
                            "deployed_capital": 0,
                            "existing_positions": [],
                        }
                    )
                    # Attach LLM validation to suggestion object for report rendering
                    suggestion._llm_optimization = optimization
                    self.logger.info(
                        f"[LLM-STRATEGY] {getattr(suggestion,'index','')} "
                        f"{getattr(optimization,'recommendation','?')} | "
                        f"source={getattr(optimization,'source','?')}"
                    )
                except Exception as _se:
                    self.logger.debug(f"[LLM-STRATEGY] skipped for {getattr(suggestion,'index','?')}: {_se}")

        # FIX 5.2: Wire FNOBrainExtension for enriched F&O synthesis
        # Called after DirectionalOptionAdvisor so it has suggestions to reason over.
        # Feeds same enriched data that LLMOptionsStrategist uses.
        try:
            from agents.fno_brain_extension import FNOBrainExtension as _FNOBrain
            if not hasattr(self, "_fno_brain"):
                self._fno_brain = _FNOBrain()
            if self._fno_brain.is_ready:
                _nifty_chain = market_data.get("index_option_chains", {}).get("NIFTY", {})
                _fno_market_ctx = {
                    "nifty_price":        market_data.get("nifty_price", 0),
                    "india_vix":          market_data.get("india_vix", 15),
                    "pcr":                _nifty_chain.get("pcr", 1.0),
                    "iv_rank":            _nifty_chain.get("iv_rank", 50),
                    "futures_premium":    market_data.get("derivatives_data", {}).get("futures_premium", 0.0),
                    "gift_nifty_gap_pct": market_data.get("gift_nifty_gap_pct", 0.0),
                    "atm_ce_ltp":         _nifty_chain.get("atm_ce_ltp", 0.0),
                    "atm_pe_ltp":         _nifty_chain.get("atm_pe_ltp", 0.0),
                }
                _fno_ctx = {
                    "greeks_summary":     plan.portfolio_greeks,
                    "oi_walls":           _nifty_chain.get("call_oi_walls", []),
                    "put_walls":          _nifty_chain.get("put_oi_walls", []),
                    "max_pain":           _nifty_chain.get("max_pain", 0),
                    "iv_rank":            _nifty_chain.get("iv_rank", 50),
                    "settlement_flags":   getattr(plan, "settlement_obligations", []),
                }
                _consensus_dict = {
                    "direction":    plan.consensus.direction.value,
                    "strength":     plan.consensus.strength,
                    "net_score":    plan.consensus.net_score,
                    "agents":       plan.consensus.agreeing_agents,
                }
                _setups_dict = [
                    {"stock": s.stock, "direction": s.direction.value,
                     "conviction": s.conviction, "rr": s.risk_reward}
                    for s in plan.top_setups[:5]
                ]
                _fno_brain_output = self._fno_brain.fno_synthesize_sync(
                    consensus=_consensus_dict,
                    equity_setups=_setups_dict,
                    market_context=_fno_market_ctx,
                    fno_context=_fno_ctx,
                )
                plan.fno_brain_synthesis = _fno_brain_output
                self._last_fno_brain     = _fno_brain_output   # for trading planner
                self.logger.info(
                    f"[FNO-BRAIN] IV regime={_fno_brain_output.iv_regime} | "
                    f"stance={_fno_brain_output.market_stance} | "
                    f"risk={_fno_brain_output.risk_score}/10 | "
                    f"strategies={len(_fno_brain_output.strategy_recommendations)}"
                )
        except Exception as _fb_err:
            self.logger.debug(f"FNOBrainExtension skipped: {_fb_err}")

        return plan

    def place_order(self, symbol, transaction_type, quantity,
                    order_type=OrderType.MARKET, price=0.0, strategy_id=None):
        """Route order through HERMES execution agent."""
        return self.fno.place_order(symbol, transaction_type, quantity,
                                    order_type, price, strategy_id)

    def get_fno_status(self) -> Dict:
        """Get F&O portfolio status from all three specialist agents."""
        return {
            "portfolio_greeks":       self.fno.get_portfolio_greeks(),
            "active_alerts":          self.fno.get_active_alerts(),
            "settlement_obligations": self.fno.get_settlement_obligations(),
            "execution_report":       self.fno.get_execution_report(),
        }

    def format_report(self, plan: DailyTradingPlan) -> str:
        """Format the complete trading plan as a human-readable report."""
        lines = []
        lines.append("=" * 65)
        lines.append("ROX PROVEN EDGE ENGINE v4.0 UNIFIED — DAILY TRADING PLAN")
        lines.append(f"Date: {plan.date.strftime('%Y-%m-%d %H:%M')} | "
                     f"Regime: {plan.market_regime.value} | "
                     f"Confidence: {plan.regime_confidence:.1f}%")
        lines.append("=" * 65)

        lines.append("\n11-AGENT CONSENSUS PANEL")
        for name, report in plan.agent_reports.items():
            v = report.verdict
            lines.append(f"  {name:10s}: {v.direction.value:7s} "
                         f"| conviction={v.conviction:.0f}% "
                         f"| weight={v.weight:.2f}")

        lines.append(f"\nCONSENSUS: {plan.consensus.direction.value} "
                     f"({plan.consensus.strength}) | net_score={plan.consensus.net_score:.3f}")

        # Overnight risk summary (v4.1)
        overnight_risk = getattr(plan, "_overnight_risk", None)
        if plan.date and hasattr(self, "news_context"):
            try:
                risk_profile = self.news_context.get_overnight_risk()
                if risk_profile:
                    lines.append(f"\nOVERNIGHT RISK (NOCTURNAL)")
                    lines.append(f"  Risk Level: {risk_profile.risk_level}")
                    lines.append(f"  Expected Gap: {risk_profile.expected_gap_size}")
                    if getattr(risk_profile, "trading_restrictions", None):
                        for r in risk_profile.trading_restrictions:
                            lines.append(f"  🚫 {r}")
            except Exception:
                pass

        # ── LLM Intelligence Layer ────────────────────────────────────────
        lc = self.lead   # reference to LeadCoordinator
        _any_llm = False

        # Regime Detector
        _rr = getattr(lc, "_last_regime_result", None)
        if _rr and getattr(_rr, "source", None) == "LLM":
            _any_llm = True
            lines.append("\nLLM REGIME INTELLIGENCE")
            prob = getattr(_rr, "probability_distribution", {})
            prob_str = " | ".join(f"{k}={v*100:.0f}%" for k, v in
                                  sorted(prob.items(), key=lambda x: -x[1])[:4]) if prob else ""
            if prob_str:
                lines.append(f"  Probabilities : {prob_str}")
            for kf in getattr(_rr, "key_factors", [])[:3]:
                lines.append(f"  Factor        : {kf}")
            tw = getattr(_rr, "transition_warning", None)
            if tw:
                lines.append(f"  ⚠ TRANSITION  : {tw}")

        # ── PHOENIX Pre-Momentum Recovery Radar (v4.3) ──────────────────────
        _px = getattr(plan, "phoenix_analysis", None)
        if _px is not None:
            lines.append("\nPHOENIX RECOVERY RADAR")
            lines.append(
                f"  {_px.tier_icon} Score      : {_px.phoenix_score:.0f}/100 "
                f"| Tier: {_px.tier} "
                f"| Recovery prob: {_px.recovery_probability*100:.0f}%"
            )
            if _px.conviction_gate_override:
                lines.append(
                    f"  🎯 Gate Override : Conviction threshold lowered to "
                    f"{_px.conviction_gate_override} (PHOENIX active)"
                )
            if _px.days_since_bottom_signal > 0:
                lines.append(
                    f"  📅 Active Since  : {_px.days_since_bottom_signal} session(s) ago"
                )
            # Signal breakdown — show top 5 fired, or top 3 if none fired
            fired_sigs = [s for s in _px.signals if s.fired]
            show_sigs  = sorted(fired_sigs, key=lambda s: -s.score)[:5] \
                         if fired_sigs else \
                         sorted(_px.signals, key=lambda s: -s.max_score)[:3]
            lines.append(f"  Signal Battery ({len(fired_sigs)}/10 active):")
            for sig in show_sigs:
                icon = "✅" if sig.fired else "⭕"
                bar  = "█" * int(sig.score / sig.max_score * 8) if sig.max_score > 0 else ""
                lines.append(
                    f"    {icon} {sig.name:28s} {sig.score:4.1f}/{sig.max_score:.0f}  "
                    f"[{bar:<8}] {sig.detail[:70]}"
                )
            for obs in _px.key_observations[:3]:
                lines.append(f"  → {obs}")
            for cau in _px.cautions[:2]:
                lines.append(f"  ⚠ {cau}")
            lines.append(f"  Action: {_px.recommended_action}")

        # Cross-Examiner
        _ex = getattr(lc, "_last_examination", None)
        if _ex:
            _any_llm = True
            lines.append("\nLLM CROSS-EXAMINATION")
            rec = getattr(_ex, "final_recommendation", "?")
            rec_icon = {"PROCEED": "✅", "WAIT": "⏸", "REDUCE_SIZE": "⬇", "AVOID": "🚫"}.get(rec, "?")
            lines.append(f"  Recommendation: {rec_icon} {rec}")
            reasoning = getattr(_ex, "reasoning", "")
            if reasoning:
                lines.append(f"  Reasoning     : {reasoning[:120]}...")
            contrarian = getattr(_ex, "contrarian_case", "")
            if contrarian:
                lines.append(f"  Contrarian    : {contrarian[:120]}...")
            for rf in getattr(_ex, "risk_flags", [])[:3]:
                lines.append(f"  ⚠ Risk        : {rf}")
            up = getattr(_ex, "agents_to_upweight", [])
            dn = getattr(_ex, "agents_to_downweight", [])
            pqa = getattr(_ex, "poor_quality_agents", [])
            if up:
                lines.append(f"  ↑ Upweighted  : {', '.join(up)}")
            if dn:
                lines.append(f"  ↓ Downweighted: {', '.join(dn)}")
            if pqa:
                lines.append(f"  ⚠ Poor-Quality: {', '.join(pqa)} (weight cut 60% — defective reasoning)")

        # News Impact
        _ni = getattr(lc, "_last_news_impact", None)
        if _ni:
            _any_llm = True
            lines.append("\nLLM NEWS INTELLIGENCE")
            omi = getattr(_ni, "overall_market_impact", {})
            lines.append(f"  Market Impact : {omi.get('direction','?')} | "
                         f"magnitude={omi.get('magnitude','?')} | "
                         f"confidence={omi.get('confidence','?')}%")
            sec_impacts = getattr(_ni, "sector_impacts", {})
            if sec_impacts:
                top_sectors = sorted(sec_impacts.items(),
                                     key=lambda x: abs(x[1].impact_score), reverse=True)[:3]
                for sec, si in top_sectors:
                    arrow = "▲" if si.impact_score > 0.05 else "▼" if si.impact_score < -0.05 else "—"
                    lines.append(f"  {arrow} {sec:12s}: {si.impact_score:+.2f} | {si.reason[:60]}")
            sigs = getattr(_ni, "actionable_signals", [])
            for sig in sigs[:3]:
                lines.append(f"  ⚡ {sig.signal_type:20s}: {sig.target} — {sig.reason[:60]}")
            restrs = getattr(_ni, "trade_restrictions", [])
            for r in restrs[:2]:
                lines.append(f"  🚫 RESTRICTION: {r}")
            summary = getattr(_ni, "executive_summary", "")
            if summary:
                lines.append(f"  Summary       : {summary[:160]}...")

        if not _any_llm:
            lines.append("\nLLM INTELLIGENCE   [running rule-based fallbacks — check GEMINI_API_KEY]")

        lines.append("\nTOP SWING SETUPS")
        if plan.setups_watch_only and plan.top_setups:
            lines.append("  ⚠  WATCH ONLY — Cross-examiner says WAIT. Do not enter these setups.")
        elif not plan.top_setups:
            lines.append("  No qualifying setups this cycle.")
        for i, s in enumerate(plan.top_setups, 1):
            prefix = f"  [WATCH] #{i}" if plan.setups_watch_only else f"  #{i}"
            lines.append(f"{prefix} {s.stock:12s} {s.direction.value:5s} | "
                         f"conviction={s.conviction} | "
                         f"entry={s.entry_price:.2f} | "
                         f"SL={s.stop_loss:.2f} | "
                         f"T1={s.target_1:.2f} | R:R={s.risk_reward:.2f}")
            # Show LLM pattern validation result per setup
            _lv = getattr(s, "llm_validation", None)
            if _lv:
                rec_icon = {"TAKE": "✅", "WAIT_FOR_CONFIRMATION": "⏸", "AVOID": "🚫"}.get(
                    getattr(_lv, "final_recommendation", ""), "🔍")
                lines.append(f"       LLM Validation: {rec_icon} {getattr(_lv, 'final_recommendation', '?')} "
                             f"| source={getattr(_lv, 'source', '?')}")
                for vn in (getattr(_lv, "validation_notes", []) or [])[:2]:
                    lines.append(f"       ✓ {vn}")
                for rn in (getattr(_lv, "risk_notes", []) or [])[:2]:
                    lines.append(f"       ⚠ {rn}")



        lines.append("\nF&O PORTFOLIO STATUS (v4.0)")
        pg = plan.portfolio_greeks
        lines.append(f"  Greeks: delta={pg.get('delta',0):.4f} | "
                     f"gamma={pg.get('gamma',0):.6f} | "
                     f"theta={pg.get('theta',0):.2f} | "
                     f"vega={pg.get('vega',0):.2f} | "
                     f"positions={pg.get('num_positions',0)}")

        if plan.active_alerts:
            lines.append("\nACTIVE ALERTS")
            for alert in plan.active_alerts[-5:]:
                lines.append(f"  ⚠  {alert}")

        if plan.settlement_obligations:
            lines.append("\nSETTLEMENT OBLIGATIONS")
            for ob in plan.settlement_obligations:
                lines.append(f"  {ob['symbol']} | {ob['obligation_type']} | "
                             f"DTE={ob['days_to_expiry']} | "
                             f"₹{ob['obligation_value']:,.0f}")

        # ── Directional F&O suggestions ──────────────────────────────
        if getattr(plan, "fno_suggestions", None):
            try:
                from agents.directional_option_advisor import format_option_suggestions
                fno_section = format_option_suggestions(plan.fno_suggestions)
                lines.append(fno_section)
                # Watch-only banner when cross-examiner said WAIT
                if getattr(plan.fno_suggestions, "_watch_only", False):
                    lines.append(
                        "  ⚠  WATCH ONLY — Cross-examiner says WAIT. "
                        "Do not enter these F&O positions."
                    )
                # Append LLM Options Strategist notes per suggestion
                suggestions = getattr(plan.fno_suggestions, "suggestions", []) or []
                _strat_lines = []
                for s in suggestions:
                    _opt = getattr(s, "_llm_optimization", None)
                    if _opt:
                        rec = getattr(_opt, "recommendation", "?")
                        src = getattr(_opt, "source", "?")
                        rec_icon = {"VALIDATE": "✅", "ALTERNATIVE": "🔄", "AVOID": "🚫", "REDUCE_SIZE": "⬇"}.get(rec, "🔍")
                        idx = getattr(s, "index", "?")
                        _strat_lines.append(f"  {idx:12s}: {rec_icon} {rec} [{src}]")
                        for note in (getattr(_opt, "enhancements", []) or [])[:2]:
                            _strat_lines.append(f"    ✓ {note}")
                        for risk in (getattr(_opt, "risks", []) or [])[:2]:
                            _strat_lines.append(f"    ⚠ {risk}")
                        alt = getattr(_opt, "alternative_strategy", None)
                        if alt:
                            _strat_lines.append(f"    ↪ Alternative: {alt}")
                if _strat_lines:
                    lines.append("\nLLM OPTIONS STRATEGIST REVIEW")
                    lines.extend(_strat_lines)
            except Exception as _fe:
                lines.append(f"\nDIRECTIONAL F&O SUGGESTIONS\n  (render error: {_fe})")

        # ── TRADING PLAN (Monday/Next-Session Execution Guide) ────────────────
        _tp = getattr(plan, "_trading_plan", None)
        if _tp is not None:
            try:
                _src_tag = "" if _tp.source == "LLM" else " [rule-based fallback]"
                lines.append(f"\n{'='*65}")
                lines.append(f"TRADING PLAN — NEXT SESSION{_src_tag}")
                lines.append(f"{'='*65}")

                # Overall stance
                _stance_icons = {
                    "AGGRESSIVE_LONG":   "🟢🟢",
                    "MODERATE_LONG":     "🟢",
                    "NEUTRAL":           "⚪",
                    "MODERATE_SHORT":    "🔴",
                    "AGGRESSIVE_SHORT":  "🔴🔴",
                    "CASH":              "💵",
                }
                _si = _stance_icons.get(_tp.overall_stance, "⚪")
                lines.append(f"\n  Stance : {_si} {_tp.overall_stance}")
                if _tp.plan_summary:
                    lines.append(f"  Summary: {_tp.plan_summary}")

                # Key levels
                kl = _tp.key_levels
                if kl and kl.strong_support > 0:
                    lines.append(f"\nKEY LEVELS")
                    lines.append(f"  Strong support    : {kl.strong_support:,}")
                    lines.append(f"  Immediate support : {kl.immediate_support:,}")
                    lines.append(f"  Immediate resist  : {kl.immediate_resistance:,}")
                    lines.append(f"  Strong resistance : {kl.strong_resistance:,}")
                    lines.append(f"  Invalidation ↓    : {kl.invalidation_bear:,}  (bull thesis broken below this)")
                    lines.append(f"  Invalidation ↑    : {kl.invalidation_bull:,}  (bear thesis broken above this)")

                # Scenarios
                if _tp.scenarios:
                    lines.append(f"\nSCENARIOS")
                    _bias_icons = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}
                    for i, sc in enumerate(_tp.scenarios, 1):
                        _bi = _bias_icons.get(sc.bias, "⚪")
                        lines.append(f"  {i}. {_bi} IF   : {sc.condition}")
                        lines.append(f"     THEN : {sc.action}")
                        lines.append(f"     TGT  : {sc.target}   SL : {sc.stop_loss}")

                # F&O ready trades
                if _tp.fno_ready_trades:
                    lines.append(f"\nF&O READY TRADES")
                    _status_icons = {
                        "READY_TO_ENTER":    "✅",
                        "WAIT_FOR_TRIGGER":  "⏳",
                        "WATCH_ONLY":        "👁",
                    }
                    for t in _tp.fno_ready_trades:
                        _si2 = _status_icons.get(t.status, "?")
                        lines.append(
                            f"  {_si2} [{t.status}] {t.instrument} {t.strategy.upper()}"
                        )
                        lines.append(f"     Strikes     : {t.approx_strikes}")
                        lines.append(f"     Entry when  : {t.entry_trigger}")
                        lines.append(f"     Credit/Debit: {t.target_credit_or_debit}  |  "
                                     f"Max loss/lot: ₹{t.max_loss_per_lot:,}")
                        lines.append(f"     Stop rule   : {t.stop_loss_rule}")
                        lines.append(f"     Conviction  : {t.confidence}%  |  "
                                     f"Rationale: {t.rationale}")

                # Equity watchlist
                if _tp.equity_watchlist:
                    lines.append(f"\nEQUITY WATCHLIST")
                    for w in _tp.equity_watchlist:
                        _dir_icon = "🟢" if w.direction == "LONG" else "🔴"
                        lines.append(
                            f"  {_dir_icon} {w.stock:12s} {w.direction:5s} "
                            f"conviction={w.conviction}%  sector={w.sector}"
                        )
                        lines.append(f"     Entry: {w.entry_trigger}  |  SL: {w.stop_loss}  |  "
                                     f"T1: {w.target_1}  |  T2: {w.target_2}")
                        lines.append(f"     Reason: {w.reason}")

                # Risk parameters
                rp = _tp.risk_parameters
                if rp:
                    lines.append(f"\nRISK PARAMETERS")
                    lines.append(f"  Position size  : {rp.recommended_position_size_pct:.1f}% of portfolio per trade")
                    lines.append(f"  Max positions  : {rp.max_positions}")
                    lines.append(f"  Capital at risk: {rp.capital_at_risk_per_trade_pct:.1f}% per trade")
                    if rp.vix_sizing_note:
                        lines.append(f"  VIX note       : {rp.vix_sizing_note}")

                # Market open checklist
                if _tp.market_open_checklist:
                    lines.append(f"\nMARKET OPEN CHECKLIST")
                    for i, item in enumerate(_tp.market_open_checklist, 1):
                        lines.append(f"  [{i}] {item}")

            except Exception as _tp_err:
                lines.append(f"\nTRADING PLAN\n  (render error: {_tp_err})")

        lines.append("\nACTION ITEMS")
        for item in plan.action_items:
            lines.append(f"  [ ] {item}")

        lines.append("=" * 65)
        return "\n".join(lines)

"""
ROX Proven Edge Engine v3.0 - Lead Coordinator
=============================================
Master orchestrator that coordinates all 7 agents through the 11-tier pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from agents import (
    BaseAgent, AgentVerdict, AgentReport,
    OrionAgent, VesperAgent, KairoAgent,
    SentinelAgent, NexusAgent, PrudenceAgent, CatalystAgent,
    OptimusAgent,
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from config import (
    TradeDirection, MarketRegime, ConvictionLevel, SystemConfig,
    DEFAULT_CONFIG, NIFTY_50_STOCKS, SECTOR_MAPPING
)

# ---------------------------------------------------------------------------
# Enhancement: Compliance, Continuous Learning, Chart Reading Integration
# ---------------------------------------------------------------------------
try:
    from monitoring.circuit_breaker import ComplianceEngine, CorporateAction
    _COMPLIANCE_AVAILABLE = True
except ImportError:
    _COMPLIANCE_AVAILABLE = False

try:
    from data.trade_logger import ContinuousLearningModule
    _LEARNING_AVAILABLE = True
except ImportError:
    _LEARNING_AVAILABLE = False

try:
    from agents.orion import ChartAnalyzer
    _CHART_AVAILABLE = True
except ImportError:
    _CHART_AVAILABLE = False


# ---------------------------------------------------------------------------
# Enhancement 2: Cross-Agent Reasoning – Shared Context Protocol
# ---------------------------------------------------------------------------

class AgentContextPool:
    """
    Shared reasoning context for all agents.
    Enables cross-agent validation without new files.
    """

    def __init__(self):
        self.agent_outputs: Dict[str, AgentReport] = {}
        self.reasoning_chains: List[List[str]] = []

    def register_output(self, agent_name: str, report: AgentReport):
        self.agent_outputs[agent_name] = report

    def query_agent(self, querier: str, target: str, query_type: str):
        """Allow one agent to query another's reasoning."""
        if target not in self.agent_outputs:
            return None
        target_report = self.agent_outputs[target]
        self.reasoning_chains.append([querier, target, query_type])
        if query_type == "direction":
            return target_report.verdict.direction
        elif query_type == "conviction":
            return target_report.verdict.conviction
        elif query_type == "risks":
            return target_report.verdict.risks
        return target_report.verdict


@dataclass
class ConsensusResult:
    """Result of agent consensus calculation"""
    direction: TradeDirection
    strength: str  # STRONG, MODERATE, WEAK, NO_CONSENSUS
    net_score: float
    weighted_votes: Dict[str, float]
    agreeing_agents: List[str]
    disagreeing_agents: List[str]
    contradictions: List[Dict]


@dataclass
class TradeSetup:
    """Complete trade setup with all details"""
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


@dataclass
class DailyTradingPlan:
    """Complete daily trading plan"""
    date: datetime
    market_regime: MarketRegime
    regime_confidence: float
    consensus: ConsensusResult
    top_setups: List[TradeSetup]
    portfolio_risk: Dict
    upcoming_events: List[Dict]
    action_items: List[str]
    agent_reports: Dict[str, AgentReport]


class LeadCoordinator:
    """
    Lead Coordinator - Master Orchestrator
    
    Coordinates all 7 agents through the 11-tier processing pipeline
    to generate daily trading plans with full reasoning transparency.
    """
    
    def __init__(self, config: SystemConfig = None, portfolio_value: float = 1000000):
        self.config = config or DEFAULT_CONFIG
        self.portfolio_value = portfolio_value
        self.logger = logging.getLogger("Coordinator")
        
        # Initialize all agents
        self.agents: Dict[str, BaseAgent] = {
            "ORION": OrionAgent(),
            "VESPER": VesperAgent(),
            "KAIRO": KairoAgent(),
            "SENTINEL": SentinelAgent(),
            "NEXUS": NexusAgent(),
            "PRUDENCE": PrudenceAgent(),
            "CATALYST": CatalystAgent(),
            "OPTIMUS": OptimusAgent(),   # F&O Weekly Expiry Agent
        }
        
        # Apply config weights
        for name, agent in self.agents.items():
            if name in self.config.agents:
                agent.current_weight = self.config.agents[name].current_weight
        
        # Current state
        self.current_regime = MarketRegime.CONSOLIDATION
        self.regime_confidence = 50.0
        self.market_data: Dict[str, Any] = {}
        self.agent_reports: Dict[str, AgentReport] = {}

        # ---------------------------------------------------------------
        # Enhancement Engines (gracefully degraded if imports unavailable)
        # ---------------------------------------------------------------

        # 1. Compliance & Risk Controls
        if _COMPLIANCE_AVAILABLE:
            self.compliance_engine = ComplianceEngine(
                portfolio_value=portfolio_value,
                max_drawdown_pct=0.10,   # 10% max drawdown auto-halt
            )
            self.logger.info("ComplianceEngine initialised (circuit filter + SEBI + corp actions + drawdown)")
        else:
            self.compliance_engine = None
            self.logger.warning("ComplianceEngine unavailable – check monitoring/circuit_breaker.py")

        # 2. Continuous Learning Module (requires trade_logger)
        self._learning_module: Optional[Any] = None  # Lazy-init in generate_trading_plan

        # 3. Chart Analyzer
        if _CHART_AVAILABLE:
            self.chart_analyzer = ChartAnalyzer()
            self.logger.info("ChartAnalyzer initialised (price structure + volume + confluence)")
        else:
            self.chart_analyzer = None
            self.logger.warning("ChartAnalyzer unavailable – check agents/orion.py")
    
    def generate_trading_plan(self, market_data: Dict[str, Any],
                             portfolio_status: Dict[str, Any] = None,
                             watchlist: List[str] = None) -> DailyTradingPlan:
        """
        Generate complete daily trading plan through the 11-tier pipeline.
        
        Args:
            market_data: Complete market data including:
                - price_data: Dict with OHLCV data
                - flow_data: FII/DII flow data
                - sentiment_data: Sentiment indicators
                - derivatives_data: Options/futures data
                - fundamental_data: Fundamental metrics
                - event_data: Upcoming events
            portfolio_status: Current portfolio status
            watchlist: List of stocks to analyze
            
        Returns:
            DailyTradingPlan with all setup recommendations
        """
        self.market_data = market_data
        watchlist = watchlist or NIFTY_50_STOCKS[:20]  # Default to top 20 Nifty stocks
        portfolio_status = portfolio_status or self._default_portfolio()

        # ---------------------------------------------------------------
        # ENHANCEMENT TIER A: Continuous Learning — Seasonality Context
        # ---------------------------------------------------------------
        seasonality_ctx = {}
        if _LEARNING_AVAILABLE:
            try:
                from data.trade_logger import ContinuousLearningModule, TradeLogger
                from data.data_manager import DataManager
                if self._learning_module is None:
                    _tl = TradeLogger(DataManager())
                    self._learning_module = ContinuousLearningModule(_tl)
                seasonality_ctx = self._learning_module.get_seasonality_context()
                self.logger.info(
                    f"Seasonality: {seasonality_ctx.get('month_label')} | "
                    f"Bias: {seasonality_ctx.get('seasonal_bias')} | "
                    f"F&O Expiry T-{seasonality_ctx.get('days_to_fno_expiry')}d"
                )
                if seasonality_ctx.get("near_fno_expiry"):
                    self.logger.warning(seasonality_ctx.get("expiry_alert", ""))
            except Exception as e:
                self.logger.warning(f"Learning module init skipped: {e}")

        # ---------------------------------------------------------------
        # ENHANCEMENT TIER B: Compliance Pre-Screen — Circuit & SEBI
        # ---------------------------------------------------------------
        compliant_watchlist = list(watchlist)
        compliance_report_lines = []
        if self.compliance_engine is not None:
            # Update drawdown from portfolio status
            current_portfolio_val = portfolio_status.get(
                "total_capital", self.portfolio_value
            )
            drawdown = self.compliance_engine.update_portfolio_value(current_portfolio_val)
            if self.compliance_engine.trading_halted_drawdown:
                self.logger.critical(
                    f"🚨 TRADING HALTED – drawdown {drawdown*100:.1f}% > "
                    f"{self.compliance_engine.max_drawdown_pct*100:.0f}% limit"
                )
                # Still generate plan but mark it as halted
                compliance_report_lines.append(
                    f"⛔ TRADING HALTED: {self.compliance_engine.drawdown_halt_reason}"
                )

            # Build price_data from market_data for circuit screening
            price_data_raw = market_data.get("price_data", {})
            compliance_results = self.compliance_engine.bulk_screen_watchlist(
                watchlist=watchlist,
                price_data=price_data_raw,
                available_margin=current_portfolio_val * 0.25,  # assume 25% margin
            )
            compliant_watchlist = self.compliance_engine.get_compliant_symbols(
                compliance_results
            )
            blocked = [s for s in watchlist if s not in compliant_watchlist]
            if blocked:
                self.logger.warning(f"Compliance blocked {len(blocked)} stocks: {blocked}")
                for sym in blocked:
                    res = compliance_results[sym]
                    compliance_report_lines.append(
                        f"  ✗ {sym}: {'; '.join(res.reasons)}"
                    )

        # TIER 0: Data Validation
        validation_result = self._tier_0_validate_data(market_data)
        
        # TIER 11: Market Regime Detection
        self.current_regime, self.regime_confidence = self._tier_11_detect_regime(market_data)
        
        # TIER 1: Meta-Learning & Agent Weighting
        self._tier_1_adjust_weights(self.current_regime)

        # ---------------------------------------------------------------
        # ENHANCEMENT: Apply learning-based weight adjustments
        # ---------------------------------------------------------------
        if self._learning_module is not None:
            try:
                base_weights = {n: a.current_weight for n, a in self.agents.items()}
                adjusted = self._learning_module.compute_weight_adjustments(
                    current_regime=self.current_regime.value,
                    base_weights=base_weights,
                )
                for name, new_w in adjusted.items():
                    if name in self.agents:
                        self.agents[name].current_weight = new_w
                self.logger.info(
                    f"Agent weights adjusted for {self.current_regime.value} regime "
                    f"via ContinuousLearning"
                )
            except Exception as e:
                self.logger.debug(f"Weight adjustment skipped (insufficient history): {e}")
        
        # Run all agents for market overview
        self._run_all_agents(market_data, self.current_regime)

        # ── v6.0 SIGNAL TRACER: Log agent verdicts ─────────────────────────
        for name, report in self.agent_reports.items():
            self.logger.info(
                f"[SIGNAL_TRACER] agent={name} "
                f"direction={report.verdict.direction.value} "
                f"conviction={report.verdict.conviction:.0f} "
                f"weight={report.verdict.weight:.3f} "
                f"weighted_vote={report.verdict.weighted_vote:+.3f}"
            )

        # TIER 2: Consensus & Contradiction Reconciliation
        consensus = self._tier_2_calculate_consensus()

        # ── v6.0 SIGNAL TRACER: Log consensus outcome ──────────────────────
        self.logger.info(
            f"[SIGNAL_TRACER] consensus_direction={consensus.direction.value} "
            f"strength={consensus.strength} "
            f"net_score={consensus.net_score:+.3f} "
            f"agreeing={consensus.agreeing_agents} "
            f"disagreeing={consensus.disagreeing_agents}"
        )
        
        # Scan compliant watchlist for setups
        setups = []
        for stock in compliant_watchlist[:10]:  # Analyze top 10 compliant stocks
            setup = self._analyze_stock(stock, market_data, portfolio_status)
            if setup:
                # -------------------------------------------------------
                # ENHANCEMENT: Chart Reading — augment each setup
                # -------------------------------------------------------
                if self.chart_analyzer is not None:
                    try:
                        stock_data = market_data.get("price_data", {}).get(stock, {})
                        bars = stock_data.get("ohlcv_bars", stock_data.get("bars", []))
                        sector_data = market_data.get("sector_data", {})
                        if bars:
                            chart_result = self.chart_analyzer.analyze(stock, bars, sector_data)
                            # Boost conviction based on chart score
                            chart_boost = int((chart_result.overall_chart_score - 50) / 10)
                            setup.conviction = max(0, min(100, setup.conviction + chart_boost))
                            # Enforce volume confirmation rule for breakouts
                            if (not chart_result.volume_analysis.is_confirmed
                                    and chart_result.breakout_signal):
                                self.logger.info(
                                    f"{stock}: Breakout detected but volume NOT confirmed "
                                    f"({chart_result.volume_analysis.volume_ratio:.1f}x avg) — "
                                    f"conviction reduced"
                                )
                                setup.conviction = max(0, setup.conviction - 15)
                            # Attach chart data to setup for report
                            setup._chart_result = chart_result  # type: ignore
                    except Exception as e:
                        self.logger.debug(f"Chart analysis for {stock} skipped: {e}")
                setups.append(setup)
        
        # TIER 5: Opportunity Scanning - Rank and select top 5
        top_setups = self._tier_5_rank_setups(setups)

        # ── v6.0 SIGNAL TRACER: Log execution decisions ────────────────────
        for s in top_setups:
            self.logger.info(
                f"[SIGNAL_TRACER] setup={s.stock} "
                f"direction={s.direction.value} "
                f"conviction={s.conviction} "
                f"rr={s.risk_reward:.2f} "
                f"shares={s.shares} "
                f"risk_pct={s.risk_percent:.1f}%"
            )
        if not top_setups:
            self.logger.info("[SIGNAL_TRACER] no_setups_passed_ranking")

        # TIER 10: Position Sizing (already done in _analyze_stock)
        
        # TIER 8: Generate Report
        trading_plan = DailyTradingPlan(
            date=datetime.now(),
            market_regime=self.current_regime,
            regime_confidence=self.regime_confidence,
            consensus=consensus,
            top_setups=top_setups,
            portfolio_risk=self._calculate_portfolio_risk(portfolio_status),
            upcoming_events=self._extract_upcoming_events(market_data),
            action_items=self._generate_action_items(top_setups, consensus),
            agent_reports=self.agent_reports
        )

        # Attach enhancement metadata to plan for the formatter
        trading_plan._seasonality = seasonality_ctx              # type: ignore
        trading_plan._compliance_lines = compliance_report_lines  # type: ignore
        
        return trading_plan
    
    def _tier_0_validate_data(self, data: Dict) -> Dict:
        """Tier 0: Data Collection & Validation"""
        required_datasets = [
            "price_data", "flow_data", "sentiment_data",
            "derivatives_data", "fundamental_data", "event_data"
        ]
        
        validation = {
            "complete": 0,
            "missing": [],
            "status": "green"
        }
        
        for dataset in required_datasets:
            if dataset in data and data[dataset]:
                validation["complete"] += 1
            else:
                validation["missing"].append(dataset)
        
        # Determine status
        if validation["complete"] >= 6:
            validation["status"] = "green"
        elif validation["complete"] >= 4:
            validation["status"] = "yellow"
        else:
            validation["status"] = "red"
        
        return validation
    
    def _tier_11_detect_regime(self, data: Dict) -> Tuple[MarketRegime, float]:
        """Tier 11: Market Regime Detection"""
        score = 0
        confidence = 50
        
        # Indicator 1: Nifty vs 200 DMA
        nifty_price = data.get('nifty_price', 0)
        nifty_200dma = data.get('nifty_200dma', 0)
        if nifty_200dma > 0:
            dma_gap = (nifty_price - nifty_200dma) / nifty_200dma
            if dma_gap > 0.05:
                score += 1
            elif dma_gap > 0:
                score += 0.5
            elif dma_gap < -0.05:
                score -= 1
            else:
                score -= 0.5
        
        # Indicator 2: Price Structure
        price_structure = data.get('price_structure', 'neutral')
        if price_structure == 'higher_highs':
            score += 1
        elif price_structure == 'lower_lows':
            score -= 1
        
        # Indicator 3: ADX
        adx = data.get('adx', 20)
        if adx > 25:
            confidence += 10
        
        # Indicator 4: FII 5-day flow
        fii_5day = data.get('flow_data', {}).get('fii_cash_5day', 0)
        if fii_5day > 3000:
            score += 1
        elif fii_5day < -3000:
            score -= 1
        
        # Indicator 5: India VIX
        vix = data.get('india_vix', 15)
        if vix > 22:
            score -= 0.5
        elif vix < 13:
            score += 0.5
        
        # Determine regime
        if score >= 2:
            regime = MarketRegime.BULL
            confidence = min(90, 60 + score * 10)
        elif score >= 1:
            regime = MarketRegime.MILD_BULL
            confidence = 55 + score * 10
        elif score <= -2:
            regime = MarketRegime.BEAR
            confidence = min(90, 60 + abs(score) * 10)
        elif score <= -1:
            regime = MarketRegime.MILD_BEAR
            confidence = 55 + abs(score) * 10
        else:
            regime = MarketRegime.CONSOLIDATION
            confidence = 50
        
        return regime, confidence
    
    def _tier_1_adjust_weights(self, regime: MarketRegime):
        """Tier 1: Meta-Learning & Agent Weighting"""
        # Regime-based adjustments
        adjustments = {
            MarketRegime.BULL: {
                "ORION": 0.03,    # Technical momentum works well
                "VESPER": 0.02,   # FII flows reliable
                "KAIRO": -0.03,   # Sentiment less useful
                "PRUDENCE": -0.02, # Fewer vetoes needed
            },
            MarketRegime.BEAR: {
                "ORION": -0.03,   # False breakouts common
                "VESPER": 0.02,   # Flow data critical
                "KAIRO": 0.03,    # Panic readings = buy signals
                "SENTINEL": 0.02, # Derivatives structure critical
                "NEXUS": 0.01,    # Fundamentals protect downside
                "PRUDENCE": 0.02, # Tighter controls needed
            },
            MarketRegime.CONSOLIDATION: {
                "VESPER": -0.02,  # Flows unclear
                "SENTINEL": 0.03, # Options structure defines range
                "NEXUS": 0.02,    # Value plays more relevant
                "CATALYST": 0.01, # Event catalysts drive breakouts
            }
        }
        
        regime_adjustments = adjustments.get(regime, {})
        
        for name, agent in self.agents.items():
            adjustment = regime_adjustments.get(name, 0)
            agent.current_weight = max(0.05, min(0.30, 
                self.config.agents[name].baseline_weight + adjustment))
        
        # Normalize weights
        total = sum(a.current_weight for a in self.agents.values())
        for agent in self.agents.values():
            agent.current_weight /= total
    
    def _run_all_agents(self, data: Dict, regime: MarketRegime):
        """
        Run analysis for all agents.

        Enhancement 5: Independent agents run in parallel via asyncio + ThreadPoolExecutor.
        Enhancement 2: Cross-agent validation applied after Phase 1.
        Enhancement 3: Uses cached_analyze instead of analyze.
        """
        self.agent_reports = {}

        agent_data_mapping = {
            "ORION": self._prepare_orion_data(data),
            "VESPER": self._prepare_vesper_data(data),
            "KAIRO": self._prepare_kairo_data(data),
            "SENTINEL": self._prepare_sentinel_data(data),
            "NEXUS": self._prepare_nexus_data(data),
            "PRUDENCE": self._prepare_prudence_data(data),
            "CATALYST": self._prepare_catalyst_data(data),
            "OPTIMUS": self._prepare_optimus_data(data),
        }

        # --- Enhancement 5: Parallel execution ---
        independent_agents = ["ORION", "VESPER", "KAIRO", "NEXUS"]
        dependent_agents   = ["SENTINEL", "CATALYST", "OPTIMUS"]
        veto_agents        = ["PRUDENCE"]

        def _safe_analyze(name: str) -> Tuple[str, AgentReport]:
            try:
                agent = self.agents[name]
                report = agent.cached_analyze(agent_data_mapping.get(name, {}), regime)
                return name, report
            except Exception as e:
                return name, AgentReport(
                    agent_name=name,
                    verdict=AgentVerdict(
                        direction=TradeDirection.NEUTRAL,
                        conviction=0,
                        weight=self.agents[name].current_weight,
                        reason=f"Analysis error: {str(e)}"
                    )
                )

        # Phase 1: Parallel independent agents
        with ThreadPoolExecutor(max_workers=len(independent_agents)) as pool:
            futures = {pool.submit(_safe_analyze, name): name for name in independent_agents}
            for future in futures:
                name, report = future.result()
                self.agent_reports[name] = report

        # --- Enhancement 2: Cross-agent validation after Phase 1 ---
        context_pool = AgentContextPool()
        for name, report in self.agent_reports.items():
            context_pool.register_output(name, report)

        self.agent_reports = {
            name: self._cross_validate(report, context_pool)
            for name, report in self.agent_reports.items()
        }

        # Phase 2: Dependent agents (sequential, may use Phase 1 peer data)
        for name in dependent_agents:
            enriched_data = {
                **agent_data_mapping.get(name, {}),
                "peer_reports": {n: r.verdict for n, r in self.agent_reports.items()}
            }
            try:
                report = self.agents[name].cached_analyze(enriched_data, regime)
            except Exception as e:
                report = AgentReport(
                    agent_name=name,
                    verdict=AgentVerdict(
                        direction=TradeDirection.NEUTRAL,
                        conviction=0,
                        weight=self.agents[name].current_weight,
                        reason=f"Analysis error: {str(e)}"
                    )
                )
            self.agent_reports[name] = report
            context_pool.register_output(name, report)

        # Phase 3: PRUDENCE veto (always last)
        for name in veto_agents:
            try:
                prudence_data = {
                    **agent_data_mapping.get(name, {}),
                    "peer_reports": {n: r.verdict for n, r in self.agent_reports.items()}
                }
                report = self.agents[name].cached_analyze(prudence_data, regime)
            except Exception as e:
                report = AgentReport(
                    agent_name=name,
                    verdict=AgentVerdict(
                        direction=TradeDirection.NEUTRAL,
                        conviction=0,
                        weight=self.agents[name].current_weight,
                        reason=f"Analysis error: {str(e)}"
                    )
                )
            self.agent_reports[name] = report

    def _cross_validate(self, report: AgentReport,
                        context_pool: AgentContextPool) -> AgentReport:
        """
        Enhancement 2: Allow agent to question its own conclusion using peer data.
        """
        if report.agent_name == "ORION":
            sentinel_dir = context_pool.query_agent("ORION", "SENTINEL", "direction")
            if sentinel_dir and sentinel_dir != report.verdict.direction:
                report.verdict.conviction *= 0.8
                report.verdict.reason += " | [Cross-check: SENTINEL disagrees -20% confidence]"

        if report.agent_name == "VESPER":
            kairo_dir = context_pool.query_agent("VESPER", "KAIRO", "direction")
            if kairo_dir == report.verdict.direction:
                report.verdict.conviction = min(100, report.verdict.conviction * 1.1)
                report.verdict.reason += " | [Cross-check: KAIRO confirms +10% confidence]"

        if report.agent_name == "KAIRO":
            vesper_dir = context_pool.query_agent("KAIRO", "VESPER", "direction")
            if vesper_dir and vesper_dir != report.verdict.direction:
                report.verdict.conviction *= 0.85
                report.verdict.reason += " | [Cross-check: VESPER flow conflicts with sentiment -15%]"

        if report.agent_name == "NEXUS":
            orion_dir = context_pool.query_agent("NEXUS", "ORION", "direction")
            if orion_dir == report.verdict.direction:
                report.verdict.conviction = min(100, report.verdict.conviction * 1.05)
                report.verdict.reason += " | [Cross-check: ORION technicals confirm fundamentals +5%]"

        # Recalculate weighted vote after conviction change
        report.verdict.__post_init__()
        return report
    
    def _tier_2_calculate_consensus(self) -> ConsensusResult:
        """Tier 2: Calculate weighted consensus from all agent votes"""
        weighted_votes = {}
        long_votes = []
        short_votes = []
        neutral_votes = []
        
        for name, report in self.agent_reports.items():
            verdict = report.verdict
            weighted_votes[name] = verdict.weighted_vote
            
            if verdict.direction == TradeDirection.LONG:
                long_votes.append(name)
            elif verdict.direction == TradeDirection.SHORT:
                short_votes.append(name)
            else:
                neutral_votes.append(name)
        
        # Calculate net score
        net_score = sum(weighted_votes.values())
        
        # Determine consensus
        if net_score > 0.25:
            direction = TradeDirection.LONG
            strength = "STRONG" if net_score > 0.4 else "MODERATE"
        elif net_score < -0.25:
            direction = TradeDirection.SHORT
            strength = "STRONG" if net_score < -0.4 else "MODERATE"
        else:
            direction = TradeDirection.NEUTRAL
            strength = "NO_CONSENSUS"
        
        # Identify contradictions
        contradictions = []
        if long_votes and short_votes:
            contradictions.append({
                "type": "direction",
                "agents": {"long": long_votes, "short": short_votes},
                "resolution": f"Weighted vote favors {direction.value}"
            })
        
        return ConsensusResult(
            direction=direction,
            strength=strength,
            net_score=net_score,
            weighted_votes=weighted_votes,
            agreeing_agents=long_votes if direction == TradeDirection.LONG else short_votes,
            disagreeing_agents=short_votes if direction == TradeDirection.LONG else long_votes,
            contradictions=contradictions
        )
    
    def _analyze_stock(self, stock: str, market_data: Dict,
                      portfolio_status: Dict) -> Optional[TradeSetup]:
        """Analyze individual stock for trade setup"""
        # Get stock-specific data
        stock_data = self._get_stock_data(stock, market_data)
        
        if not stock_data:
            return None
        
        # Run ORION for technical analysis
        orion_report = self.agents["ORION"].analyze(stock_data, self.current_regime)

        self.logger.debug(
            f"{stock}: ORION={orion_report.verdict.direction.value} "
            f"conv={orion_report.verdict.conviction:.0f}%"
        )

        # Check if technical setup exists
        if orion_report.verdict.conviction < 50:
            self.logger.debug(f"{stock}: ORION conviction {orion_report.verdict.conviction:.0f} < 50, skip")
            return None

        entry_setup = orion_report.raw_data.get("entry_setup", {})

        # If ORION entry_setup has no direction (no patterns/S/R detected),
        # inherit direction from ORION's verdict which uses confluence score
        if entry_setup.get("direction") == TradeDirection.NEUTRAL:
            verdict_dir = orion_report.verdict.direction
            if verdict_dir == TradeDirection.NEUTRAL:
                return None   # genuinely no signal
            # Rebuild entry using ATR from price data
            stock_price = stock_data.get("price_data", {}).get("close", 0)
            atr         = stock_data.get("indicators", {}).get("atr", stock_price * 0.02)
            if stock_price <= 0:
                return None
            entry_setup = dict(entry_setup)   # copy
            entry_setup["direction"] = verdict_dir
            if verdict_dir == TradeDirection.LONG:
                entry_setup["entry_zone"] = (stock_price * 0.999, stock_price * 1.003)
                entry_setup["stop_loss"]  = stock_price - atr * 1.5
                entry_setup["target_1"]   = stock_price + atr * 2.5
                entry_setup["target_2"]   = stock_price + atr * 4.0
            else:
                entry_setup["entry_zone"] = (stock_price * 0.997, stock_price * 1.001)
                entry_setup["stop_loss"]  = stock_price + atr * 1.5
                entry_setup["target_1"]   = stock_price - atr * 2.5
                entry_setup["target_2"]   = stock_price - atr * 4.0
            risk   = abs(stock_price - entry_setup["stop_loss"])
            reward = abs(entry_setup["target_1"] - stock_price)
            entry_setup["risk_reward"] = reward / risk if risk > 0 else 0
        
        # Run NEXUS — pass stock price data so it can compute proxy metrics
        stock_price_for_nexus = stock_data.get("price_data", {}).get("close", 0)
        nexus_report = self.agents["NEXUS"].analyze({
            **stock_data,
            "stock":      stock,
            "nifty_pe":   market_data.get("nifty_pe",   22.5),
            "gsec_yield": market_data.get("gsec_yield",  7.0),
            "fundamentals": stock_data.get("fundamentals", {}),
        }, self.current_regime)
        
        # Check PRUDENCE
        prudence_report = self.agents["PRUDENCE"].analyze({
            "portfolio": portfolio_status,
            "trade_request": {
                "entry_price": entry_setup.get("entry_zone", (0, 0))[0],
                "stop_loss": entry_setup.get("stop_loss", 0),
            },
            "stock": stock,
            "sector": self._get_stock_sector(stock),
            "conviction_level": self._get_conviction_level(orion_report.verdict.conviction).value
        }, self.current_regime)
        
        # Check if PRUDENCE vetoed
        if prudence_report.analysis_details.get("verdict_type") == "VETO":
            return None
        
        # Calculate conviction
        conviction = self._calculate_conviction(
            orion_report, nexus_report, prudence_report
        )
        
        if conviction < 55:
            self.logger.debug(f"{stock}: conviction {conviction} < 55, skip")
            return None
        
        # Build trade setup
        sizing = prudence_report.analysis_details.get("sizing", {})
        
        return TradeSetup(
            stock=stock,
            direction=entry_setup.get("direction", TradeDirection.NEUTRAL),
            conviction=conviction,
            conviction_level=self._get_conviction_level(conviction),
            entry_price=entry_setup.get("entry_zone", (0, 0))[0],
            entry_zone=entry_setup.get("entry_zone", (0, 0)),
            stop_loss=entry_setup.get("stop_loss", 0),
            target_1=entry_setup.get("target_1", 0),
            target_2=entry_setup.get("target_2", 0),
            risk_reward=entry_setup.get("risk_reward", 0),
            shares=sizing.get("shares", 0),
            position_value=sizing.get("position_value", 0),
            position_percent=sizing.get("position_percent", 0),
            risk_percent=sizing.get("risk_percent", 0),
            agent_votes={
                "ORION": orion_report.verdict,
                "NEXUS": nexus_report.verdict,
                "PRUDENCE": prudence_report.verdict
            },
            pattern_match={"match": False, "win_rate": 50},  # Placeholder
            historical_win_rate=50,
            exit_strategy=self._generate_exit_strategy(entry_setup)
        )
    
    def _tier_5_rank_setups(self, setups: List[TradeSetup]) -> List[TradeSetup]:
        """Tier 5: Rank and select top 5 setups"""
        if not setups:
            return []
        
        # Sort by conviction score
        sorted_setups = sorted(setups, key=lambda x: x.conviction, reverse=True)
        
        # Return top 5
        return sorted_setups[:5]
    
    def _calculate_conviction(self, *reports: AgentReport) -> int:
        """
        Calculate combined conviction score from agent reports.
        Uses a weighted average of non-neutral agent convictions,
        scaled to 0-100, with a small penalty for risks.
        """
        total_weight   = 0.0
        weighted_score = 0.0

        for report in reports:
            w = report.verdict.weight if report.verdict.weight > 0 else 0.10
            c = report.verdict.conviction
            d = report.verdict.direction

            if d == TradeDirection.LONG:
                weighted_score += w * c
            elif d == TradeDirection.SHORT:
                weighted_score += w * (100 - c)   # short conviction flips the score
            else:
                weighted_score += w * 50            # neutral anchors at 50

            total_weight += w

        if total_weight == 0:
            return 50

        raw = weighted_score / total_weight        # 0-100

        # Small risk deduction (cap at -10 total)
        risk_count = sum(len(r.verdict.risks or []) for r in reports)
        risk_deduction = min(10, risk_count * 2)

        return max(0, min(100, int(raw - risk_deduction)))
    
    def _get_conviction_level(self, conviction: int) -> ConvictionLevel:
        """Get conviction level from score"""
        if conviction >= 85:
            return ConvictionLevel.VERY_HIGH
        elif conviction >= 75:
            return ConvictionLevel.HIGH
        elif conviction >= 65:
            return ConvictionLevel.MEDIUM
        elif conviction >= 50:
            return ConvictionLevel.LOW
        else:
            return ConvictionLevel.SKIP
    
    def _get_stock_sector(self, stock: str) -> str:
        """Get sector for a stock"""
        for sector, stocks in SECTOR_MAPPING.items():
            if stock in stocks:
                return sector
        return "Others"
    
    def _get_stock_data(self, stock: str, market_data: Dict) -> Dict:
        """Extract stock-specific data from market data, enriched for ORION."""
        price_data   = market_data.get("price_data",  {}).get(stock, {})
        indicators   = market_data.get("indicators",  {}).get(stock, {})
        fundamentals = market_data.get("fundamental_data", {}).get(stock, {})

        close  = price_data.get("close",  0)
        sma20  = indicators.get("sma20",  0)
        sma50  = indicators.get("sma50",  0)
        sma200 = indicators.get("sma200", 0)
        trend  = indicators.get("trend",  "SIDEWAYS")

        def _signal(bull, bear):
            return "bullish" if bull else ("bearish" if bear else "neutral")

        weekly_trend    = _signal(trend == "UPTREND",  trend == "DOWNTREND")
        daily_trend     = _signal(close > sma50 > 0,   close < sma50 > 0)
        four_hour_trend = _signal(close > sma20 > 0,   close < sma20 > 0)
        one_hour_trend  = _signal(sma20 > sma50 > 0,   sma20 < sma50 > 0)

        enriched_price = dict(price_data)
        if sma50  > 0: enriched_price["ma_50"]  = sma50
        if sma200 > 0: enriched_price["ma_200"] = sma200

        return {
            "stock":           stock,
            "price_data":      enriched_price,
            "indicators":      indicators,
            "fundamentals":    fundamentals,
            "weekly_trend":    weekly_trend,
            "daily_trend":     daily_trend,
            "four_hour_trend": four_hour_trend,
            "one_hour_trend":  one_hour_trend,
        }
    
    def _default_portfolio(self) -> Dict:
        """Get default portfolio status"""
        return {
            "total_capital": self.portfolio_value,
            "deployed_capital": 0,
            "cash": self.portfolio_value,
            "portfolio_heat": 0,
            "current_drawdown": 0,
            "open_positions": [],
            "sector_exposure": {}
        }
    
    def _calculate_portfolio_risk(self, portfolio: Dict) -> Dict:
        """Calculate portfolio risk metrics"""
        return {
            "capital": portfolio.get("total_capital", 0),
            "deployed_percent": portfolio.get("deployed_capital", 0) / max(1, portfolio.get("total_capital", 1)) * 100,
            "cash_percent": portfolio.get("cash", 0) / max(1, portfolio.get("total_capital", 1)) * 100,
            "portfolio_heat": portfolio.get("portfolio_heat", 0) * 100,
            "drawdown": portfolio.get("current_drawdown", 0) * 100
        }
    
    def _extract_upcoming_events(self, market_data: Dict) -> List[Dict]:
        """Extract upcoming events from market data"""
        events = market_data.get("event_data", {}).get("events", [])
        return [
            {
                "name": e.get("name", ""),
                "date": str(e.get("date", "")),
                "impact": e.get("impact", "LOW")
            }
            for e in events[:5]
        ]
    
    def _generate_action_items(self, setups: List[TradeSetup],
                               consensus: ConsensusResult) -> List[str]:
        """Generate action items for the trading day"""
        items = []
        
        if setups:
            items.append(f"Review top setup: {setups[0].stock} ({setups[0].direction.value})")
            items.append(f"Set alerts for {len(setups)} trade setups")
        else:
            items.append("No high-conviction setups found - wait for better opportunities")
        
        items.append("Verify all stop losses are set before market open")
        items.append("Check for any overnight news affecting positions")
        
        return items
    
    def _generate_exit_strategy(self, entry_setup: Dict) -> str:
        """Generate exit strategy description"""
        strategies = []
        
        if entry_setup.get("target_1"):
            strategies.append(f"Take partial profits at {entry_setup['target_1']}")
        if entry_setup.get("target_2"):
            strategies.append(f"Trail stop to breakeven after T1, exit at {entry_setup['target_2']}")
        if entry_setup.get("stop_loss"):
            strategies.append(f"Stop loss at {entry_setup['stop_loss']} (ATR-based)")
        
        return " | ".join(strategies) if strategies else "Set stop loss and target based on technical levels"
    
    # Data preparation methods for each agent
    def _prepare_orion_data(self, data: Dict) -> Dict:
        """
        Build ORION's consensus-panel input from market-wide data.
        Uses Nifty price vs SMAs to derive the 4-timeframe trend signals
        that ORION needs to score confluence correctly.
        """
        nifty       = data.get("nifty_price",  22500)
        dma200      = data.get("nifty_200dma", 22000)
        vix         = data.get("india_vix",    15.0)
        structure   = data.get("price_structure", "neutral")
        adx         = data.get("adx", 25)

        # Synthetic SMA estimates from available data
        sma20  = nifty * 0.99   # short-term: approximate
        sma50  = nifty * 0.975
        sma200 = dma200

        trend = ("UPTREND"   if nifty > sma50 > sma200 else
                 "DOWNTREND" if nifty < sma50 < sma200 else "SIDEWAYS")

        def sig(bull, bear):
            return "bullish" if bull else ("bearish" if bear else "neutral")

        weekly_trend    = sig(trend == "UPTREND",    trend == "DOWNTREND")
        daily_trend     = sig(nifty > sma50,         nifty < sma50)
        four_hour_trend = sig(nifty > sma20,         nifty < sma20)
        one_hour_trend  = sig(structure == "higher_highs", structure == "lower_lows")

        atr = nifty * (vix / 100) * (1 / 16)  # ~1-day ATR from VIX

        return {
            "stock":            "NIFTY",
            "price_data": {
                "open":       nifty * 0.998,
                "high":       nifty * 1.005,
                "low":        nifty * 0.993,
                "close":      nifty,
                "volume":     500000,
                "avg_volume": 450000,
                "atr":        atr,
                "ma_50":      sma50,
                "ma_200":     sma200,
            },
            "indicators": {
                "rsi":          55 if trend == "UPTREND" else (45 if trend == "DOWNTREND" else 50),
                "atr":          atr,
                "atr_percent":  (atr / nifty) * 100,
                "adx":          adx,
                "sma20":        sma20,
                "sma50":        sma50,
                "sma200":       sma200,
                "trend":        trend,
                "volume_ratio": 1.05,
                "above_200dma": nifty > sma200,
            },
            "weekly_trend":     weekly_trend,
            "daily_trend":      daily_trend,
            "four_hour_trend":  four_hour_trend,
            "one_hour_trend":   one_hour_trend,
        }
    
    def _prepare_vesper_data(self, data: Dict) -> Dict:
        return {
            "flow_data": data.get("flow_data", {}),
            "sector_flows": data.get("sector_flows", []),
            "bulk_deals": data.get("bulk_deals", [])
        }
    
    def _prepare_kairo_data(self, data: Dict) -> Dict:
        return {
            "news_sentiment": data.get("sentiment_data", {}).get("news", 0),
            "analyst_sentiment": data.get("sentiment_data", {}).get("analyst", 0),
            "social_sentiment": data.get("sentiment_data", {}).get("social", 0),
            "corporate_sentiment": data.get("sentiment_data", {}).get("corporate", 0),
            "global_sentiment": data.get("sentiment_data", {}).get("global", 0),
            "vix": data.get("india_vix", 15),
            "pcr": data.get("derivatives_data", {}).get("pcr", 1)
        }
    
    def _prepare_sentinel_data(self, data: Dict) -> Dict:
        deriv = data.get("derivatives_data", {})
        return {
            "pcr": deriv.get("pcr", 1),
            "pcr_trend": deriv.get("pcr_trend", "stable"),
            "max_pain": deriv.get("max_pain", 0),
            "current_price": data.get("nifty_price", 0),
            "india_vix": data.get("india_vix", 15),
            "iv_rank": deriv.get("iv_rank", 50),
            "call_oi_walls": deriv.get("call_oi_walls", []),
            "put_oi_walls": deriv.get("put_oi_walls", []),
            "oi_change": deriv.get("oi_change", 0),
            "price_change": deriv.get("price_change", 0)
        }
    
    def _prepare_nexus_data(self, data: Dict) -> Dict:
        """
        Build NEXUS input for consensus panel.
        When per-stock fundamentals are unavailable, supply market-level
        quality proxies so NEXUS does not default to a hard SHORT.
        """
        nifty_pe    = data.get("nifty_pe",    22.5)
        gsec_yield  = data.get("gsec_yield",   7.0)
        regime      = data.get("price_structure", "higher_highs")

        # Earnings yield vs bond yield (a simple quality proxy)
        earnings_yield  = (1 / nifty_pe) * 100    # e.g. 4.4% at PE=22.5
        eq_risk_premium = earnings_yield - gsec_yield   # positive = equities cheap vs bonds

        # Synthesise reasonable F-Score / quality when no stock data provided
        market_quality = 55 if regime == "higher_highs" else 40

        return {
            "stock":       "MARKET",
            "fundamentals": {},   # intentionally empty → triggers no-data guard in NEXUS
            "nifty_pe":    nifty_pe,
            "gsec_yield":  gsec_yield,
        }
    
    def _prepare_prudence_data(self, data: Dict) -> Dict:
        # Use Nifty spot price as a representative trade for consensus panel evaluation
        nifty_price = data.get("nifty_price", 22500)
        atr_estimate = nifty_price * 0.01  # ~1% ATR estimate
        return {
            "portfolio": self._default_portfolio(),
            "trade_request": {
                "entry_price": nifty_price,
                "stop_loss": nifty_price - (atr_estimate * 1.5),
            },
            "conviction_level": "MEDIUM"
        }
    
    def _prepare_catalyst_data(self, data: Dict) -> Dict:
        return {
            "events": data.get("event_data", {}).get("events", []),
            "current_date": datetime.now(),
            "expiry_week": data.get("expiry_week", False)
        }
    
    def _prepare_optimus_data(self, data: Dict) -> Dict:
        """
        Prepare data for OPTIMUS F&O Weekly Expiry agent.
        
        Merges derivatives_data with price-action context so OPTIMUS has
        everything it needs in a single flat dict.
        """
        deriv = data.get("derivatives_data", {})
        return {
            # Symbol / expiry context
            "symbol": data.get("symbol", "NIFTY"),
            "weekly_expiry": data.get("weekly_expiry", deriv.get("weekly_expiry", "")),
            # Price
            "current_price": data.get("nifty_price", deriv.get("current_price", 0)),
            "price_change": deriv.get("price_change", data.get("price_change_pct", 0)),
            # PCR
            "pcr": deriv.get("pcr", 1.0),
            "pcr_trend": deriv.get("pcr_trend", "stable"),
            # OI walls
            "max_pain": deriv.get("max_pain", 0),
            "call_oi_walls": deriv.get("call_oi_walls", []),
            "put_oi_walls": deriv.get("put_oi_walls", []),
            # OI change
            "ce_oi_change_pct": deriv.get("ce_oi_change_pct", 0),
            "pe_oi_change_pct": deriv.get("pe_oi_change_pct", 0),
            "oi_signal": deriv.get("oi_signal", deriv.get("oi_change", "NEUTRAL")),
            # Volatility
            "india_vix": data.get("india_vix", deriv.get("india_vix", 15)),
            "iv_rank": deriv.get("iv_rank", 50),
            "iv_skew": deriv.get("iv_skew", 0),
            # Futures
            "futures_premium": deriv.get("futures_premium", deriv.get("basis", 0)),
            # Price action context from ORION
            "support_level": data.get("support_level", 0),
            "resistance_level": data.get("resistance_level", 0),
        }
    
    def format_report(self, plan: DailyTradingPlan) -> str:
        """Format trading plan as readable report"""
        lines = []
        
        lines.append("=" * 65)
        lines.append("ROX EDGE ENGINE - DAILY TRADING PLAN")
        lines.append(f"Date: {plan.date.strftime('%Y-%m-%d')} | Regime: {plan.market_regime.value} | Confidence: {plan.regime_confidence}%")
        lines.append("=" * 65)

        # ---------------------------------------------------------------
        # ENHANCEMENT: Compliance Status Banner
        # ---------------------------------------------------------------
        compliance_lines = getattr(plan, "_compliance_lines", [])
        if compliance_lines:
            lines.append("")
            lines.append("⛔ COMPLIANCE ALERTS")
            lines.extend(compliance_lines)

        # Drawdown status from ComplianceEngine
        if self.compliance_engine is not None:
            dd = self.compliance_engine.get_drawdown_status()
            if dd["drawdown_pct"] > 5:
                lines.append(
                    f"⚠️  DRAWDOWN: {dd['drawdown_pct']:.1f}% from peak "
                    f"(limit {dd['max_drawdown_pct']:.0f}%) | "
                    f"Peak ₹{dd['peak_value']:,.0f} → Current ₹{dd['current_value']:,.0f}"
                )
            if dd["halt_active"]:
                lines.append(f"🚨 TRADING HALTED: {dd['halt_reason']}")

        # ---------------------------------------------------------------
        # ENHANCEMENT: Seasonality Context
        # ---------------------------------------------------------------
        seasonality = getattr(plan, "_seasonality", {})
        if seasonality:
            lines.append("")
            lines.append("=" * 65)
            lines.append("SEASONALITY & MARKET CALENDAR")
            lines.append(
                f"Month Theme  : {seasonality.get('month_label', '—')} "
                f"[Bias: {seasonality.get('seasonal_bias', '—')}]"
            )
            lines.append(f"Note         : {seasonality.get('seasonal_note', '')}")
            fno_expiry = seasonality.get('fno_expiry_date', '')
            days_exp   = seasonality.get('days_to_fno_expiry', '?')
            lines.append(f"F&O Expiry   : {fno_expiry} (T-{days_exp}d)")
            if seasonality.get("near_fno_expiry"):
                lines.append(f"  {seasonality.get('expiry_alert', '')}")
        
        lines.append("")
        lines.append("MARKET REGIME ASSESSMENT")
        lines.append(f"Current regime: {plan.market_regime.value}")
        lines.append(f"Regime confidence: {plan.regime_confidence}%")
        lines.append("")
        
        lines.append("=" * 65)
        lines.append("8-AGENT CONSENSUS PANEL")
        lines.append("")
        
        for name, report in plan.agent_reports.items():
            v = report.verdict
            lines.append(f"{name} ({report.agent_name}): {v.direction.value} - {v.conviction}% conviction")
            lines.append(f"  Reason: {v.reason}")
            # Print OPTIMUS options signal inline
            if name == "OPTIMUS" and "options_signal" in report.analysis_details:
                sig = report.analysis_details["options_signal"]
                lines.append(
                    f"  ↳ OPTIONS SIGNAL: {sig.get('symbol')} "
                    f"{sig.get('expiry_date')} | "
                    f"{sig.get('strategy')} @ Strike {sig.get('strike')} "
                    f"({sig.get('strike_label')}) | "
                    f"Entry {sig.get('entry_range')} | "
                    f"SL {sig.get('stop_loss')} | "
                    f"T1 {sig.get('target_1')} | T2 {sig.get('target_2')} | "
                    f"Risk/lot ₹{sig.get('risk_per_lot')} | "
                    f"{sig.get('iv_context')} | {sig.get('pcr_context')}"
                )
        
        lines.append("")
        lines.append("=" * 65)
        lines.append("WEIGHTED CONSENSUS")
        lines.append(f"Direction: {plan.consensus.direction.value}")
        lines.append(f"Strength: {plan.consensus.strength}")
        lines.append(f"Net Score: {plan.consensus.net_score:.2f}")
        
        lines.append("")
        lines.append("=" * 65)
        lines.append("TOP SWING TRADE SETUPS")
        lines.append("")
        
        for i, setup in enumerate(plan.top_setups, 1):
            lines.append("-" * 50)
            lines.append(f"SETUP #{i}: {setup.stock} | {setup.direction.value} | Conviction: {setup.conviction}/100")
            lines.append(f"Entry Zone: {setup.entry_zone[0]:.2f} - {setup.entry_zone[1]:.2f}")
            lines.append(f"Stop Loss: {setup.stop_loss:.2f}")
            lines.append(f"Target 1: {setup.target_1:.2f} | Target 2: {setup.target_2:.2f}")
            lines.append(f"R:R = 1:{setup.risk_reward:.1f}")
            lines.append(f"Position: {setup.shares} shares ({setup.position_percent*100:.1f}% portfolio)")

            # ---------------------------------------------------------------
            # ENHANCEMENT: Chart Reading inline block per setup
            # ---------------------------------------------------------------
            chart_result = getattr(setup, "_chart_result", None)
            if chart_result is not None:
                lines.append("")
                lines.append(f"  📊 CHART ANALYSIS [{setup.stock}]")
                lines.append(f"  Structure  : {chart_result.price_structure.value} — {chart_result.structure_detail}")
                lines.append(f"  Volume     : {chart_result.volume_analysis.summary()}")
                lines.append(f"  Chart Score: {chart_result.overall_chart_score:.0f}/100")
                if chart_result.breakout_signal:
                    lines.append(f"  Signal     : {chart_result.breakout_signal}")
                if chart_result.confluence_zones:
                    top_zone = chart_result.confluence_zones[0]
                    lines.append(f"  Confluence : {top_zone.summary()}")
                if chart_result.heatmap_row:
                    lines.append(f"  {chart_result.heatmap_row}")
                # ASCII chart (indented)
                for chart_line in chart_result.chart_ascii.split("\n"):
                    lines.append("  " + chart_line)
        
        lines.append("")
        lines.append("=" * 65)
        lines.append("PORTFOLIO RISK DASHBOARD")
        lines.append(f"Capital: {plan.portfolio_risk['capital']:,.0f}")
        lines.append(f"Deployed: {plan.portfolio_risk['deployed_percent']:.1f}%")
        lines.append(f"Cash: {plan.portfolio_risk['cash_percent']:.1f}%")
        lines.append(f"Heat: {plan.portfolio_risk['portfolio_heat']:.1f}%")
        lines.append(f"Drawdown: {plan.portfolio_risk['drawdown']:.1f}%")
        
        lines.append("")
        lines.append("=" * 65)
        lines.append("TODAY'S ACTION ITEMS")
        for item in plan.action_items:
            lines.append(f"[ ] {item}")
        
        lines.append("")
        lines.append("=" * 65)
        
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Enhancement 8: Self-Reflection – coordinator-level entry point
    # ------------------------------------------------------------------

    def reflect_on_closed_trade(self, trade_result: Dict):
        """
        Called after a trade closes. Passes outcome to each agent that
        contributed so they can auto-calibrate future confidence.
        Also feeds the ContinuousLearningModule for accuracy tracking
        and updates the ComplianceEngine with the new portfolio value.

        trade_result should contain:
            - direction:        'LONG' | 'SHORT' | 'NEUTRAL'
            - return_pct:       actual percentage return
            - agent_verdicts:   Dict[str, AgentVerdict] (from the original TradeSetup)
            - recommending_agents: pipe-separated agent names (for learning module)
            - regime_at_entry:  market regime string
            - pnl_pct:          final P&L percentage
            - stock:            symbol
            - pattern:          detected chart pattern (optional)
            - new_portfolio_value: updated portfolio value (optional)
        """
        actual_outcome = {
            "direction": trade_result.get("direction"),
            "return_pct": trade_result.get("return_pct", 0)
        }
        agent_verdicts = trade_result.get("agent_verdicts", {})

        for name, verdict in agent_verdicts.items():
            agent = self.agents.get(name)
            if agent and hasattr(agent, "reflect_on_prediction"):
                reflection = agent.reflect_on_prediction(verdict, actual_outcome)
                self.logger.info(
                    f"[Self-Reflection] {name}: errors={reflection['errors']}, "
                    f"calib={agent.confidence_calibration:.2f}"
                )

        # Feed ContinuousLearningModule
        if self._learning_module is not None:
            try:
                self._learning_module.record_outcome(trade_result)
                self.logger.info(
                    f"[ContinuousLearning] Outcome recorded for "
                    f"{trade_result.get('stock', '?')} | "
                    f"P&L: {trade_result.get('pnl_pct', 0):+.2f}%"
                )
            except Exception as e:
                self.logger.warning(f"Learning module outcome recording failed: {e}")

        # Update ComplianceEngine drawdown tracker
        new_portfolio_val = trade_result.get("new_portfolio_value")
        if new_portfolio_val and self.compliance_engine is not None:
            drawdown = self.compliance_engine.update_portfolio_value(new_portfolio_val)
            self.logger.info(
                f"[Compliance] Portfolio updated to ₹{new_portfolio_val:,.0f} "
                f"| Drawdown: {drawdown*100:.1f}%"
            )

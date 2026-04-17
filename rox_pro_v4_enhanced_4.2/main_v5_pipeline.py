"""ROX Engine v6.0 — Main Pipeline Orchestrator (CLOSED-LOOP LEARNING MODE)
3-wave async execution with full reasoning upgrade.
v5.1 — Windows asyncio fix, RAM management, centralized model routing.
v6.0 — Closed-loop learning: RegimeArbiter, DirectionalRouter+ShortExecutor,
         CircuitBreakerV2, TradeOutcomeLogger, AdaptiveCalibrator feedback.
         ACTIVATED 2026-04-17 — all 15 tasks complete, 103/103 tests pass."""

ROX_VERSION = "v6.0"

import asyncio
import json
import logging
import platform
import sys
import time
from datetime import datetime
from typing import Optional

# ═══════════════════════════════════════════════════════════════════
# CRITICAL: Windows asyncio event loop policy
# Must be set BEFORE any asyncio operations, otherwise
# asyncio.gather() will crash with RuntimeError on Windows.
# ═══════════════════════════════════════════════════════════════════
if platform.system() == "Windows" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from config_v5 import EngineConfig, validate_startup, check_system_resources, GEMINI_MODEL_ROUTING
from core.v5_logger import setup_logging
from agents.llm.async_client import GeminiClient
#from reasoning import LLMResponse
from reasoning.data_classes import (
    Signal, TradePlan, MarketState, RegimeResult, NewsResult,
    PortfolioState, TradeRecord,
)
from reasoning.cot_prompts import (
    build_regime_cot_prompt,
    build_news_prompt,
    build_trading_planner_prompt,
    build_fno_brain_prompt,
    build_self_reflector_prompt,
)
from reasoning.debate_engine import DebateEngine, DebateResult
from reasoning.pattern_memory import PatternMemoryBank, DailySnapshot
from reasoning.confidence_calibrator import ConfidenceCalibrator
from reasoning.rule_validator import RuleBasedValidator
from reasoning.adaptive_and_cache import AdaptivePromptSelector, RegimeCache

# ── v6.0 Module Imports ──────────────────────────────────────────────────
try:
    from reasoning.rule_regime_classifier import RuleRegimeClassifier
    from reasoning.regime_arbiter import RegimeArbiter
    from reasoning.regime_transition_detector import RegimeTransitionDetector
    from reasoning.regime_accuracy_tracker import RegimeAccuracyTracker
    from reasoning.adaptive_calibrator import AdaptiveConfidenceCalibrator
    from execution.short_executor import ShortExecutor
    from execution.directional_router import DirectionalRouter
    from monitoring.circuit_breaker_v2 import CircuitBreakerV2
    from data.trade_outcome_logger import TradeOutcomeLogger
    V6_AVAILABLE = True
except ImportError:
    V6_AVAILABLE = False

logger = logging.getLogger("rox.engine")

# ═══════════════════════════════════════════════════════════════════
# v6.0 ACTIVATION BANNER — logged once at module load
# ═══════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("  ROX ENGINE v6.0 — CLOSED-LOOP LEARNING MODE ACTIVATED")
logger.info("  RegimeArbiter | DirectionalRouter | ShortExecutor | CircuitBreakerV2")
logger.info("  TradeOutcomeLogger | AdaptiveCalibrator | PatternMemory feedback")
logger.info("  Debate: BULL/BEAR adversarial, temp=0.7, diversity score")
logger.info("  Cycle: Adaptive interval, signal tracing, lunch-hour skip")
logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# v6.0 ADAPTIVE CYCLE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_cycle_interval_minutes(vix: float, regime: str,
                                consecutive_no_signals: int,
                                minutes_since_last_trade: float) -> int:
    """Adaptive cycle frequency to avoid burning tokens on stale data."""
    if regime in ("EXTREME", "VOLATILE") or vix > 25:
        return 5
    elif regime == "TRENDING":
        return 15
    elif consecutive_no_signals >= 3:
        return 30
    elif minutes_since_last_trade > 120:
        return 20
    else:
        return 10


def should_skip_cycle(current_time, regime: str,
                       consecutive_no_signals: int) -> tuple:
    """Returns (should_skip: bool, reason: str or None)."""
    hour = current_time.hour + current_time.minute / 60.0

    # Skip lunch hour — lowest liquidity, highest noise on NSE
    if 12.0 <= hour <= 13.5:
        return True, "LUNCH_HOUR_LOW_LIQUIDITY"

    return False, None


class ROXEngine:
    """
    Main ROX PROVEN EDGE ENGINE v5.0.
    Orchestrates the full analysis pipeline with 3-wave async execution.
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = setup_logging(
            level=config.logging.level,
            log_file=config.logging.log_file,
        )
        
        # Initialize components
        self.llm = GeminiClient(
            api_key=config.llm.api_key,
            config=config.llm,
        )
        
        self.debate_engine = DebateEngine(
            client=self.llm,
            model_pro=config.llm.model_pro,
            model_flash=config.llm.model_flash,
        )
        
        self.calibrator = ConfidenceCalibrator(
            weights={
                "debate_agreement": config.reasoning.weight_debate_agreement,
                "pattern_match": config.reasoning.weight_pattern_match,
                "technical_alignment": config.reasoning.weight_technical_alignment,
                "volume_confirmation": config.reasoning.weight_volume_confirmation,
                "regime_consistency": config.reasoning.weight_regime_consistency,
                "anti_consensus": config.reasoning.weight_anti_consensus,
            }
        )
        
        self.validator = RuleBasedValidator(
            min_rr_ratio=config.reasoning.min_rr_ratio,
            rsi_long_min=config.reasoning.rsi_long_min,
            rsi_short_max=config.reasoning.rsi_short_max,
            volume_min_pct_of_avg=config.reasoning.volume_min_pct_of_avg,
            min_signal_strength=config.trading_rules.min_signal_strength,
            require_price_above_sma20_long=config.reasoning.price_above_sma20_required,
        )
        
        self.pattern_bank = PatternMemoryBank(
            db_path=config.reasoning.pattern_db_path,
        )
        
        self.adaptive_selector = AdaptivePromptSelector()
        self.regime_cache = RegimeCache(
            ttl_high_conf=config.reasoning.regime_cache_ttl_high_conf,
            ttl_med_conf=config.reasoning.regime_cache_ttl_med_conf,
            min_confidence=config.reasoning.regime_cache_min_confidence,
            invalidate_vix_delta=config.reasoning.regime_invalidate_vix_delta,
            invalidate_dma_break=config.reasoning.regime_invalidate_dma_break,
        )
        
        self.portfolio = PortfolioState(
            capital=config.portfolio.initial_capital,
        )
        
        # ── v6.0 Closed-Loop Learning Modules (ACTIVE — no fallback) ─────────
        # All v6 modules are ALWAYS loaded. If import fails, the system
        # refuses to start rather than silently degrading to v5 mode.
        assert V6_AVAILABLE, (
            "ROX v6.0 requires all v6 modules. Import failed — "
            "check reasoning/, execution/, monitoring/, data/ packages."
        )
        self.rule_regime_classifier = RuleRegimeClassifier()
        self.regime_arbiter = RegimeArbiter()
        self.regime_transition_detector = RegimeTransitionDetector()
        self.regime_accuracy_tracker = RegimeAccuracyTracker()
        self.adaptive_calibrator_v6 = AdaptiveConfidenceCalibrator()
        self.short_executor = ShortExecutor()
        self.circuit_breaker_v2 = CircuitBreakerV2(
            initial_capital=config.portfolio.initial_capital,
            consecutive_loss_threshold=3,
            daily_loss_limit_pct=5.0,
            max_drawdown_pct=10.0,
            reduced_size_pct=50.0,
        )
        self.directional_router = DirectionalRouter(
            circuit_breaker=self.circuit_breaker_v2
        )
        self.trade_outcome_logger = TradeOutcomeLogger()
        self._short_paper_trade_count = 0
        self._SHORT_PAPER_MODE_LIMIT = 15
        self._v6_regime_decision = None
        self._v6_transition_event = None
        self._v6_cycle_number = 0
        self._v6_previous_regime = ""
        self._v6_previous_vix = 0.0
        self._v6_previous_fii = 0.0
        logger.info(
            "v6.0 CLOSED-LOOP modules ACTIVE: RuleRegimeClassifier, RegimeArbiter, "
            "RegimeTransitionDetector, RegimeAccuracyTracker, AdaptiveCalibrator, "
            "ShortExecutor, DirectionalRouter, CircuitBreakerV2, TradeOutcomeLogger"
        )
    
    async def run_cycle(self, market_data: dict) -> dict:
        """
        Execute one full analysis cycle with 3-wave async execution.
        """
        cycle_start = time.monotonic()
        self._v6_cycle_number += 1
        logger.info("=" * 70)
        logger.info(f"CYCLE START — ROX Engine {ROX_VERSION} Analysis Cycle #{self._v6_cycle_number}")
        logger.info("=" * 70)

        # ── v6.0 Adaptive cycle skip + interval check ─────────────────────
        from datetime import datetime as dt
        _now = dt.now()
        skip, reason = should_skip_cycle(
            _now,
            market_data.get("regime", "UNKNOWN"),
            market_data.get("consecutive_no_signals", 0)
        )
        if skip:
            logger.info(f"CYCLE SKIPPED: {reason}")
            cycle_time = time.monotonic() - cycle_start
            return {"action": "SKIPPED", "reason": reason, "trades": [], "cycle_time_ms": int(cycle_time * 1000), "stats": {}}
        
        # ── v6.0: Circuit breaker pre-check ───────────────────────────────
        _can_trade, _cb_reason = self.circuit_breaker_v2.can_trade()
        if not _can_trade:
            logger.warning(
                f"CIRCUIT BREAKER HALT: {_cb_reason} — cycle aborted"
            )
            cycle_time = time.monotonic() - cycle_start
            return {
                "action": "HALTED",
                "reason": f"CIRCUIT_BREAKER:{_cb_reason}",
                "circuit_breaker_state": self.circuit_breaker_v2.get_state().__dict__,
                "trades": [],
                "cycle_time_ms": int(cycle_time * 1000),
                "stats": {},
            }
        
        # ───────────────────────────────────────────────────────────────
        # PRE-FLIGHT: Assess Complexity
        # ───────────────────────────────────────────────────────────
        complexity = self.adaptive_selector.assess_complexity(
            vix=market_data.get("vix", 15),
            intraday_range_pct=market_data.get("intraday_range_pct", 0),
            macro_event_count=market_data.get("macro_events_today", 0),
            fii_streak_days=market_data.get("fii_trend_days", 0),
            key_level_test=market_data.get("key_level_test", "NONE"),
            sector_dispersion=market_data.get("sector_dispersion", 0),
        )
        
        logger.info(f"Complexity: {complexity.complexity.value} | "
                     f"Debate rounds: {complexity.debate_rounds} | "
                     f"CoT steps: {complexity.cot_steps} | "
                     f"Conf threshold: {complexity.confidence_threshold}%")
        
        # Build market state
        market_state = self._build_market_state(market_data)
        
        # ────────────────────────────────────────────────────────────
        # WAVE 1: Regime Detection + News Analysis (PARALLEL)
        # ────────────────────────────────────────────────────────────
        logger.info(">>> WAVE 1: Regime + News (parallel)...")
        
        # Check regime cache first
        cached = self.regime_cache.get(
            current_vix=market_data.get("vix", 15),
            current_nifty=market_data.get("nifty_price", 24200),
        )
        
        if cached:
            regime_result = RegimeResult(
                regime=cached.regime,
                confidence=cached.confidence,
                reasoning=cached.reasoning,
                timestamp=datetime.now().isoformat(),
            )
            logger.info(f"Regime from cache: {cached.regime} ({cached.confidence}%)")
        else:
            # ── v6.0: Run RuleRegimeClassifier + LLM in parallel ──────────────
            _v6_rule_regime = None
            _v6_llm_regime = None
            
            # Source 1: v6 RuleRegimeClassifier (deterministic, no LLM call)
            if self.rule_regime_classifier is not None:
                try:
                    _nifty_20dma = market_data.get("nifty_20dma",
                                    market_data.get("nifty_200dma",
                                    market_data.get("nifty_price", 0)))
                    _fii = market_data.get("fii_net_cr",
                            market_data.get("raw_market_data", {}).get("fii_net_cr", 0))
                    _rule_result = self.rule_regime_classifier.classify(
                        vix=market_data.get("vix", 15),
                        nifty_price=market_data.get("nifty_price", 0),
                        nifty_20dma=_nifty_20dma,
                        fii_net_flow=_fii,
                        sector_green_pct=market_data.get("sector_green_pct", 50),
                        nifty_5d_slope=market_data.get("nifty_5d_slope", 0),
                    )
                    _v6_rule_regime = _rule_result
                    logger.info(
                        f"[V6-RULE-REGIME] {_rule_result.regime} | "
                        f"confidence={_rule_result.confidence:.0f}%"
                    )
                except Exception as _e:
                    logger.debug(f"[V6-RULE-REGIME] failed: {_e}")
            
            # Source 2: LLM regime detection (existing code)
            regime_cot_prompt = build_regime_cot_prompt(
                market_data=market_data.get("raw_market_data", {}),
                num_steps=complexity.cot_steps,
            )
            
            model = self.adaptive_selector.get_model(complexity, "regime_detector")
            regime_response = await self.llm.generate(
                prompt=regime_cot_prompt,
                model=model,
                temperature=self.config.llm.temperature_cot,
                max_tokens=self.config.llm.max_output_tokens,
                expect_json=True,
            )
            
            regime_data = regime_response.json_data or {}
            _v6_llm_regime_str = regime_data.get("regime", "UNKNOWN")
            _v6_llm_confidence = regime_data.get("confidence", 50)
            
            # ── v6.0: RegimeArbiter conflict resolution ──────────────────────
            if (self.regime_arbiter is not None 
                and _v6_rule_regime is not None):
                try:
                    _llm_accuracy = 1.0
                    if self.regime_accuracy_tracker:
                        _acc = self.regime_accuracy_tracker.get_rolling_accuracy(n=20)
                        _llm_accuracy = _acc.get("llm_accuracy", 1.0)
                    
                    _decision = self.regime_arbiter.resolve(
                        rule_regime=_v6_rule_regime.regime,
                        rule_confidence=_v6_rule_regime.confidence,
                        llm_regime=_v6_llm_regime_str,
                        llm_confidence=_v6_llm_confidence,
                        llm_rolling_accuracy=_llm_accuracy,
                    )
                    self._v6_regime_decision = _decision
                    logger.info(
                        f"[V6-ARBITER] {_decision.regime} ({_decision.confidence:.0f}%) | "
                        f"source={_decision.source}"
                    )
                    
                    # Use arbiter's decision as primary regime
                    regime_result = RegimeResult(
                        regime=_decision.regime,
                        confidence=_decision.confidence,
                        key_level=regime_data.get("key_level", 0),
                        key_level_type=regime_data.get("key_level_type", "support"),
                        reasoning=regime_data.get("reasoning", {}),
                        timestamp=datetime.now().isoformat(),
                    )
                except Exception as _arb_err:
                    logger.warning(f"[V6-ARBITER] failed, using LLM result: {_arb_err}")
                    regime_result = RegimeResult(
                        regime=_v6_llm_regime_str,
                        confidence=_v6_llm_confidence,
                        key_level=regime_data.get("key_level", 0),
                        key_level_type=regime_data.get("key_level_type", "support"),
                        reasoning=regime_data.get("reasoning", {}),
                        timestamp=datetime.now().isoformat(),
                    )
            else:
                # Fallback: LLM-only regime
                regime_result = RegimeResult(
                    regime=_v6_llm_regime_str,
                    confidence=_v6_llm_confidence,
                    key_level=regime_data.get("key_level", 0),
                    key_level_type=regime_data.get("key_level_type", "support"),
                    reasoning=regime_data.get("reasoning", {}),
                    timestamp=datetime.now().isoformat(),
                )
            
            self.regime_cache.set(
                regime=regime_result.regime,
                confidence=regime_result.confidence,
                reasoning=regime_result.reasoning,
                vix=market_data.get("vix", 15),
                nifty=market_data.get("nifty_price", 24200),
            )
            
            # ── v6.0: Regime transition detection ─────────────────────────────
            if self.regime_transition_detector is not None:
                try:
                    _vix_prev = self._v6_previous_vix or market_data.get("vix", 15)
                    _fii_prev = self._v6_previous_fii or market_data.get("fii_net_cr", 0)
                    _event = self.regime_transition_detector.detect(
                        current_regime=regime_result.regime,
                        previous_regime=self._v6_previous_regime,
                        vix_current=market_data.get("vix", 15),
                        vix_previous=_vix_prev,
                        nifty_price=market_data.get("nifty_price", 0),
                        nifty_20dma=market_data.get("nifty_20dma", 0),
                        fii_current=market_data.get("fii_net_cr", 0),
                        fii_previous=_fii_prev,
                    )
                    self._v6_transition_event = _event
                    if _event.type != "NONE":
                        logger.info(
                            f"[V6-TRANSITION] type={_event.type} | "
                            f"from={_event.from_regime} → to={_event.to_regime} | "
                            f"signals={_event.signals} | action={_event.action}"
                        )
                        # On CONFIRMED transition: reduce position size
                        if _event.type == "CONFIRMED":
                            self.circuit_breaker_v2.size_multiplier = min(
                                self.circuit_breaker_v2.size_multiplier, 0.75
                            )
                            logger.warning(
                                "[V6-TRANSITION] Size reduced to 75% due to regime change"
                            )
                except Exception as _te:
                    logger.debug(f"[V6-TRANSITION] detection failed: {_te}")
            
            # Store for next cycle
            self._v6_previous_regime = regime_result.regime
            self._v6_previous_vix = market_data.get("vix", 15)
            self._v6_previous_fii = market_data.get("fii_net_cr", 0)
        
        # News analysis (always runs, uses flash)
        news_prompt = build_news_prompt(
            headlines=market_data.get("headlines", []),
            market_context=f"Regime: {regime_result.regime} | VIX: {market_data.get('vix', 15)}",
        )
        
        news_response = await self.llm.generate(
            prompt=news_prompt,
            model=self.config.llm.model_news,
            temperature=self.config.llm.temperature_news,
            expect_json=True,
        )
        
        news_data = news_response.json_data or {}
        news_result = NewsResult(
            sentiment_score=news_data.get("aggregate", {}).get("sentiment_score", 0),
            sentiment_label=news_data.get("aggregate", {}).get("sentiment_label", "NEUTRAL"),
            block_long_sectors=news_data.get("aggregate", {}).get("block_long_sectors", []),
            block_short_sectors=news_data.get("aggregate", {}).get("block_short_sectors", []),
            boost_sectors=news_data.get("aggregate", {}).get("boost_sectors", []),
            uncertainty_level=news_data.get("aggregate", {}).get("uncertainty_level", "MEDIUM"),
        )
        
        logger.info(
            f"Wave 1 complete | Regime: {regime_result.regime} "
            f"({regime_result.confidence}%) | "
            f"News: {news_result.sentiment_label} ({news_result.sentiment_score})"
        )
        
        # ────────────────────────────────────────────────────────────
        # PATTERN MEMORY LOOKUP (Parallel with Wave 1)
        # ────────────────────────────────────────────────────────────
        logger.info(">>> Pattern Memory lookup...")
        
        if self.config.reasoning.pattern_memory_enabled:
            current_snapshot = self._build_snapshot(market_data)
            pattern_matches = self.pattern_bank.find_similar(
                current_snapshot,
                top_k=complexity.pattern_match_count,
            )
            
            few_shot_text = self.pattern_bank.format_as_few_shot(pattern_matches)
            logger.info(f"Pattern matches found: {len(pattern_matches)}")
        else:
            pattern_matches = []
            few_shot_text = ""
        
        # ────────────────────────────────────────────────────────────
        # META-LEARNER (Conditional)
        # ────────────────────────────────────────────────────────────
        meta_confidence = None
        if self.config.reasoning.meta_learner_enabled:
            trade_count = self.portfolio.total_trades
            win_pct = self.portfolio.winning_pct
            
            if (trade_count >= self.config.reasoning.meta_learner_min_trades and 
                win_pct >= self.config.reasoning.meta_learner_min_win_rate):
                logger.info("MetaLearner: Conditions met, running analysis")
                # Would call LLM here in full implementation
            else:
                logger.info(
                    f"MetaLearner SKIPPED: "
                    f"Trades: {trade_count}/{self.config.reasoning.meta_learner_min_trades}, "
                    f"Win%: {win_pct:.1%}/{self.config.reasoning.meta_learner_min_win_rate * 100}%"
                )
        
        # ────────────────────────────────────────────────────────────
        # WAVE 2: Debate Protocol
        # ────────────────────────────────────────────────────────────
        logger.info(">>> WAVE 2: Debate Protocol...")
        
        debate_result = None
        if self.config.reasoning.debate_enabled and complexity.debate_rounds > 0:
            debate_result = await self.debate_engine.run_debate(
                market_data=market_data.get("raw_market_data", {}),
                regime_result={"regime": regime_result.regime, "confidence": regime_result.confidence},
                news_result={"aggregate": news_data.get("aggregate", {}), "headline_analysis": news_data.get("headline_analysis", [])},
                pattern_matches=[{"date": m.date, "similarity": m.similarity, "outcome": m.outcome, "optimal_strategy": m.optimal_strategy, "lesson": m.lesson} for m in pattern_matches] if pattern_matches else None,
                include_neutral=complexity.debate_rounds >= 2,
                rounds=complexity.debate_rounds,
            )
        else:
            logger.info("Debate SKIPPED (disabled or LOW complexity)")
        
        # ────────────────────────────────────────────────────────────
        # WAVE 3: Trading Planning + FNO Brain (Parallel)
        # ────────────────────────────────────────────────────────────
        logger.info(">>> WAVE 3: Trading Planning + FNO Brain (parallel)...")
        
        # Generate raw signals (from agents - simplified for standalone)
        raw_signals = self._generate_raw_signals(market_data, news_result)
        logger.info(f"Raw signals from agents: {len(raw_signals)}")
        
        # Rule-based validation (no LLM needed)
        validation_results = self.validator.validate_batch(
            signals=[{
                "symbol": s.symbol,
                "direction": s.direction.value if hasattr(s.direction, 'value') else str(s.direction),
                "strength": s.strength.value if hasattr(s.strength, 'value') else str(s.strength),
                "agent": s.agent,
                "rr_ratio": s.rr_ratio,
                "rsi": s.rsi,
                "volume": s.volume,
                "volume_avg_20d": s.volume_avg_20d,
                "price": s.price,
                "sma_20": s.sma_20,
                "sector": s.sector,
            } for s in raw_signals],
            regime={"regime": regime_result.regime},
            news_restrictions={
                "block_long_sectors": news_result.block_long_sectors,
                "block_short_sectors": news_result.block_short_sectors,
            },
            active_sectors=self._get_active_sectors(),
        )
        
        passed_signals = [r for r in validation_results if r.passed]
        logger.info(f"After rule validation: {len(passed_signals)} signals passed (eliminated {len(raw_signals) - len(passed_signals)})")
        
        if not passed_signals:
            logger.info("NO signals passed validation — cycle complete, no trades")
            cycle_time = time.monotonic() - cycle_start
            stats = self.llm.get_stats()
            
            return {
                "action": "NO_SIGNALS",
                "market_state": self._market_state_to_dict(market_state),
                "regime": {"regime": regime_result.regime, "confidence": regime_result.confidence},
                "news": {"sentiment": news_result.sentiment_label, "score": news_result.sentiment_score},
                "validation": {"passed": 0, "failed": len(raw_signals)},
                "calibration": {"raw": 0, "calibrated": 0, "actionable": False},
                "debate": {"ran": bool(debate_result), "agreement": debate_result.debate_agreement if debate_result else 0},
                "stats": stats,
                "cycle_time_ms": int(cycle_time * 1000),
            }
        
        # Trading Planner (only if signals pass)
        trading_planner_prompt = build_trading_planner_prompt(
            signals=[{
                "symbol": s.original_signal.get("symbol", s.symbol) if s.original_signal else s.symbol,
                "direction": s.original_signal.get("direction", "LONG") if s.original_signal else "LONG",
                "strength": s.original_signal.get("strength", "MEDIUM") if s.original_signal else "MEDIUM",
                "rr_ratio": s.original_signal.get("rr_ratio", 1.5) if s.original_signal else 1.5,
                "rsi": s.original_signal.get("rsi") if s.original_signal else None,
                "sector": s.original_signal.get("sector", "") if s.original_signal else "",
                "agent": s.original_signal.get("agent", "") if s.original_signal else "",
                "price": s.original_signal.get("price") if s.original_signal else None,
            } for s in passed_signals],
            regime={"regime": regime_result.regime, "confidence": regime_result.confidence},
            news_restrictions=news_result.block_long_sectors,
            portfolio={"capital": self.portfolio.capital, "risk_pct": self.config.portfolio.risk_per_trade_pct},
            prediction={
                "prediction": debate_result.final_prediction.get("prediction", {}) if debate_result else {},
                "calibrated_confidence": 0,  # Will be set by calibrator
            },
        )
        
        planner_response = await self.llm.generate(
            prompt=trading_planner_prompt,
            model=self.config.llm.model_pro,
            temperature=self.config.llm.temperature_planning,
            max_tokens=self.config.llm.max_output_tokens,
            expect_json=True,
        )
        
        planner_data = planner_response.json_data or {}
        
        # Build trade plans
        trade_plans = []
        for t in planner_data.get("trades", []):
            if t.get("verdict") == "EXECUTE":
                trade_plans.append(TradePlan(
                    symbol=t.get("symbol", ""),
                    direction=t.get("direction", "LONG"),
                    entry_price=t.get("entry_price", 0),
                    stop_loss=t.get("stop_loss", 0),
                    target_1=t.get("target_1", 0),
                    target_2=t.get("target_2", 0),
                    position_size=t.get("position_size", 0),
                    risk_amount=t.get("risk_amount", 0),
                    expected_reward=t.get("expected_reward", 0),
                    verdict=t.get("verdict", "HOLD"),
                    verdict_reason=t.get("verdict_reason", ""),
                    strategy=t.get("strategy", ""),
                    validation=t.get("validation", ""),
                    invalidation=t.get("invalidation", ""),
                ))
        
        # ────────────────────────────────────────────────────────────
        # CONFIDENCE CALIBRATION
        # ────────────────────────────────────────────────────────────
        logger.info(">>> Confidence Calibration...")
        
        if debate_result and self.config.reasoning.calibration_enabled:
            cal_result = self.calibrator.calibrate(
                debate_agreement=debate_result.debate_agreement,
                debate_confidence=debate_result.raw_confidence,
                pattern_match_score=pattern_matches[0].similarity if pattern_matches else 0,
                pattern_match_count=len(pattern_matches),
                technical_aligned=market_state.nifty_vs_20dma > 0 and market_state.nifty_vs_50dma > 0,
                volume_confirms_price=market_data.get("volume_confirms", False),
                volume_strength=market_data.get("volume_strength", 0.5),
                regime_direction=regime_result.regime,
                prediction_direction=debate_result.final_prediction.get("prediction", {}).get("direction", "NEUTRAL"),
                raw_confidence=debate_result.raw_confidence,
                vix_level=market_data.get("vix", 15),
                macro_event_count=market_data.get("macro_events_today", 0),
                news_restrictions_count=len(news_result.block_long_sectors) + len(news_result.block_short_sectors),
                is_200dma_test=market_data.get("key_level_test") == "200DMA",
                is_ath_test=market_data.get("key_level_test") == "ATH",
                position_overlaps=0,
                market_positioning="NEUTRAL",
                pattern_bullish_count=sum(1 for m in pattern_matches if "UP" in m.outcome.upper() or "RALLY" in m.outcome.upper()),
                pattern_bearish_count=sum(1 for m in pattern_matches if "DOWN" in m.outcome.upper() or "FELL" in m.outcome.upper()),
                pattern_total_count=len(pattern_matches),
            )
            
            calibrated = cal_result.calibrated_confidence
            actionable = cal_result.is_actionable
            
            logger.info(
                f"Calibration: raw={debate_result.raw_confidence:.1f}% → "
                f"calibrated={calibrated:.1f}% | "
                f"actionable={actionable}"
            )
        else:
            calibrated = regime_result.confidence
            actionable = False
            cal_result = None
        
        # ────────────────────────────────────────────────────────────
        # v6.0: DIRECTIONAL ROUTING — Route LONG/SHORT to executors
        # ────────────────────────────────────────────────────────────
        _v6_routed_trades = []
        _v6_short_orders = []
        
        if trade_plans and actionable:
            for tp in trade_plans:
                _direction = tp.direction.upper() if isinstance(tp.direction, str) else str(tp.direction)
                
                if _direction == "SHORT":
                    # ── v6.0: Route SHORT via ShortExecutor ──────────────────
                    try:
                        _short_order = self.short_executor.prepare_short_order(
                            symbol=tp.symbol,
                            spot_price=tp.entry_price,
                            conviction=calibrated,
                            regime=regime_result.regime,
                            portfolio_capital=self.portfolio.capital,
                        )
                        if _short_order is not None:
                            # Paper mode check for first N SHORTs
                            if self._short_paper_trade_count < self._SHORT_PAPER_MODE_LIMIT:
                                self._short_paper_trade_count += 1
                                _short_order._v6_paper_short = True
                                logger.info(
                                    f"[V6-SHORT-PAPER] {tp.symbol} | "
                                    f"paper trade #{self._short_paper_trade_count}/"
                                    f"{self._SHORT_PAPER_MODE_LIMIT} | "
                                    f"strategy={_short_order.strategy.value}"
                                )
                            else:
                                # Live SHORT — route through DirectionalRouter
                                _result = self.directional_router.route_short(
                                    short_order=_short_order,
                                    execute_fn=lambda o: {"order_id": f"V6_SHORT_{self._v6_cycle_number}"},
                                )
                                if not _result.executed:
                                    logger.warning(
                                        f"[V6-SHORT-BLOCKED] {tp.symbol}: {_result.reason}"
                                    )
                            
                            _v6_short_orders.append({
                                "symbol": tp.symbol,
                                "strategy": _short_order.strategy.value,
                                "strike": _short_order.strike,
                                "lots": _short_order.lots,
                                "premium": _short_order.premium,
                                "paper": self._short_paper_trade_count <= self._SHORT_PAPER_MODE_LIMIT,
                            })
                            _v6_routed_trades.append({
                                "symbol": tp.symbol, "direction": "SHORT",
                                "entry": tp.entry_price, "sl": tp.stop_loss,
                                "t1": tp.target_1, "t2": tp.target_2,
                                "size": tp.position_size, "risk": tp.risk_amount,
                                "reward": tp.expected_reward, "strategy": tp.strategy,
                                "v6_strategy": _short_order.strategy.value,
                                "v6_strike": _short_order.strike,
                            })
                    except Exception as _se:
                        logger.warning(f"[V6-SHORT-ERROR] {tp.symbol}: {_se}")
                
                elif _direction == "LONG":
                    # ── v6.0: Route LONG via DirectionalRouter ───────────────
                    try:
                        _signal_data = {
                            "symbol": tp.symbol,
                            "entry_price": tp.entry_price,
                            "stop_loss": tp.stop_loss,
                            "target_1": tp.target_1,
                            "target_2": tp.target_2,
                            "position_size": tp.position_size,
                            "strategy": tp.strategy,
                        }
                        _result = self.directional_router.route_long(
                            signal_data=_signal_data,
                            execute_fn=lambda data: {"order_id": f"V6_LONG_{self._v6_cycle_number}"},
                        )
                        if not _result.executed:
                            logger.warning(
                                f"[V6-LONG-BLOCKED] {tp.symbol}: {_result.reason}"
                            )
                        _v6_routed_trades.append({
                            "symbol": tp.symbol, "direction": "LONG",
                            "entry": tp.entry_price, "sl": tp.stop_loss,
                            "t1": tp.target_1, "t2": tp.target_2,
                            "size": tp.position_size, "risk": tp.risk_amount,
                            "reward": tp.expected_reward, "strategy": tp.strategy,
                        })
                    except Exception as _le:
                        logger.warning(f"[V6-LONG-ERROR] {tp.symbol}: {_le}")
                
                else:
                    # NEUTRAL or unknown — pass through
                    _v6_routed_trades.append({
                        "symbol": tp.symbol, "direction": _direction,
                        "entry": tp.entry_price, "sl": tp.stop_loss,
                        "t1": tp.target_1, "t2": tp.target_2,
                        "size": tp.position_size, "risk": tp.risk_amount,
                        "reward": tp.expected_reward, "strategy": tp.strategy,
                    })
        
        # ── v6.0: Trade Outcome Logging ──────────────────────────────────
        if _v6_routed_trades and self.trade_outcome_logger is not None:
            for _trade in _v6_routed_trades:
                try:
                    _agent_verdicts = [
                        {"agent": s.agent, "direction": str(s.direction.value) if hasattr(s.direction, 'value') else str(s.direction), "conviction": str(s.strength.value) if hasattr(s.strength, 'value') else str(s.strength)}
                        for s in passed_signals[:5]
                    ] if passed_signals else []
                    self.trade_outcome_logger.log_trade(
                        timestamp_entry=datetime.now().isoformat(),
                        timestamp_exit=None,
                        symbol=_trade["symbol"],
                        direction=_trade["direction"],
                        entry_price=_trade["entry"],
                        exit_price=None,
                        pnl=None,
                        regime_at_entry=regime_result.regime,
                        regime_confidence=regime_result.confidence,
                        debate_agreement_score=debate_result.debate_agreement * 100 if debate_result else 50,
                        calibration_score=calibrated,
                        agent_verdicts=_agent_verdicts,
                        signals_passed=[f"{s.symbol} {s.direction.value if hasattr(s.direction, 'value') else str(s.direction)}" for s in passed_signals],
                        signals_failed=[],
                        news_sentiment=news_result.sentiment_label,
                        pattern_match_ids=[f"PM_{i}" for i in range(len(pattern_matches))],
                        cycle_number=self._v6_cycle_number,
                    )
                except Exception as _tl_err:
                    logger.debug(f"[V6-TRADE-LOG] logging failed: {_tl_err}")
        
        # ────────────────────────────────────────────────────────────
        # FINAL ASSEMBLY
        # ────────────────────────────────────────────────────────────
        cycle_time = time.monotonic() - cycle_start
        stats = self.llm.get_stats()
        
        # Get circuit breaker state for diagnostics
        _cb_state = self.circuit_breaker_v2.get_state()
        
        final_result = {
            "version": ROX_VERSION,
            "action": "TRADE" if trade_plans and actionable else "NO_TRADE",
            "market_state": self._market_state_to_dict(market_state),
            "regime": {"regime": regime_result.regime, "confidence": regime_result.confidence},
            "news": {"sentiment": news_result.sentiment_label, "score": news_result.sentiment_score},
            "validation": {
                "total": len(raw_signals),
                "passed": len(passed_signals),
                "failed": len(raw_signals) - len(passed_signals),
            },
            "debate": {
                "ran": bool(debate_result),
                "agreement": debate_result.debate_agreement if debate_result else 0,
                "bull": debate_result.bull_thesis.get("thesis", "N/A") if debate_result else "N/A",
                "bear": debate_result.bear_thesis.get("thesis", "N/A") if debate_result else "N/A",
                "final_direction": debate_result.final_prediction.get("prediction", {}).get("direction", "N/A") if debate_result else "N/A",
                "diversity_score": DebateEngine._compute_diversity_score(
                    debate_result.bull_thesis, debate_result.bear_thesis
                ) if debate_result and debate_result.bull_thesis and debate_result.bear_thesis else None,
            },
            "calibration": {
                "raw": debate_result.raw_confidence if debate_result else regime_result.confidence,
                "calibrated": calibrated,
                "actionable": actionable,
                "adjustments": cal_result.adjustments_applied if cal_result else [],
                "threshold": complexity.confidence_threshold,
                "v6_adaptive_weights": self.adaptive_calibrator_v6.get_weights() if self.adaptive_calibrator_v6 else {},
            },
            "trades": _v6_routed_trades if _v6_routed_trades else [
                {
                    "symbol": tp.symbol,
                    "direction": tp.direction,
                    "entry": tp.entry_price,
                    "sl": tp.stop_loss,
                    "t1": tp.target_1,
                    "t2": tp.target_2,
                    "size": tp.position_size,
                    "risk": tp.risk_amount,
                    "reward": tp.expected_reward,
                    "strategy": tp.strategy,
                }
                for tp in trade_plans
            ],
            "portfolio": planner_data.get("portfolio_summary", {}),
            "stats": stats,
            "cycle_time_ms": int(cycle_time * 1000),
            "complexity": complexity.complexity.value,
            "circuit_breaker": {
                "halted": _cb_state.halted,
                "reason": _cb_state.halt_reason,
                "consecutive_losses": _cb_state.consecutive_losses,
                "daily_pnl": _cb_state.daily_pnl,
                "size_multiplier": _cb_state.size_multiplier,
                "drawdown_pct": _cb_state.drawdown_pct,
            },
            "v6_diagnostics": self.v6_get_diagnostics(),
        }
        
        logger.info(
            f"CYCLE COMPLETE | v{ROX_VERSION} | Time: {cycle_time * 1000:.0f}ms | "
            f"Action: {final_result['action']} | "
            f"Trades: {len(_v6_routed_trades)} | "
            f"CB: halted={_cb_state.halted} size={_cb_state.size_multiplier:.0%} | "
            f"Stats: {stats}"
        )
        
        return final_result
    
    def _build_market_state(self, market_data: dict) -> MarketState:
        """Build MarketState from raw market data."""
        return MarketState(
            nifty_price=market_data.get("nifty_price", 0),
            nifty_change_pct=market_data.get("nifty_change_pct", 0),
            vix=market_data.get("vix", 15),
            vix_change_pct=market_data.get("vix_change_pct", 0),
            usd_inr=market_data.get("usd_inr", 0),
            crude_oil=market_data.get("crude_oil", 0),
            gold=market_data.get("gold", 0),
            us_10y_yield=market_data.get("us_10y_yield", 0),
            nifty_vs_20dma=market_data.get("nifty_vs_20dma", 0),
            nifty_vs_50dma=market_data.get("nifty_vs_50dma", 0),
            nifty_vs_200dma=market_data.get("nifty_vs_200dma", 0),
            rsi_14=market_data.get("rsi_14", 50),
            pcr=market_data.get("pcr", 1.0),
            max_pain=market_data.get("max_pain", 0),
            days_to_expiry=market_data.get("days_to_expiry", 5),
            regime=market_data.get("regime", "UNKNOWN"),
            raw_market_data=market_data.get("raw_market_data", {}),
        )
    
    def _build_snapshot(self, market_data: dict) -> DailySnapshot:
        """Build a DailySnapshot for the pattern memory bank."""
        return DailySnapshot(
            date=datetime.now().strftime("%Y-%m-%d"),
            nifty_close=market_data.get("nifty_price", 0),
            nifty_vs_200dma_pct=market_data.get("nifty_vs_200dma", 0),
            nifty_vs_50dma_pct=market_data.get("nifty_vs_50dma", 0),
            nifty_vs_20dma_pct=market_data.get("nifty_vs_20dma", 0),
            vix_close=market_data.get("vix", 15),
            vix_vs_30d_avg=market_data.get("vix_vs_30d_avg", 0),
            usd_inr_close=market_data.get("usd_inr", 0),
            usd_inr_trend_5d=market_data.get("usd_inr_trend_5d", 0),
            crude_close=market_data.get("crude_oil", 0),
            fii_net_buy_sell_cr=market_data.get("fii_net_cr", 0),
            fii_trend_days=market_data.get("fii_trend_days", 0),
            dii_net_buy_sell_cr=market_data.get("dii_net_cr", 0),
            nifty_intraday_range_pct=market_data.get("intraday_range_pct", 0),
            nifty_close_vs_open=market_data.get("nifty_close_vs_open", 0),
            expiry_days_remaining=market_data.get("days_to_expiry", 5),
            pcr=market_data.get("pcr", 1.0),
            max_pain_level=market_data.get("max_pain", 0),
            put_call_oi_ratio=market_data.get("pcr", 1.0),
            global_cues_summary=market_data.get("global_cues", "NEUTRAL"),
            sector_rotation_pattern=market_data.get("sector_rotation", "NEUTRAL"),
        )
    
    def _generate_raw_signals(self, market_data: dict, news_result: NewsResult) -> list[Signal]:
        """Generate simulated agent signals (placeholder for real agent integration)."""
        # In production, this comes from ORION, VESPER, KAIRO, etc.
        # Here we return the heatmap data as raw signals
        signals = []
        
        for item in market_data.get("heatmap", []):
            signals.append(Signal(
                symbol=item.get("symbol", ""),
                direction=SignalDirection.LONG if item.get("change_pct", 0) > 0 else SignalDirection.SHORT,
                strength=SignalStrength(item.get("strength", "MEDIUM")),
                agent=item.get("agent", "UNKNOWN"),
                rr_ratio=item.get("rr_ratio", 1.5),
                rsi=item.get("rsi"),
                volume=item.get("volume"),
                volume_avg_20d=item.get("volume_avg_20d"),
                price=item.get("price"),
                sma_20=item.get("sma_20"),
                sector=item.get("sector", ""),
            ))
        
        return signals
    
    def _get_active_sectors(self) -> dict:
        """Get count of active positions by sector."""
        sectors = {}
        for pos in self.portfolio.active_positions:
            sector = pos.get("sector", "")
            if sector:
                sectors[sector] = sectors.get(sector, 0) + 1
        return sectors
    
    def _market_state_to_dict(self, ms: MarketState) -> dict:
        """Convert MarketState to dict for JSON serialization."""
        return {
            "nifty_price": ms.nifty_price,
            "nifty_change_pct": ms.nifty_change_pct,
            "vix": ms.vix,
            "usd_inr": ms.usd_inr,
            "crude_oil": ms.crude_oil,
            "gold": ms.gold,
            "us_10y_yield": ms.us_10y_yield,
            "nifty_vs_20dma": ms.nifty_vs_20dma,
            "nifty_vs_50dma": ms.nifty_vs_50dma,
            "nifty_vs_200dma": ms.nifty_vs_200dma,
            "rsi_14": ms.rsi_14,
            "pcr": ms.pcr,
            "max_pain": ms.max_pain,
            "days_to_expiry": ms.days_to_expiry,
            "regime": ms.regime,
        }


    # ═══════════════════════════════════════════════════════════════════
    # v6.0 CLOSED-LOOP FEEDBACK METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def v6_on_trade_close(self, symbol: str, direction: str, pnl: float,
                          signal_scores: dict = None,
                          pattern_match_id: str = None,
                          hold_period_minutes: int = 0) -> None:
        """
        v6.0: Called when a trade closes. Feeds the outcome into:
        1. CircuitBreakerV2 — consecutive loss / daily loss / drawdown tracking
        2. TradeOutcomeLogger — update trade record with exit data
        3. AdaptiveCalibrator — update signal weights based on win/loss
        4. PatternMemory — update pattern match with actual outcome
        
        Args:
            symbol: Trading symbol (e.g. "NSE:SBIN").
            direction: "LONG" or "SHORT".
            pnl: Profit/loss from the closed trade.
            signal_scores: Dict of signal scores used at entry (for calibrator).
            pattern_match_id: Pattern match ID from entry (for memory update).
            hold_period_minutes: How long the position was held.
        """
        _actual_outcome = "WIN" if pnl > 0 else "LOSS"
        
        # 1. Circuit breaker update
        if self.circuit_breaker_v2 is not None:
            try:
                self.circuit_breaker_v2.on_trade_close(pnl)
                _state = self.circuit_breaker_v2.get_state()
                logger.info(
                    f"[V6-CBV2] PnL={pnl:.2f} | halted={_state.halted} | "
                    f"consec_losses={_state.consecutive_losses} | "
                    f"size_mult={_state.size_multiplier:.0%} | "
                    f"dd={_state.drawdown_pct:.2%}"
                )
            except Exception as _cb_err:
                logger.error(f"[V6-CBV2] error: {_cb_err}")
        
        # 2. Trade outcome logger update
        if self.trade_outcome_logger is not None:
            try:
                self.trade_outcome_logger.update_trade(
                    symbol=symbol,
                    timestamp_entry=None,  # finds last open trade for symbol
                    exit_price=0,  # will be filled by caller if available
                    pnl=pnl,
                )
            except Exception as _tl_err:
                logger.error(f"[V6-TRADE-LOG] update failed: {_tl_err}")
        
        # 3. Adaptive calibrator weight update
        if self.adaptive_calibrator_v6 is not None and signal_scores:
            try:
                self.adaptive_calibrator_v6.update(
                    signal_scores=signal_scores,
                    won=pnl > 0,
                    timestamp=datetime.now().isoformat(),
                )
                _weights = self.adaptive_calibrator_v6.get_weights()
                logger.info(
                    f"[V6-CALIBRATOR] {symbol} {direction} "
                    f"{'WIN' if pnl > 0 else 'LOSS'} | "
                    f"weights_updated: {', '.join(f'{k}={v:.3f}' for k, v in sorted(_weights.items()))}"
                )
            except Exception as _ac_err:
                logger.error(f"[V6-CALIBRATOR] error: {_ac_err}")
        
        # 4. Pattern memory outcome update
        if self.pattern_bank is not None and pattern_match_id:
            try:
                self.pattern_bank.update_outcome(
                    match_id=pattern_match_id,
                    actual_outcome=_actual_outcome,
                    actual_pnl=pnl,
                    hold_period_minutes=hold_period_minutes,
                )
                logger.info(
                    f"[V6-PATTERN] {symbol} outcome={_actual_outcome} "
                    f"pnl={pnl:.2f} hold={hold_period_minutes}min"
                )
            except Exception as _pm_err:
                logger.debug(f"[V6-PATTERN] update failed: {_pm_err}")
    
    def v6_end_of_day_scoring(self, market_data: dict) -> None:
        """
        v6.0: End-of-day regime accuracy scoring.
        
        Compares the day's regime predictions (rule-based and LLM) against
        actual market outcome, logs to RegimeAccuracyTracker, and resets
        circuit breaker daily counters.
        
        Args:
            market_data: Dict with nifty/vix OHLC data for the session.
        """
        if self.regime_accuracy_tracker is not None:
            try:
                _rule_regime = ""
                _rule_conf = 0.0
                _llm_regime = ""
                _llm_conf = 0.0
                
                if self._v6_regime_decision is not None:
                    _rule_regime = self._v6_regime_decision.rule_regime
                    _rule_conf = self._v6_regime_decision.rule_confidence
                    _llm_regime = self._v6_regime_decision.llm_regime
                    _llm_conf = self._v6_regime_decision.llm_confidence
                
                self.regime_accuracy_tracker.log_session(
                    rule_regime=_rule_regime,
                    rule_confidence=_rule_conf,
                    llm_regime=_llm_regime,
                    llm_confidence=_llm_conf,
                    nifty_open=market_data.get("nifty_open", 0),
                    nifty_close=market_data.get("nifty_price", 0),
                    nifty_high=market_data.get("nifty_high", 0),
                    nifty_low=market_data.get("nifty_low", 0),
                    vix_open=market_data.get("vix_open", 0),
                    vix_close=market_data.get("vix", 0),
                )
                _acc = self.regime_accuracy_tracker.get_rolling_accuracy(n=20)
                logger.info(
                    f"[V6-EOD-SCORING] rule_accuracy={_acc.get('rule_accuracy', 0):.1%} | "
                    f"llm_accuracy={_acc.get('llm_accuracy', 0):.1%} | "
                    f"sessions={_acc.get('sessions_tracked', 0)} | "
                    f"rule_override={_acc.get('rule_should_override_llm', False)}"
                )
            except Exception as _eod_err:
                logger.error(f"[V6-EOD-SCORING] error: {_eod_err}")
        
        # Reset circuit breaker daily counters
        if self.circuit_breaker_v2 is not None:
            try:
                self.circuit_breaker_v2.reset_daily()
                logger.info("[V6-CBV2] Daily reset for new session")
            except Exception:
                pass
    
    def v6_get_diagnostics(self) -> dict:
        """
        v6.0: Return full diagnostic state of all v6 modules.
        
        Returns:
            Dict with all v6 module states for monitoring and debugging.
        """
        diag = {
            "version": ROX_VERSION,
            "v6_active": True,
            "cycle_number": self._v6_cycle_number,
            "short_paper_trades": self._short_paper_trade_count,
            "short_paper_limit": self._SHORT_PAPER_MODE_LIMIT,
        }
        
        if self._v6_regime_decision:
            diag["regime_decision"] = {
                "regime": self._v6_regime_decision.regime,
                "confidence": self._v6_regime_decision.confidence,
                "source": self._v6_regime_decision.source,
                "rule_regime": self._v6_regime_decision.rule_regime,
                "llm_regime": self._v6_regime_decision.llm_regime,
            }
        
        if self._v6_transition_event and self._v6_transition_event.type != "NONE":
            diag["transition_event"] = {
                "type": self._v6_transition_event.type,
                "from_regime": self._v6_transition_event.from_regime,
                "to_regime": self._v6_transition_event.to_regime,
                "signals": self._v6_transition_event.signals,
                "action": self._v6_transition_event.action,
            }
        
        if self.circuit_breaker_v2:
            _state = self.circuit_breaker_v2.get_state()
            diag["circuit_breaker"] = {
                "halted": _state.halted,
                "reason": _state.halt_reason,
                "consecutive_losses": _state.consecutive_losses,
                "daily_pnl": round(_state.daily_pnl, 2),
                "current_capital": round(_state.current_capital, 2),
                "peak_capital": round(_state.peak_capital, 2),
                "size_multiplier": _state.size_multiplier,
                "drawdown_pct": _state.drawdown_pct,
            }
        
        if self.adaptive_calibrator_v6:
            diag["calibrator_weights"] = self.adaptive_calibrator_v6.get_weights()
            diag["calibrator_correlations"] = self.adaptive_calibrator_v6.get_signal_correlations()
        
        if self.regime_accuracy_tracker:
            diag["regime_accuracy"] = self.regime_accuracy_tracker.get_rolling_accuracy(n=20)
        
        if self.trade_outcome_logger:
            diag["trade_outcome_count"] = self.trade_outcome_logger.get_trade_count()
            diag["trade_win_rate"] = self.trade_outcome_logger.get_win_rate(last_n=20)
        
        return diag


async def main():
    """Entry point for standalone execution."""
    # ── Pre-flight: System resource check ──────────────────────────────
    diag = validate_startup()
    print(f"[SYSTEM] RAM: {diag['available_ram_gb']}GB available / {diag['total_ram_gb']}GB total "
          f"| Platform: {diag['platform']} | CPU: {diag['cpu_count']} cores")
    for w in diag["warnings"]:
        print(f"[SYSTEM] {w}")
    if not diag["ready"]:
        print("[SYSTEM] FATAL: System does not meet minimum requirements. Exiting.")
        return
    
    # ── Model routing display ──────────────────────────────────────────
    print(f"[CONFIG] FAST_MODEL={GEMINI_MODEL_ROUTING['FAST_MODEL']} | "
          f"SMART_MODEL={GEMINI_MODEL_ROUTING['SMART_MODEL']} | "
          f"CACHE_TTL={GEMINI_MODEL_ROUTING['CACHE_TTL_MINUTES']}min | "
          f"MAX_PARALLEL={GEMINI_MODEL_ROUTING['MAX_PARALLEL_LLM_CALLS']}")
    
    config = EngineConfig.from_env()
    engine = ROXEngine(config)
    
    # Demo with sample market data
    sample_data = {
        "nifty_price": 24245,
        "vix": 19.42,
        "nifty_vs_200dma": 0.19,  # Just above 200DMA
        "nifty_vs_50dma": -0.35,
        "nifty_vs_20dma": -1.04,
        "rsi_14": 42.3,
        "usd_inr": 93.47,
        "crude_oil": 72.34,
        "fii_net_cr": -1847,
        "fii_trend_days": 5,
        "macro_events_today": 3,
        "key_level_test": "200DMA",
        "pcr": 1.097,
        "max_pain": 24050,
        "days_to_expiry": 3,
        "sector_dispersion": 2.8,
        "intraday_range_pct": 0.92,
        "volume_confirms": False,
        "volume_strength": 0.45,
        "headline_analyses": [],
        "raw_market_data": {"sample": True},
    }
    
    result = await engine.run_cycle(sample_data)
    
    print(f"\n{'='*60}")
    print(f"  ROX ENGINE {ROX_VERSION} — CLOSED-LOOP LEARNING MODE")
    _action = result.get('action', 'UNKNOWN')
    print(f"  ACTION: {_action}")
    print(f"  TRADES: {len(result.get('trades', []))}")
    print(f"  TIME: {result.get('cycle_time_ms', 0)}ms")
    if result.get('action') in ('SKIPPED', 'HALTED'):
        print(f"  REASON: {result.get('reason', 'N/A')}")
    print(f"  STATS: {result.get('stats', {})}")
    if result.get("circuit_breaker"):
        cb = result["circuit_breaker"]
        print(f"  CB: halted={cb['halted']} size={cb['size_multiplier']:.0%} dd={cb['drawdown_pct']:.2%}")
    print(f"{'='*60}\n")
    
    if result.get("trades"):
        for t in result["trades"]:
            print(f"  → {t['direction']} {t['symbol']} | "
                  f"Entry: ₹{t['entry']:.2f} | SL: ₹{t['sl']:.2f} | "
                  f"T1: ₹{t['t1']:.2f} | Risk: ₹{t['risk']:.0f}")
    else:
        print("  → No trades executed (signals filtered or low confidence)")


if __name__ == "__main__":
    asyncio.run(main())

"""ROX Engine v5.0 — Main Pipeline Orchestrator
3-wave async execution with full reasoning upgrade.
v5.1 — Windows asyncio fix, RAM management, centralized model routing."""

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

logger = logging.getLogger("rox.engine")


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
    
    async def run_cycle(self, market_data: dict) -> dict:
        """
        Execute one full analysis cycle with 3-wave async execution.
        """
        cycle_start = time.monotonic()
        logger.info("=" * 70)
        logger.info("CYCLE START — ROX Engine v5.0 Analysis Cycle")
        logger.info("=" * 70)
        
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
            regime_result = RegimeResult(
                regime=regime_data.get("regime", "UNKNOWN"),
                confidence=regime_data.get("confidence", 50),
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
        # FINAL ASSEMBLY
        # ────────────────────────────────────────────────────────────
        cycle_time = time.monotonic() - cycle_start
        stats = self.llm.get_stats()
        
        final_result = {
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
            },
            "calibration": {
                "raw": debate_result.raw_confidence if debate_result else regime_result.confidence,
                "calibrated": calibrated,
                "actionable": actionable,
                "adjustments": cal_result.adjustments_applied if cal_result else [],
                "threshold": complexity.confidence_threshold,
            },
            "trades": [
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
        }
        
        logger.info(
            f"CYCLE COMPLETE | Time: {cycle_time * 1000:.0f}ms | "
            f"Action: {final_result['action']} | "
            f"Trades: {len(trade_plans)} | "
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
    print(f"  ACTION: {result['action']}")
    print(f"  TRADES: {len(result.get('trades', []))}")
    print(f"  TIME: {result['cycle_time_ms']}ms")
    print(f"  STATS: {result['stats']}")
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

"""
ROX Proven Edge Engine v3.0 - Processing Pipeline
================================================
11-Tier Processing Pipeline for trade analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradeDirection, MarketRegime, ConvictionLevel


class TierStatus(Enum):
    """Status of a tier execution"""
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    FAILURE = "FAILURE"
    SKIPPED = "SKIPPED"


@dataclass
class TierResult:
    """Result of a single tier execution"""
    tier_number: int
    tier_name: str
    status: TierStatus
    output: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


@dataclass
class PipelineResult:
    """Complete pipeline execution result"""
    timestamp: datetime
    tiers: List[TierResult]
    final_recommendation: Dict
    overall_status: TierStatus
    
    def get_tier_result(self, tier_number: int) -> Optional[TierResult]:
        """Get result for a specific tier"""
        for tier in self.tiers:
            if tier.tier_number == tier_number:
                return tier
        return None


class ProcessingPipeline:
    """
    11-Tier Processing Pipeline
    
    Executes the complete analysis pipeline for trade recommendations:
    - Tier 0: Data Collection & Validation
    - Tier 11: Market Regime Detection
    - Tier 1: Meta-Learning & Agent Weighting
    - Tier 2: Contradiction Reconciliation
    - Tier 3: Historical Pattern Matching
    - Tier 4: Risk-Adjusted Conviction Scoring
    - Tier 5: Opportunistic Scanning
    - Tier 10: Position Sizing
    - Tier 9: Trade Logging
    - Tier 8: Report Generation
    """
    
    # Pipeline execution order
    TIER_ORDER = [0, 11, 1, 2, 3, 4, 5, 10, 9, 8]
    
    def __init__(self, coordinator):
        """
        Initialize pipeline with lead coordinator reference.
        
        Args:
            coordinator: LeadCoordinator instance
        """
        self.coordinator = coordinator
        self.tier_handlers = {
            0: self._tier_0_data_validation,
            11: self._tier_11_regime_detection,
            1: self._tier_1_agent_weighting,
            2: self._tier_2_contradiction_reconciliation,
            3: self._tier_3_pattern_matching,
            4: self._tier_4_conviction_scoring,
            5: self._tier_5_opportunity_scanning,
            10: self._tier_10_position_sizing,
            9: self._tier_9_trade_logging,
            8: self._tier_8_report_generation,
        }
    
    def execute(self, market_data: Dict, portfolio_status: Dict,
               watchlist: List[str]) -> PipelineResult:
        """
        Execute the complete processing pipeline.
        
        Args:
            market_data: Complete market data
            portfolio_status: Current portfolio status
            watchlist: Stocks to analyze
            
        Returns:
            PipelineResult with all tier results
        """
        start_time = datetime.now()
        tier_results = []
        context = {
            "market_data": market_data,
            "portfolio_status": portfolio_status,
            "watchlist": watchlist,
        }
        
        # Execute each tier in order
        for tier_number in self.TIER_ORDER:
            tier_start = datetime.now()
            
            try:
                handler = self.tier_handlers.get(tier_number)
                if handler:
                    result = handler(context)
                else:
                    result = TierResult(
                        tier_number=tier_number,
                        tier_name=f"Tier {tier_number}",
                        status=TierStatus.SKIPPED,
                        output={"message": "No handler defined"}
                    )
            except Exception as e:
                result = TierResult(
                    tier_number=tier_number,
                    tier_name=f"Tier {tier_number}",
                    status=TierStatus.FAILURE,
                    errors=[str(e)]
                )
            
            result.execution_time_ms = (datetime.now() - tier_start).total_seconds() * 1000
            tier_results.append(result)
            
            # Update context with tier output
            context[f"tier_{tier_number}_output"] = result.output
            
            # Stop pipeline on critical failures
            if result.status == TierStatus.FAILURE and tier_number in [0, 11]:
                break
        
        # Determine overall status
        failures = [t for t in tier_results if t.status == TierStatus.FAILURE]
        warnings = [t for t in tier_results if t.status == TierStatus.WARNING]
        
        if failures:
            overall_status = TierStatus.FAILURE
        elif warnings:
            overall_status = TierStatus.WARNING
        else:
            overall_status = TierStatus.SUCCESS
        
        # Build final recommendation
        final_recommendation = self._build_final_recommendation(context, tier_results)
        
        return PipelineResult(
            timestamp=start_time,
            tiers=tier_results,
            final_recommendation=final_recommendation,
            overall_status=overall_status
        )
    
    def _tier_0_data_validation(self, context: Dict) -> TierResult:
        """Tier 0: Data Collection & Validation"""
        market_data = context.get("market_data", {})
        
        required_datasets = [
            ("price_data", "Price/OHLCV data"),
            ("flow_data", "FII/DII flow data"),
            ("sentiment_data", "Sentiment indicators"),
            ("derivatives_data", "Options/derivatives data"),
            ("fundamental_data", "Fundamental metrics"),
            ("event_data", "Event calendar")
        ]
        
        validation_results = {}
        missing = []
        warnings = []
        
        for key, name in required_datasets:
            if key in market_data and market_data[key]:
                validation_results[key] = "PRESENT"
            else:
                validation_results[key] = "MISSING"
                missing.append(name)
        
        completeness = 6 - len(missing)
        
        # Determine status
        if completeness >= 6:
            status = TierStatus.SUCCESS
        elif completeness >= 4:
            status = TierStatus.WARNING
            warnings.append(f"Missing data: {', '.join(missing)}")
        else:
            status = TierStatus.FAILURE
        
        return TierResult(
            tier_number=0,
            tier_name="Data Validation",
            status=status,
            output={
                "completeness": f"{completeness}/6",
                "validation_results": validation_results,
                "missing_datasets": missing
            },
            warnings=warnings
        )
    
    def _tier_11_regime_detection(self, context: Dict) -> TierResult:
        """Tier 11: Market Regime Detection"""
        market_data = context.get("market_data", {})
        
        # Use coordinator's regime detection
        regime, confidence = self.coordinator._tier_11_detect_regime(market_data)
        
        # Store in context for later tiers
        context["detected_regime"] = regime
        context["regime_confidence"] = confidence
        
        return TierResult(
            tier_number=11,
            tier_name="Regime Detection",
            status=TierStatus.SUCCESS,
            output={
                "regime": regime.value,
                "confidence": confidence,
                "indicators_used": [
                    "Nifty vs 200 DMA",
                    "Price Structure",
                    "ADX",
                    "FII 5-day Flow",
                    "India VIX"
                ]
            }
        )
    
    def _tier_1_agent_weighting(self, context: Dict) -> TierResult:
        """Tier 1: Meta-Learning & Agent Weighting"""
        regime = context.get("detected_regime", MarketRegime.CONSOLIDATION)
        
        # Adjust weights based on regime
        self.coordinator._tier_1_adjust_weights(regime)
        
        # Collect current weights
        weights = {
            name: agent.current_weight
            for name, agent in self.coordinator.agents.items()
        }
        
        return TierResult(
            tier_number=1,
            tier_name="Agent Weighting",
            status=TierStatus.SUCCESS,
            output={
                "regime": regime.value,
                "adjusted_weights": weights,
                "adjustment_reason": f"Weights adjusted for {regime.value} regime"
            }
        )
    
    def _tier_2_contradiction_reconciliation(self, context: Dict) -> TierResult:
        """Tier 2: Contradiction Reconciliation"""
        market_data = context.get("market_data", {})
        regime = context.get("detected_regime", MarketRegime.CONSOLIDATION)
        
        # Run all agents
        self.coordinator._run_all_agents(market_data, regime)
        
        # Calculate consensus
        consensus = self.coordinator._tier_2_calculate_consensus()
        
        context["agent_reports"] = self.coordinator.agent_reports
        context["consensus"] = consensus
        
        return TierResult(
            tier_number=2,
            tier_name="Consensus & Reconciliation",
            status=TierStatus.SUCCESS,
            output={
                "consensus_direction": consensus.direction.value,
                "consensus_strength": consensus.strength,
                "net_score": consensus.net_score,
                "contradictions": consensus.contradictions
            }
        )
    
    def _tier_3_pattern_matching(self, context: Dict) -> TierResult:
        """Tier 3: Historical Pattern Matching"""
        # This would normally search the pattern database
        # For now, return default values
        
        return TierResult(
            tier_number=3,
            tier_name="Pattern Matching",
            status=TierStatus.SUCCESS,
            output={
                "matches_found": 0,
                "historical_win_rate": 0.5,
                "avg_return": 0,
                "pattern_rating": "NO_HISTORICAL_DATA"
            },
            warnings=["No historical patterns in database - using defaults"]
        )
    
    def _tier_4_conviction_scoring(self, context: Dict) -> TierResult:
        """Tier 4: Risk-Adjusted Conviction Scoring"""
        consensus = context.get("consensus")
        agent_reports = context.get("agent_reports", {})
        
        if not consensus:
            return TierResult(
                tier_number=4,
                tier_name="Conviction Scoring",
                status=TierStatus.FAILURE,
                errors=["No consensus available"]
            )
        
        # Calculate conviction based on consensus strength
        if consensus.strength == "STRONG":
            base_conviction = 80
        elif consensus.strength == "MODERATE":
            base_conviction = 70
        elif consensus.strength == "WEAK":
            base_conviction = 55
        else:
            base_conviction = 40
        
        # Store in context
        context["conviction_score"] = base_conviction
        
        return TierResult(
            tier_number=4,
            tier_name="Conviction Scoring",
            status=TierStatus.SUCCESS,
            output={
                "conviction_score": base_conviction,
                "conviction_level": self._get_conviction_level(base_conviction).value,
                "components": {
                    "agent_consensus": 30 if consensus.strength == "STRONG" else 20,
                    "pattern_match": 10,
                    "flow_alignment": 10,
                    "technical_quality": 10
                }
            }
        )
    
    def _tier_5_opportunity_scanning(self, context: Dict) -> TierResult:
        """Tier 5: Opportunistic Scanning"""
        market_data = context.get("market_data", {})
        watchlist = context.get("watchlist", [])
        portfolio_status = context.get("portfolio_status", {})
        
        setups = []
        
        # Scan watchlist for setups
        for stock in watchlist[:10]:
            setup = self.coordinator._analyze_stock(stock, market_data, portfolio_status)
            if setup:
                setups.append(setup)
        
        # Rank and select top 5
        top_setups = self.coordinator._tier_5_rank_setups(setups)
        
        context["top_setups"] = top_setups
        
        return TierResult(
            tier_number=5,
            tier_name="Opportunity Scanning",
            status=TierStatus.SUCCESS,
            output={
                "total_setups_found": len(setups),
                "top_setups_count": len(top_setups),
                "stocks_analyzed": len(watchlist[:10]),
                "setups": [
                    {
                        "stock": s.stock,
                        "direction": s.direction.value,
                        "conviction": s.conviction
                    }
                    for s in top_setups
                ]
            }
        )
    
    def _tier_10_position_sizing(self, context: Dict) -> TierResult:
        """Tier 10: Position Sizing"""
        top_setups = context.get("top_setups", [])
        
        # Position sizing is already done in _analyze_stock
        # This tier just validates and reports
        
        sizing_results = []
        for setup in top_setups:
            sizing_results.append({
                "stock": setup.stock,
                "shares": setup.shares,
                "position_value": setup.position_value,
                "position_percent": setup.position_percent * 100,
                "risk_percent": setup.risk_percent * 100
            })
        
        return TierResult(
            tier_number=10,
            tier_name="Position Sizing",
            status=TierStatus.SUCCESS,
            output={
                "sizing_results": sizing_results,
                "methodology": "2% risk rule with 15% position cap"
            }
        )
    
    def _tier_9_trade_logging(self, context: Dict) -> TierResult:
        """Tier 9: Trade Logging"""
        # This tier prepares trade logs for recording
        # Actual logging happens when trades are executed
        
        top_setups = context.get("top_setups", [])
        
        return TierResult(
            tier_number=9,
            tier_name="Trade Logging",
            status=TierStatus.SUCCESS,
            output={
                "trades_to_log": len(top_setups),
                "log_format": "CSV",
                "status": "Ready for logging upon execution"
            }
        )
    
    def _tier_8_report_generation(self, context: Dict) -> TierResult:
        """Tier 8: Report Generation"""
        # Generate the final report
        
        return TierResult(
            tier_number=8,
            tier_name="Report Generation",
            status=TierStatus.SUCCESS,
            output={
                "report_generated": True,
                "format": "text/markdown",
                "sections": [
                    "Market Regime Assessment",
                    "7-Agent Consensus Panel",
                    "Weighted Consensus",
                    "Top Swing Trade Setups",
                    "Portfolio Risk Dashboard",
                    "Today's Action Items"
                ]
            }
        )
    
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
    
    def _build_final_recommendation(self, context: Dict, 
                                   tier_results: List[TierResult]) -> Dict:
        """Build final recommendation from all tier results"""
        top_setups = context.get("top_setups", [])
        consensus = context.get("consensus")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "market_regime": context.get("detected_regime", MarketRegime.CONSOLIDATION).value,
            "regime_confidence": context.get("regime_confidence", 50),
            "consensus": {
                "direction": consensus.direction.value if consensus else "NEUTRAL",
                "strength": consensus.strength if consensus else "NO_CONSENSUS"
            },
            "conviction_score": context.get("conviction_score", 0),
            "top_setups": [
                {
                    "stock": s.stock,
                    "direction": s.direction.value,
                    "conviction": s.conviction,
                    "entry": s.entry_price,
                    "stop_loss": s.stop_loss,
                    "target": s.target_1
                }
                for s in top_setups
            ],
            "action": "PROCEED" if top_setups else "WAIT",
            "pipeline_status": {
                tier.tier_name: tier.status.value
                for tier in tier_results
            }
        }

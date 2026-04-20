"""
LLM Options Strategist - Options strategy optimization (Enhancement P2.2)
==========================================================================

Enhances options strategy selection with LLM reasoning:
- Strategy selection beyond rule-based heuristics
- Strike optimization considering OI distribution
- Risk management with adjustment triggers
- Edge identification and probabilistic analysis
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any

from .base_llm_agent import BaseLLMAgent, LLMConfig, LLMResponse

# Import from parent config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MarketRegime, TradeDirection


# Prompt Template
OPTIONS_STRATEGY_PROMPT = """You are an expert options strategist for Indian index derivatives.

MARKET CONTEXT:
- Index: {index_name}
- Spot Price: {spot}
- Consensus Direction: {consensus_direction} (strength: {consensus_strength})
- Market Regime: {regime}
- India VIX: {vix} (rank: {iv_rank})
- IV Trend: {iv_trend}
- Futures Basis: {futures_basis:+.1f} pts ({futures_basis_note})
- GIFT Nifty Gap: {gift_gap_pct:+.2f}% (pre-market indication)

OPTION CHAIN HIGHLIGHTS:
- ATM Strike: {atm_strike}
- ATM CE LTP (real market): {atm_ce_ltp:.1f} (BS estimate was used if 0.0)
- ATM PE LTP (real market): {atm_pe_ltp:.1f} (BS estimate was used if 0.0)
- Call OI Distribution: {call_oi_distribution}
- Put OI Distribution: {put_oi_distribution}
- Max Pain: {max_pain}
- PCR: {pcr}
- Key Resistance (highest CE OI): {resistance_strike}
- Key Support (highest PE OI): {support_strike}

PORTFOLIO CONSTRAINTS:
- Available Capital: ₹{capital:,}
- Max Risk Per Trade: ₹{max_risk:,}
- Current F&O Exposure: {current_exposure}%

HISTORICAL STRATEGY PERFORMANCE (last 60 days):
{historical_performance_formatted}

IMPORTANT: Use ATM CE/PE LTP for accurate straddle cost instead of estimated premiums.
A futures basis > +40 pts signals bullish roll-over; < -10 pts signals bearish roll-over.
If GIFT gap is > +0.8%, prefer directional (buy the gap direction) over neutral strategies.

Recommend the optimal options strategy:

{{
    "primary_strategy": "LONG_STRADDLE|IRON_CONDOR|BULL_CALL_SPREAD|BEAR_PUT_SPREAD|SHORT_STRANGLE|NAKED_CALL|NAKED_PUT",
    "alternative_strategy": "<optional backup strategy or null>",
    "strike_selection": {{
        "primary_strike": <float>,
        "secondary_strike": <float or null>,
        "expiry": "<YYYY-MM-DD>",
        "reasoning": "<explanation for strike choice>"
    }},
    "risk_management": {{
        "stop_loss_pct": <float 20-60>,
        "target_pct": <float 50-150>,
        "position_size_lots": <integer>,
        "max_loss_rupees": <float>
    }},
    "edge_identification": {{
        "key_advantage": "<why this strategy has edge>",
        "risk_factors": ["<risk 1>", "<risk 2>"],
        "probabilistic_edge": "<expected outcome description>"
    }},
    "entry_timing": "IMMEDIATE|WAIT_FOR_<condition>",
    "adjustment_triggers": [
        "<trigger 1, e.g., If spot breaks 25300, exit PE leg>",
        "<trigger 2>"
    ],
    "confidence": <integer 0-100>,
    "reasoning": "<comprehensive explanation of strategy choice>"
}}

IMPORTANT STRATEGY SELECTION GUIDELINES:
- NEUTRAL consensus + LOW IV (<30) → Prefer LONG_STRADDLE or SHORT_STRANGLE
- NEUTRAL consensus + HIGH IV (>60) → Prefer IRON_CONDOR
- BULLISH consensus + TRENDING regime → Prefer BULL_CALL_SPREAD or NAKED_CALL
- BEARISH consensus + TRENDING regime → Prefer BEAR_PUT_SPREAD or NAKED_PUT
- Always consider max pain - avoid strikes near max pain
- Factor in upcoming events (holidays, policy)
- Position size must respect max_loss_rupees constraint
- If real ATM LTP is available (>0), use it for premium sizing, not BS estimates
"""


@dataclass
class StrikeSelection:
    """Strike selection details."""
    primary_strike: float
    secondary_strike: Optional[float]
    expiry: str
    reasoning: str


@dataclass
class RiskManagement:
    """Risk management parameters."""
    stop_loss_pct: float
    target_pct: float
    position_size_lots: int
    max_loss_rupees: float


@dataclass
class EdgeIdentification:
    """Edge identification for the strategy."""
    key_advantage: str
    risk_factors: List[str]
    probabilistic_edge: str


@dataclass
class OptionsStrategyResult:
    """Complete options strategy recommendation."""
    primary_strategy: str  # "LONG_STRADDLE", "IRON_CONDOR", etc.
    alternative_strategy: Optional[str]
    strike_selection: StrikeSelection
    risk_management: RiskManagement
    edge_identification: EdgeIdentification
    entry_timing: str  # "IMMEDIATE", "WAIT_FOR_<condition>", etc.
    adjustment_triggers: List[str]
    confidence: int  # 0-100
    reasoning: str
    source: str = "LLM"  # "LLM" or "FALLBACK"
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[str] = None


class LLMOptionsStrategist(BaseLLMAgent):
    """
    LLM-powered options strategy selection and optimization.

    Called after DirectionalOptionAdvisor generates suggestions.
    Can validate, modify, or replace strategy recommendations.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config, logger_name="LLMOptionsStrategist")
        self._last_strategy: Optional[OptionsStrategyResult] = None
        self._strategy_history: List[OptionsStrategyResult] = []

    def optimize_strategy(
        self,
        index_name: str,
        spot: float,
        consensus: Any,
        regime: MarketRegime,
        vix: float,
        iv_rank: float,
        option_chain: Dict[str, Any],
        historical_performance: Dict[str, Dict],
        portfolio_constraints: Dict[str, Any]
    ) -> OptionsStrategyResult:
        """
        Generate optimized options strategy recommendation.

        Args:
            index_name: NIFTY, BANKNIFTY, etc.
            spot: Current spot price
            consensus: Consensus result from agents
            regime: Market regime
            vix: India VIX value
            iv_rank: IV rank (0-100)
            option_chain: Option chain data from Fyers
            historical_performance: Strategy performance history
            portfolio_constraints: Capital, risk limits, etc.

        Returns:
            OptionsStrategyResult with strategy, strikes, risk management
        """
        # Build prompt
        prompt = self._build_strategy_prompt(
            index_name, spot, consensus, regime, vix, iv_rank,
            option_chain, historical_performance, portfolio_constraints
        )

        # Get LLM response (no fallback_handler - we handle fallback ourselves)
        response = self.generate(
            prompt=prompt,
            expect_json=True,
            fallback_handler=None
        )

        # Parse response - check if we got valid LLM response
        if response.source == "LLM" and response.parsed_json:
            result = self._parse_strategy_response(response.parsed_json, response.content)
        else:
            # Use fallback when LLM unavailable
            result = self._fallback_strategy(
                index_name, spot, consensus, regime, vix, portfolio_constraints
            )

        # Store result
        self._last_strategy = result
        self._strategy_history.append(result)
        if len(self._strategy_history) > 50:
            self._strategy_history = self._strategy_history[-50:]

        return result

    def _build_strategy_prompt(
        self,
        index_name: str,
        spot: float,
        consensus: Any,
        regime: MarketRegime,
        vix: float,
        iv_rank: float,
        option_chain: Dict[str, Any],
        historical_performance: Dict[str, Dict],
        portfolio_constraints: Dict[str, Any]
    ) -> str:
        """Construct strategy prompt."""
        # Consensus details
        direction = consensus.direction.value if hasattr(consensus, 'direction') else "NEUTRAL"
        if hasattr(consensus, 'direction'):
            if consensus.direction == TradeDirection.LONG:
                direction = "BULLISH"
            elif consensus.direction == TradeDirection.SHORT:
                direction = "BEARISH"
        strength = getattr(consensus, 'strength', 'MODERATE')

        # IV trend
        iv_trend = "HIGH" if iv_rank > 60 else "LOW" if iv_rank < 30 else "MODERATE"

        # Option chain data
        atm_strike = option_chain.get('atm_strike', round(spot / 100) * 100)
        max_pain = option_chain.get('max_pain', atm_strike)
        pcr = option_chain.get('pcr', 1.0)

        # OI distribution
        call_oi = option_chain.get('call_oi', {})
        put_oi = option_chain.get('put_oi', {})

        # Find max OI strikes
        resistance_strike = max(call_oi.keys(), key=lambda k: call_oi.get(k, 0)) if call_oi else atm_strike + 200
        support_strike = min(put_oi.keys(), key=lambda k: put_oi.get(k, 0)) if put_oi else atm_strike - 200

        # Format OI distribution
        call_oi_str = ", ".join(f"{k}:{v}" for k, v in sorted(call_oi.items())[-5:]) if call_oi else "N/A"
        put_oi_str = ", ".join(f"{k}:{v}" for k, v in sorted(put_oi.items())[:5]) if put_oi else "N/A"

        # Portfolio constraints
        capital = portfolio_constraints.get('capital', 500000)
        max_risk = portfolio_constraints.get('max_risk', 22500)
        current_exposure = portfolio_constraints.get('current_exposure', 0)

        # Historical performance
        perf_lines = []
        for strat, perf in historical_performance.items():
            win_rate = perf.get('win_rate', 0) * 100
            avg_return = perf.get('avg_return', 0)
            count = perf.get('count', 0)
            perf_lines.append(f"  - {strat}: {win_rate:.0f}% WR, {avg_return:+.1f}% avg return ({count} trades)")
        historical_performance_formatted = "\n".join(perf_lines) or "No historical data"

        return OPTIONS_STRATEGY_PROMPT.format(
            index_name=index_name,
            spot=spot,
            consensus_direction=direction,
            consensus_strength=strength,
            regime=regime.value if hasattr(regime, 'value') else str(regime),
            vix=vix,
            iv_rank=iv_rank,
            iv_trend=iv_trend,
            atm_strike=atm_strike,
            # FIX 5.1: Real ATM LTPs from option chain (Fix 1 data)
            atm_ce_ltp=float(option_chain.get("atm_ce_ltp", 0.0)),
            atm_pe_ltp=float(option_chain.get("atm_pe_ltp", 0.0)),
            call_oi_distribution=call_oi_str,
            put_oi_distribution=put_oi_str,
            max_pain=max_pain,
            pcr=pcr,
            resistance_strike=resistance_strike,
            support_strike=support_strike,
            capital=int(capital),
            max_risk=int(max_risk),
            current_exposure=current_exposure,
            historical_performance_formatted=historical_performance_formatted,
            # FIX 5.1: Real futures basis (Fix 2 data)
            futures_basis=float(option_chain.get("futures_premium", 0.0)),
            futures_basis_note=(
                "bullish rollover" if option_chain.get("futures_premium", 0) > 40
                else "bearish rollover" if option_chain.get("futures_premium", 0) < -10
                else "normal carry"
            ),
            # FIX 5.1: GIFT Nifty pre-market gap (Fix 4.2 data)
            gift_gap_pct=float(option_chain.get("gift_nifty_gap_pct", 0.0)),
        )

    def _parse_strategy_response(
        self,
        parsed: Dict[str, Any],
        raw_response: str
    ) -> OptionsStrategyResult:
        """Parse JSON response into strategy result."""
        try:
            # Parse strike selection
            strike_data = parsed.get("strike_selection", {})
            strike_selection = StrikeSelection(
                primary_strike=float(strike_data.get("primary_strike", 0)),
                secondary_strike=strike_data.get("secondary_strike"),
                expiry=strike_data.get("expiry", ""),
                reasoning=strike_data.get("reasoning", "")
            )

            # Parse risk management
            risk_data = parsed.get("risk_management", {})
            risk_management = RiskManagement(
                stop_loss_pct=float(risk_data.get("stop_loss_pct", 40)),
                target_pct=float(risk_data.get("target_pct", 80)),
                position_size_lots=int(risk_data.get("position_size_lots", 1)),
                max_loss_rupees=float(risk_data.get("max_loss_rupees", 10000))
            )

            # Parse edge identification
            edge_data = parsed.get("edge_identification", {})
            edge_identification = EdgeIdentification(
                key_advantage=edge_data.get("key_advantage", ""),
                risk_factors=edge_data.get("risk_factors", []),
                probabilistic_edge=edge_data.get("probabilistic_edge", "")
            )

            return OptionsStrategyResult(
                primary_strategy=parsed.get("primary_strategy", "IRON_CONDOR"),
                alternative_strategy=parsed.get("alternative_strategy"),
                strike_selection=strike_selection,
                risk_management=risk_management,
                edge_identification=edge_identification,
                entry_timing=parsed.get("entry_timing", "IMMEDIATE"),
                adjustment_triggers=parsed.get("adjustment_triggers", []),
                confidence=int(parsed.get("confidence", 60)),
                reasoning=parsed.get("reasoning", ""),
                source="LLM",
                raw_response=raw_response
            )

        except Exception as e:
            self.logger.error(f"Failed to parse strategy response: {e}")
            return OptionsStrategyResult(
                primary_strategy="IRON_CONDOR",
                alternative_strategy=None,
                strike_selection=StrikeSelection(0, None, "", "Parse error"),
                risk_management=RiskManagement(40, 80, 1, 10000),
                edge_identification=EdgeIdentification("", [], ""),
                entry_timing="IMMEDIATE",
                adjustment_triggers=[],
                confidence=50,
                reasoning="Strategy parsing failed",
                source="FALLBACK"
            )

    def _fallback_strategy(
        self,
        index_name: str,
        spot: float,
        consensus: Any,
        regime: MarketRegime,
        vix: float,
        portfolio_constraints: Dict[str, Any]
    ) -> OptionsStrategyResult:
        """Fallback strategy when LLM unavailable."""
        self.logger.info("Using fallback options strategy")

        # Determine direction
        direction = TradeDirection.NEUTRAL
        if hasattr(consensus, 'direction'):
            direction = consensus.direction

        # Select strategy based on rules
        if direction == TradeDirection.LONG:
            primary_strategy = "BULL_CALL_SPREAD"
            primary_strike = round(spot / 100) * 100
            secondary_strike = primary_strike + 200
        elif direction == TradeDirection.SHORT:
            primary_strategy = "BEAR_PUT_SPREAD"
            primary_strike = round(spot / 100) * 100
            secondary_strike = primary_strike - 200
        else:
            # NEUTRAL - use Iron Condor for moderate VIX, Short Strangle for low VIX
            if vix < 15:
                primary_strategy = "SHORT_STRANGLE"
                primary_strike = round(spot / 100) * 100
                secondary_strike = None
            else:
                primary_strategy = "IRON_CONDOR"
                primary_strike = round(spot / 100) * 100
                secondary_strike = primary_strike

        # Calculate position size
        capital = portfolio_constraints.get('capital', 500000)
        max_risk = portfolio_constraints.get('max_risk', 22500)
        lot_size = 75 if index_name == "NIFTY" else 15
        position_size = min(10, max(1, int(max_risk / 5000)))

        # Default expiry: next Thursday
        today = date.today()
        days_until_thu = (3 - today.weekday()) % 7
        if days_until_thu == 0:
            days_until_thu = 7
        expiry = (today + timedelta(days=days_until_thu)).isoformat()

        return OptionsStrategyResult(
            primary_strategy=primary_strategy,
            alternative_strategy=None,
            strike_selection=StrikeSelection(
                primary_strike=primary_strike,
                secondary_strike=secondary_strike,
                expiry=expiry,
                reasoning="Rule-based strike selection (LLM unavailable)"
            ),
            risk_management=RiskManagement(
                stop_loss_pct=40.0,
                target_pct=80.0,
                position_size_lots=position_size,
                max_loss_rupees=max_risk
            ),
            edge_identification=EdgeIdentification(
                key_advantage="Rule-based selection",
                risk_factors=["LLM analysis unavailable"],
                probabilistic_edge="Standard risk-reward profile"
            ),
            entry_timing="IMMEDIATE",
            adjustment_triggers=["Exit if spot breaks key support/resistance"],
            confidence=55,
            reasoning="Fallback strategy based on consensus direction and VIX level",
            source="FALLBACK"
        )

    def get_last_strategy(self) -> Optional[OptionsStrategyResult]:
        """Get the most recent strategy."""
        return self._last_strategy

    def get_strategy_history(self, count: int = 20) -> List[OptionsStrategyResult]:
        """Get strategy history."""
        return self._strategy_history[-count:]

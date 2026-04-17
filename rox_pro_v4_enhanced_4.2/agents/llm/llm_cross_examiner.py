"""
LLM Cross-Examiner Patch — Regime-Aware Thresholds
===================================================
DROP-IN REPLACEMENT for agents/llm/llm_cross_examiner.py

Key changes vs original:
1. In BEAR/CORRECTION regimes: allow PROCEED_WITH_CAUTION when trading plan
   is SHORT with >= 50% conviction, even if consensus is NEUTRAL.
2. Fallback examination now respects regime — no longer blindly returns WAIT
   on NO_CONSENSUS in directional regimes.
3. Tracks examination streaks to detect WAIT paralysis (3+ consecutive WAITs).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .base_llm_agent import BaseLLMAgent, LLMConfig, LLMResponse

# Import from parent config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MarketRegime, TradeDirection


# ── FIX-EXAMINE-01: Regime-aware WAIT thresholds ───────────────────────────
# The original cross-examiner applied the same WAIT bar to all regimes.
# In BEAR with confirmed SHORT signals, this caused total trade suppression.
# These thresholds allow SHORT trades in bearish regimes at lower consensus
# strength while maintaining strict bars for LONG trades in the same regime.
REGIME_EXAMINE_THRESHOLDS = {
    "BEAR": {
        "min_short_conviction_to_proceed": 50,   # SHORT allowed at 50%+ 
        "min_long_conviction_to_proceed":  75,   # LONG needs very high conviction
        "auto_wait_on_neutral":            False, # Don't auto-WAIT on NEUTRAL consensus
        "allow_proceed_with_caution":      True,  # Enable PROCEED_WITH_CAUTION mode
    },
    "MILD_BEAR": {
        "min_short_conviction_to_proceed": 55,
        "min_long_conviction_to_proceed":  70,
        "auto_wait_on_neutral":            False,
        "allow_proceed_with_caution":      True,
    },
    "CORRECTION": {
        "min_short_conviction_to_proceed": 45,
        "min_long_conviction_to_proceed":  80,
        "auto_wait_on_neutral":            False,
        "allow_proceed_with_caution":      True,
    },
    "CONSOLIDATION": {
        "min_short_conviction_to_proceed": 65,
        "min_long_conviction_to_proceed":  65,
        "auto_wait_on_neutral":            True,
        "allow_proceed_with_caution":      False,
    },
    "MILD_BULL": {
        "min_short_conviction_to_proceed": 70,
        "min_long_conviction_to_proceed":  55,
        "auto_wait_on_neutral":            True,
        "allow_proceed_with_caution":      False,
    },
    "BULL": {
        "min_short_conviction_to_proceed": 75,
        "min_long_conviction_to_proceed":  50,
        "auto_wait_on_neutral":            False,
        "allow_proceed_with_caution":      True,
    },
}

# Prompt Template (unchanged from original)
CROSS_EXAMINATION_PROMPT = """You are a senior trading desk manager reviewing your team's analysis.

TEAM CONSENSUS: {consensus_direction} (strength: {consensus_strength}, net_score: {net_score:.3f})

AGENT REPORTS:
{agent_reports_formatted}

HISTORICAL ACCURACY (last 30 days):
{historical_accuracy_formatted}

THIS ENGINE'S TRADE HISTORY:
{trade_history}

RECENT SYSTEM FIXES (applied this week — factor into your assessment):
{system_changes_context}

MARKET CONTEXT:
- Current Regime: {regime}
- India VIX: {vix}
- Recent Consensus History: {recent_consensus_history}
- Brent Crude: ${crude_usd:.1f}/bbl
- USD/INR: ₹{usd_inr:.2f}
- Gift Nifty Gap (overnight): {gift_nifty_gap_pct:+.2f}%
- External Catalyst: {external_catalyst}
- Total Resolved Trades in History: {resolved_trade_count}  ← use for statistical validity guard above

ACTIVE NEWS RESTRICTIONS (enforce these — they override consensus):
{news_restrictions}
If restrictions include HALT_NEW_LONGS_IN_FINANCIALS: any LONG consensus on banking stocks
must be flagged as a restriction violation in risk_flags — do NOT recommend PROCEED for
those stocks. Convert to WATCH_ONLY instead.

CROSS-EXAMINATION TASKS:
1. Is the consensus direction correct given the contradictions?
2. Are any agents over/under-weighted based on recent performance?
3. What is the contrarian case (why might this be wrong)?
4. Is this a high-conviction or low-conviction situation?
5. Are there any hidden risks not captured in agent analysis?
6. Does the engine's own trade history support or contradict this signal?

IMPORTANT NOTES ON PERFORMANCE HISTORY:
- Historical win rate reflects trades made under the OLD engine with structural bugs now fixed.
- PRUDENCE previously hardcoded direction=LONG regardless of market; this is now corrected.
- CATALYST previously output conviction=80 with NEUTRAL direction; now scaled to 60-70.
- Do NOT permanently AVOID solely due to pre-fix historical win rate — factor in the fixes above.
- If you recommend AVOID, state which specific CURRENT signals (not historical stats alone) justify it.
- SENTINEL uses contrarian PCR logic: high PCR (>1.1) = bearish sentiment = contrarian LONG signal. This is correct options market interpretation, not a reasoning error.

STATISTICAL VALIDITY GUARD (FIX-AVOID-02):
- If total resolved trades < 30, the historical win rate is STATISTICALLY INSUFFICIENT to justify
  AVOID on its own. With < 30 trades, a 0% win rate could be bad luck on 5 trades — not a
  signal. Use win rate only as a TIEBREAKER, not a primary justification for AVOID.
- When the engine has < 30 resolved trades, prefer WAIT or REDUCE_SIZE over AVOID unless
  CURRENT market signals (not past stats) independently justify AVOID.
- A self-reinforcing AVOID loop — where AVOID prevents new trades, which keeps win rate low,
  which triggers more AVOID — is a known failure mode. Break it with REDUCE_SIZE, not AVOID.

Respond ONLY with valid JSON:
{{
    "examined_direction": "LONG|SHORT|NEUTRAL",
    "confidence_adjustment": <integer -20 to +20>,
    "agents_to_upweight": ["AGENT_NAME1", "AGENT_NAME2"],
    "agents_to_downweight": ["AGENT_NAME3"],
    "poor_quality_agents": ["AGENT_NAME4"],
    "contrarian_case": "<description of what could go wrong>",
    "final_recommendation": "PROCEED|WAIT|REDUCE_SIZE|AVOID",
    "reasoning": "<2-3 sentences explaining the decision>",
    "risk_flags": [
        "<risk 1>",
        "<risk 2>"
    ]
}}

IMPORTANT:
- If consensus is NEUTRAL with strong LONG votes, consider suggesting LONG
- If agents strongly contradict each other, recommend WAIT
- Factor in historical accuracy when deciding whose view to trust
- Be conservative: recommend WAIT when uncertain
- "poor_quality_agents": list agents whose REASONING QUALITY is defective —
  e.g. circular logic, data that directly contradicts their own signal,
  or conclusions that don't follow from their stated evidence.
  These agents must also appear in "agents_to_downweight".
  Leave empty [] if no agents have reasoning quality issues.
"""


@dataclass
class CrossExaminationResult:
    """Result of LLM consensus cross-examination."""
    examined_direction: TradeDirection
    confidence_adjustment: int  # -20 to +20
    agents_to_upweight: List[str]
    agents_to_downweight: List[str]
    contrarian_case: str
    final_recommendation: str  # "PROCEED", "WAIT", "REDUCE_SIZE", "AVOID"
    reasoning: str
    risk_flags: List[str]
    poor_quality_agents: List[str] = field(default_factory=list)
    source: str = "LLM"  # "LLM" or "FALLBACK"
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[str] = None
    # NEW: regime override flag
    regime_override: bool = False  # True if recommendation was adjusted by regime logic


class LLMCrossExaminer(BaseLLMAgent):
    """
    LLM-powered consensus cross-examination.

    Called after consensus is calculated in LeadCoordinator.
    Can override consensus or adjust agent weights.

    FIX-EXAMINE-01: Regime-aware thresholds prevent WAIT paralysis
    in directional regimes (BEAR, BULL) where consensus is naturally
    fragmented but market direction is clear.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config, logger_name="LLMCrossExaminer")
        self._last_examination: Optional[CrossExaminationResult] = None
        self._examination_history: List[CrossExaminationResult] = []
        # FIX-EXAMINE-02: WAIT paralysis tracker
        self._consecutive_waits: int = 0
        self._WAIT_PARALYSIS_LIMIT: int = 3

    def examine_consensus(
        self,
        agent_reports: Dict[str, Any],
        consensus: Any,
        regime: MarketRegime,
        historical_accuracy: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> CrossExaminationResult:
        """
        Cross-examine the agent consensus with regime-aware thresholds.
        """
        # Build prompt
        prompt = self._build_examination_prompt(
            agent_reports, consensus, regime, historical_accuracy, market_context
        )

        # Get LLM response
        response = self.generate(
            prompt=prompt,
            expect_json=True,
            fallback_handler=None
        )

        # Parse response
        if response.source == "LLM" and response.parsed_json:
            result = self._parse_examination_response(response.parsed_json, consensus, response.content)
        else:
            result = self._fallback_examination(consensus, regime)

        # ── FIX-EXAMINE-01: Apply regime-aware post-processing ─────────
        result = self._apply_regime_override(result, consensus, regime, market_context)

        # ── FIX-EXAMINE-02: WAIT paralysis detection ──────────────────
        result = self._check_wait_paralysis(result)

        # Store result
        self._last_examination = result
        self._examination_history.append(result)
        if len(self._examination_history) > 100:
            self._examination_history = self._examination_history[-100:]

        return result

    def _apply_regime_override(
        self,
        result: CrossExaminationResult,
        consensus: Any,
        regime: MarketRegime,
        market_context: Dict[str, Any]
    ) -> CrossExaminationResult:
        """
        FIX-EXAMINE-01: Override LLM recommendation when regime logic
        clearly supports a different decision.

        In BEAR with SHORT trading plan at 50%+ conviction:
          WAIT → PROCEED_WITH_CAUTION (mapped to REDUCE_SIZE)
          AVOID → REDUCE_SIZE

        This breaks the paralysis loop where the LLM says WAIT because
        "team is fragmented" even though the regime is clearly directional.
        """
        regime_name = regime.value if hasattr(regime, 'value') else str(regime)
        threshold = REGIME_EXAMINE_THRESHOLDS.get(regime_name, REGIME_EXAMINE_THRESHOLDS["CONSOLIDATION"])

        # Only override when regime allows it
        if not threshold.get("allow_proceed_with_caution", False):
            return result

        # Extract trading plan stance from market context
        trading_plan_stance = market_context.get("trading_plan_stance", "")
        net_score = market_context.get("consensus_net_score", 0.0)
        plan_short = trading_plan_stance in ("MODERATE_SHORT", "AGGRESSIVE_SHORT")
        plan_long = trading_plan_stance in ("MODERATE_LONG", "AGGRESSIVE_LONG")

        # Get max conviction from SHORT or LONG agents
        max_short_conviction = 0
        max_long_conviction = 0
        for name, report in (market_context.get("agent_reports_for_override") or {}).items():
            if hasattr(report, 'verdict'):
                v = report.verdict
                if v.direction == TradeDirection.SHORT:
                    max_short_conviction = max(max_short_conviction, v.conviction)
                elif v.direction == TradeDirection.LONG:
                    max_long_conviction = max(max_long_conviction, v.conviction)

        # BEAR/CORRECTION regime: SHORT plan with decent conviction → override WAIT
        if regime_name in ("BEAR", "MILD_BEAR", "CORRECTION"):
            # Check both plan stance AND net_score direction
            _short_signal = plan_short or (net_score < -0.02 and max_short_conviction >= 40)
            
            if (_short_signal 
                and max_short_conviction >= threshold["min_short_conviction_to_proceed"]
                and result.final_recommendation in ("WAIT", "AVOID")):
                self.logger.info(
                    f"[FIX-EXAMINE-01] Regime override: {result.final_recommendation} → "
                    f"REDUCE_SIZE (BEAR regime + SHORT plan at {max_short_conviction}% conviction, "
                    f"threshold={threshold['min_short_conviction_to_proceed']}%)"
                )
                result.final_recommendation = "REDUCE_SIZE"
                result.regime_override = True
                if not result.reasoning:
                    result.reasoning = ""
                result.reasoning += (
                    f" [REGIME OVERRIDE: {regime_name} regime with SHORT plan "
                    f"conviction {max_short_conviction}% >= {threshold['min_short_conviction_to_proceed']}% "
                    f"threshold — relaxing from {result.final_recommendation} to REDUCE_SIZE]"
                )

        # BULL regime: LONG plan with decent conviction → override WAIT  
        elif regime_name in ("BULL", "MILD_BULL"):
            if (plan_long
                and max_long_conviction >= threshold["min_long_conviction_to_proceed"]
                and result.final_recommendation in ("WAIT", "AVOID")):
                self.logger.info(
                    f"[FIX-EXAMINE-01] Regime override: {result.final_recommendation} → "
                    f"REDUCE_SIZE (BULL regime + LONG plan at {max_long_conviction}% conviction)"
                )
                result.final_recommendation = "REDUCE_SIZE"
                result.regime_override = True

        return result

    def _check_wait_paralysis(self, result: CrossExaminationResult) -> CrossExaminationResult:
        """
        FIX-EXAMINE-02: If the cross-examiner fires WAIT for 3+ consecutive
        cycles, escalate to REDUCE_SIZE for one cycle to break the loop.
        This is complementary to the AVOID deadlock circuit breaker in coordinator.py.
        """
        if result.final_recommendation == "WAIT":
            self._consecutive_waits += 1
            if self._consecutive_waits >= self._WAIT_PARALYSIS_LIMIT:
                self.logger.warning(
                    f"[FIX-EXAMINE-02] WAIT paralysis detected: {self._consecutive_waits} "
                    f"consecutive WAITs — escalating to REDUCE_SIZE for one cycle"
                )
                result.final_recommendation = "REDUCE_SIZE"
                result.regime_override = True
                self._consecutive_waits = 0
        else:
            self._consecutive_waits = 0

        return result

    def _build_examination_prompt(
        self,
        agent_reports: Dict[str, Any],
        consensus: Any,
        regime: MarketRegime,
        historical_accuracy: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> str:
        """Construct cross-examination prompt."""
        direction = consensus.direction.value if hasattr(consensus.direction, 'value') else str(consensus.direction)
        strength = getattr(consensus, 'strength', 'MODERATE')
        net_score = getattr(consensus, 'net_score', 0.0)

        # Format agent reports
        agent_lines = []
        for name, report in agent_reports.items():
            if hasattr(report, 'verdict'):
                v = report.verdict
                agent_lines.append(
                    f"  - {name}: {v.direction.value} (conviction: {v.conviction}, weight: {v.weight:.2f})"
                )
                if hasattr(v, 'reason') and v.reason:
                    agent_lines.append(f"    Reason: {v.reason[:80]}...")
                if hasattr(v, 'risks') and v.risks:
                    agent_lines.append(f"    Risks: {', '.join(v.risks[:2])}")
        agent_reports_formatted = "\n".join(agent_lines)

        # Format historical accuracy
        accuracy_lines = []
        for agent, wr in historical_accuracy.items():
            emoji = "✓" if wr >= 0.55 else "✗" if wr < 0.45 else "○"
            accuracy_lines.append(f"  - {agent}: {wr*100:.1f}% {emoji}")
        historical_accuracy_formatted = "\n".join(accuracy_lines) or "No historical data"

        # System changes context
        system_changes_context = market_context.get(
            "system_changes_context",
            "- PRUDENCE direction bias fixed (was hardcoding LONG regardless of market)\n"
            "- CATALYST conviction de-anchored (was hardcoded 80, now scales with event horizon)\n"
            "- SENTINEL PCR reasoning clarified (contrarian interpretation is intentional)\n"
            "- AVOID deadlock circuit breaker active (streak counter prevents infinite suppression)\n"
            "- Historical losses predate these fixes — treat win rate as partially stale."
        )

        # Market context
        vix = market_context.get("india_vix", 15.0)
        recent_history = market_context.get("recent_consensus_history", "No recent history")
        trade_history  = market_context.get("trade_history", "No resolved trade history yet.")

        market_data_raw = market_context.get("market_data", {})
        crude_usd = (
            market_context.get("crude_brent_usd")
            or market_data_raw.get("crude_brent_usd", 0.0)
        )
        usd_inr = (
            market_context.get("usd_inr")
            or market_data_raw.get("usd_inr", 84.0)
        )
        gift_nifty_gap_pct = (
            market_context.get("gift_nifty_gap_pct")
            or market_data_raw.get("gift_nifty_gap_pct", 0.0)
        )
        external_catalyst = (
            market_context.get("geopolitical_summary")
            or market_data_raw.get("geopolitical_summary", "None identified")
        )
        resolved_trade_count = market_context.get("resolved_trade_count", "unknown")
        if resolved_trade_count == "unknown" and trade_history:
            import re as _re
            m = _re.search(r"(\d+)\s+resolved", trade_history)
            resolved_trade_count = int(m.group(1)) if m else "unknown"

        _news_restrictions = market_context.get("news_restrictions", [])
        if not _news_restrictions:
            _news_impact = market_context.get("news_impact")
            if _news_impact and hasattr(_news_impact, "trade_restrictions"):
                _news_restrictions = list(_news_impact.trade_restrictions)
        if _news_restrictions:
            news_restrictions_str = "\n".join(f"  - {r}" for r in _news_restrictions)
        else:
            news_restrictions_str = "  (No active restrictions)"

        return CROSS_EXAMINATION_PROMPT.format(
            consensus_direction=direction,
            consensus_strength=strength,
            net_score=net_score,
            agent_reports_formatted=agent_reports_formatted,
            historical_accuracy_formatted=historical_accuracy_formatted,
            system_changes_context=system_changes_context,
            regime=regime.value if hasattr(regime, 'value') else str(regime),
            vix=vix,
            recent_consensus_history=recent_history,
            trade_history=trade_history,
            crude_usd=crude_usd if crude_usd else 0.0,
            usd_inr=usd_inr if usd_inr else 84.0,
            gift_nifty_gap_pct=gift_nifty_gap_pct if gift_nifty_gap_pct else 0.0,
            external_catalyst=external_catalyst,
            resolved_trade_count=resolved_trade_count,
            news_restrictions=news_restrictions_str,
        )

    def _parse_examination_response(
        self,
        parsed: Dict[str, Any],
        consensus: Any,
        raw_response: str
    ) -> CrossExaminationResult:
        """Parse JSON response."""
        try:
            dir_str = parsed.get("examined_direction", "NEUTRAL").upper()
            direction_map = {
                "LONG": TradeDirection.LONG,
                "SHORT": TradeDirection.SHORT,
                "NEUTRAL": TradeDirection.NEUTRAL,
            }
            examined_direction = direction_map.get(dir_str, TradeDirection.NEUTRAL)

            conf_adj = int(parsed.get("confidence_adjustment", 0))
            conf_adj = max(-20, min(20, conf_adj))

            recommendation = parsed.get("final_recommendation", "PROCEED").upper()
            if recommendation not in ["PROCEED", "WAIT", "REDUCE_SIZE", "AVOID"]:
                recommendation = "PROCEED"

            downweight_set = set(parsed.get("agents_to_downweight", []))
            raw_pqa = parsed.get("poor_quality_agents", [])
            poor_quality_agents = [a for a in raw_pqa if a in downweight_set]

            return CrossExaminationResult(
                examined_direction=examined_direction,
                confidence_adjustment=conf_adj,
                agents_to_upweight=parsed.get("agents_to_upweight", []),
                agents_to_downweight=parsed.get("agents_to_downweight", []),
                poor_quality_agents=poor_quality_agents,
                contrarian_case=parsed.get("contrarian_case", ""),
                final_recommendation=recommendation,
                reasoning=parsed.get("reasoning", "LLM cross-examination completed"),
                risk_flags=parsed.get("risk_flags", []),
                source="LLM",
                raw_response=raw_response
            )

        except Exception as e:
            self.logger.error(f"Failed to parse cross-examination response: {e}")
            return self._fallback_examination(consensus, None)

    def _fallback_examination(self, consensus: Any, regime: Optional[MarketRegime] = None) -> CrossExaminationResult:
        """
        FIX-EXAMINE-01: Regime-aware fallback when LLM unavailable.
        
        Original fallback blindly returned WAIT on NO_CONSENSUS.
        Now it checks regime: in BEAR, NO_CONSENSUS with SHORT-leaning
        net_score returns REDUCE_SIZE instead of WAIT.
        """
        self.logger.info("Using fallback cross-examination (regime-aware)")

        direction = consensus.direction if hasattr(consensus, 'direction') else TradeDirection.NEUTRAL
        strength = getattr(consensus, 'strength', 'MODERATE')
        contradictions = getattr(consensus, 'contradictions', [])
        net_score = getattr(consensus, 'net_score', 0.0)

        recommendation = "PROCEED"
        risk_flags = []
        conf_adj = 0

        # Check for contradictions
        if contradictions:
            risk_flags.append("Agent contradictions detected")
            if len(contradictions) > 2:
                recommendation = "WAIT"
                conf_adj = -10

        # Check strength
        if strength == "NO_CONSENSUS":
            regime_name = regime.value if regime else "CONSOLIDATION"
            
            # FIX: In BEAR/CORRECTION, a negative net_score means the team
            # leans SHORT even without formal consensus. Don't auto-WAIT.
            if regime_name in ("BEAR", "MILD_BEAR", "CORRECTION") and net_score < -0.02:
                recommendation = "REDUCE_SIZE"
                conf_adj = -5
                risk_flags.append(f"No formal consensus but SHORT-leaning in {regime_name} regime")
                self.logger.info(
                    f"[FIX-EXAMINE-01] Fallback: NO_CONSENSUS in {regime_name} with "
                    f"net_score={net_score:.3f} → REDUCE_SIZE (not WAIT)"
                )
            elif regime_name in ("BULL", "MILD_BULL") and net_score > 0.05:
                recommendation = "REDUCE_SIZE"
                conf_adj = -5
                risk_flags.append(f"No formal consensus but LONG-leaning in {regime_name} regime")
            else:
                recommendation = "WAIT"
                conf_adj = -15
                risk_flags.append("No clear consensus among agents")
        elif strength == "WEAK":
            recommendation = "REDUCE_SIZE"
            conf_adj = -5
            risk_flags.append("Weak consensus strength")

        return CrossExaminationResult(
            examined_direction=direction,
            confidence_adjustment=conf_adj,
            agents_to_upweight=[],
            agents_to_downweight=[],
            contrarian_case="LLM cross-examination unavailable - basic validation applied",
            final_recommendation=recommendation,
            reasoning="Fallback validation based on consensus structure and regime",
            risk_flags=risk_flags,
            source="FALLBACK"
        )

    def get_last_examination(self) -> Optional[CrossExaminationResult]:
        """Get the most recent examination result."""
        return self._last_examination

    def get_examination_history(self, count: int = 20) -> List[CrossExaminationResult]:
        """Get examination history."""
        return self._examination_history[-count:]

"""
LLM Meta-Learner - Weekly performance analysis (Enhancement P3.2)
===================================================================

Performs weekly analysis of prediction performance to generate
improvement recommendations:
- Agent weight adjustments
- Regime-specific rules
- Pattern adjustments
- Sector insights
- Systemic improvements
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any

from .base_llm_agent import LLMConfig, LLMResponse

# Import from parent config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MarketRegime


# Prompt Template
META_LEARNING_PROMPT = """You are a trading system performance analyst reviewing weekly results.

WEEKLY PERFORMANCE SUMMARY:
- Period: {week_start} to {week_end}
- Total Predictions: {total_predictions}
- Win Rate: {win_rate}%
- Average Win: +{avg_win_pct}%
- Average Loss: -{avg_loss_pct}%
- Profit Factor: {profit_factor}

PREDICTION BREAKDOWN BY AGENT:
{agent_performance_formatted}

FAILURE ANALYSIS (losing trades):
{failure_analysis_formatted}

SUCCESS ANALYSIS (winning trades):
{success_analysis_formatted}

REGIME-SPECIFIC PERFORMANCE:
{regime_performance_formatted}

PATTERN PERFORMANCE:
{pattern_performance_formatted}

SECTOR PERFORMANCE:
{sector_performance_formatted}

Generate improvement recommendations:

{{
    "agent_weight_adjustments": [
        {{
            "agent_name": "<AGENT_NAME>",
            "action": "INCREASE|DECREASE",
            "amount": <float 0.01-0.05>,
            "reason": "<explanation>"
        }}
    ],
    "regime_specific_rules": [
        {{
            "regime": "BULL|BEAR|CONSOLIDATION|MILD_BULL|MILD_BEAR",
            "rule": "<specific rule to apply>",
            "reason": "<explanation>"
        }}
    ],
    "pattern_adjustments": [
        "<adjustment 1, e.g., Flag patterns in CONSOLIDATION: reduce confidence by 10>"
    ],
    "sector_insights": [
        "<insight 1, e.g., Banking sector showing 68% win rate - consider overweighting>"
    ],
    "systemic_improvements": [
        "<improvement 1, e.g., Add volume confirmation check before taking CONSOLIDATION setups>"
    ],
    "next_week_focus": "<specific focus area for next week>",
    "confidence_in_recommendations": <integer 0-100>
}}

IMPORTANT ANALYSIS PRINCIPLES:
1. Only recommend adjustments with clear statistical basis
2. Consider sample size (min 15 occurrences for confidence)
3. Balance between adaptation and overfitting
4. Focus on systemic issues, not individual trade outcomes
5. Preserve conservative bias - only increase risk when evidence is strong

DATA SUFFICIENCY CHECK (enforce these rules strictly):
- If total_predictions < 5: cap confidence_in_recommendations at 15. Add note: "Insufficient data for reliable analysis — minimum 5 trades needed."
- If win_rate < 20% AND total_predictions >= 10: this is a CRITICAL FAILURE STATE.
  You MUST output at least 3 concrete, named improvements in systemic_improvements.
  Each improvement MUST name a specific agent and the specific threshold/behavior to change
  (e.g., "ORION: increase EMA crossover confirmation period from 5 to 10 bars in CONSOLIDATION").
  Generic text like "improve accuracy" is NOT acceptable.
- If win_rate is 20–40%: this is a WARNING state.
  You MUST output at least 2 concrete, named improvements in systemic_improvements.
- HARD RULE: NEVER output an empty systemic_improvements list if win_rate < 40%.
  If the data is poor, the system needs MORE guidance, not silence.
"""


@dataclass
class AgentWeightAdjustment:
    """Recommendation for agent weight adjustment."""
    agent_name: str
    action: str  # "INCREASE" or "DECREASE"
    amount: float  # 0.01 to 0.05
    reason: str


@dataclass
class RegimeSpecificRule:
    """Rule specific to a market regime."""
    regime: str
    rule: str
    reason: str


@dataclass
class MetaLearningResult:
    """Complete meta-learning analysis result."""
    agent_weight_adjustments: List[AgentWeightAdjustment]
    regime_specific_rules: List[RegimeSpecificRule]
    pattern_adjustments: List[str]
    sector_insights: List[str]
    systemic_improvements: List[str]
    next_week_focus: str
    confidence_in_recommendations: int  # 0-100
    analysis_timestamp: datetime
    source: str = "LLM"  # "LLM" or "FALLBACK"
    week_start: Optional[str] = None
    week_end: Optional[str] = None
    raw_response: Optional[str] = None


class LLMMetaLearner:
    """
    LLM-powered meta-learning for system improvement.

    Runs as weekly batch job.
    Analyzes performance patterns and generates recommendations.
    """

    def __init__(self, config: LLMConfig, data_manager: Any = None):
        self.config = config
        self.data_manager = data_manager
        self.logger = logging.getLogger("LLMMetaLearner")
        self._last_analysis: Optional[MetaLearningResult] = None
        self._analysis_history: List[MetaLearningResult] = []

        # Import BaseLLMAgent here to create an instance
        from .base_llm_agent import BaseLLMAgent
        self._llm_agent = BaseLLMAgent(config, logger_name="LLMMetaLearner")

    def analyze_weekly_performance(
        self,
        week_start: date,
        week_end: date,
        performance_data: Optional[Dict[str, Any]] = None
    ) -> MetaLearningResult:
        """
        Analyze weekly performance and generate recommendations.

        Args:
            week_start: Start date of analysis period
            week_end: End date of analysis period
            performance_data: Pre-collected performance data (optional)

        Returns:
            MetaLearningResult with improvement recommendations
        """
        # Collect performance data if not provided
        if performance_data is None:
            performance_data = self._collect_performance_data(week_start, week_end)

        # Build prompt
        prompt = self._build_meta_prompt(performance_data, week_start, week_end)

        # Get LLM response (we handle fallback ourselves after)
        response = self._llm_agent.generate(
            prompt=prompt,
            expect_json=True,
            fallback_handler=None  # We handle fallback ourselves
        )

        # Parse response - check if we got valid LLM response
        if response.source == "LLM" and response.parsed_json:
            result = self._parse_meta_response(
                response.parsed_json, week_start, week_end, response.content
            )
        else:
            # Use fallback analysis
            result = self._fallback_analysis(performance_data, week_start, week_end)

        # Store result
        self._last_analysis = result
        self._analysis_history.append(result)
        if len(self._analysis_history) > 52:  # Keep 1 year
            self._analysis_history = self._analysis_history[-52:]

        return result

    def _collect_performance_data(
        self,
        week_start: date,
        week_end: date
    ) -> Dict[str, Any]:
        """Collect all performance data for the week."""
        data = {
            "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(),
            "total_predictions": 0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "agent_performance": {},
            "failure_analysis": [],
            "success_analysis": [],
            "regime_performance": {},
            "pattern_performance": {},
            "sector_performance": {},
        }

        # Try to get data from data_manager if available
        if self.data_manager:
            try:
                # Get trades for the period
                trades = self.data_manager.get_trades_for_period(week_start, week_end)
                if trades:
                    wins = [t for t in trades if t.get('outcome') == 'WIN']
                    losses = [t for t in trades if t.get('outcome') == 'LOSS']

                    data["total_predictions"] = len(trades)
                    data["win_rate"] = len(wins) / len(trades) * 100 if trades else 0

                    if wins:
                        data["avg_win_pct"] = sum(t.get('return_pct', 0) for t in wins) / len(wins)
                    if losses:
                        data["avg_loss_pct"] = sum(abs(t.get('return_pct', 0)) for t in losses) / len(losses)

                    total_wins = sum(t.get('return_pct', 0) for t in wins)
                    total_losses = sum(abs(t.get('return_pct', 0)) for t in losses)
                    data["profit_factor"] = total_wins / total_losses if total_losses > 0 else 0

            except Exception as e:
                self.logger.warning(f"Could not collect performance data: {e}")

        return data

    def _build_meta_prompt(
        self,
        performance_data: Dict[str, Any],
        week_start: date,
        week_end: date
    ) -> str:
        """Construct meta-learning prompt."""
        # Format agent performance
        agent_perf_lines = []
        for agent, stats in performance_data.get("agent_performance", {}).items():
            wr = stats.get('win_rate', 0) * 100
            count = stats.get('count', 0)
            agent_perf_lines.append(f"  - {agent}: {wr:.1f}% WR ({count} trades)")
        agent_perf_str = "\n".join(agent_perf_lines) or "No agent data"

        # Format failure analysis
        failure_lines = []
        for fail in performance_data.get("failure_analysis", [])[:5]:
            stock = fail.get('stock', 'N/A')
            direction = fail.get('direction', 'N/A')
            reason = fail.get('reason', 'Unknown')
            failure_lines.append(f"  - {stock} ({direction}): {reason}")
        failure_str = "\n".join(failure_lines) or "No failure data"

        # Format success analysis
        success_lines = []
        for succ in performance_data.get("success_analysis", [])[:5]:
            stock = succ.get('stock', 'N/A')
            direction = succ.get('direction', 'N/A')
            reason = succ.get('reason', 'Unknown')
            success_lines.append(f"  - {stock} ({direction}): {reason}")
        success_str = "\n".join(success_lines) or "No success data"

        # Format regime performance
        regime_lines = []
        for regime, stats in performance_data.get("regime_performance", {}).items():
            wr = stats.get('win_rate', 0) * 100
            count = stats.get('count', 0)
            regime_lines.append(f"  - {regime}: {wr:.1f}% WR ({count} trades)")
        regime_str = "\n".join(regime_lines) or "No regime data"

        # Format pattern performance
        pattern_lines = []
        for pattern, stats in performance_data.get("pattern_performance", {}).items():
            wr = stats.get('win_rate', 0) * 100
            count = stats.get('count', 0)
            pattern_lines.append(f"  - {pattern}: {wr:.1f}% WR ({count} trades)")
        pattern_str = "\n".join(pattern_lines) or "No pattern data"

        # Format sector performance
        sector_lines = []
        for sector, stats in performance_data.get("sector_performance", {}).items():
            wr = stats.get('win_rate', 0) * 100
            count = stats.get('count', 0)
            sector_lines.append(f"  - {sector}: {wr:.1f}% WR ({count} trades)")
        sector_str = "\n".join(sector_lines) or "No sector data"

        return META_LEARNING_PROMPT.format(
            week_start=week_start.isoformat(),
            week_end=week_end.isoformat(),
            total_predictions=performance_data.get("total_predictions", 0),
            win_rate=performance_data.get("win_rate", 0),
            avg_win_pct=performance_data.get("avg_win_pct", 0),
            avg_loss_pct=performance_data.get("avg_loss_pct", 0),
            profit_factor=performance_data.get("profit_factor", 0),
            agent_performance_formatted=agent_perf_str,
            failure_analysis_formatted=failure_str,
            success_analysis_formatted=success_str,
            regime_performance_formatted=regime_str,
            pattern_performance_formatted=pattern_str,
            sector_performance_formatted=sector_str
        )

    def _parse_meta_response(
        self,
        parsed: Dict[str, Any],
        week_start: date,
        week_end: date,
        raw_response: str
    ) -> MetaLearningResult:
        """Parse JSON response into meta-learning result."""
        try:
            # Parse agent weight adjustments
            agent_adjustments = []
            for adj in parsed.get("agent_weight_adjustments", []):
                agent_adjustments.append(AgentWeightAdjustment(
                    agent_name=adj.get("agent_name", ""),
                    action=adj.get("action", "INCREASE"),
                    amount=float(adj.get("amount", 0.02)),
                    reason=adj.get("reason", "")
                ))

            # Parse regime rules
            regime_rules = []
            for rule in parsed.get("regime_specific_rules", []):
                regime_rules.append(RegimeSpecificRule(
                    regime=rule.get("regime", ""),
                    rule=rule.get("rule", ""),
                    reason=rule.get("reason", "")
                ))

            return MetaLearningResult(
                agent_weight_adjustments=agent_adjustments,
                regime_specific_rules=regime_rules,
                pattern_adjustments=parsed.get("pattern_adjustments", []),
                sector_insights=parsed.get("sector_insights", []),
                systemic_improvements=parsed.get("systemic_improvements", []),
                next_week_focus=parsed.get("next_week_focus", ""),
                confidence_in_recommendations=max(0, min(100, int(parsed.get("confidence_in_recommendations", 50)))),
                analysis_timestamp=datetime.now(),
                source="LLM",
                week_start=week_start.isoformat(),
                week_end=week_end.isoformat(),
                raw_response=raw_response
            )

        except Exception as e:
            self.logger.error(f"Failed to parse meta-learning response: {e}")
            return self._fallback_analysis({}, week_start, week_end)

    def _fallback_analysis(
        self,
        performance_data: Dict[str, Any],
        week_start: date,
        week_end: date
    ) -> MetaLearningResult:
        """Fallback analysis when LLM unavailable."""
        self.logger.info("Using fallback meta-learning analysis")

        # Simple rule-based recommendations
        win_rate = performance_data.get("win_rate", 50)

        agent_adjustments = []
        if win_rate < 45:
            agent_adjustments.append(AgentWeightAdjustment(
                agent_name="PRUDENCE",
                action="INCREASE",
                amount=0.03,
                reason="Increase risk oversight after poor performance"
            ))
        elif win_rate > 60:
            agent_adjustments.append(AgentWeightAdjustment(
                agent_name="ORION",
                action="INCREASE",
                amount=0.02,
                reason="Strong performance - increase technical weight"
            ))

        return MetaLearningResult(
            agent_weight_adjustments=agent_adjustments,
            regime_specific_rules=[],
            pattern_adjustments=[],
            sector_insights=[],
            systemic_improvements=["Consider manual review of LLM performance analysis"],
            next_week_focus="Monitor trade execution quality",
            confidence_in_recommendations=40,
            analysis_timestamp=datetime.now(),
            source="FALLBACK",
            week_start=week_start.isoformat(),
            week_end=week_end.isoformat()
        )

    def apply_recommendations(
        self,
        recommendations: MetaLearningResult,
        config: Any
    ) -> bool:
        """
        Apply approved recommendations to system.

        Args:
            recommendations: The recommendations to apply
            config: SystemConfig to modify

        Returns:
            True if successfully applied
        """
        try:
            applied_count = 0

            # Apply agent weight adjustments
            for adj in recommendations.agent_weight_adjustments:
                if adj.agent_name in config.agents:
                    agent_cfg = config.agents[adj.agent_name]
                    old_weight = agent_cfg.current_weight
                    if adj.action == "INCREASE":
                        agent_cfg.current_weight = min(
                            0.30,
                            agent_cfg.current_weight + adj.amount
                        )
                    else:
                        agent_cfg.current_weight = max(
                            0.05,
                            agent_cfg.current_weight - adj.amount
                        )
                    self.logger.info(
                        f"Adjusted {adj.agent_name} weight: {adj.action} by {adj.amount:.2f} "
                        f"({old_weight:.3f} → {agent_cfg.current_weight:.3f}) — {adj.reason}"
                    )
                    applied_count += 1

            # Normalize weights
            config.normalize_weights()

            # FIX-META-APPLY: Log regime rules and systemic improvements.
            # These cannot be auto-applied (they require code/config changes)
            # but logging them at INFO makes them searchable and actionable.
            if recommendations.regime_specific_rules:
                self.logger.info(
                    f"[META] {len(recommendations.regime_specific_rules)} regime rule(s) "
                    f"require manual implementation:"
                )
                for rule in recommendations.regime_specific_rules:
                    self.logger.info(
                        f"  [REGIME-RULE][{rule.regime}] {rule.rule} — {rule.reason}"
                    )

            if recommendations.systemic_improvements:
                self.logger.info(
                    f"[META] {len(recommendations.systemic_improvements)} systemic "
                    f"improvement(s) flagged:"
                )
                for i, imp in enumerate(recommendations.systemic_improvements, 1):
                    self.logger.info(f"  [SYSTEMIC-{i}] {imp}")

            if recommendations.next_week_focus:
                self.logger.info(
                    f"[META] Next-week focus: {recommendations.next_week_focus}"
                )

            self.logger.info(
                f"apply_recommendations complete: {applied_count} weight adjustment(s) applied | "
                f"confidence={recommendations.confidence_in_recommendations}%"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply recommendations: {e}")
            return False

    def get_pending_recommendations(self) -> Optional[MetaLearningResult]:
        """
        Return the last analysis result pending manual review.

        Use from REPL to inspect and apply:
            result = coordinator.lead.llm_meta.get_pending_recommendations()
            for imp in result.systemic_improvements: print(imp)
            coordinator.lead.llm_meta.apply_recommendations(result, coordinator.lead.config)
        """
        return self._last_analysis

    def get_last_analysis(self) -> Optional[MetaLearningResult]:
        """Get the most recent analysis."""
        return self._last_analysis

    def get_analysis_history(self, weeks: int = 12) -> List[MetaLearningResult]:
        """Get analysis history."""
        return self._analysis_history[-weeks:]

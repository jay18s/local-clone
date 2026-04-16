"""
ROX Engine v5.0 — Multi-Agent Debate Protocol
Bull + Bear + (optional Neutral) parallel debate → Cross-Examination → Final Synthesis
"""

import logging
import json
from typing import Optional
from dataclasses import dataclass

from agents.llm.async_client import GeminiClient, AsyncLLMResponse
from reasoning_v5.cot_prompts import (
    build_cross_exam_prompt,
    build_final_arbiter_prompt,
)

logger = logging.getLogger("reasoning.debate")


@dataclass
class DebateResult:
    """Result of the full debate process."""
    bull_thesis: dict
    bear_thesis: dict
    neutral_thesis: Optional[dict]
    cross_examination: dict
    final_prediction: dict
    raw_confidence: float
    debate_agreement: float  # 0.0-1.0 (how much all sides agree)
    all_directions: list[str]


# ═══════════════════════════════════════════════════════════════════
# DEBATE AGENT PROMPTS
# ═══════════════════════════════════════════════════════════════════

BULL_SYSTEM = """You are a PERPETUAL BULL — your job is to find the STRONGEST bullish case for the next 1-5 sessions. Even in terrible markets, find what could go right. Be specific with price levels and catalysts."""

BEAR_SYSTEM = """You are a PERPETUAL BEAR — your job is to find the STRONGEST bearish case for the next 1-5 sessions. Even in raging bull markets, find what could go wrong. Be specific with risk levels and triggers."""

NEUTRAL_SYSTEM = """You are a STRICT NEUTRAL ANALYST — your job is to objectively weigh both sides without bias. You represent the median probability. Be honest about the range of outcomes."""


def _build_bull_prompt(market_data: dict) -> str:
    return f"""{BULL_SYSTEM}

Market Data:
{json.dumps(market_data, indent=2, default=str)}

Find the STRONGEST bullish case:
1. At least 3 bullish factors (even if weak)
2. The single best bullish catalyst
3. The price level that would CONFIRM the bull case
4. A specific target price with rationale
5. If you genuinely cannot find ANY bullish factor, explain why the bear case is overwhelming

Respond in JSON:
{{
  "thesis": "STRONGLY_BULLISH|BULLISH|CAUTIOUSLY_BULLISH|NEUTRAL|CAUTIOUSLY_BEARISH",
  "bullish_factors": ["...", "...", "..."],
  "best_catalyst": "...",
  "confirmation_level": 24300,
  "target": 24450,
  "target_rationale": "...",
  "confidence": 65,
  "weakness_in_bull_case": "..."
}}"""


def _build_bear_prompt(market_data: dict) -> str:
    return f"""{BEAR_SYSTEM}

Market Data:
{json.dumps(market_data, indent=2, default=str)}

Find the STRONGEST bearish case:
1. At least 3 bearish factors (even if weak)
2. The single worst risk
3. The price level that would CONFIRM the bear case
4. A specific downside target with rationale
5. If you genuinely cannot find ANY bearish factor, explain why the bull case is overwhelming

Respond in JSON:
{{
  "thesis": "STRONGLY_BEARISH|BEARISH|CAUTIOUSLY_BEARISH|NEUTRAL|CAUTIOUSLY_BULLISH",
  "bearish_factors": ["...", "...", "..."],
  "worst_risk": "...",
  "confirmation_level": 24100,
  "target": 23950,
  "target_rationale": "...",
  "confidence": 55,
  "weakness_in_bear_case": "..."
}}"""


def _build_neutral_prompt(market_data: dict) -> str:
    return f"""{NEUTRAL_SYSTEM}

Market Data:
{json.dumps(market_data, indent=2, default=str)}

Objectively assess both sides:
1. Bull case strength (0-100)
2. Bear case strength (0-100)
3. Which side has more evidence?
4. Expected range for the next 1-5 sessions
5. Probability of range-bound vs trending day

Respond in JSON:
{{
  "thesis": "NEUTRAL|RANGE_BOUND|MILDLY_BULLISH|MILDLY_BEARISH",
  "bull_strength": 45,
  "bear_strength": 55,
  "expected_range_high": 24350,
  "expected_range_low": 24150,
  "range_probability": 70,
  "trend_probability": 30,
  "trend_direction_if_trending": "DOWN",
  "confidence": 60,
  "observation": "..."
}}"""


class DebateEngine:
    """
    Multi-Agent Debate Protocol with parallel execution.
    Runs Bull + Bear + (optional) Neutral in parallel,
    then Cross-Examiner, then Final Arbiter.
    """
    
    def __init__(self, client: GeminiClient, model_pro: str, model_flash: str):
        self.client = client
        self.model_pro = model_pro
        self.model_flash = model_flash
    
    async def run_debate(
        self,
        market_data: dict,
        regime_result: dict,
        news_result: dict,
        pattern_matches: list[dict] = None,
        include_neutral: bool = True,
        rounds: int = 2,
    ) -> DebateResult:
        """
        Run the full debate protocol.
        
        Args:
            market_data: Complete market data dictionary.
            regime_result: Previous regime detection result.
            news_result: Previous news analysis result.
            pattern_matches: Historical pattern matches.
            include_neutral: Whether to include neutral agent.
            rounds: Number of debate rounds (0=skip, 1=bull+bear, 2=full).
        
        Returns:
            DebateResult with all thesis and final prediction.
        """
        logger.info(f"Starting Debate Protocol (rounds={rounds}, neutral={include_neutral})")
        
        if rounds == 0:
            # Skip debate entirely
            return self._skip_debate(regime_result, news_result, pattern_matches)
        
        # ═════════════════════════════════════════════════════════
        # WAVE 1: Parallel debate (Bull + Bear + optional Neutral)
        # ═════════════════════════════════════════════════════════
        debate_calls = [
            {"prompt": _build_bull_prompt(market_data), "model": self.model_flash,
             "temperature": 0.6, "expect_json": True},
            {"prompt": _build_bear_prompt(market_data), "model": self.model_flash,
             "temperature": 0.6, "expect_json": True},
        ]
        
        if include_neutral:
            debate_calls.append(
                {"prompt": _build_neutral_prompt(market_data), "model": self.model_flash,
                 "temperature": 0.3, "expect_json": True},
            )
        
        logger.info(f"Launching {len(debate_calls)} debate agents in parallel...")
        responses = await self.client.generate_parallel(debate_calls)
        
        # Parse responses
        bull_response = responses[0]
        bear_response = responses[1]
        neutral_response = responses[2] if include_neutral and len(responses) > 2 else None
        
        bull_thesis = bull_response.json_data or {"thesis": "NEUTRAL", "confidence": 50, "weakness_in_bull_case": "Failed to parse"}
        bear_thesis = bear_response.json_data or {"thesis": "NEUTRAL", "confidence": 50, "weakness_in_bear_case": "Failed to parse"}
        neutral_thesis = neutral_response.json_data if neutral_response else None
        
        if not bull_thesis.get("thesis"):
            bull_thesis["thesis"] = "NEUTRAL"
        if not bear_thesis.get("thesis"):
            bear_thesis["thesis"] = "NEUTRAL"
        
        logger.info(
            f"Debate results: Bull={bull_thesis.get('thesis')} ({bull_thesis.get('confidence', 0)}%), "
            f"Bear={bear_thesis.get('thesis')} ({bear_thesis.get('confidence', 0)}%)"
            + (f", Neutral={neutral_thesis.get('thesis')} ({neutral_thesis.get('confidence', 0)}%)" if neutral_thesis else "")
        )
        
        # ═════════════════════════════════════════════════════════
        # WAVE 2: Cross-Examination (Pro model)
        # ═════════════════════════════════════════════════════════
        if rounds >= 2:
            cross_prompt = build_cross_exam_prompt(
                bull_thesis=bull_thesis,
                bear_thesis=bear_thesis,
                neutral_thesis=neutral_thesis,
                market_data=market_data,
            )
            
            logger.info("Running Cross-Examination (Pro model)...")
            cross_response = await self.client.generate(
                prompt=cross_prompt,
                model=self.model_pro,
                temperature=0.3,
                expect_json=True,
            )
            
            cross_exam = cross_response.json_data or {
                "overall_assessment": "Failed to parse cross-examination"
            }
            
            logger.info(f"Cross-Exam: {cross_exam.get('overall_assessment', 'N/A')}")
            
            # Adjust confidences based on cross-exam
            bull_conf = bull_thesis.get("confidence", 50)
            bear_conf = bear_thesis.get("confidence", 50)
            
            if "bull_critique" in cross_exam:
                bull_adj = cross_exam["bull_critique"].get("confidence_adjustment", 0)
                bull_conf = max(5, min(95, bull_conf + bull_adj))
            
            if "bear_critique" in cross_exam:
                bear_adj = cross_exam["bear_critique"].get("confidence_adjustment", 0)
                bear_conf = max(5, min(95, bear_conf + bear_adj))
            
            bull_thesis["adjusted_confidence"] = bull_conf
            bear_thesis["adjusted_confidence"] = bear_conf
        else:
            cross_exam = {"overall_assessment": "Cross-examination skipped (rounds=1)"}
            bull_conf = bull_thesis.get("confidence", 50)
            bear_conf = bear_thesis.get("confidence", 50)
        
        # ═════════════════════════════════════════════════════════
        # WAVE 3: Final Arbiter (Pro model)
        # ═════════════════════════════════════════════════════════
        final_prompt = build_final_arbiter_prompt(
            regime_result=regime_result,
            news_result=news_result,
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis,
            cross_exam=cross_exam,
            pattern_matches=pattern_matches,
        )
        
        logger.info("Running Final Arbiter (Pro model)...")
        final_response = await self.client.generate(
            prompt=final_prompt,
            model=self.model_pro,
            temperature=0.3,
            expect_json=True,
        )
        
        final_prediction = final_response.json_data or {
            "prediction": {"direction": "NEUTRAL", "confidence": 50}
        }
        
        # Calculate debate agreement
        directions = []
        for t in [bull_thesis, bear_thesis]:
            thesis = t.get("thesis", "").upper()
            if "BULL" in thesis and "BEAR" not in thesis:
                directions.append("BULL")
            elif "BEAR" in thesis and "BULL" not in thesis:
                directions.append("BEAR")
            else:
                directions.append("NEUTRAL")
        
        if neutral_thesis:
            n_thesis = neutral_thesis.get("thesis", "").upper()
            if "BULL" in n_thesis and "BEAR" not in n_thesis:
                directions.append("BULL")
            elif "BEAR" in n_thesis and "BULL" not in n_thesis:
                directions.append("BEAR")
            else:
                directions.append("NEUTRAL")
        
        bull_count = directions.count("BULL")
        bear_count = directions.count("BEAR")
        total = len(directions)
        
        if bull_count == total:
            agreement = 1.0  # 100% bullish agreement (rare)
        elif bear_count == total:
            agreement = 1.0  # 100% bearish agreement (rare)
        elif bull_count == 0 and bear_count == 0:
            agreement = 0.5  # All neutral
        else:
            agreement = max(bull_count, bear_count) / total
        
        raw_confidence = final_prediction.get("prediction", {}).get("confidence", 50)
        
        logger.info(
            f"Final prediction: {final_prediction.get('prediction', {}).get('direction')} "
            f"(raw confidence: {raw_confidence}%, debate agreement: {agreement:.0%})"
        )
        
        return DebateResult(
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis,
            neutral_thesis=neutral_thesis,
            cross_examination=cross_exam,
            final_prediction=final_prediction,
            raw_confidence=raw_confidence,
            debate_agreement=agreement,
            all_directions=directions,
        )
    
    def _skip_debate(self, regime_result, news_result, pattern_matches):
        """Return a DebateResult that wraps the regime detection without debate."""
        return DebateResult(
            bull_thesis={"thesis": "SKIPPED", "confidence": 0},
            bear_thesis={"thesis": "SKIPPED", "confidence": 0},
            neutral_thesis=None,
            cross_examination={"overall_assessment": "Debate skipped"},
            final_prediction={
                "prediction": {
                    "direction": regime_result.get("regime", "NEUTRAL"),
                    "confidence": regime_result.get("confidence", 50),
                }
            },
            raw_confidence=regime_result.get("confidence", 50),
            debate_agreement=0.5,
            all_directions=[regime_result.get("regime", "NEUTRAL")],
        )

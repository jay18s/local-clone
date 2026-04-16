"""
ROX Engine v5.0 — Chain-of-Thought Prompt Templates
Structured reasoning prompts for all LLM modules.
"""

import json

# ═══════════════════════════════════════════════════════════════════
# REGIME DETECTOR — 7-Step CoT
# ═══════════════════════════════════════════════════════════════════

REGIME_DETECTOR_SYSTEM = """You are a senior quantitative market analyst specializing in Indian equities (NSE). Your analysis must be rigorous, evidence-based, and honest about uncertainty. Never express more confidence than the data warrants."""

def build_regime_cot_prompt(market_data: dict, num_steps: int = 7) -> str:
    """Build the CoT prompt with dynamic number of steps."""
    
    steps = []
    
    if num_steps >= 1:
        steps.append(f"""STEP 1 — TREND ANALYSIS
- Where is Nifty price relative to key moving averages (20, 50, 200 DMA)?
- Is the trend accelerating, decelerating, or reversing?
- What does the sequence of higher highs/lower lows tell us?
- Are there any notable chart patterns (head & shoulders, double top/bottom, etc.)?
- Verdict: STRONG UPTREND / WEAK UPTREND / RANGE_BOUND / WEAK DOWNTREND / STRONG DOWNTREND""")
    
    if num_steps >= 2:
        steps.append(f"""STEP 2 — MOMENTUM ASSESSMENT
- RSI(14) reading — Overbought (>70), Oversold (<30), or Neutral?
- MACD configuration — Line vs Signal, histogram direction
- Stochastic(14,3,3) — %K and %D values
- Is momentum confirming or diverging from price?
- Verdict: STRONG BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG BEARISH""")
    
    if num_steps >= 3:
        steps.append(f"""STEP 3 — VOLATILITY CONTEXT
- India VIX level and percentile (30-day, 90-day)
- Is VIX rising or falling? For how many sessions?
- ATM IV level and IV Skew (Call premium vs Put premium)
- Is volatility expanding (trending market) or contracting (range-bound)?
- Verdict: EXPANDING / STABLE / CONTRACTING""")
    
    if num_steps >= 4:
        steps.append(f"""STEP 4 — INTERMARKET ANALYSIS
- USD/INR trend — Strengthening (bearish) or Weakening (bullish) for Indian equities?
- Crude Oil trend — Inflationary or deflationary pressure?
- Gold trend — Risk-on or risk-off signal?
- US 10Y Yield — Rising (tightening) or falling (accommodative)?
- Global indices (DOW, Nasdaq, FTSE, Nikkei) direction
- How many of 6 intermarket signals are bullish?
- Verdict: STRONGLY BULLISH / BULLISH / MIXED / BEARISH / STRONGLY BEARISH""")
    
    if num_steps >= 5:
        steps.append(f"""STEP 5 — VOLUME & INSTITUTIONAL FLOW
- Market volume vs 20-day average — Elevated or normal?
- FII activity — Buying or selling? For how many consecutive days?
- DII activity — Absorbing or distributing?
- FII Index Futures positioning — Net long or net short?
- Is high volume on down days (distribution) or up days (accumulation)?
- Verdict: ACCUMULATION / NEUTRAL / DISTRIBUTION""")
    
    if num_steps >= 6:
        steps.append(f"""STEP 6 — SYNTHESIS
Based on Steps 1-{min(num_steps-1, 5)}, determine the MOST LIKELY market regime.
Consider: BULLISH, BEARISH, CONSOLIDATION, TRANSITIONAL, VOLATILITY_SPIKE, TRENDING
Assign confidence (0-100%) and explain your reasoning.
Also assess regime shift probabilities (probability of transitioning to each other regime).""")
    
    if num_steps >= 7:
        steps.append(f"""STEP 7 — CONTRARIAN CHECK
What could make your prediction WRONG? List the top 2-3 risks.
What would need to happen to FLIP your regime assessment?
Be honest about uncertainty — markets are inherently unpredictable.""")
    
    steps_text = "\n\n".join(steps)
    
    return f"""{REGIME_DETECTOR_SYSTEM}

Analyze the following Indian market data step-by-step:

{steps_text}

MARKET DATA:
{market_data}

CRITICAL RULES:
1. Each step must be completed before moving to the next
2. Base your analysis ONLY on the provided data — do not invent information
3. Confidence above 85% is almost never justified — be honest about uncertainty
4. The CONTRARIAN CHECK is mandatory — every bullish thesis needs its weakness identified

Respond in this exact JSON format:
{{
  "reasoning": {{
    "trend_analysis": "<your detailed analysis for Step 1>",
    "momentum_assessment": "<your detailed analysis for Step 2>",
    "volatility_context": "<your detailed analysis for Step 3>",
    "intermarket_analysis": "<your detailed analysis for Step 4>",
    "volume_flow": "<your detailed analysis for Step 5>",
    "synthesis": "<your synthesis combining all steps>",
    "contrarian_risks": ["<risk 1>", "<risk 2>", "<risk 3>"]
  }},
  "regime": "BULLISH|BEARISH|CONSOLIDATION|TRANSITIONAL|VOLATILITY_SPIKE|TRENDING",
  "confidence": 75,
  "regime_shift_probability": {{
    "to_bullish": 20,
    "to_bearish": 30,
    "stay_same": 50
  }},
  "key_level": 24200,
  "key_level_type": "support|resistance"
}}"""


# ═══════════════════════════════════════════════════════════════════
# NEWS IMPACT ANALYZER
# ═══════════════════════════════════════════════════════════════════

NEWS_ANALYZER_SYSTEM = """You are a financial news impact analyst. Your job is to objectively assess how each news item affects the Indian stock market (NSE) in the short term (intraday to 1 week). Be specific about which sectors and stocks are affected."""

def build_news_prompt(headlines: list[dict], market_context: str = "") -> str:
    """Build the news analysis prompt."""
    headlines_text = "\n".join(
        f"{i+1}. [{h.get('sentiment', 'NEUTRAL')}] {h.get('title', '')} — "
        f"Impact: {h.get('impact', 'UNKNOWN')} | Source: {h.get('source', '')}"
        for i, h in enumerate(headlines)
    )
    
    return f"""{NEWS_ANALYZER_SYSTEM}

Analyze these news headlines for their impact on Indian markets:

{headlines_text}

MARKET CONTEXT:
{market_context}

For EACH headline, assess:
1. Direction: BULLISH, BEARISH, or NEUTRAL for Indian equities
2. Impact magnitude: HIGH (±1%+), MEDIUM (±0.3-1%), LOW (<±0.3%)
3. Sectors affected specifically
4. Any trade restrictions this news imposes (e.g., "avoid X sector longs")

AGGREGATE ASSESSMENT:
- Overall news sentiment score (-100 to +100)
- Sectors to BLOCK from new long positions
- Sectors to BLOCK from new short positions
- Sectors with boosted confidence

Respond in JSON:
{{
  "headline_analysis": [
    {{
      "headline": "...",
      "direction": "BULLISH|BEARISH|NEUTRAL",
      "impact": "HIGH|MEDIUM|LOW",
      "sectors_affected": ["...", "..."],
      "trade_restrictions": ["<if any>"]
    }}
  ],
  "aggregate": {{
    "sentiment_score": -30,
    "sentiment_label": "MILDLY_BEARISH",
    "block_long_sectors": ["..."],
    "block_short_sectors": ["..."],
    "boost_sectors": ["..."],
    "uncertainty_level": "LOW|MEDIUM|HIGH"
  }}
}}"""


# ═══════════════════════════════════════════════════════════════════
# CROSS-EXAMINER
# ═══════════════════════════════════════════════════════════════════

CROSS_EXAMINER_SYSTEM = """You are a HARSH CROSS-EXAMINER with deep expertise in behavioral finance and cognitive biases. Your job is to find LOGICAL ERRORS, MISSING FACTORS, and BIASES in market analyses. Be ruthless but fair."""

def build_cross_exam_prompt(
    bull_thesis: dict,
    bear_thesis: dict,
    neutral_thesis: dict = None,
    market_data: dict = "",
) -> str:
    """Build the cross-examination prompt."""
    
    sections = []
    
    sections.append(f"""BULL THESIS:
Verdict: {bull_thesis.get('thesis', 'N/A')}
Confidence: {bull_thesis.get('confidence', 'N/A')}%
Key Factors: {bull_thesis.get('factors', 'N/A')}
Best Catalyst: {bull_thesis.get('catalyst', 'N/A')}
Target: {bull_thesis.get('target', 'N/A')}
Weakness in Bull Case: {bull_thesis.get('weakness', 'N/A')}""")
    
    sections.append(f"""BEAR THESIS:
Verdict: {bear_thesis.get('thesis', 'N/A')}
Confidence: {bear_thesis.get('confidence', 'N/A')}%
Key Factors: {bear_thesis.get('factors', 'N/A')}
Worst Risk: {bear_thesis.get('risk', 'N/A')}
Target: {bear_thesis.get('target', 'N/A')}
Weakness in Bear Case: {bear_thesis.get('weakness', 'N/A')}""")
    
    if neutral_thesis:
        sections.append(f"""NEUTRAL THESIS:
Verdict: {neutral_thesis.get('thesis', 'N/A')}
Bull Strength: {neutral_thesis.get('bull_strength', 'N/A')}/100
Bear Strength: {neutral_thesis.get('bear_strength', 'N/A')}/100
Expected Range: {neutral_thesis.get('range', 'N/A')}
Key Observation: {neutral_thesis.get('observation', 'N/A')}""")
    
    thesis_text = "\n\n".join(sections)
    
    return f"""{CROSS_EXAMINER_SYSTEM}

Three market analyses have been presented. CROSS-EXAMINE each one for errors:

{thesis_text}

For EACH thesis, evaluate:
1. LOGICAL ERRORS: Any logical fallacies, survivorship bias, recency bias, or causation errors?
2. MISSING FACTORS: Important data points they IGNORED or didn't consider?
3. OVERCONFIDENCE: Are they too sure? Or too cautious?
4. WEAKEST ARGUMENT: Their single most vulnerable claim
5. IMPACTFUL MISSING DATA: What ONE additional data point would most change their view?

Also identify CONSENSUS FACTORS that ALL theses agree on (these are the most reliable signals).

Respond in JSON:
{{
  "bull_critique": {{
    "logical_errors": ["...", "..."],
    "missing_factors": ["...", "..."],
    "confidence_adjustment": -10,
    "weakest_argument": "...",
    "most_impactful_missing_data": "..."
  }},
  "bear_critique": {{
    "logical_errors": ["...", "..."],
    "missing_factors": ["...", "..."],
    "confidence_adjustment": -5,
    "weakest_argument": "...",
    "most_impactful_missing_data": "..."
  }},
  "neutral_critique": {{
    "logical_errors": ["...", "..."],
    "missing_factors": ["...", "..."],
    "confidence_adjustment": -3,
    "weakest_argument": "...",
    "most_impactful_missing_data": "..."
  }},
  "consensus_factors": ["...", "..."],
  "most_reliable_signal": "...",
  "overall_assessment": "..."
}}"""


# ═══════════════════════════════════════════════════════════════════
# FINAL ARBITER
# ═══════════════════════════════════════════════════════════════════

FINAL_ARBITER_SYSTEM = """You are the FINAL ARBITER of the ROX Engine. You have received multiple analyses and a cross-examination. Your job is to produce ONE calibrated prediction that the trading engine will ACT on. Be precise, be honest about uncertainty, and be actionable."""

def build_final_arbiter_prompt(
    regime_result: dict,
    news_result: dict,
    bull_thesis: dict,
    bear_thesis: dict,
    cross_exam: dict,
    pattern_matches: list[dict] = None,
) -> str:
    """Build the final arbiter prompt."""
    
    pattern_text = ""
    if pattern_matches:
        pattern_text = "\n".join(
            f"- {p['date']} ({p['similarity']}% similar): {p['outcome']} "
            f"→ Best strategy: {p['optimal']} → Lesson: {p['lesson']}"
            for p in pattern_matches[:5]
        )
        pattern_text = f"HISTORICAL PATTERNS:\n{pattern_text}"
    
    return f"""{FINAL_ARBITER_SYSTEM}

You have received:
1. REGIME DETECTOR: {regime_result.get('regime', 'N/A')} (confidence: {regime_result.get('confidence', 0)}%)
2. NEWS ANALYSIS: Sentiment {news_result.get('aggregate', {}).get('sentiment_label', 'N/A')} ({news_result.get('aggregate', {}).get('sentiment_score', 0)}/100)
3. BULL THESIS: {bull_thesis.get('thesis', 'N/A')} (confidence: {bull_thesis.get('confidence', 0)}%)
4. BEAR THESIS: {bear_thesis.get('thesis', 'N/A')} (confidence: {bear_thesis.get('confidence', 0)}%)
5. CROSS-EXAMINATION: {cross_exam.get('overall_assessment', 'N/A')}

{pattern_text}

RULES:
1. Weight the cross-examination HEAVILY — if a thesis has logical errors, reduce its influence
2. Consensus factors should be given 2x weight
3. If all theses agree on direction, confidence should be 75-90%
4. If they disagree, confidence should be 30-50%
5. NEVER give confidence above 90% — markets are inherently uncertain
6. HISTORICAL PATTERNS provide grounding — if 3+ similar cases agree, that's strong evidence

Respond in JSON:
{{
  "prediction": {{
    "direction": "BULLISH|BEARISH|NEUTRAL|STRONGLY_BULLISH|STRONGLY_BEARISH",
    "confidence": 55,
    "timeframe": "intraday|1-3 sessions|multi-week",
    "target_range": {{
      "upper": 24400,
      "lower": 24100
    }}
  }},
  "key_levels": {{
    "breakout_above": 24350,
    "breakdown_below": 24100,
    "critical": 24200,
    "critical_type": "support|resistance"
  }},
  "catalysts_to_watch": ["...", "..."],
  "reasoning_summary": "<2-3 sentence explanation>",
  "biggest_risk": "<the single most important risk>",
  "alternative_scenario": "<what happens if prediction is wrong>"
}}"""


# ═══════════════════════════════════════════════════════════════════
# TRADING PLANNER
# ═══════════════════════════════════════════════════════════════════

TRADING_PLANNER_SYSTEM = """You are a professional trading planner. Design specific, executable trade plans with precise entries, stop losses, targets, and position sizes. Every plan must include a clear INVALIDATION condition — the scenario that would prove the thesis wrong."""

def build_trading_planner_prompt(
    signals: list[dict],
    regime: dict,
    news_restrictions: list[str] = None,
    portfolio: dict = None,
    prediction: dict = None,
) -> str:
    """Build the trading planner prompt."""
    
    signals_text = "\n".join(
        f"  {s['symbol']}: {s['direction']} | Agent: {s['agent']} | "
        f"Strength: {s['strength']} | R:R {s['rr_ratio']} | "
        f"RSI: {s.get('rsi', 'N/A')} | Sector: {s.get('sector', 'N/A')}"
        for s in signals
    )
    
    restrictions_text = "\n".join(f"  - {r}" for r in (news_restrictions or []))
    
    return f"""{TRADING_PLANNER_SYSTEM}

REGIME: {regime.get('regime', 'N/A')} (confidence: {regime.get('confidence', 0)}%)
PREDICTION: {prediction.get('prediction', {}).get('direction', 'N/A')} 
              (calibrated confidence: {prediction.get('calibrated_confidence', 'N/A')}%)

SIGNALS TO EVALUATE:
{signals_text}

NEWS RESTRICTIONS:
{restrictions_text if restrictions_text else "  None"}

PORTFOLIO: Capital = {portfolio.get('capital', 'N/A')} | Max risk/trade = {portfolio.get('risk_pct', 'N/A')}%

For EACH signal, provide:
1. Entry price and method (market/limit)
2. Stop loss level and rationale
3. Target(s) with rationale
4. Position size (shares) based on risk management
5. Risk amount (INR) and expected reward
6. VALIDATION: Why this signal is reliable given current regime
7. INVALIDATION: What would make this trade wrong
8. FINAL VERDICT: EXECUTE or HOLD (with reason)

Portfolio constraints:
- Max total positions: 6
- Max same-sector trades: 2
- Max single position: 25% of capital
- Total portfolio risk must stay below 3%

Respond in JSON:
{{
  "trades": [
    {{
      "symbol": "...",
      "direction": "LONG|SHORT",
      "entry_price": 0,
      "stop_loss": 0,
      "target_1": 0,
      "target_2": 0,
      "position_size": 0,
      "risk_amount": 0,
      "expected_reward": 0,
      "validation": "...",
      "invalidation": "...",
      "verdict": "EXECUTE|HOLD",
      "verdict_reason": "...",
      "strategy": "..."
    }}
  ],
  "portfolio_summary": {{
    "total_capital_deployed": 0,
    "cash_remaining": 0,
    "total_risk_pct": 0,
    "expected_reward_pct": 0,
    "blended_rr": 0
  }}
}}"""


# ═══════════════════════════════════════════════════════════════════
# FNO BRAIN (Options Strategy)
# ═══════════════════════════════════════════════════════════════════

FNO_BRAIN_SYSTEM = """You are an options strategy designer for Indian index derivatives (NIFTY, BANKNIFTY). Design strategies that match the market regime with appropriate risk management. Always provide max profit, max loss, and breakeven for each strategy."""

def build_fno_brain_prompt(
    market_view: dict,
    options_chain: dict,
    expiry_info: dict,
    regime: dict,
) -> str:
    """Build the FNO options strategy prompt."""
    
    return f"""{FNO_BRAIN_SYSTEM}

MARKET VIEW: {market_view.get('direction', 'N/A')} 
              (confidence: {market_view.get('confidence', 'N/A')}%)

OPTIONS CHAIN DATA:
{json.dumps(options_chain, indent=2)}

EXPIRY INFO: Days to expiry: {expiry_info.get('days_to_expiry', 'N/A')}
Max Pain: {expiry_info.get('max_pain', 'N/A')}
PCR: {expiry_info.get('pcr', 'N/A')}
VIX: {expiry_info.get('vix', 'N/A')}

REGIME: {regime.get('regime', 'N/A')}

Design 2-3 options strategies ranked by confidence:
1. PRIMARY strategy (highest conviction)
2. INCOME strategy (if appropriate)
3. HEDGING strategy (if needed)

For each strategy provide:
- Strategy type (e.g., Bull Call Spread, Iron Condor, Short Straddle)
- Exact strikes and premiums
- Max profit, max loss, breakeven
- Conditions for entry and exit
- Why this strategy fits the current regime

Respond in JSON:
{{
  "strategies": [
    {{
      "name": "...",
      "type": "BULL_CALL_SPREAD|BEAR_PUT_SPREAD|IRON_CONDOR|SHORT_STRADDLE|...",
      "confidence": 60,
      "risk_level": "LOW|MEDIUM|HIGH",
      "legs": [
        {{
          "action": "BUY|SELL",
          "instrument": "NIFTY 25000 CE",
          "premium": 150,
          "quantity": 75
        }}
      ],
      "net_premium": -50,
      "max_profit": 5000,
      "max_loss": 3000,
      "breakeven": 25050,
      "entry_conditions": ["..."],
      "exit_conditions": ["..."],
      "rationale": "..."
    }}
  ]
}}"""


# ═══════════════════════════════════════════════════════════════════
# SELF-REFLECTOR
# ═══════════════════════════════════════════════════════════════════

SELF_REFLECTOR_SYSTEM = """You are a trading coach analyzing a COMPLETED trade. Be brutally honest. The goal is to extract actionable lessons that will improve future decisions."""

def build_self_reflector_prompt(
    trade_record: dict,
    market_before: dict,
    market_after: dict,
) -> str:
    """Build the post-trade reflection prompt."""
    
    outcome = "PROFIT" if trade_record.get("pnl", 0) > 0 else "LOSS"
    
    return f"""{SELF_REFLECTOR_SYSTEM}

TRADE RECORD:
- Symbol: {trade_record.get('symbol', 'N/A')}
- Direction: {trade_record.get('direction', 'N/A')}
- Entry: {trade_record.get('entry', 'N/A')} at {trade_record.get('entry_time', 'N/A')}
- Exit: {trade_record.get('exit', 'N/A')} at {trade_record.get('exit_time', 'N/A')}
- P&L: INR {trade_record.get('pnl', 0)} ({trade_record.get('pnl_pct', 0)}%)
- Initial confidence: {trade_record.get('confidence', 'N/A')}%
- Original reasoning: {trade_record.get('reasoning', 'N/A')}
- Outcome: {outcome}

MARKET CONDITIONS AT ENTRY:
{json.dumps(market_before, indent=2)}

MARKET CONDITIONS AT EXIT:
{json.dumps(market_after, indent=2)}

Analyze:
1. PRIMARY REASON: Why did this trade {outcome}?
2. MISSED SIGNALS: What signal did we miss that could have predicted this?
3. IDEAL DECISION: What should we have done with hindsight?
   Options: SHOULD_NOT_TRADE | SMALLER_SIZE | TIGHTER_SL | OPPOSITE | PERFECT
4. NEW RULE: What specific, implementable rule should be added?
5. CONFIDENCE CALIBRATION: Was the original confidence correct? What should it have been?

Respond in JSON:
{{
  "primary_reason": "...",
  "missed_signals": ["...", "..."],
  "ideal_decision": "SHOULD_NOT_TRADE|SMALLER_SIZE|TIGHTER_SL|OPPOSITE",
  "new_rule": {{
    "condition": "...",
    "action": "..."
  }},
  "confidence_was": "TOO_HIGH|TOO_LOW|ABOUT_RIGHT",
  "confidence_should_have_been": 40,
  "lesson": "...",
  "pattern_recognized": true,
  "pattern_description": "..."
}}"""

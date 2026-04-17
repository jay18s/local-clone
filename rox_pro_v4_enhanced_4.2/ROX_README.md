# ROX PROVEN EDGE ENGINE v6.0

> Multi-Agent LLM-Powered Indian Market Trading System  
> **CLOSED-LOOP LEARNING MODE — ACTIVATED 2026-04-17**  
> 15 Agents | 7 LLM Modules | 9 v6 Modules | 103/103 Tests Pass

---

## What's New in v6.0

v6.0 is a structural upgrade from v5.0 that fixes **three root-cause bugs** responsible for a 16.7% win rate, and activates **closed-loop learning** so the system improves from its own trade outcomes. All 15 implementation tasks across 4 phases are complete with 103/103 tests passing.

### Three Root-Cause Bugs Fixed

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | **Regime Misclassification** — LLM-only regime calls were inconsistent, especially in low-volatility markets | Single data source with no cross-check | RuleRegimeClassifier (deterministic) + LLM parallel → RegimeArbiter with 5-case conflict resolution |
| 2 | **No SHORT Execution** — system only went LONG, missing 50% of bearish opportunities | No SHORT execution path existed | ShortExecutor (BUY_PUT, SELL_CALL, BEAR_SPREAD) + DirectionalRouter + CircuitBreakerV2 |
| 3 | **No Learning Loop** — the system never learned from outcomes; same mistakes repeated forever | No feedback path from trade results to signal weights | TradeOutcomeLogger → PatternMemory.update_outcome() → AdaptiveCalibrator.update_weights() → MetaLearner |

### v6.0 Core Runtime Activations (5 Changes)

| # | Activation | Components | What It Does |
|---|-----------|-----------|-------------|
| 1 | **Regime Engine** | RuleRegimeClassifier + LLM parallel → RegimeArbiter + RegimeAccuracyTracker | Dual-source regime with conflict resolution; accuracy tracking feeds back to arbiter |
| 2 | **SHORT Execution** | DirectionalRouter + ShortExecutor + CircuitBreakerV2 | Routes SHORT → F&O (BUY_PUT ATM, SELL_CALL ATM, BEAR_SPREAD); paper mode for first 15 trades; 3-layer capital protection |
| 3 | **Learning Loop** | TradeOutcomeLogger (JSONL) + PatternMemory.update_outcome() + AdaptiveCalibrator.update_weights() + MetaLearner (threshold 20/0.10) | Full closed-loop: log → learn → recalibrate weights → better decisions |
| 4 | **Debate Protocol** | BULL_SYSTEM_PROMPT + BEAR_SYSTEM_PROMPT, temperature=0.7, diversity score | Adversarial prompts force genuine disagreement; diversity score detects echo chambers |
| 5 | **Cycle Management** | get_cycle_interval_minutes() + should_skip_cycle() + signal tracing in coordinator | Adaptive frequency (5-30 min based on regime); lunch-hour skip; end-of-day signal audit trail |

---

## Architecture

```
+=====================================================================+
|                    ROX ENGINE v6.0                                  |
|              CLOSED-LOOP LEARNING MODE                              |
+=====================================================================+
|                                                                     |
|  +----------+  +----------+  +----------+  +----------+            |
|  |  ORION   |  |  VESPER  |  |  KAIRO   |  | SENTINEL |            |
|  |  (Tech)  |  |  (Flow)  |  |(Sentiment)| | (Deriv)  |            |
|  +----------+  +----------+  +----------+  +----------+            |
|  +----------+  +----------+  +----------+  +----------+            |
|  |  NEXUS   |  |PRUDENCE  |  | CATALYST |  | OPTIMUS  |            |
|  |(Fundmnt) |  |  (Risk)  |  | (Events) |  |  (F&O)   |            |
|  +----------+  +----------+  +----------+  +----------+            |
|  +----------+  +----------+  +----------+                          |
|  |  HERMES  |  |  THETA   |  |  DELTA   |  F&O Agents             |
|  |  (Exec)  |  |(Greeks)  |  |(Settle)  |                          |
|  +----------+  +----------+  +----------+                          |
|                                                                     |
|  +---------- 7 LLM Intelligence Modules -----------------+          |
|  | RegimeDetector | NewsAnalyzer | CrossExaminer       |          |
|  | TradingPlanner | PatternValidator | MetaLearner     |          |
|  | OptionsStrategist | HistoryAnalyzer                 |          |
|  +------------------------------------------------------+          |
|                                                                     |
|  +---------- v5.0 Reasoning Layer ----------------------+          |
|  | CoT Prompts | Debate Engine | Pattern Memory        |          |
|  | Confidence Calibrator | Rule Validator | Adaptive    |          |
|  +------------------------------------------------------+          |
|                                                                     |
|  +========== v6.0 CLOSED-LOOP LEARNING LAYER ===========+          |
|  |                                                       |          |
|  | [REGIME ENGINE]                                       |          |
|  |  RuleRegimeClassifier (VIX 30%, DMA 25%, FII 20%,   |          |
|  |  Sector 15%, Momentum 10%)                            |          |
|  |  + LLM Regime (parallel)                              |          |
|  |  -> RegimeArbiter (5-case conflict resolution)        |          |
|  |  -> RegimeTransitionDetector (SUSPECTED/CONFIRMED)    |          |
|  |  -> RegimeAccuracyTracker (end-of-day vs NIFTY)       |          |
|  |                                                       |          |
|  | [SHORT EXECUTION]                                     |          |
|  |  ShortExecutor: BUY_PUT ATM, SELL_CALL ATM,           |          |
|  |                 BEAR_SPREAD                           |          |
|  |  DirectionalRouter: LONG -> LongExecutor              |          |
|  |                     SHORT -> ShortExecutor             |          |
|  |  CircuitBreakerV2: 3 consec loss / 5% daily /         |          |
|  |                     10% max DD -> halt; resume @50%   |          |
|  |  Paper Mode: First 15 SHORTs are paper-only           |          |
|  |  Filters: OI>1000, spread<2%                         |          |
|  |                                                       |          |
|  | [LEARNING LOOP]                                       |          |
|  |  TradeOutcomeLogger -> JSONL (full context)           |          |
|  |  PatternMemory.update_outcome() -> accuracy feedback  |          |
|  |  AdaptiveCalibrator.update_weights() -> Bayesian EMA  |          |
|  |  MetaLearner: threshold 20 trades / 0.10 min win rate |          |
|  |                                                       |          |
|  | [DEBATE PROTOCOL]                                     |          |
|  |  BULL_SYSTEM_PROMPT (momentum-following)              |          |
|  |  BEAR_SYSTEM_PROMPT (mean-reversion)                  |          |
|  |  temperature=0.7 for genuine diversity                |          |
|  |  _compute_diversity_score() (0=identical, 1=opposite) |          |
|  |                                                       |          |
|  | [CYCLE MANAGEMENT]                                    |          |
|  |  get_cycle_interval_minutes(): 5-30min adaptive       |          |
|  |  should_skip_cycle(): lunch-hour skip (12:00-13:30)   |          |
|  |  Signal tracing: 3-block audit in coordinator         |          |
|  +=======================================================+          |
+=====================================================================+
```

---

## Project Structure

```
rox_pro_v5_enhanced_5.0/
+-- main_v5_pipeline.py          # v6.0 async 3-wave orchestrator (MAIN ENTRY)
|                                  #   - RegimeArbiter, DirectionalRouter, ShortExecutor
|                                  #   - CircuitBreakerV2, TradeOutcomeLogger, AdaptiveCalibrator
|                                  #   - get_cycle_interval_minutes(), should_skip_cycle()
+-- main.py                      # v4.0 production entry point
+-- main_production.py           # Production runner with Fyers API
+-- coordinator.py               # UnifiedCoordinator v6.0
|                                  #   LeadCoordinator + FnoCoordinator + v6 modules
|                                  #   Signal tracing blocks, v6 runtime assertions
+-- config_v5.py                 # v6.0 Master Configuration
|                                  #   MetaLearner: min_trades=20, min_win_rate=0.10
|                                  #   GEMINI_MODEL_ROUTING, RAM management
+-- api_server.py                # FastAPI dashboard server
+-- fyers_login.py               # Fyers OAuth login
+-- setup.py                     # Package installer (v6.0.0)
+-- requirements.txt             # Dependencies
|
+-- reasoning/                   # v6.0 Reasoning Layer
|   +-- __init__.py
|   +-- rule_regime_classifier.py   # [NEW] VIX 30%, DMA 25%, FII 20%, Sector 15%, Mom 10%
|   +-- regime_arbiter.py           # [NEW] 5-case conflict resolution (rule vs LLM)
|   +-- regime_transition_detector.py # [NEW] SUSPECTED/CONFIRMED transition events
|   +-- regime_accuracy_tracker.py   # [NEW] End-of-day regime accuracy vs NIFTY
|   +-- adaptive_calibrator.py       # [NEW] Bayesian EMA weight updating from outcomes
|   +-- rule_validator.py            # Deterministic signal validator (<1ms)
|   +-- cot_prompts.py              # 7-step Chain-of-Thought prompt builders
|   +-- debate_engine.py            # Bull/Bear + BULL/BEAR_SYSTEM_PROMPT, temp=0.7,
|   |                                 #   _compute_diversity_score()
|   +-- pattern_memory.py           # DuckDB Pattern Memory + update_outcome()
|   +-- confidence_calibrator.py    # 6-Signal weighted calibrator
|   +-- adaptive_and_cache.py       # Adaptive prompt selector + regime cache
|   +-- data_classes.py             # Signal, RegimeResult, NewsResult, TradePlan, etc.
|
+-- execution/                   # Order Execution (v6.0 enhanced)
|   +-- __init__.py
|   +-- short_executor.py           # [NEW] BUY_PUT ATM, SELL_CALL ATM, BEAR_SPREAD
|   |                                 #   OI>1000 & spread<2% filters
|   +-- directional_router.py       # [NEW] Routes LONG/SHORT to appropriate executor
|   +-- fno_execution_engine.py     # F&O execution engine
|   +-- order_manager.py            # Order lifecycle management
|   +-- execution_algorithms.py     # TWAP/VWAP algorithms
|   +-- slippage_control.py         # Slippage monitoring
|
+-- monitoring/                  # Risk Management (v6.0 enhanced)
|   +-- __init__.py
|   +-- circuit_breaker_v2.py       # [NEW] 3-layer: consec loss / daily loss / max DD
|   |                                 #   Resume at 50% size; size_multiplier
|   +-- circuit_breaker.py          # Original circuit breaker
|   +-- risk_monitor.py             # Portfolio risk monitoring
|   +-- performance_filter.py       # Performance-based filtering
|   +-- risk_dashboard.py           # Risk visualization
|
+-- data/                        # Data Management (v6.0 enhanced)
|   +-- __init__.py
|   +-- trade_outcome_logger.py     # [NEW] JSONL full-context trade lifecycle logging
|   +-- shadow_trade_logger.py      # Suppressed setup logging for meta-learner
|   +-- data_manager.py             # Core data management
|   +-- pattern_database.py         # Pattern database
|   +-- scorecard.py                # Agent scorecard
|   +-- macro_fetcher.py            # Macro data fetcher
|   +-- fyers_fetcher.py            # Fyers API data fetcher
|   +-- trade_logger.py             # Trade logging
|   +-- calibration/                # Calibration data
|   +-- meta_learning/              # Meta-learning data
|
+-- agents/                      # All Trading Agents
|   +-- orion.py, vesper.py, kairo.py, sentinel.py, nexus.py
|   +-- prudence.py, catalyst.py, optimus.py
|   +-- phoenix_agent.py, nocturnal_agent.py
|   +-- hermes_agent.py, theta_agent.py, delta_agent.py
|   +-- fno_brain_extension.py, directional_option_advisor.py
|   +-- news_core.py, strategy_builders.py, ai_brain.py
|   +-- base_agent.py               # Base class with ReAct mixin
|   +-- calibration/                # Agent calibration store
|   +-- llm/                        # 7 LLM Intelligence Modules
|       +-- async_client.py         # Async Gemini client with semaphore + retry
|       +-- base_llm_agent.py       # Base class with dual SDK support
|       +-- llm_regime_detector.py
|       +-- llm_news_analyzer.py
|       +-- llm_cross_examiner.py
|       +-- llm_pattern_validator.py
|       +-- llm_trading_planner.py
|       +-- llm_meta_learner.py
|       +-- llm_options_strategist.py
|       +-- llm_history_analyzer.py
|
+-- infrastructure/              # Data Feed & Fyers Integration
|   +-- data_feed.py, coordinator.py, cache.py
|   +-- fno_instrument_manager.py, greeks_calculator.py
|   +-- margin_calculator.py, option_chain_stream.py
|   +-- historical_data_manager.py, websocket_handler.py
|   +-- mwpl_monitor.py, physical_settlement_manager.py
|   +-- event_bus.py, data_normalizer.py
|
+-- core/                        # Core Infrastructure
|   +-- config.py, v5_logger.py, logger.py, news_fetcher.py
|
+-- monitoring/                  # Risk & Circuit Breakers
|
+-- alerts/                      # Notification System
|   +-- alert_manager.py, channels.py
|
+-- pipeline/                    # v4.0 Processing Pipeline
|   +-- pipeline.py
|
+-- utils/                       # Utilities
|   +-- helpers.py, platform_utils.py
|
+-- tests/                       # Test Suite
|   +-- test_v6_integration.py      # [NEW] 103 comprehensive v6 tests
|   +-- test_v5_integration.py      # v5.0 comprehensive tests (40 tests)
|   +-- test_optimus.py
|   +-- test_greeks_calculator.py
|   +-- test_fno_extension.py
|
+-- scripts/                     # Maintenance Scripts
|   +-- daily_reconcile.py, weekly_meta_analysis.py
|
+-- logs/                        # Runtime logs
+-- data/                        # Runtime data (DuckDB, JSONL, CSV)
```

---

## v6.0 Module Reference

### 1. Regime Engine

| Module | File | Purpose |
|--------|------|---------|
| **RuleRegimeClassifier** | `reasoning/rule_regime_classifier.py` | Deterministic regime classification using weighted indicators: VIX (30%), DMA position (25%), FII flow (20%), Sector breadth (15%), 5d Momentum (10%). No LLM call required. |
| **RegimeArbiter** | `reasoning/regime_arbiter.py` | 5-case conflict resolution between rule-based and LLM regime calls: (1) both agree, (2) rule more confident, (3) LLM more confident, (4) disagree with accuracy tilt, (5) no confidence — default to rule. Uses LLM rolling accuracy from RegimeAccuracyTracker. |
| **RegimeTransitionDetector** | `reasoning/regime_transition_detector.py` | Detects SUSPECTED and CONFIRMED regime transitions. On CONFIRMED transition, reduces position size to 75% via CircuitBreakerV2.size_multiplier. |
| **RegimeAccuracyTracker** | `reasoning/regime_accuracy_tracker.py` | End-of-day regime accuracy scoring vs NIFTY actual move. Tracks rolling LLM accuracy over last N days. Feeds back to RegimeArbiter to weight LLM vs rule confidence. |

### 2. SHORT Execution

| Module | File | Purpose |
|--------|------|---------|
| **ShortExecutor** | `execution/short_executor.py` | Generates SHORT orders via three strategies: BUY_PUT ATM (default), SELL_CALL ATM (high conviction), BEAR_SPREAD (moderate conviction). Filters: OI > 1000, bid-ask spread < 2%. |
| **DirectionalRouter** | `execution/directional_router.py` | Routes LONG signals to LongExecutor and SHORT signals to ShortExecutor. Checks CircuitBreakerV2 before routing. Returns RoutingResult with executed/blocked status and reason. |
| **CircuitBreakerV2** | `monitoring/circuit_breaker_v2.py` | 3-layer capital preservation: (1) 3 consecutive losses → halt, (2) 5% daily loss → halt, (3) 10% max drawdown → halt. On resume: size reduced to 50% (size_multiplier). On regime transition: size reduced to 75%. |

### 3. Learning Loop

| Module | File | Purpose |
|--------|------|---------|
| **TradeOutcomeLogger** | `data/trade_outcome_logger.py` | JSONL-based full-context trade lifecycle logging. Records: entry/exit timestamps, prices, PnL, regime at entry, regime confidence, debate agreement, calibration score, agent verdicts, signals passed/failed, news sentiment, pattern match IDs, cycle number. |
| **PatternMemory.update_outcome()** | `reasoning/pattern_memory.py` | Closed-loop feedback: after trade resolves, updates pattern accuracy with actual_outcome, actual_pnl, hold_period_minutes. Future pattern lookups now reflect real outcomes. |
| **AdaptiveCalibrator** | `reasoning/adaptive_calibrator.py` | Bayesian weight updating via EMA (alpha=0.1). After each trade, computes signal-outcome correlation over rolling 20-trade window. Clamps weights: 0.05 min, 0.35 max, all sum to 1.0. Default priors: debate_agreement=0.25, pattern_match=0.20, technical_alignment=0.20, volume_confirmation=0.15, regime_consistency=0.10, anti_consensus=0.10. |
| **MetaLearner** | `agents/llm/llm_meta_learner.py` | Threshold lowered: min_trades 50→20, min_win_rate 0.15→0.10. Enables meta-learning after just 20 trades with any positive win rate. |

### 4. Debate Protocol Enhancement

| Module | File | Purpose |
|--------|------|---------|
| **BULL_SYSTEM_PROMPT** | `reasoning/debate_engine.py` | Adversarial prompt: "You are a MOMENTUM-FOLLOWING trader. Your thesis: the market's current direction will CONTINUE." Forces genuine bullish arguments. |
| **BEAR_SYSTEM_PROMPT** | `reasoning/debate_engine.py` | Adversarial prompt: "You are a MEAN-REVERSION risk manager. Your thesis: the market's current move will REVERSE or STALL." Forces genuine bearish arguments. |
| **temperature=0.7** | `reasoning/debate_engine.py` | Debate agents run at temperature=0.7 (up from 0.5) to ensure diverse, non-templated responses. Cross-examination and arbiter stay at 0.3 for analytical precision. |
| **diversity score** | `reasoning/debate_engine.py` | `_compute_diversity_score()` measures genuine disagreement: direction_score (0.7 weight) + confidence_divergence (0.3 weight). Returns 0.0 (identical) to 1.0 (opposite). Low diversity triggers a warning log. |

### 5. Cycle Management

| Module | File | Purpose |
|--------|------|---------|
| **get_cycle_interval_minutes()** | `main_v5_pipeline.py` | Adaptive cycle frequency: EXTREME/VOLATILE regime or VIX>25 → 5 min; TRENDING → 15 min; 3+ consecutive no-signal → 30 min; 120+ min since last trade → 20 min; default → 10 min. |
| **should_skip_cycle()** | `main_v5_pipeline.py` | Skip cycles during lunch hour (12:00-13:30 IST) when NSE liquidity is lowest and noise is highest. |
| **Signal tracing** | `coordinator.py` | 3-block audit trail in coordinator: (1) signal entry with regime+consensus context, (2) routing decision (LONG/SHORT/blocked), (3) execution result with order ID. Full trace logged per cycle. |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export FYERS_APP_ID="your-fyers-app-id"
export FYERS_APP_SECRET="your-fyers-app-secret"
export FYERS_ACCESS_TOKEN="your-fyers-access-token"
```

### 3. Run v6.0 Pipeline

```bash
python main_v5_pipeline.py
```

The v6.0 activation banner will confirm closed-loop learning mode:

```
======================================================================
  ROX ENGINE v6.0 — CLOSED-LOOP LEARNING MODE ACTIVATED
  RegimeArbiter | DirectionalRouter | ShortExecutor | CircuitBreakerV2
  TradeOutcomeLogger | AdaptiveCalibrator | PatternMemory feedback
  Debate: BULL/BEAR adversarial, temp=0.7, diversity score
  Cycle: Adaptive interval, signal tracing, lunch-hour skip
======================================================================
```

### 4. Run Tests

```bash
python -m pytest tests/test_v6_integration.py -v
# 103 tests passing
```

### 5. Run Production (with Fyers)

```bash
python main_production.py
```

---

## v6.0 Execution Flow (3-Wave Async + Closed Loop)

```
+=====================================================================+
| PRE-FLIGHT: Complexity Assessment + Circuit Breaker Check           |
|  -> AdaptivePromptSelector.assess_complexity()                      |
|  -> CircuitBreakerV2.can_trade() — halt if 3 consec loss /         |
|     5% daily loss / 10% max drawdown                                |
|  -> should_skip_cycle() — skip lunch hour (12:00-13:30)            |
+========================================+============================+
                                         |
+========================================v============================+
| WAVE 1: Dual Regime Detection + News Analysis (PARALLEL)           |
|                                                                      |
|  [v6 NEW] RuleRegimeClassifier (deterministic, <1ms)                |
|            VIX 30% + DMA 25% + FII 20% + Sector 15% + Mom 10%     |
|                        |                                             |
|            LLM Regime Detector (CoT 7-step prompt)                  |
|                        |                                             |
|            v         RegimeArbiter.resolve()  v                     |
|            -> 5-case conflict resolution + accuracy tracking        |
|            -> RegimeTransitionDetector (SUSPECTED/CONFIRMED)        |
|            -> On CONFIRMED: reduce size to 75%                      |
|                                                                      |
|  News Impact Analyzer (flash model)                                 |
|  Pattern Memory Bank (DuckDB lookup)                                |
|  MetaLearner (conditional: skip if < 20 trades or < 10% win rate)  |
+========================================+============================+
                                         |
+========================================v============================+
| WAVE 2: Debate Protocol (v6 Enhanced)                               |
|                                                                      |
|  [v6 NEW] BULL agent (BULL_SYSTEM_PROMPT, temp=0.7)                 |
|           "MOMENTUM-FOLLOWING trader — direction will CONTINUE"     |
|                        +                                             |
|           BEAR agent (BEAR_SYSTEM_PROMPT, temp=0.7)                 |
|           "MEAN-REVERSION manager — move will REVERSE"              |
|                        +                                             |
|           Neutral agent (optional, temp=0.3)                        |
|                        |                                             |
|           Cross-Examination (Pro model, temp=0.3)                   |
|           Final Arbiter (Pro model, temp=0.3)                       |
|                        |                                             |
|           [v6 NEW] _compute_diversity_score()                       |
|           0.0 = identical (echo chamber) -> 1.0 = opposite (good)  |
+========================================+============================+
                                         |
+========================================v============================+
| WAVE 3: Validation + Planning + Routing (v6 Enhanced)               |
|                                                                      |
|  Rule-Based Validator (<1ms, no LLM)                                |
|  Trading Planner (only if signals pass)                             |
|  Confidence Calibration (6-signal weighted)                          |
|                                                                      |
|  [v6 NEW] DIRECTIONAL ROUTING:                                      |
|           LONG  -> DirectionalRouter.route_long() -> LongExecutor   |
|           SHORT -> ShortExecutor.prepare_short_order()               |
|                   -> BUY_PUT ATM / SELL_CALL ATM / BEAR_SPREAD      |
|                   -> Paper mode for first 15 SHORTs                  |
|                   -> DirectionalRouter.route_short() after paper     |
|                   -> Filters: OI>1000, spread<2%                    |
|                                                                      |
|  [v6 NEW] CircuitBreakerV2 pre-check before every route             |
+========================================+============================+
                                         |
+========================================v============================+
| POST-CYCLE: Closed-Loop Learning (v6 NEW)                           |
|                                                                      |
|  TradeOutcomeLogger.log_trade() -> JSONL file                        |
|    Records: regime, confidence, debate, calibration, agents,         |
|             signals, news, patterns, cycle number                    |
|                                                                      |
|  PatternMemory.update_outcome() -> pattern accuracy feedback         |
|    actual_outcome, actual_pnl, hold_period_minutes                  |
|                                                                      |
|  AdaptiveCalibrator.update_weights() -> Bayesian EMA                 |
|    Signal-outcome correlation over rolling 20-trade window           |
|    Clamped: 0.05 min, 0.35 max, sum = 1.0                          |
|                                                                      |
|  RegimeAccuracyTracker -> end-of-day scoring vs NIFTY               |
|    Feeds back to RegimeArbiter for next day's LLM vs rule weighting |
+=====================================================================+
```

---

## Configuration

All settings are in `config_v5.py`. Key parameters:

```python
# Portfolio
initial_capital = 10,00,000  # INR
risk_per_trade_pct = 1.5%
max_positions = 6

# Gemini Models (centralized routing)
FAST_MODEL = "gemini-2.0-flash"
SMART_MODEL = "gemini-3-flash-preview"
NEWS_MODEL = "gemini-2.0-flash"
CACHE_TTL_MINUTES = 5
MAX_PARALLEL_LLM_CALLS = 7

# v6.0 Calibration Weights (initial priors, updated by AdaptiveCalibrator)
debate_agreement = 0.25
pattern_match = 0.20
technical_alignment = 0.20
volume_confirmation = 0.15
regime_consistency = 0.10
anti_consensus = 0.10

# v6.0 MetaLearner (lowered thresholds)
meta_learner_min_trades = 20       # was 50 in v5
meta_learner_min_win_rate = 0.10   # was 0.15 in v5

# v6.0 CircuitBreakerV2
consecutive_loss_threshold = 3
daily_loss_limit_pct = 5.0
max_drawdown_pct = 10.0
reduced_size_pct = 50.0

# v6.0 ShortExecutor
short_paper_mode_limit = 15  # first 15 SHORTs are paper-only
min_open_interest = 1000
max_bid_ask_spread_pct = 2.0

# v6.0 Cycle Management
cycle_interval_extreme = 5 min
cycle_interval_trending = 15 min
cycle_interval_default = 10 min
lunch_hour_skip = 12:00-13:30 IST
```

---

## Gemini Model Strategy

| Module | Model | Temperature | v6 Note |
|--------|-------|-------------|---------|
| Regime Detection (LLM) | gemini-3-flash-preview | 0.3 | Now runs parallel with RuleRegimeClassifier |
| News Analysis | gemini-2.0-flash | 0.2 | Unchanged |
| Debate Bull Agent | gemini-2.0-flash | **0.7** | Increased from 0.5 for diversity |
| Debate Bear Agent | gemini-2.0-flash | **0.7** | Increased from 0.5 for diversity |
| Debate Neutral Agent | gemini-2.0-flash | 0.3 | Unchanged |
| Cross-Examination | gemini-3-flash-preview | 0.3 | Unchanged |
| Trading Planner | gemini-3-flash-preview | 0.4 | Unchanged |
| Pattern Validator | None (Rule-Based) | — | <1ms deterministic |
| MetaLearner | gemini-2.0-flash | 0.2 | Lowered thresholds: 20 trades / 10% win rate |

---

## v6.0 Data Flow Diagram

```
Market Data (Fyers API)
     |
     v
+----+----+     +-----------------+
| Regime  |     | News Analyzer   |  (Parallel, Wave 1)
| Engine  |     | (flash model)   |
+----+----+     +--------+--------+
     |                   |
     v                   v
+----+----+     +--------+--------+
| Pattern |     | MetaLearner     |  (Conditional)
| Memory  |     | (20 trade gate) |
+----+----+     +--------+--------+
     |                   |
     +--------+----------+
              |
              v
     +--------+--------+
     | Debate Protocol  |  (Wave 2)
     | Bull/Bear/Neutral|
     | temp=0.7         |
     | diversity score  |
     +--------+---------+
              |
              v
     +--------+---------+
     | Rule Validator    |  (Wave 3)
     | Trading Planner   |
     | Calibrator        |
     +--------+---------+
              |
              v
     +--------+---------+     +-------------------+
     | DirectionalRouter |---->| LongExecutor      |  LONG
     | (v6 NEW)         |     +-------------------+
     |                   |
     |                   |---->+-------------------+
     |                   |     | ShortExecutor     |  SHORT
     |                   |     | BUY_PUT / SELL_   |
     |                   |     | CALL / BEAR_SPREAD|
     |                   |     | Paper mode (15)   |
     +-------------------+     +-------------------+
              |
              v
     +--------+---------+
     | CircuitBreakerV2  |  (Pre-trade gate)
     | 3-layer halt      |
     | Resume @50% size  |
     +--------+---------+
              |
              v
     +--------+---------+
     | TradeOutcomeLogger|  (JSONL)
     | Full context log  |
     +--------+---------+
              |
              v
     +--------+---------+     +-------------------+
     | Pattern Memory    |     | AdaptiveCalibrator|
     | .update_outcome() |     | .update_weights() |
     +-------------------+     +-------------------+
              |                         |
              v                         v
     +--------+---------+     +--------+---------+
     | Better pattern    |     | Recalibrated     |
     | matching next     |     | signal weights   |
     | cycle             |     | next cycle       |
     +-------------------+     +------------------+
```

---

## Portfolio & Risk

- **Capital**: 10,00,000 INR
- **Risk per Trade**: 1.5% of portfolio (15,000 INR)
- **Max Portfolio Risk**: 3.0%
- **Max Positions**: 6
- **Max Sector Allocation**: 35%
- **Min R:R Ratio**: 1.5:1
- **Default Stop Loss**: 2% from entry
- **v6.0 CircuitBreakerV2**: 3 consecutive losses / 5% daily loss / 10% max drawdown → halt; resume at 50% size
- **v6.0 SHORT Paper Mode**: First 15 SHORT trades are paper-only (logged but not executed)
- **v6.0 SHORT Filters**: Open Interest > 1000, Bid-Ask Spread < 2%

---

## Tracked Symbols (51 NSE Stocks)

RELIANCE, TCS, HDFCBANK, ICICIBANK, SBIN, INFY, BAJFINANCE, KOTAKBANK, AXISBANK, HCLTECH, WIPRO, LT, TATASTEEL, JSWSTEEL, HINDALCO, TATAMOTORS, MARUTI, SUNPHARMA, ITC, ULTRACEMCO, TITAN, and 30 more across Banking, IT, Metals, Auto, Pharma, FMCG, Energy, and Infrastructure sectors.

---

## v5.0 Features (Retained)

All v5.0 features remain active in v6.0:

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Async Parallel LLM Execution** | Wave-based `asyncio.gather` — cycle time: 2m12s → ~55s |
| 2 | **Rule-Based PatternValidator** | Pure Python, zero LLM calls — validation: ~30s → <1ms |
| 3 | **7-Step Chain-of-Thought Prompts** | Structured JSON reasoning for deeper analysis |
| 4 | **Pattern Memory Bank** | DuckDB historical pattern matching with few-shot learning |
| 5 | **6-Signal Confidence Calibration** | Weighted multi-signal scoring (now adaptive via v6) |
| 6 | **Self-Reflection Loop** | Post-trade analysis feedback |
| 7 | **Adaptive Prompting** | Complexity-based model/depth selection |
| 8 | **API Call Deduplication** | Eliminates duplicate Fyers calls |
| 9 | **Conditional MetaLearner Skip** | Skip below threshold (now 20 trades / 10% win rate) |
| 10 | **Regime Cache** | TTL-based caching with VIX/DMA invalidation |

---

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_v6_integration.py` | **103** | All v6 modules: RuleRegimeClassifier, RegimeArbiter, RegimeTransitionDetector, RegimeAccuracyTracker, ShortExecutor, DirectionalRouter, CircuitBreakerV2, TradeOutcomeLogger, AdaptiveCalibrator, DebateEngine diversity, cycle management, end-to-end pipeline |
| `test_v5_integration.py` | **40** | All v5 features: async execution, debate protocol, pattern memory, calibration, rule validation |

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v3.2 | 2025-Q3 | 8-agent swing engine, LeadCoordinator |
| v4.0 | 2025-Q4 | F&O agents (HERMES, THETA, DELTA), LLM intelligence layer |
| v4.1 | 2026-Q1 | News intelligence, directional F&O advisor, cross-examiner |
| v4.2 | 2026-Q1 | Cross-examiner gate (AVOID/WAIT/REDUCE_SIZE/PROCEED) |
| v4.3 | 2026-Q1 | PHOENIX pre-momentum recovery radar |
| v5.0 | 2026-03 | Async 3-wave execution, debate protocol, pattern memory, confidence calibration |
| v5.1 | 2026-04 | Windows asyncio fix, RAM management, centralized model routing |
| **v6.0** | **2026-04-17** | **Closed-loop learning: RegimeArbiter, ShortExecutor, CircuitBreakerV2, TradeOutcomeLogger, AdaptiveCalibrator, debate diversity, cycle management** |

---

## License

MIT License — ROX Trading Systems

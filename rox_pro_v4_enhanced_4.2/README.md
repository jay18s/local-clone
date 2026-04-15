# ROX PROVEN EDGE ENGINE v5.0

> Multi-Agent LLM-Powered Indian Market Trading System  
> 15 Agents | 7 LLM Modules | Async Wave-Based Execution | Gemini API

## What's New in v5.0

v5.0 is a major upgrade over v4.0 with **10 critical improvements** addressing performance, accuracy, and intelligence:

| # | Upgrade | Impact |
|---|---------|--------|
| 1 | **Async Parallel LLM Execution** — Wave-based `asyncio.gather` | Cycle time: 2m12s → ~55s |
| 2 | **Rule-Based PatternValidator** — Pure Python, zero LLM calls | Validation: ~30s → <1ms |
| 3 | **7-Step Chain-of-Thought Prompts** — Structured JSON reasoning | Deeper analysis quality |
| 4 | **Bull/Bear Debate Protocol** — Multi-perspective adversarial analysis | Prevents echo chambers |
| 5 | **Pattern Memory Bank** — SQLite historical pattern matching | Few-shot learning from history |
| 6 | **6-Signal Confidence Calibration** — Weighted multi-signal scoring | Reliable confidence scores |
| 7 | **Self-Reflection Loop** — Post-trade analysis feedback | Continuous improvement |
| 8 | **Adaptive Prompting** — Complexity-based model/depth selection | Cost optimization |
| 9 | **API Call Deduplication** — Eliminates duplicate Fyers calls | Fewer API errors |
| 10 | **Conditional MetaLearner Skip** — Skip below 50-trade threshold | Eliminates noise |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ROX Engine v5.0                          │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  ORION   │  │  VESPER  │  │  KAIRO   │  │ SENTINEL │    │
│  │  (Tech)  │  │  (Flow)  │  │(Sentiment)│ │ (Deriv)  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  NEXUS   │  │PRUDENCE  │  │ CATALYST │  │ OPTIMUS  │    │
│  │(Fundmnt) │  │  (Risk)  │  │ (Events) │  │  (F&O)   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  HERMES  │  │  THETA   │  │  DELTA   │  F&O Agents   │
│  │  (Exec)  │  │(Greeks)  │  │(Settle)  │                │
│  └──────────┘  └──────────┘  └──────────┘                  │
│                                                             │
│  ┌──────────── 7 LLM Intelligence Modules ────────────┐     │
│  │ RegimeDetector │ NewsAnalyzer │ CrossExaminer   │     │
│  │ TradingPlanner │ PatternValidator │ MetaLearner  │     │
│  │ OptionsStrategist │ HistoryAnalyzer               │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  ┌──────────── v5.0 Reasoning Layer ─────────────────┐     │
│  │ CoT Prompts │ Debate Engine │ Pattern Memory      │     │
│  │ Confidence Calibrator │ Rule Validator │ Adaptive   │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rox_pro_v5_enhanced_5.0/
├── main_v5_pipeline.py          # v5.0 async 3-wave orchestrator (MAIN ENTRY)
├── main.py                      # v4.0 production entry point
├── main_production.py           # Production runner with Fyers API
├── coordinator.py               # LeadCoordinator + FnoCoordinator + UnifiedCoordinator
├── config.py                    # v4.0 configuration (MarketRegime, SystemConfig, etc.)
├── config_v5.py                 # v5.0 unified configuration (EngineConfig)
├── api_server.py                # FastAPI dashboard server
├── fyers_login.py               # Fyers OAuth login
├── setup.py                     # Package installer (v5.0.0)
├── requirements.txt             # Dependencies
│
├── reasoning_v5/                # v5.0 Reasoning Layer (NEW)
│   ├── __init__.py
│   ├── data_classes.py          # Signal, RegimeResult, NewsResult, TradePlan, etc.
│   ├── cot_prompts.py           # 7-step Chain-of-Thought prompt builders
│   ├── debate_engine.py         # Bull/Bear Debate Protocol + Cross-examination
│   ├── pattern_memory.py        # SQLite Pattern Memory Bank
│   ├── confidence_calibrator.py # 6-Signal weighted calibrator
│   ├── rule_validator.py        # Deterministic signal validator (<1ms)
│   └── adaptive_and_cache.py    # Adaptive prompt selector + regime cache
│
├── agents/                      # All Trading Agents
│   ├── orion.py, vesper.py, kairo.py, sentinel.py, nexus.py
│   ├── prudence.py, catalyst.py, optimus.py
│   ├── phoenix_agent.py, nocturnal_agent.py
│   ├── hermes_agent.py, theta_agent.py, delta_agent.py
│   ├── fno_brain_extension.py, directional_option_advisor.py
│   ├── news_core.py, strategy_builders.py, ai_brain.py
│   ├── base_agent.py            # Base class with ReAct mixin
│   ├── calibration/             # Agent calibration store
│   └── llm/                     # 7 LLM Intelligence Modules
│       ├── async_client.py      # Async Gemini client with semaphore + retry
│       ├── base_llm_agent.py    # Base class with dual SDK support
│       ├── llm_regime_detector.py
│       ├── llm_news_analyzer.py
│       ├── llm_cross_examiner.py
│       ├── llm_pattern_validator.py
│       ├── llm_trading_planner.py
│       ├── llm_meta_learner.py
│       ├── llm_options_strategist.py
│       └── llm_history_analyzer.py
│
├── infrastructure/              # Data Feed & Fyers Integration
│   ├── data_feed.py, coordinator.py, cache.py
│   ├── fno_instrument_manager.py, greeks_calculator.py
│   ├── margin_calculator.py, option_chain_stream.py
│   ├── historical_data_manager.py, websocket_handler.py
│   ├── mwpl_monitor.py, physical_settlement_manager.py
│   └── event_bus.py
│
├── execution/                   # Order Execution
│   ├── fno_execution_engine.py, order_manager.py
│   ├── execution_algorithms.py, slippage_control.py
│   └── __init__.py
│
├── data/                        # Data Management
│   ├── data_manager.py, fyers_fetcher.py, trade_logger.py
│   ├── pattern_database.py, scorecard.py, macro_fetcher.py
│   └── calibration/, meta_learning/
│
├── monitoring/                  # Risk Management
│   ├── risk_monitor.py, circuit_breaker.py
│   ├── performance_filter.py, risk_dashboard.py
│   └── __init__.py
│
├── ml_pipeline/                 # ML Pattern Recognition
│   ├── streaming_indicators.py, feature_engineering.py
│   ├── pattern_recognition.py, ml_models.py
│   └── __init__.py
│
├── alerts/                      # Notification System
│   ├── alert_manager.py, channels.py
│   └── __init__.py
│
├── pipeline/                    # v4.0 Processing Pipeline
│   ├── pipeline.py, __init__.py
│
├── utils/                       # Utilities
│   ├── helpers.py, platform_utils.py
│   └── __init__.py
│
├── tests/                       # Test Suite
│   ├── test_v5_integration.py   # v5.0 comprehensive tests (40 tests)
│   ├── test_optimus.py, test_greeks_calculator.py, test_fno_extension.py
│   └── __init__.py
│
├── scripts/                     # Maintenance Scripts
│   ├── daily_reconcile.py, weekly_meta_analysis.py
│
└── logs/, data/                 # Runtime directories
```

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

### 3. Run v5.0 Pipeline

```bash
python main_v5_pipeline.py
```

### 4. Run Tests

```bash
python tests/test_v5_integration.py
```

### 5. Run v4.0 Production (with Fyers)

```bash
python main_production.py
```

## v5.0 Execution Flow (3-Wave Async)

```
┌─────────────────────────────────────────────────┐
│ PRE-FLIGHT: Complexity Assessment               │
│ → AdaptivePromptSelector.assess_complexity()   │
│ → Determine CoT steps, debate rounds, model    │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│ WAVE 1: Regime + News (PARALLEL)                │
│ → RegimeDetector (CoT 7-step prompt)           │
│ → NewsImpactAnalyzer (flash model)             │
│ → Pattern Memory Bank (SQLite lookup)           │
│ → Conditional MetaLearner (skip <50 trades)     │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│ WAVE 2: Debate Protocol                         │
│ → Bull thesis (Gemini Pro)                      │
│ → Bear thesis (Gemini Pro)                      │
│ → Cross-examination synthesis                    │
│ → Agreement percentage calculation               │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│ WAVE 3: Validation + Planning                   │
│ → Rule-Based Validator (<1ms, no LLM)           │
│ → Trading Planner (only if signals pass)        │
│ → Confidence Calibration (6-signal weighted)     │
│ → Final Assembly + Decision                     │
└─────────────────────────────────────────────────┘
```

## Configuration

All settings are in `config_v5.py`. Key parameters:

```python
# Portfolio
initial_capital = ₹10,00,000
risk_per_trade_pct = 1.5%
max_positions = 6

# Gemini Models
model_pro = "gemini-2.5-flash-preview-05-20"
model_flash = "gemini-2.0-flash"

# Calibration Weights (sum to 1.0)
debate_agreement = 0.25
pattern_match = 0.20
technical_alignment = 0.20
volume_confirmation = 0.15
regime_consistency = 0.10
anti_consensus = 0.10
```

## Gemini Model Strategy

| Module | Model | Reason |
|--------|-------|--------|
| Regime Detection | gemini-2.5-flash-preview | Pro/Flash based on complexity |
| News Analysis | gemini-2.0-flash | Speed + factual extraction |
| Cross-Examination | gemini-2.5-flash-preview | Deep reasoning needed |
| Debate (Bull/Bear) | gemini-2.5-flash-preview | Diverse perspectives |
| Trading Planner | gemini-2.5-flash-preview | Critical decisions |
| Pattern Validator | None (Rule-Based) | <1ms deterministic |
| MetaLearner | gemini-2.0-flash | Weekly, low priority |

## Portfolio & Risk

- **Capital**: ₹10,00,000
- **Risk per Trade**: 1.5% of portfolio (₹15,000)
- **Max Portfolio Risk**: 3.0%
- **Max Positions**: 6
- **Max Sector Allocation**: 35%
- **Min R:R Ratio**: 1.5:1
- **Default Stop Loss**: 2% from entry

## Tracked Symbols (51 NSE Stocks)

RELIANCE, TCS, HDFCBANK, ICICIBANK, SBIN, INFY, BAJFINANCE, KOTAKBANK, AXISBANK, HCLTECH, WIPRO, LT, TATASTEEL, JSWSTEEL, HINDALCO, TATAMOTORS, MARUTI, SUNPHARMA, ITC, ULTRACEMCO, TITAN, and 31 more.

## License

MIT License — ROX Trading Systems

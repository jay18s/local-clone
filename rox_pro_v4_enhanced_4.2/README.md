# ROX Proven Edge Engine v4.0 Unified

**A production-grade 11-agent algorithmic trading system for Indian equities and F&O markets.**

This release merges two previously separate codebases:
- **rox_v32_enhanced** — 8-agent swing trading engine (ORION, VESPER, KAIRO, SENTINEL, NEXUS, PRUDENCE, CATALYST, OPTIMUS)
- **rox_pro_v4** — F&O specialist extension (HERMES, THETA, DELTA) with Black-76 Greeks, MWPL monitoring, physical settlement compliance

---

## Architecture

```
UnifiedCoordinator
├── LeadCoordinator  (v3.2 swing engine)
│   ├── ORION     — Technical Analysis (price action, S/R, patterns)
│   ├── VESPER    — FII/DII Flow Analysis
│   ├── KAIRO     — Sentiment (news, social, analyst)
│   ├── SENTINEL  — Derivatives Analysis (PCR, OI walls, VIX)
│   ├── NEXUS     — Fundamental Analysis (PE, quality scores)
│   ├── PRUDENCE  — Risk Management & Position Sizing
│   ├── CATALYST  — Event Calendar (expiries, results, macro)
│   └── OPTIMUS   — F&O Weekly Expiry (options signal generation)
│
└── FnoCoordinator  (v4.0 specialists)
    ├── HERMES    — Order Execution & Slippage Tracking
    ├── THETA     — Portfolio Greeks Management & Hedging
    └── DELTA     — Physical Settlement & SEBI Compliance
```

### Key Infrastructure

| Module | Description |
|--------|-------------|
| `infrastructure/greeks_calculator.py` | Black-76 model: delta/gamma/theta/vega/rho, IV solver |
| `infrastructure/option_chain_stream.py` | Real-time PCR, max pain, OI walls, expiry detection |
| `infrastructure/physical_settlement_manager.py` | ITM detection, delivery capital, settlement blocking |
| `infrastructure/fno_instrument_manager.py` | NSE contract lookup, lot sizes, strike spacing |
| `infrastructure/margin_calculator.py` | SPAN + exposure margin estimation |
| `infrastructure/mwpl_monitor.py` | Market-Wide Position Limits (SEBI disclosure tracking) |
| `infrastructure/fno_strategy_builders.py` | Iron Condor, Calendar Spread, Bull/Bear Spread, Collar |
| `ml_pipeline/` | Pattern recognition, feature engineering, streaming indicators |
| `execution/` | Smart order routing, slippage control |
| `monitoring/` | Circuit breaker, risk dashboard, alert manager |
| `data/` | Data manager, trade logger, pattern database, agent scorecard |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (synthetic data, no broker needed)
python main.py --mode demo

# Paper trading (requires market data source)
python main.py --mode paper --portfolio-value 500000

# Backtest
python main.py --mode backtest --start-date 2024-01-01 --end-date 2024-12-31

# Live trading (broker API keys required in .env)
python main.py --mode live
```

---

## Configuration

Copy `.env.example` → `.env` and fill in your credentials:

```bash
# Broker (Fyers or Zerodha — only one needed)
FYERS_API_KEY=...
FYERS_ACCESS_TOKEN=...
FYERS_ENABLED=true

# Portfolio
PORTFOLIO_VALUE=1000000

# Risk overrides
MAX_OPTIONS_EXPOSURE=0.10
MAX_OPTION_PREMIUM=0.02

# Logging
LOG_LEVEL=INFO
```

All parameters are also configurable via `config.py` → `SystemConfig`.

---

## Project Structure

```
rox_pro_v4_unified/
├── main.py                    # Unified entry point (all modes)
├── coordinator.py             # UnifiedCoordinator, LeadCoordinator, FnoCoordinator
├── config.py                  # Merged SystemConfig (11 agents, FnoConfig, FnoRiskLimits)
├── agents/
│   ├── __init__.py            # All 11 agents exported
│   ├── base_agent.py          # BaseAgent, AgentVerdict, AgentReport
│   ├── orion.py  vesper.py  kairo.py  sentinel.py  nexus.py
│   ├── prudence.py  catalyst.py  optimus.py
│   ├── hermes_agent.py        # (v4.0) Execution
│   ├── theta_agent.py         # (v4.0) Greeks Management
│   └── delta_agent.py         # (v4.0) Settlement/Compliance
├── infrastructure/
│   ├── __init__.py            # Lazy-import facade for all infra
│   ├── greeks_calculator.py   # Black-76 model
│   ├── option_chain_stream.py
│   ├── physical_settlement_manager.py
│   ├── fno_instrument_manager.py  # (v4.0)
│   ├── margin_calculator.py       # (v4.0)
│   ├── mwpl_monitor.py            # (v4.0)
│   ├── fno_strategy_builders.py   # (v4.0) IronCondor, Calendar…
│   ├── data_feed.py  websocket_handler.py  event_bus.py
│   ├── cache.py  data_normalizer.py  historical_data_manager.py
│   └── config.py  coordinator.py
├── data/
│   ├── data_manager.py  pattern_database.py
│   ├── scorecard.py  trade_logger.py
│   └── __init__.py
├── ml_pipeline/               # Machine learning pipeline
├── execution/                 # Order management, algorithms
├── monitoring/                # Circuit breaker, risk dashboard, alerts
├── alerts/                    # Multi-channel alert manager
├── pipeline/                  # Data pipeline components
├── core/                      # Shared utilities, logging
├── utils/                     # Helper utilities
├── scripts/                   # Maintenance scripts
├── tests/
│   ├── test_greeks_calculator.py   # Black-76 validation
│   ├── test_fno_extension.py       # F&O components (57 tests)
│   └── test_optimus.py             # OPTIMUS agent tests
├── requirements.txt
└── README.md
```

---

## Testing

```bash
# Full test suite
python -m pytest tests/ -v

# Individual suites
python -m pytest tests/test_greeks_calculator.py -v    # 5 tests
python -m pytest tests/test_fno_extension.py -v        # 57 tests
python -m pytest tests/test_optimus.py -v
```

---

## Integration Notes

### v3.2 → v4.0 Migration

| v3.2 | v4.0 Unified |
|------|-------------|
| `coordinator.LeadCoordinator` | `coordinator.UnifiedCoordinator` (recommended) |
| `coordinator.LeadCoordinator` | still available as `LeadCoordinator` (unchanged) |
| `config.SystemConfig` (8 agents) | `config.SystemConfig` (11 agents, `FnoConfig` added) |
| `infrastructure/__init__` | now exports all v4.0 infra via lazy loading |

### DailyTradingPlan Additions (v4.0)
```python
plan.portfolio_greeks       # {'delta', 'gamma', 'theta', 'vega', 'num_positions'}
plan.active_alerts          # list of alert strings from THETA, DELTA, MWPL
plan.settlement_obligations # list of settlement obligation dicts
```

---

## Risk Disclaimer

This software is for **educational and research purposes only**.
Nothing in this codebase constitutes financial advice.
Always comply with your local regulations and your broker's policies.
SEBI regulations apply to all F&O positions in Indian markets.

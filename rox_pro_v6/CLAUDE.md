# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run in different modes
python main.py --mode demo          # Synthetic data, no broker needed
python main.py --mode paper --portfolio-value 500000
python main.py --mode backtest --start-date 2024-01-01 --end-date 2024-12-31
python main.py --mode live         # Requires broker API keys in .env

# Run tests
python -m pytest tests/ -v
python -m pytest tests/test_greeks_calculator.py -v    # 5 tests
python -m pytest tests/test_fno_extension.py -v        # 57 tests
python -m pytest tests/test_optimus.py -v
```

## Architecture

This is an 11-agent algorithmic trading system for Indian equities and F&O markets.

```
UnifiedCoordinator
├── LeadCoordinator (v3.2 swing engine)
│   ├── ORION     — Technical Analysis
│   ├── VESPER    — FII/DII Flow Analysis
│   ├── KAIRO     — Sentiment Analysis
│   ├── SENTINEL  — Derivatives Analysis
│   ├── NEXUS     — Fundamental Analysis
│   ├── PRUDENCE  — Risk Management & Position Sizing
│   ├── CATALYST  — Event Calendar
│   └── OPTIMUS   — F&O Weekly Expiry
│
└── FnoCoordinator (v4.0 specialists)
    ├── HERMES    — Order Execution & Slippage Tracking
    ├── THETA     — Portfolio Greeks Management & Hedging
    └── DELTA     — Physical Settlement & SEBI Compliance
```

Key infrastructure modules in `infrastructure/`: greeks_calculator.py (Black-76 model), option_chain_stream.py, physical_settlement_manager.py, margin_calculator.py, mwpl_monitor.py, fno_strategy_builders.py.

## Configuration

Copy `.env.example` → `.env`. Required for live trading: FYERS_API_KEY, FYERS_ACCESS_TOKEN, FYERS_ENABLED (or Zerodha equivalent). All parameters also configurable via `config.py` → `SystemConfig`.

## Migration Notes

v3.2 → v4.0: Use `coordinator.UnifiedCoordinator` instead of `LeadCoordinator`. `config.SystemConfig` now includes 11 agents and `FnoConfig`. `infrastructure/__init__` exports all v4.0 infra via lazy loading.

## Live Deployment & Validation

To test the ROX Engine in live mode with real-time predictions:

1. Run the engine in live mode:
   ```bash
   python main.py --mode live
   ```
2. Capture the output in `logs/live.log`.
3. Compare predictions with live market data from your broker API.
4. Validate accuracy by:
   - Checking profit/loss against executed trades.
   - Reviewing any alerts or order modifications.
   - Using EdgeQuake to identify hidden patterns influencing predictions.

For detailed debugging, see the `debug/` directory.
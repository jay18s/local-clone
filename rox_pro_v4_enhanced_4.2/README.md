# ROX Proven Edge Engine v4.2 Enhanced — Fixes v2

## Version: 4.2 + Fixes 7-11 (2026-04-17)

Multi-Agent LLM-Powered Indian Market Trading System with closed-loop learning.

---

## What's in This Package

All uploaded v1-fixed files (Fixes 1-6, already applied) plus **5 new modules** implementing Fixes 7-11 from `rox_fixes_v2_master_prompt.md`.

### V1 Fixes (Already Applied — Preserved)
| Fix | File | Description |
|-----|------|-------------|
| FIX-JSON-01 | `agents/fno_brain_extension.py` | JSON retry (2x), schema validator, 8192 tokens |
| FIX-ARBITER-01 | `reasoning/regime_arbiter.py` | Semantic proximity, directional merge, 7-case logic |
| FIX-EXAMINE-03 | `agents/llm/llm_cross_examiner.py` | Soft-allow under WAIT, RANGE_BOUND override |
| FIX-EXAMINE-03 | `coordinator.py` | Gate respects soft_allow for equity + F&O |
| FIX-DYNAMIC-RISK | `agents/directional_option_advisor.py` | Conviction-based risk: 3.5%/4.5%/5.5% |
| FIX-IC-TRIGGER | `execution/ic_trigger_monitor.py` | IC auto-fire on range sustain |

### V2 Fixes (New in This Release)
| Fix | File | Description |
|-----|------|-------------|
| FIX 7 | `execution/theta_time_stop.py` | **NEW** — Theta time-stop for straddles (3-day max, 40% decay threshold, expiry risk) |
| FIX 8 | `monitoring/correlation_risk.py` | **NEW** — Correlation risk checker (blocks >0.90, reduces cap >0.75) |
| FIX 9 | `tests/test_fixes_v1.py` | **NEW** — 34 unit tests covering all v1+v2 modules |
| FIX 10 | `scripts/run_simulation.py` | **NEW** — 5-day simulation framework with success criteria |
| FIX 11 | `utils/rejection_logger.py` | **NEW** — Structured rejection logging with daily summary |

### Integration Changes
- **`coordinator.py`** — Added theta time-stop check + rejection summary logging after F&O suggestions
- **`directional_option_advisor.py`** — Added correlation risk check before returning suggestions

---

## File Structure

```
rox_pro_v4_enhanced_4.2/
├── coordinator.py                      ← MODIFIED (v1 + v2 integration)
├── directional_option_advisor.py       ← MODIFIED (v1 + v2 integration)
├── regime_arbiter.py                   ← MODIFIED (v1)
├── llm_cross_examiner.py               ← MODIFIED (v1)
├── fno_brain_extension.py              ← MODIFIED (v1)
├── ic_trigger_monitor.py               ← NEW (v1)
├── rox_fixes_v2_master_prompt.md       — Fix specifications
│
├── execution/
│   ├── __init__.py
│   └── theta_time_stop.py              ← NEW (FIX 7)
├── monitoring/
│   ├── __init__.py
│   └── correlation_risk.py             ← NEW (FIX 8)
├── utils/
│   ├── __init__.py
│   └── rejection_logger.py             ← NEW (FIX 11)
├── tests/
│   ├── __init__.py
│   └── test_fixes_v1.py                ← NEW (FIX 9)
└── scripts/
    ├── __init__.py
    └── run_simulation.py               ← NEW (FIX 10)
```

---

## Fix 7 — Theta Time Stop

Tracks open straddle/strangle positions and exits when:

| Rule | Condition | Priority |
|------|-----------|----------|
| EXPIRY_RISK | DTE ≤ 2 | HIGH |
| THETA_DECAY_THRESHOLD | theta_eaten ≥ 40% of entry cost | HIGH |
| MAX_HOLD_DAYS | trading days held ≥ 3 | MEDIUM |

**Usage:**
```python
from execution.theta_time_stop import ThetaTimeStop

stop = ThetaTimeStop()
stop.register_position(
    index="NIFTY", strategy="LONG_STRADDLE",
    entry_cost_per_unit=10000, lot_size=25,
    daily_theta=-150, breakeven_low=22000, breakeven_high=22500,
)
signals = stop.check_exits(spot_prices={"NIFTY": 22300})
```

---

## Fix 8 — Correlation Risk Check

Prevents doubling exposure on correlated indices.

| Correlation | Action |
|-------------|--------|
| > 0.90 | BLOCK (cannot add second position) |
| > 0.75 | REDUCE_CAP (combined max_loss to 6%, not 9%) |
| ≤ 0.75 | ALLOW |

**Usage:**
```python
from monitoring.correlation_risk import CorrelationRiskChecker

checker = CorrelationRiskChecker()
result = checker.check("BANKNIFTY", ["NIFTY"])
# result.action == "BLOCK" (correlation 0.92)
```

---

## Fix 9 — Unit Tests

34 tests across 8 test classes. Run with:
```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from tests.test_fixes_v1 import *
# ... run test classes
"
```

**Coverage:** FNOBrain JSON, Regime Arbiter, Cross-Examiner, Dynamic Risk, Strategy Selection, Theta Time Stop, Correlation Risk, Rejection Logger.

---

## Fix 10 — Simulation Framework

5-day simulation with 25 cycles total. Validates:
- ✅ 2-4 trades/day average
- ✅ 0 paralysis days
- ✅ 0 RANGE_BOUND fallbacks from regime arbiter
- ✅ Soft-allow fires at least once

```bash
python3 scripts/run_simulation.py
```

---

## Fix 11 — Enhanced Rejection Logging

Structured log format:
```
[REJECT] module=RULE_VALIDATOR | symbol=ADANIPORTS | strategy=BUY_CE | reason="R:R below 1.5:1" | threshold=">=1.5" | actual="0.63" | action=REJECTED
```

Daily summary:
```
[DAILY-REJECT-SUMMARY] date=2026-04-17 | total_rejections=14
  By module: RULE_VALIDATOR=4 | CHECKLIST=3 | CROSS_EXAMINER=3
  By symbol: NIFTY=3 | ADANIPORTS=2 | AXISBANK=2
  By reason: R:R_fail=4 | max_loss_breach=3 | WAIT_hard=2
```

**Usage:**
```python
from utils.rejection_logger import log_rejection, get_rejection_logger

log_rejection(
    module="RULE_VALIDATOR",
    symbol="ADANIPORTS", strategy="BUY_CE",
    reason="R:R below 1.5:1", threshold=">=1.5", actual="0.63",
    action="REJECTED",
)

# End of day summary
print(get_rejection_logger().get_daily_summary())
```

---

## Configuration (Unchanged from v4.2)

| Parameter | Value |
|-----------|-------|
| Initial Capital | ₹10,00,000 |
| Risk per Trade | 1.5% (dynamic: 3.5-5.5%) |
| Max Positions | 6 |
| Theta Max Hold Days | 3 |
| Theta Decay Threshold | 40% |
| Correlation Block | > 0.90 |
| Correlation Reduce | > 0.75 |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v4.2 | 2026-Q1 | Cross-examiner gate, PHOENIX recovery radar |
| v4.2+v1 | 2026-04-17 | JSON retry, arbiter merge, soft-allow, dynamic risk, IC trigger |
| **v4.2+v2** | **2026-04-17** | **Theta time stop, correlation risk, rejection logger, tests, simulation** |

---

## License

MIT License — ROX Trading Systems

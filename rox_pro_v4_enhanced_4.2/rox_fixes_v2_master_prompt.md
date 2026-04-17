# MASTER PROMPT — ROX Engine Fixes v2 (Fixes 7-11)
# Generated: 2026-04-17 18:20 IST
# ================================================
# Feed this entire file to a fresh session to continue implementation.
# All v1 fixes (1-6) are ALREADY APPLIED to the codebase.

---

## CONTEXT

You are continuing fixes to the ROX trading engine located at:
```
/root/.openclaw/workspace/local-clone/rox_pro_v4_enhanced_4.2/
```

The engine is a multi-agent Indian F&O trading system with:
- 13 agents (8 swing + 3 F&O + PHOENIX + NOCTURNAL)
- 7 LLM modules (Gemini)
- v6.0 closed-loop learning (RegimeArbiter, ShortExecutor, CircuitBreakerV2)
- Portfolio: ₹10,00,000

## V1 FIXES ALREADY APPLIED (DO NOT RE-IMPLEMENT)

| Fix | File | What was done |
|-----|------|---------------|
| FIX-JSON-01 | `agents/fno_brain_extension.py` | JSON retry (2x), schema validator, 8192 tokens |
| FIX-ARBITER-01 | `reasoning/regime_arbiter.py` | Semantic proximity, directional merge, 7-case logic |
| FIX-EXAMINE-03 | `agents/llm/llm_cross_examiner.py` | Soft-allow under WAIT, RANGE_BOUND override |
| FIX-EXAMINE-03 | `coordinator.py` | Gate respects soft_allow for equity + F&O |
| FIX-DYNAMIC-RISK | `agents/directional_option_advisor.py` | Conviction-based risk: 3.5%/4.5%/5.5% |
| FIX-IC-TRIGGER | `execution/ic_trigger_monitor.py` | NEW — IC auto-fire on range sustain |

---

## YOUR TASK: IMPLEMENT FIXES 7-11

### FIX 7 — THETA TIME STOP

**Problem:** Long straddles (BANKNIFTY, SENSEX) have theta burn of ₹82-106/day on ₹23.8k cost. After 3 days with no move, the position is a guaranteed loser.

**Requirements:**
1. Create `execution/theta_time_stop.py` — NEW module
2. Track open straddle/strangle positions with:
   - entry_time, entry_cost, daily_theta, breakeven_range
   - max_hold_trading_days (default: 3)
   - theta_burn_threshold (exit if theta > 40% of entry cost)
3. `check_exits(spot_prices, current_time)` → returns list of positions to exit
4. Integration point: call from the main loop after `generate_trading_plan()`
5. Log each exit with: reason, P&L, hold_duration, theta_eaten

**Data structure:**
```python
@dataclass
class ThetaPosition:
    index: str
    strategy: str  # "LONG_STRADDLE" | "LONG_STRANGLE"
    entry_time: datetime
    entry_cost_per_unit: float
    lot_size: int
    daily_theta: float  # negative number (decay per day)
    breakeven_low: float
    breakeven_high: float
    max_hold_days: int = 3
    strike: float
    expiry: date
    dte_at_entry: int

@dataclass
class ThetaExitSignal:
    position: ThetaPosition
    reason: str  # "MAX_HOLD_DAYS" | "THETA_DECAY_THRESHOLD" | "EXPIRY_RISK"
    current_spot: float
    hold_days: int
    theta_eaten: float  # total theta decay since entry
    unrealized_pnl: float
    urgency: str  # "HIGH" | "MEDIUM" | "LOW"
```

**Exit rules:**
- `MAX_HOLD_DAYS`: trading days since entry >= max_hold_days (default 3)
- `THETA_DECAY_THRESHOLD`: abs(theta_eaten) >= entry_cost * 0.40
- `EXPIRY_RISK`: dte <= 2 (gamma risk explodes)
- Priority: EXPIRY_RISK > THETA_DECAY > MAX_HOLD_DAYS

---

### FIX 8 — CORRELATION RISK CHECK

**Problem:** BANKNIFTY and SENSEX are ~85% correlated. Taking straddles on both doubles exposure to the same macro move without doubling the hedge.

**Requirements:**
1. Create `monitoring/correlation_risk.py` — NEW module
2. Hardcoded correlation matrix (updated quarterly):
```python
INDEX_CORRELATIONS = {
    ("NIFTY", "BANKNIFTY"):    0.92,
    ("NIFTY", "SENSEX"):       0.95,
    ("NIFTY", "FINNIFTY"):     0.90,
    ("NIFTY", "BANKEX"):       0.88,
    ("BANKNIFTY", "SENSEX"):   0.85,
    ("BANKNIFTY", "BANKEX"):   0.93,
    ("SENSEX", "BANKEX"):      0.87,
    ("FINNIFTY", "BANKNIFTY"): 0.88,
}
```
3. `CorrelationRiskChecker` class:
   - `check(new_position, existing_positions)` → `CorrelationRiskResult`
   - If combined correlation > 0.75: reduce combined max_loss cap to 6% (not 9%)
   - If combined correlation > 0.90: block second position entirely
4. Integration: call from `DirectionalOptionAdvisor.advise()` before returning suggestions
5. Log: which pair triggered, correlation value, action taken

**Data structures:**
```python
@dataclass
class CorrelationRiskResult:
    is_correlated: bool
    pair: Tuple[str, str]
    correlation: float
    action: str  # "ALLOW" | "REDUCE_CAP" | "BLOCK"
    combined_risk_cap: float  # adjusted cap if REDUCE_CAP
    message: str
```

---

### FIX 9 — UNIT TESTS

**Requirements:**
Create `tests/test_fixes_v1.py` with these test functions:

```python
# 1. FNOBrain JSON parsing
def test_fno_json_validation_valid():
    """Valid JSON passes schema validation"""
    
def test_fno_json_validation_missing_field():
    """Missing required field fails validation"""

def test_fno_json_validation_bad_iv_regime():
    """Invalid iv_regime value fails validation"""

def test_fno_json_validation_bad_risk_score():
    """risk_score outside 1-10 fails validation"""

def test_fno_json_retry_on_invalid():
    """Retry mechanism activates on parse failure"""

# 2. Regime Arbiter
def test_arbiter_same_regime():
    """Same regime → BOTH_AGREE"""

def test_arbiter_same_direction_merge():
    """CAUTIOUS + CORRECTION → DIRECTIONAL_MERGE (not RANGE_BOUND)"""

def test_arbiter_cross_spectrum():
    """BULLISH vs BEARISH → higher confidence wins or bearish bias"""

def test_arbiter_llm_degraded():
    """LLM accuracy < 0.55 → RULE_OVERRIDE"""

def test_arbiter_conflict_default_reduced():
    """RANGE_BOUND is last resort, not frequent"""

# 3. Cross-Examiner
def test_cross_examiner_soft_allow_range_bound():
    """RANGE_BOUND + WAIT → soft_allow when conviction >= 55"""

def test_cross_examiner_no_soft_allow_consolidation():
    """CONSOLIDATION + WAIT → no soft_allow"""

def test_cross_examiner_regime_override_bear():
    """BEAR + SHORT 50% → REDUCE_SIZE"""

def test_cross_examiner_wait_paralysis():
    """3 consecutive WAITs → escalate to REDUCE_SIZE"""

# 4. Dynamic Risk
def test_dynamic_risk_high_conviction():
    """conviction=70 → risk_pct=5.5%"""

def test_dynamic_risk_medium_conviction():
    """conviction=60 → risk_pct=4.5%"""

def test_dynamic_risk_low_conviction():
    """conviction=50 → risk_pct=3.5%"""

def test_checklist_passes_high_conviction():
    """NIFTY straddle at 70% conviction passes risk check"""

# 5. Strategy Selection
def test_straddle_selected_no_consensus_low_iv():
    """NO_CONSENSUS + LOW IV → Long Straddle"""

def test_iron_condor_selected_high_iv():
    """HIGH IV → Iron Condor"""

def test_spread_selected_non_trending():
    """Non-trending + LONG → Bull Spread (not naked call)"""
```

Each test must be runnable with `pytest tests/test_fixes_v1.py -v`.
Mock any external dependencies (LLM calls, Fyers API).
Tests should import directly from the source modules.

---

### FIX 10 — SIMULATION TEST FRAMEWORK

**Requirements:**
Create `scripts/run_simulation.py` that:

1. Loads 5 days of historical market data (can use mock/replay data)
2. Runs the full engine pipeline for each day:
   - Data fetch (mocked)
   - Regime detection (real rule-based, mocked LLM)
   - Agent consensus (real)
   - Cross-examination (mocked with deterministic responses)
   - F&O suggestion generation (real)
   - Rule validation (real)
3. Tracks per-cycle:
```python
@dataclass
class SimulationCycle:
    day: int
    cycle: int
    regime: str
    regime_source: str
    consensus: str
    consensus_strength: str
    cross_examiner_rec: str
    soft_allow: bool
    equity_setups: List[str]  # stock names
    equity_setups_passed: int
    fno_suggestions: List[str]  # strategy names
    fno_suggestions_passed: int
    trades_executed: int
    rejection_reasons: List[str]
```
4. Prints summary:
```
=== SIMULATION SUMMARY (5 days) ===
Total cycles:        25
Trades generated:    12
Trades executed:     8
Trades blocked:      4
  - WAIT (hard):     1
  - R:R fail:        2
  - Max loss breach: 1
Avg trades/day:      1.6
Paralysis days:      0  ← must be 0
Regime conflicts:    3
  → RANGE_BOUND:     0  ← must be 0 (Fix 2 working)
  → MERGE:           3
```
5. SUCCESS CRITERIA:
   - At least 2-4 trades generated per day (not necessarily executed)
   - Zero full-day paralysis (0 trades in a day)
   - Zero RANGE_BOUND fallbacks from regime arbiter
   - Soft-allow fires at least once

Mock data should simulate:
- Day 1: VIX=14, Nifty above 20-DMA (BULLISH regime)
- Day 2: VIX=17, Nifty below 200-DMA (CAUTIOUS/CORRECTION conflict)
- Day 3: VIX=22, gap down (BEAR regime)
- Day 4: VIX=16, range-bound (CONSOLIDATION)
- Day 5: VIX=13, recovery (MILD_BULL)

---

### FIX 11 — ENHANCED REJECTION LOGGING

**Problem:** When a trade is rejected, the logs don't clearly show WHY, WHICH module blocked it, and WHAT threshold failed.

**Requirements:**
1. Create a `RejectionLogger` utility class (can be in `utils/rejection_logger.py` or inline)
2. Every rejection point must call:
```python
log_rejection(
    module: str,        # "RULE_VALIDATOR" | "CROSS_EXAMINER" | "CHECKLIST" | "CORRELATION" | etc.
    symbol: str,        # "NIFTY" | "ADANIPORTS" | etc.
    strategy: str,      # "LONG_STRADDLE" | "BUY_CE" | etc.
    reason: str,        # "R:R below threshold" 
    threshold: str,     # "R:R >= 1.5"
    actual_value: str,  # "R:R = 0.63"
    action: str,        # "REJECTED" | "WATCH_ONLY" | "REDUCED_SIZE"
    conviction: int = 0,
    extra: dict = None,
)
```
3. Output format (structured log line):
```
[REJECT] module=RULE_VALIDATOR | symbol=ADANIPORTS | strategy=BUY_CE | reason="R:R below 1.5:1" | threshold=">=1.5" | actual="0.63" | action=REJECTED
```
4. Add rejection logging to ALL existing rejection points:
   - `rule_validator.py` — R:R, RSI, volume failures
   - `directional_option_advisor.py` — checklist failures (max_loss, delta, liquidity, max_pain)
   - `coordinator.py` — cross-examiner AVOID/WAIT, regime transition blocks
   - `llm_cross_examiner.py` — AVOID, WAIT (soft-allow or not)
   - `correlation_risk.py` — correlation blocks (new)
   - `theta_time_stop.py` — time-stop exits (new)
5. End-of-day summary log:
```
[DAILY-REJECT-SUMMARY] date=2026-04-17 | total_rejections=14
  By module: RULE_VALIDATOR=4 | CHECKLIST=3 | CROSS_EXAMINER=3 | CORRELATION=2 | THETA_EXIT=2
  By symbol: NIFTY=3 | ADANIPORTS=2 | AXISBANK=2 | BANKNIFTY=2 | SENSEX=2 | DRREDDY=1 | BHARTIARTL=1 | FINNIFTY=1
  By reason: R:R_fail=4 | max_loss_breach=3 | WAIT_hard=2 | correlation_block=2 | theta_time_stop=2 | delta_fail=1
```

---

## FILE STRUCTURE AFTER ALL FIXES

```
rox_pro_v4_enhanced_4.2/
├── agents/
│   ├── fno_brain_extension.py          ← MODIFIED (v1)
│   ├── directional_option_advisor.py   ← MODIFIED (v1)
│   └── llm/
│       └── llm_cross_examiner.py       ← MODIFIED (v1)
├── reasoning/
│   └── regime_arbiter.py               ← MODIFIED (v1)
├── execution/
│   ├── ic_trigger_monitor.py           ← NEW (v1)
│   └── theta_time_stop.py             ← NEW (v2) ← YOU CREATE
├── monitoring/
│   └── correlation_risk.py            ← NEW (v2) ← YOU CREATE
├── utils/
│   └── rejection_logger.py            ← NEW (v2) ← YOU CREATE
├── tests/
│   └── test_fixes_v1.py               ← NEW (v2) ← YOU CREATE
├── scripts/
│   └── run_simulation.py              ← NEW (v2) ← YOU CREATE
└── coordinator.py                      ← MODIFIED (v1 + v2 integration)
```

---

## INTEGRATION POINTS FOR FIXES 7-8

### Fix 7 (Theta Time Stop) integration into coordinator.py:

In `UnifiedCoordinator.generate_trading_plan()`, after F&O suggestions are generated:
```python
# FIX-THETA: Check existing positions for time-stop exits
try:
    from execution.theta_time_stop import ThetaTimeStop
    if not hasattr(self, '_theta_stop'):
        self._theta_stop = ThetaTimeStop()
    
    # Register new straddle positions from this cycle
    for sug in (plan.fno_suggestions.suggestions if plan.fno_suggestions else []):
        if sug.strategy in ("LONG_STRADDLE", "LONG_STRANGLE") and sug.proceed:
            self._theta_stop.register_position(sug, market_data)
    
    # Check existing positions
    exit_signals = self._theta_stop.check_exits(
        spot_prices={idx: market_data.get(f"{idx.lower()}_price", 0) 
                     for idx in ("NIFTY","BANKNIFTY","SENSEX","FINNIFTY","BANKEX")},
        current_time=datetime.now()
    )
    for signal in exit_signals:
        logger.warning(
            f"[THETA-EXIT] {signal.position.index} {signal.position.strategy} → "
            f"{signal.reason} | hold={signal.hold_days}d | theta_eaten=₹{signal.theta_eaten:,.0f} | "
            f"P&L=₹{signal.unrealized_pnl:,.0f} | urgency={signal.urgency}"
        )
        plan.action_items.append(f"EXIT {signal.position.index} straddle: {signal.reason}")
except Exception as e:
    logger.debug(f"Theta time stop check skipped: {e}")
```

### Fix 8 (Correlation Risk) integration into directional_option_advisor.py:

In `DirectionalOptionAdvisor.advise()`, before returning suggestions:
```python
# FIX-CORRELATION: Check for correlated index exposure
try:
    from monitoring.correlation_risk import CorrelationRiskChecker
    checker = CorrelationRiskChecker()
    checked_suggestions = []
    for sug in suggestions:
        result = checker.check(sug.index, [s.index for s in checked_suggestions])
        if result.action == "BLOCK":
            logger.warning(f"[CORRELATION] BLOCKED: {sug.index} — {result.message}")
            skipped.append(f"{sug.index}: {result.message}")
        elif result.action == "REDUCE_CAP":
            logger.info(f"[CORRELATION] REDUCE_CAP: {sug.index} — {result.message}")
            checked_suggestions.append(sug)
        else:
            checked_suggestions.append(sug)
    suggestions = checked_suggestions
except ImportError:
    pass  # correlation module not yet created
```

---

## CONSTRAINTS

1. **Do not modify v1-fixed files** except for integration hooks in coordinator.py and directional_option_advisor.py
2. **All new modules must be self-contained** — import from existing config/agents, don't create new config files
3. **No new pip dependencies** — use only stdlib + existing project deps
4. **Use existing logging patterns** — `logging.getLogger("rox.xxx")`
5. **Match existing code style** — dataclasses, type hints, docstrings
6. **Tests must be runnable** with `pytest` — mock external calls

---

## OUTPUT FORMAT

After implementing all fixes, return:

1. **Each new file's full content** (theta_time_stop.py, correlation_risk.py, rejection_logger.py, test_fixes_v1.py, run_simulation.py)
2. **Diff of integration changes** to coordinator.py and directional_option_advisor.py
3. **Test results** — run `pytest tests/test_fixes_v1.py -v` and include output
4. **Simulation results** — run `python scripts/run_simulation.py` and include output
5. **Fix summary table** — before/after for each fix

---

## SUCCESS CRITERIA (ALL MUST PASS)

- [ ] Theta time stop exits straddles after 3 days or 40% theta decay
- [ ] Correlation checker blocks or caps correlated index pairs
- [ ] All 15+ unit tests pass
- [ ] Simulation shows 2-4 trades/day, 0 paralysis days
- [ ] Every rejection has a structured log line with module/threshold/actual
- [ ] Zero RANGE_BOUND fallbacks from regime arbiter in simulation
- [ ] No new pip dependencies
- [ ] All existing tests still pass (v6 integration tests)

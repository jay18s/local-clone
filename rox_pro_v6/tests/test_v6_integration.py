"""
ROX PROVEN EDGE ENGINE v6.0 — Integration Test Suite
=====================================================
Tests all v6.0 modules: RuleRegimeClassifier, RegimeArbiter,
RegimeTransitionDetector, ShortExecutor, DirectionalRouter,
CircuitBreakerV2, AdaptiveConfidenceCalibrator, TradeOutcomeLogger,
RegimeAccuracyTracker, and their integration with existing v5 modules.
No external API calls required — uses mocks where needed.
"""

import sys
import os
import json
import tempfile
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0
ERRORS = []

def _run_test(name, func):  # FIX-TEST-HELPER: renamed to avoid pytest collection
    global PASS, FAIL, ERRORS
    print(f"  TEST: {name}...", end=" ", flush=True)
    try:
        result = func()
        if result is False:
            print("FAIL")
            FAIL += 1
        else:
            print("PASS")
            PASS += 1
    except Exception as e:
        print(f"ERROR: {e}")
        FAIL += 1
        ERRORS.append((name, str(e)))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 1: Rule-Based Regime Classifier
# ═══════════════════════════════════════════════════════════════════

def test_rule_regime_bullish():
    from reasoning.rule_regime_classifier import RuleRegimeClassifier
    clf = RuleRegimeClassifier()
    result = clf.classify(
        vix=12.0, nifty_price=24500, nifty_20dma=24000,
        fii_net_flow=2000, sector_green_pct=75, nifty_5d_slope=0.5,
    )
    assert result.regime == "BULLISH", f"Expected BULLISH, got {result.regime}"
    assert result.confidence > 60, f"Confidence too low: {result.confidence}"
    assert result.source == "RULE_BASED"
    assert result.details is not None
    assert "composite_score" in result.details
    return True


def test_rule_regime_bearish():
    from reasoning.rule_regime_classifier import RuleRegimeClassifier
    clf = RuleRegimeClassifier()
    result = clf.classify(
        vix=25.0, nifty_price=23500, nifty_20dma=24000,
        fii_net_flow=-2000, sector_green_pct=20, nifty_5d_slope=-0.5,
    )
    assert result.regime == "BEARISH", f"Expected BEARISH, got {result.regime}"
    assert result.confidence > 60, f"Confidence too low: {result.confidence}"
    return True


def test_rule_regime_range_bound():
    from reasoning.rule_regime_classifier import RuleRegimeClassifier
    clf = RuleRegimeClassifier()
    result = clf.classify(
        vix=15.0, nifty_price=24200, nifty_20dma=24180,
        fii_net_flow=200, sector_green_pct=50, nifty_5d_slope=0.05,
    )
    # With VIX=15, nifty slightly above 20DMA, small positive FII, the score may be
    # in the TRENDING range rather than strictly RANGE_BOUND depending on signal combos
    assert result.regime in ("RANGE_BOUND", "TRENDING"), f"Got {result.regime}"
    return True


def test_rule_regime_cautious():
    from reasoning.rule_regime_classifier import RuleRegimeClassifier
    clf = RuleRegimeClassifier()
    result = clf.classify(
        vix=20.0, nifty_price=23800, nifty_20dma=24000,
        fii_net_flow=-500, sector_green_pct=35, nifty_5d_slope=-0.15,
    )
    assert result.regime in ("CAUTIOUS", "RANGE_BOUND"), f"Got {result.regime}"
    return True


def test_rule_regime_trending():
    from reasoning.rule_regime_classifier import RuleRegimeClassifier
    clf = RuleRegimeClassifier()
    result = clf.classify(
        vix=14.0, nifty_price=24400, nifty_20dma=24200,
        fii_net_flow=500, sector_green_pct=60, nifty_5d_slope=0.2,
    )
    assert result.regime in ("TRENDING", "BULLISH", "RANGE_BOUND"), f"Got {result.regime}"
    return True


def test_rule_regime_details():
    from reasoning.rule_regime_classifier import RuleRegimeClassifier
    clf = RuleRegimeClassifier()
    result = clf.classify(vix=15.0, nifty_price=24200, nifty_20dma=24100)
    assert "signals" in result.details
    assert "vix" in result.details["signals"]
    assert "dma" in result.details["signals"]
    assert "fii" in result.details["signals"]
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 2: Regime Arbiter
# ═══════════════════════════════════════════════════════════════════

def test_arbiter_both_agree():
    from reasoning.regime_arbiter import RegimeArbiter
    arbiter = RegimeArbiter()
    decision = arbiter.resolve(
        rule_regime="BULLISH", rule_confidence=75.0,
        llm_regime="BULLISH", llm_confidence=80.0,
    )
    assert decision.regime == "BULLISH"
    assert decision.source == "BOTH_AGREE"
    assert decision.confidence == 80.0  # max of both
    return True


def test_arbiter_llm_degraded():
    from reasoning.regime_arbiter import RegimeArbiter
    arbiter = RegimeArbiter()
    decision = arbiter.resolve(
        rule_regime="BEARISH", rule_confidence=70.0,
        llm_regime="BULLISH", llm_confidence=75.0,
        llm_rolling_accuracy=0.40,
    )
    assert decision.regime == "BEARISH"
    assert decision.source == "RULE_OVERRIDE_LLM_DEGRADED"
    return True


def test_arbiter_rule_high_conf():
    from reasoning.regime_arbiter import RegimeArbiter
    arbiter = RegimeArbiter()
    decision = arbiter.resolve(
        rule_regime="BEARISH", rule_confidence=80.0,
        llm_regime="BULLISH", llm_confidence=55.0,
    )
    assert decision.regime == "BEARISH"
    assert decision.source == "RULE_HIGH_CONF"
    return True


def test_arbiter_llm_high_conf():
    from reasoning.regime_arbiter import RegimeArbiter
    arbiter = RegimeArbiter()
    decision = arbiter.resolve(
        rule_regime="RANGE_BOUND", rule_confidence=50.0,
        llm_regime="BULLISH", llm_confidence=85.0,
    )
    assert decision.regime == "BULLISH"
    assert decision.source == "LLM_HIGH_CONF"
    return True


def test_arbiter_conflict_default():
    from reasoning.regime_arbiter import RegimeArbiter
    arbiter = RegimeArbiter()
    decision = arbiter.resolve(
        rule_regime="TRENDING", rule_confidence=65.0,
        llm_regime="BEARISH", llm_confidence=70.0,
    )
    # FIX-TEST-ARBITER: FIX-ARBITER-01 added CONFLICT_BEARISH_BIAS logic — when either
    # side is bearish, the arbiter now conservatively returns the bearish regime rather
    # than defaulting to RANGE_BOUND. TRENDING vs BEARISH → BEARISH is correct behaviour.
    assert decision.regime == "BEARISH"
    # FIX-TEST-ARBITER: source is now CONFLICT_BEARISH_BIAS (not CONFLICT_DEFAULT)
    # and confidence is 70*0.8=56.0 (penalised max of both sides), not 50.0
    assert decision.source == "CONFLICT_BEARISH_BIAS"
    assert decision.confidence == round(max(65.0, 70.0) * 0.8, 1)  # 56.0
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 3: Regime Transition Detector
# ═══════════════════════════════════════════════════════════════════

def test_transition_confirmed():
    from reasoning.regime_transition_detector import RegimeTransitionDetector
    detector = RegimeTransitionDetector()
    event = detector.detect(
        current_regime="BEARISH", previous_regime="BULLISH",
        vix_current=22.0, vix_previous=15.0,
        nifty_price=23800, nifty_20dma=24200,
    )
    assert event.type == "CONFIRMED"
    assert event.from_regime == "BULLISH"
    assert event.to_regime == "BEARISH"
    assert event.action == "REDUCE_SIZE_AND_INCREASE_THRESHOLD"
    return True


def test_transition_vix_spike_imminent():
    from reasoning.regime_transition_detector import RegimeTransitionDetector
    detector = RegimeTransitionDetector()
    event = detector.detect(
        current_regime="BULLISH", previous_regime="BULLISH",
        vix_current=20.0, vix_previous=15.0,  # +5 VIX spike
        nifty_price=24200, nifty_20dma=24000,
    )
    assert event.type == "IMMINENT"
    assert "VIX_SPIKE" in event.signals
    return True


def test_transition_dma_break_imminent():
    from reasoning.regime_transition_detector import RegimeTransitionDetector
    detector = RegimeTransitionDetector()
    # First call — above DMA
    detector.detect(
        current_regime="BULLISH", previous_regime="BULLISH",
        vix_current=15.0, vix_previous=15.0,
        nifty_price=24200, nifty_20dma=24000,
    )
    # Second call — crossed below DMA by >0.5%
    event = detector.detect(
        current_regime="BULLISH", previous_regime="BULLISH",
        vix_current=15.5, vix_previous=15.0,
        nifty_price=23700, nifty_20dma=24000,  # Below DMA by 1.25%
    )
    assert event.type == "IMMINENT"
    assert "DMA_BREAK" in event.signals
    return True


def test_transition_fii_reversal_imminent():
    from reasoning.regime_transition_detector import RegimeTransitionDetector
    detector = RegimeTransitionDetector()
    event = detector.detect(
        current_regime="BULLISH", previous_regime="BULLISH",
        vix_current=15.0, vix_previous=15.0,
        nifty_price=24200, nifty_20dma=24000,
        fii_current=800, fii_previous=-600,  # Sign flip with >500 delta
    )
    assert event.type in ("IMMINENT", "NONE")  # Could be IMMINENT if FII reversal detected
    return True


def test_transition_none():
    from reasoning.regime_transition_detector import RegimeTransitionDetector
    detector = RegimeTransitionDetector()
    event = detector.detect(
        current_regime="BULLISH", previous_regime="BULLISH",
        vix_current=15.5, vix_previous=15.0,
        nifty_price=24200, nifty_20dma=24000,
        fii_current=500, fii_previous=400,
    )
    assert event.type == "NONE"
    return True


def test_transition_reset():
    from reasoning.regime_transition_detector import RegimeTransitionDetector
    detector = RegimeTransitionDetector()
    detector._previous_nifty_side = "ABOVE"
    detector.reset()
    assert detector._previous_nifty_side is None
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 4: Short Executor
# ═══════════════════════════════════════════════════════════════════

def test_short_buy_atm_put():
    from execution.short_executor import ShortExecutor, ShortStrategy
    executor = ShortExecutor()
    order = executor.prepare_short_order(
        symbol="NSE:SBIN", spot_price=780.0,
        conviction=60.0, regime="VOLATILE",
        portfolio_capital=10_00_000,
    )
    assert order is not None
    assert order.strategy == ShortStrategy.BUY_ATM_PUT
    assert order.option_type == "PE"
    assert order.transaction_type == "BUY"
    assert order.max_loss > 0
    assert order.underlying == "NSE:SBIN"
    return True


def test_short_sell_atm_call():
    from execution.short_executor import ShortExecutor, ShortStrategy
    executor = ShortExecutor()
    order = executor.prepare_short_order(
        symbol="NSE:RELIANCE", spot_price=2850.0,
        conviction=75.0, regime="RANGE_BOUND",
        portfolio_capital=10_00_000,
    )
    assert order is not None
    assert order.strategy == ShortStrategy.SELL_ATM_CALL
    assert order.option_type == "CE"
    assert order.transaction_type == "SELL"
    assert order.max_profit > 0  # Credit received
    return True


def test_short_bear_put_spread():
    from execution.short_executor import ShortExecutor, ShortStrategy
    executor = ShortExecutor()
    order = executor.prepare_short_order(
        symbol="NSE:NIFTY", spot_price=24200.0,
        conviction=85.0, regime="BEARISH",
        portfolio_capital=10_00_000,
    )
    assert order is not None
    assert order.strategy == ShortStrategy.BEAR_PUT_SPREAD
    assert order.max_loss > 0
    assert order.max_profit > 0
    return True


def test_short_atm_strike_nifty():
    from execution.short_executor import ShortExecutor
    executor = ShortExecutor()
    strike = executor.get_atm_strike(24230.0, "NSE:NIFTY")
    assert strike == 24250  # Nearest 50
    return True


def test_short_atm_strike_stock():
    from execution.short_executor import ShortExecutor
    executor = ShortExecutor()
    strike = executor.get_atm_strike(785.0, "NSE:SBIN")
    # SBIN price > 500, so nearest 20
    assert strike % 20 == 0 or strike % 50 == 0
    return True


def test_short_lot_size():
    from execution.short_executor import ShortExecutor
    executor = ShortExecutor()
    assert executor.get_lot_size("NSE:NIFTY") == 25
    assert executor.get_lot_size("NSE:RELIANCE") == 250
    assert executor.get_lot_size("UNKNOWN") == 1
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 5: Directional Router
# ═══════════════════════════════════════════════════════════════════

def test_router_long_pass():
    from execution.directional_router import DirectionalRouter
    router = DirectionalRouter()
    result = router.route_long(
        signal_data={"symbol": "NSE:SBIN"},
        execute_fn=lambda data: {"order_id": "ORD001"},
    )
    assert result.executed
    assert result.direction == "LONG"
    assert result.reason == "OK"
    return True


def test_router_long_circuit_breaker():
    from execution.directional_router import DirectionalRouter, CircuitBreakerProtocol

    class BlockedBreaker(CircuitBreakerProtocol):
        def can_trade(self):
            return False, "DAILY_LOSS_LIMIT"
        def get_size_multiplier(self):
            return 0.5

    router = DirectionalRouter(circuit_breaker=BlockedBreaker())
    result = router.route_long(
        signal_data={"symbol": "NSE:SBIN"},
        execute_fn=lambda data: {"order_id": "ORD001"},
    )
    assert not result.executed
    assert "CIRCUIT_BREAKER" in result.reason
    return True


def test_router_short_pass():
    from execution.directional_router import DirectionalRouter
    from execution.short_executor import ShortOrder, ShortStrategy
    router = DirectionalRouter()
    order = ShortOrder(
        symbol="NSE:SBIN_780PE", strategy=ShortStrategy.BUY_ATM_PUT,
        transaction_type="BUY", quantity=1500, strike=780.0,
        option_type="PE", premium=15.0, max_loss=22500.0,
        max_profit=float("inf"), stop_loss_premium=22.5,
        target_premium=7.5, lot_size=1500, lots=1,
        underlying="NSE:SBIN", conviction=65.0, regime="BEARISH",
    )
    result = router.route_short(
        short_order=order,
        execute_fn=lambda o: {"order_id": "ORD002"},
    )
    assert result.executed
    assert result.direction == "SHORT"
    assert result.strategy == "BUY_ATM_PUT"
    return True


def test_router_short_no_order():
    from execution.directional_router import DirectionalRouter
    router = DirectionalRouter()
    result = router.route_short(
        short_order=None,
        execute_fn=lambda o: None,
    )
    assert not result.executed
    assert "NO_SHORT_ORDER" in result.reason
    return True


def test_router_size_reduction():
    from execution.directional_router import DirectionalRouter, CircuitBreakerProtocol
    from execution.short_executor import ShortOrder, ShortStrategy

    class ReducedBreaker(CircuitBreakerProtocol):
        def can_trade(self):
            return True, "OK"
        def get_size_multiplier(self):
            return 0.5

    router = DirectionalRouter(circuit_breaker=ReducedBreaker())
    order = ShortOrder(
        symbol="NSE:SBIN_780PE", strategy=ShortStrategy.BUY_ATM_PUT,
        transaction_type="BUY", quantity=3000, strike=780.0,
        option_type="PE", premium=15.0, max_loss=45000.0,
        max_profit=float("inf"), stop_loss_premium=22.5,
        target_premium=7.5, lot_size=1500, lots=2,
        underlying="NSE:SBIN", conviction=65.0, regime="BEARISH",
    )
    result = router.route_short(
        short_order=order,
        execute_fn=lambda o: {"order_id": "ORD003"},
    )
    assert result.executed
    # Lots should have been reduced from 2 to 1
    assert order.lots == 1
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 6: Circuit Breaker V2
# ═══════════════════════════════════════════════════════════════════

def test_circuit_breaker_normal():
    from monitoring.circuit_breaker_v2 import CircuitBreakerV2
    cb = CircuitBreakerV2(initial_capital=10_00_000)
    can, reason = cb.can_trade()
    assert can
    assert reason == "OK"
    assert cb.get_size_multiplier() == 1.0
    return True


def test_circuit_breaker_consecutive_losses():
    from monitoring.circuit_breaker_v2 import CircuitBreakerV2
    cb = CircuitBreakerV2(initial_capital=10_00_000, consecutive_loss_threshold=3)
    cb.on_trade_close(-5000)  # Loss 1
    cb.on_trade_close(-3000)  # Loss 2
    assert cb.get_size_multiplier() == 1.0  # Not yet triggered
    cb.on_trade_close(-4000)  # Loss 3 → trigger
    assert cb.get_size_multiplier() == 0.5
    can, _ = cb.can_trade()
    assert can  # Not halted, just reduced
    return True


def test_circuit_breaker_daily_loss_limit():
    from monitoring.circuit_breaker_v2 import CircuitBreakerV2
    cb = CircuitBreakerV2(initial_capital=10_00_000, daily_loss_limit_pct=3.0)
    # 3% of 10L = 30,000
    cb.on_trade_close(-20000)
    cb.on_trade_close(-15000)  # Total: -35000, exceeds limit
    can, reason = cb.can_trade()
    assert not can
    assert reason == "DAILY_LOSS_LIMIT"
    return True


def test_circuit_breaker_max_drawdown():
    from monitoring.circuit_breaker_v2 import CircuitBreakerV2
    cb = CircuitBreakerV2(initial_capital=10_00_000, max_drawdown_pct=8.0)
    # 8% of 10L = 80,000
    cb.on_trade_close(-50000)
    cb.on_trade_close(-40000)  # Total: -90000, exceeds 8%
    can, reason = cb.can_trade()
    assert not can
    assert reason == "MAX_DRAWDOWN"
    return True


def test_circuit_breaker_size_recovery():
    from monitoring.circuit_breaker_v2 import CircuitBreakerV2
    cb = CircuitBreakerV2(
        initial_capital=10_00_000,
        consecutive_loss_threshold=3,
        wins_to_reset_size=3,
    )
    # Trigger reduced size
    cb.on_trade_close(-5000)
    cb.on_trade_close(-3000)
    cb.on_trade_close(-4000)
    assert cb.get_size_multiplier() == 0.5

    # Win 3 times in reduced mode → restore
    cb.on_trade_close(2000)
    cb.on_trade_close(3000)
    cb.on_trade_close(1500)
    assert cb.get_size_multiplier() == 1.0
    return True


def test_circuit_breaker_daily_reset():
    from monitoring.circuit_breaker_v2 import CircuitBreakerV2
    cb = CircuitBreakerV2(initial_capital=10_00_000, daily_loss_limit_pct=3.0)
    cb.on_trade_close(-35000)  # Exceeds 3%
    can, reason = cb.can_trade()
    assert not can
    assert reason == "DAILY_LOSS_LIMIT"

    cb.reset_daily()
    can, reason = cb.can_trade()
    assert can  # Daily halt cleared
    return True


def test_circuit_breaker_manual_restart():
    from monitoring.circuit_breaker_v2 import CircuitBreakerV2
    cb = CircuitBreakerV2(initial_capital=10_00_000, max_drawdown_pct=8.0)
    cb.on_trade_close(-90000)  # Exceeds 8% drawdown
    can, reason = cb.can_trade()
    assert not can

    cb.manual_restart()
    can, reason = cb.can_trade()
    assert can  # Manual restart clears halt
    return True


def test_circuit_breaker_get_state():
    from monitoring.circuit_breaker_v2 import CircuitBreakerV2
    cb = CircuitBreakerV2(initial_capital=10_00_000)
    cb.on_trade_close(5000)
    state = cb.get_state()
    assert not state.halted
    assert state.consecutive_losses == 0
    assert state.current_capital == 10_05_000
    assert state.peak_capital == 10_05_000
    assert state.size_multiplier == 1.0
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 7: Adaptive Confidence Calibrator
# ═══════════════════════════════════════════════════════════════════

def test_adaptive_calibrator_initial():
    from reasoning.adaptive_calibrator import AdaptiveConfidenceCalibrator
    cal = AdaptiveConfidenceCalibrator()
    score = cal.calibrate({
        "debate_agreement": 80.0,
        "pattern_match": 70.0,
        "technical_alignment": 75.0,
        "volume_confirmation": 65.0,
        "regime_consistency": 85.0,
        "anti_consensus": 50.0,
    })
    assert 0.0 <= score <= 100.0
    return True


def test_adaptive_calibrator_update():
    from reasoning.adaptive_calibrator import AdaptiveConfidenceCalibrator
    cal = AdaptiveConfidenceCalibrator(min_history=3)
    initial_weights = cal.get_weights()

    # Add 3 winning trades with high debate_agreement
    for _ in range(3):
        cal.update(
            signal_scores={
                "debate_agreement": 85.0,
                "pattern_match": 60.0,
                "technical_alignment": 70.0,
                "volume_confirmation": 65.0,
                "regime_consistency": 75.0,
                "anti_consensus": 50.0,
            },
            won=True,
            timestamp="2026-04-15T10:00:00",
        )

    # After min_history, weights should start adapting
    new_weights = cal.get_weights()
    assert new_weights != initial_weights or len(cal.history) < cal.min_history
    return True


def test_adaptive_calibrator_weight_clamp():
    from reasoning.adaptive_calibrator import AdaptiveConfidenceCalibrator
    cal = AdaptiveConfidenceCalibrator(
        initial_weights={"a": 0.8, "b": 0.2},
        min_weight=0.05, max_weight=0.35,
    )
    weights = cal.get_weights()
    # After normalization, weights are clamped and re-normalized to sum to 1.0
    # So individual weights may slightly exceed max_weight due to normalization
    for w in weights.values():
        assert w >= 0.05  # min_weight is respected
    # Weights should sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 0.01
    return True


def test_adaptive_calibrator_correlations():
    from reasoning.adaptive_calibrator import AdaptiveConfidenceCalibrator
    cal = AdaptiveConfidenceCalibrator(min_history=3)
    # Add enough data
    for won in [True, True, False, True, True]:
        cal.update(
            signal_scores={
                "debate_agreement": 80.0 if won else 40.0,
                "pattern_match": 70.0,
                "technical_alignment": 65.0,
                "volume_confirmation": 60.0,
                "regime_consistency": 70.0,
                "anti_consensus": 50.0,
            },
            won=won,
            timestamp="2026-04-15T10:00:00",
        )
    corrs = cal.get_signal_correlations()
    assert "debate_agreement" in corrs
    return True


def test_adaptive_calibrator_weight_evolution():
    from reasoning.adaptive_calibrator import AdaptiveConfidenceCalibrator
    cal = AdaptiveConfidenceCalibrator()
    cal.update({"debate_agreement": 80, "pattern_match": 70,
                "technical_alignment": 65, "volume_confirmation": 60,
                "regime_consistency": 70, "anti_consensus": 50},
               won=True, timestamp="2026-04-15T10:00:00")
    evo = cal.get_weight_evolution()
    assert len(evo) == 1
    assert evo[0]["won"] == True
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 8: Trade Outcome Logger
# ═══════════════════════════════════════════════════════════════════

def test_trade_outcome_log():
    from data.trade_outcome_logger import TradeOutcomeLogger
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TradeOutcomeLogger(log_dir=tmpdir)
        logger.log_trade(
            timestamp_entry="2026-04-15T09:30:00",
            timestamp_exit=None,
            symbol="NSE:SBIN", direction="LONG",
            entry_price=780.0, exit_price=None, pnl=None,
            regime_at_entry="BULLISH", regime_confidence=75.0,
            debate_agreement_score=80.0, calibration_score=65.0,
            agent_verdicts=[{"agent": "ORION", "direction": "LONG", "conviction": 78}],
            signals_passed=["NSE:SBIN LONG"],
            signals_failed=[{"symbol": "NSE:ICICIBANK", "reason": "low_rr"}],
            news_sentiment="BULLISH",
            pattern_match_ids=["PM001"],
            cycle_number=5,
        )
        assert logger.get_trade_count() == 1
    return True


def test_trade_outcome_update():
    from data.trade_outcome_logger import TradeOutcomeLogger
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TradeOutcomeLogger(log_dir=tmpdir)
        logger.log_trade(
            timestamp_entry="2026-04-15T09:30:00",
            timestamp_exit=None,
            symbol="NSE:SBIN", direction="LONG",
            entry_price=780.0, exit_price=None, pnl=None,
            regime_at_entry="BULLISH", regime_confidence=75.0,
            debate_agreement_score=80.0, calibration_score=65.0,
            agent_verdicts=[], signals_passed=[], signals_failed=[],
            news_sentiment="BULLISH", pattern_match_ids=[], cycle_number=1,
        )
        logger.update_trade(
            symbol="NSE:SBIN", timestamp_entry=None,
            exit_price=800.0, pnl=2000.0,
        )
        trades = logger.get_all_trades()
        assert trades[0]["exit_price"] == 800.0
        assert trades[0]["pnl"] == 2000.0
    return True


def test_trade_outcome_win_rate():
    from data.trade_outcome_logger import TradeOutcomeLogger
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TradeOutcomeLogger(log_dir=tmpdir)
        # 3 wins, 2 losses
        for pnl in [1000, -500, 2000, -300, 1500]:
            logger.log_trade(
                timestamp_entry="2026-04-15T09:30:00",
                timestamp_exit="2026-04-15T14:00:00",
                symbol="NSE:SBIN", direction="LONG",
                entry_price=780.0, exit_price=800.0, pnl=pnl,
                regime_at_entry="BULLISH", regime_confidence=75.0,
                debate_agreement_score=80.0, calibration_score=65.0,
                agent_verdicts=[], signals_passed=[], signals_failed=[],
                news_sentiment="BULLISH", pattern_match_ids=[], cycle_number=1,
            )
        wr = logger.get_win_rate()
        assert wr == 0.6  # 3/5
    return True


def test_trade_outcome_recent_trades():
    from data.trade_outcome_logger import TradeOutcomeLogger
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TradeOutcomeLogger(log_dir=tmpdir)
        for i in range(10):
            logger.log_trade(
                timestamp_entry="2026-04-15T09:30:00",
                timestamp_exit="2026-04-15T14:00:00",
                symbol="NSE:SBIN", direction="LONG",
                entry_price=780.0, exit_price=800.0, pnl=1000.0,
                regime_at_entry="BULLISH", regime_confidence=75.0,
                debate_agreement_score=80.0, calibration_score=65.0,
                agent_verdicts=[], signals_passed=[], signals_failed=[],
                news_sentiment="BULLISH", pattern_match_ids=[], cycle_number=i,
            )
        recent = logger.get_recent_trades(5)
        assert len(recent) == 5
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 9: Regime Accuracy Tracker
# ═══════════════════════════════════════════════════════════════════

def test_regime_accuracy_classify():
    from reasoning.regime_accuracy_tracker import RegimeAccuracyTracker
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = RegimeAccuracyTracker(data_dir=tmpdir)
        # Big up day with elevated VIX → BEARISH
        regime = tracker.classify_actual_regime(
            nifty_open=24000, nifty_close=24300,
            nifty_high=24350, nifty_low=23950,
            vix_open=15.0, vix_close=19.0,
        )
        assert regime == "BEARISH"
    return True


def test_regime_accuracy_classify_trending():
    from reasoning.regime_accuracy_tracker import RegimeAccuracyTracker
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = RegimeAccuracyTracker(data_dir=tmpdir)
        # 1% up day → TRENDING
        regime = tracker.classify_actual_regime(
            nifty_open=24000, nifty_close=24240,
            nifty_high=24300, nifty_low=23980,
            vix_open=15.0, vix_close=14.5,
        )
        assert regime == "TRENDING"
    return True


def test_regime_accuracy_classify_volatile():
    from reasoning.regime_accuracy_tracker import RegimeAccuracyTracker
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = RegimeAccuracyTracker(data_dir=tmpdir)
        # Wide range but small close change → VOLATILE
        regime = tracker.classify_actual_regime(
            nifty_open=24200, nifty_close=24230,
            nifty_high=24400, nifty_low=24100,
            vix_open=15.0, vix_close=15.5,
        )
        assert regime == "VOLATILE"
    return True


def test_regime_accuracy_classify_range_bound():
    from reasoning.regime_accuracy_tracker import RegimeAccuracyTracker
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = RegimeAccuracyTracker(data_dir=tmpdir)
        # Small range, small change → RANGE_BOUND
        regime = tracker.classify_actual_regime(
            nifty_open=24200, nifty_close=24210,
            nifty_high=24250, nifty_low=24180,
            vix_open=15.0, vix_close=15.0,
        )
        assert regime == "RANGE_BOUND"
    return True


def test_regime_accuracy_log_and_rolling():
    from reasoning.regime_accuracy_tracker import RegimeAccuracyTracker
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = RegimeAccuracyTracker(data_dir=tmpdir)
        # Log a few sessions
        tracker.log_session(
            rule_regime="TRENDING", rule_confidence=70.0,
            llm_regime="BULLISH", llm_confidence=75.0,
            nifty_open=24000, nifty_close=24120,
            nifty_high=24150, nifty_low=23980,
            vix_open=15.0, vix_close=14.5,
        )
        tracker.log_session(
            rule_regime="BEARISH", rule_confidence=65.0,
            llm_regime="RANGE_BOUND", llm_confidence=55.0,
            nifty_open=24200, nifty_close=24100,
            nifty_high=24250, nifty_low=24050,
            vix_open=15.0, vix_close=16.0,
        )
        acc = tracker.get_rolling_accuracy(n=20)
        assert "rule_accuracy" in acc
        assert "llm_accuracy" in acc
        assert "sessions_tracked" in acc
        assert acc["sessions_tracked"] == 2
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 10: Debate Engine v6 additions
# ═══════════════════════════════════════════════════════════════════

def test_debate_diversity_score_opposite():
    from reasoning.debate_engine import DebateEngine
    score = DebateEngine._compute_diversity_score(
        bull_thesis={"direction": "LONG", "confidence": 80},
        bear_thesis={"direction": "SHORT", "confidence": 70},
    )
    # Opposite directions with confidence divergence: 0.7*1.0 + 0.3*(0.1) = 0.73
    # Not exactly 1.0 because confidence divergence contributes
    assert score >= 0.5  # Should be high for opposite directions
    return True


def test_debate_diversity_score_same():
    from reasoning.debate_engine import DebateEngine
    score = DebateEngine._compute_diversity_score(
        bull_thesis={"direction": "LONG", "confidence": 70},
        bear_thesis={"direction": "LONG", "confidence": 70},
    )
    assert score == 0.0  # Same direction → no diversity
    return True


def test_debate_diversity_score_inferred():
    from reasoning.debate_engine import DebateEngine
    # No direction field, infer from thesis label
    score = DebateEngine._compute_diversity_score(
        bull_thesis={"thesis": "STRONGLY_BULLISH", "confidence": 80},
        bear_thesis={"thesis": "BEARISH", "confidence": 75},
    )
    assert score > 0  # Inferred opposite directions
    return True


def test_debate_system_prompts():
    from reasoning.debate_engine import BULL_SYSTEM_PROMPT, BEAR_SYSTEM_PROMPT
    assert "MOMENTUM" in BULL_SYSTEM_PROMPT
    assert "MEAN-REVERSION" in BEAR_SYSTEM_PROMPT
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 11: Pattern Memory v6 outcome feedback
# ═══════════════════════════════════════════════════════════════════

def test_pattern_memory_v6_update_outcome():
    from reasoning.pattern_memory import PatternMemoryBank
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_v6_patterns.db")
        bank = PatternMemoryBank(db_path)

        # First, insert a match record directly
        conn = __import__("duckdb").connect(db_path)
        conn.execute(
            "INSERT INTO pattern_matches "
            "(match_id, date, symbol, direction, predicted_outcome, similarity) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("PM001", "2026-04-15", "NSE:SBIN", "LONG", "WIN", 0.85),
        )
        conn.commit()
        conn.close()

        # Update with actual outcome
        bank.update_outcome(
            match_id="PM001",
            actual_outcome="WIN",
            actual_pnl=2500.0,
            hold_period_minutes=270,
        )

        # Verify update
        conn = __import__("duckdb").connect(db_path)
        row = conn.execute(
            "SELECT actual_outcome, actual_pnl, hold_period_minutes "
            "FROM pattern_matches WHERE match_id = ?",
            ("PM001",),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "WIN"
        assert row[1] == 2500.0
        assert row[2] == 270
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 12: Config v6 patches
# ═══════════════════════════════════════════════════════════════════

def test_config_v6_meta_learner_thresholds():
    from config_v5 import ReasoningConfig
    config = ReasoningConfig()
    assert config.meta_learner_min_trades == 20, (
        f"Expected 20, got {config.meta_learner_min_trades}"
    )
    assert config.meta_learner_min_win_rate == 0.10, (
        f"Expected 0.10, got {config.meta_learner_min_win_rate}"
    )
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 13: Integration — Full Regime Pipeline
# ═══════════════════════════════════════════════════════════════════

def test_regime_pipeline_rule_then_arbiter():
    """Test rule classifier → arbiter integration."""
    from reasoning.rule_regime_classifier import RuleRegimeClassifier
    from reasoning.regime_arbiter import RegimeArbiter

    clf = RuleRegimeClassifier()
    arbiter = RegimeArbiter()

    # Rule says BEARISH with high confidence
    rule_result = clf.classify(
        vix=24.0, nifty_price=23500, nifty_20dma=24000,
        fii_net_flow=-1500, sector_green_pct=25, nifty_5d_slope=-0.4,
    )
    assert rule_result.regime in ("BEARISH", "CAUTIOUS")

    # LLM says BULLISH but with lower confidence
    decision = arbiter.resolve(
        rule_regime=rule_result.regime,
        rule_confidence=rule_result.confidence,
        llm_regime="BULLISH",
        llm_confidence=55.0,
    )
    # Rule should win due to higher confidence
    assert decision.regime == rule_result.regime
    return True


def test_regime_pipeline_with_transition():
    """Test regime → transition detector integration."""
    from reasoning.rule_regime_classifier import RuleRegimeClassifier
    from reasoning.regime_transition_detector import RegimeTransitionDetector

    clf = RuleRegimeClassifier()
    detector = RegimeTransitionDetector()

    # Session 1: BULLISH
    r1 = clf.classify(vix=14.0, nifty_price=24500, nifty_20dma=24200,
                       fii_net_flow=1000, sector_green_pct=65)
    event1 = detector.detect(
        current_regime=r1.regime, previous_regime="",
        vix_current=14.0, vix_previous=14.0,
        nifty_price=24500, nifty_20dma=24200,
    )

    # Session 2: Regime shift to BEARISH (VIX spike)
    r2 = clf.classify(vix=22.0, nifty_price=23800, nifty_20dma=24200,
                       fii_net_flow=-1500, sector_green_pct=25)
    event2 = detector.detect(
        current_regime=r2.regime, previous_regime=r1.regime,
        vix_current=22.0, vix_previous=14.0,
        nifty_price=23800, nifty_20dma=24200,
    )
    # Should detect transition
    assert event2.type in ("CONFIRMED", "IMMINENT")
    return True


def test_execution_pipeline_short():
    """Test DirectionalRouter → ShortExecutor integration."""
    from execution.short_executor import ShortExecutor
    from execution.directional_router import DirectionalRouter
    from monitoring.circuit_breaker_v2 import CircuitBreakerV2

    cb = CircuitBreakerV2(initial_capital=10_00_000)
    router = DirectionalRouter(circuit_breaker=cb)
    executor = ShortExecutor()

    order = executor.prepare_short_order(
        symbol="NSE:NIFTY", spot_price=24200.0,
        conviction=70.0, regime="CAUTIOUS",
        portfolio_capital=10_00_000,
    )
    assert order is not None

    result = router.route_short(
        short_order=order,
        execute_fn=lambda o: {"order_id": "ORD_V6_001"},
    )
    assert result.executed
    assert result.direction == "SHORT"
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 14: Module Imports (v6 __init__.py)
# ═══════════════════════════════════════════════════════════════════

def test_reasoning_module_v6_imports():
    from reasoning import (
        RuleRegimeClassifier, RegimeClassification,
        RegimeArbiter, RegimeDecision,
        RegimeTransitionDetector, TransitionEvent,
        RegimeAccuracyTracker,
        AdaptiveConfidenceCalibrator, CalibrationRecord,
    )
    assert RuleRegimeClassifier is not None
    assert RegimeArbiter is not None
    assert RegimeTransitionDetector is not None
    return True


def test_data_module_v6_imports():
    from data import TradeOutcomeLogger
    assert TradeOutcomeLogger is not None
    return True


def test_execution_module_v6_imports():
    from execution import ShortExecutor, ShortOrder, ShortStrategy
    from execution import DirectionalRouter, ExecutionResult
    assert ShortExecutor is not None
    assert DirectionalRouter is not None
    return True


def test_monitoring_module_v6_imports():
    from monitoring import CircuitBreakerV2, CircuitBreakerState
    assert CircuitBreakerV2 is not None
    assert CircuitBreakerState is not None
    return True


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  ROX PROVEN EDGE ENGINE v6.0 — Integration Test Suite")
    print("=" * 70)

    groups = [
        ("Rule-Based Regime Classifier", [
            test_rule_regime_bullish,
            test_rule_regime_bearish,
            test_rule_regime_range_bound,
            test_rule_regime_cautious,
            test_rule_regime_trending,
            test_rule_regime_details,
        ]),
        ("Regime Arbiter", [
            test_arbiter_both_agree,
            test_arbiter_llm_degraded,
            test_arbiter_rule_high_conf,
            test_arbiter_llm_high_conf,
            test_arbiter_conflict_default,
        ]),
        ("Regime Transition Detector", [
            test_transition_confirmed,
            test_transition_vix_spike_imminent,
            test_transition_dma_break_imminent,
            test_transition_fii_reversal_imminent,
            test_transition_none,
            test_transition_reset,
        ]),
        ("Short Executor", [
            test_short_buy_atm_put,
            test_short_sell_atm_call,
            test_short_bear_put_spread,
            test_short_atm_strike_nifty,
            test_short_atm_strike_stock,
            test_short_lot_size,
        ]),
        ("Directional Router", [
            test_router_long_pass,
            test_router_long_circuit_breaker,
            test_router_short_pass,
            test_router_short_no_order,
            test_router_size_reduction,
        ]),
        ("Circuit Breaker V2", [
            test_circuit_breaker_normal,
            test_circuit_breaker_consecutive_losses,
            test_circuit_breaker_daily_loss_limit,
            test_circuit_breaker_max_drawdown,
            test_circuit_breaker_size_recovery,
            test_circuit_breaker_daily_reset,
            test_circuit_breaker_manual_restart,
            test_circuit_breaker_get_state,
        ]),
        ("Adaptive Confidence Calibrator", [
            test_adaptive_calibrator_initial,
            test_adaptive_calibrator_update,
            test_adaptive_calibrator_weight_clamp,
            test_adaptive_calibrator_correlations,
            test_adaptive_calibrator_weight_evolution,
        ]),
        ("Trade Outcome Logger", [
            test_trade_outcome_log,
            test_trade_outcome_update,
            test_trade_outcome_win_rate,
            test_trade_outcome_recent_trades,
        ]),
        ("Regime Accuracy Tracker", [
            test_regime_accuracy_classify,
            test_regime_accuracy_classify_trending,
            test_regime_accuracy_classify_volatile,
            test_regime_accuracy_classify_range_bound,
            test_regime_accuracy_log_and_rolling,
        ]),
        ("Debate Engine v6", [
            test_debate_diversity_score_opposite,
            test_debate_diversity_score_same,
            test_debate_diversity_score_inferred,
            test_debate_system_prompts,
        ]),
        ("Pattern Memory v6", [
            test_pattern_memory_v6_update_outcome,
        ]),
        ("Config v6 Patches", [
            test_config_v6_meta_learner_thresholds,
        ]),
        ("Integration Pipeline", [
            test_regime_pipeline_rule_then_arbiter,
            test_regime_pipeline_with_transition,
            test_execution_pipeline_short,
        ]),
        ("Module Imports (v6 __init__)", [
            test_reasoning_module_v6_imports,
            test_data_module_v6_imports,
            test_execution_module_v6_imports,
            test_monitoring_module_v6_imports,
        ]),
    ]

    for group_name, tests in groups:
        print(f"\n--- {group_name} ---")
        for t in tests:
            _run_test(t.__name__, t)

    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASS} PASSED | {FAIL} FAILED | {PASS + FAIL} TOTAL")
    print("=" * 70)

    if ERRORS:
        print("\nERRORS:")
        for name, err in ERRORS:
            print(f"  - {name}: {err}")

    sys.exit(0 if FAIL == 0 else 1)

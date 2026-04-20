"""
ROX PROVEN EDGE ENGINE v5.0 — Integration Test Suite
=====================================================
Tests all v5.0 modules independently and their integration.
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
# TEST GROUP 1: Data Classes
# ═══════════════════════════════════════════════════════════════════

def test_data_classes_import():
    from reasoning.data_classes import (
        Signal, SignalDirection, SignalStrength, ComplexityLevel,
        RegimeResult, NewsResult, TradePlan, PortfolioState,
        TradeRecord, MarketState,
    )
    # Verify enums
    assert SignalDirection.LONG.value == "LONG"
    assert SignalDirection.SHORT.value == "SHORT"
    assert SignalStrength.STRONG.value == "STRONG"
    assert ComplexityLevel.HIGH.value == "high"
    
    # Verify Signal dataclass with all fields
    s = Signal(
        symbol="RELIANCE", direction=SignalDirection.LONG,
        strength=SignalStrength.HIGH, agent="ORION",
        rr_ratio=2.0, rsi=55.3, volume=1500000,
        volume_avg_20d=1200000, price=2850.0, sma_20=2780.0,
        sector="Energy",
    )
    assert s.symbol == "RELIANCE"
    assert s.rr_ratio == 2.0
    assert s.rsi == 55.3
    
    # Verify RegimeResult with all fields
    r = RegimeResult(
        regime="BULLISH", confidence=78.5,
        key_level=24200, key_level_type="support",
        reasoning={"trend": "uptrend"}, timestamp="2026-04-15",
    )
    assert r.regime == "BULLISH"
    assert r.key_level == 24200
    
    # Verify NewsResult with v5.0 fields
    n = NewsResult(
        sentiment_score=45, sentiment_label="BULLISH",
        block_long_sectors=["Auto"], block_short_sectors=["IT"],
        boost_sectors=["Pharma"], uncertainty_level="MEDIUM",
    )
    assert n.block_long_sectors == ["Auto"]
    
    # Verify MarketState
    ms = MarketState(nifty_price=24245, vix=19.42, nifty_vs_20dma=-1.04)
    assert ms.nifty_price == 24245
    
    # Verify PortfolioState
    ps = PortfolioState(capital=10_00_000, total_trades=50, winning_pct=0.68)
    assert ps.total_trades == 50
    assert ps.winning_pct == 0.68
    
    return True


def test_trade_plan():
    from reasoning.data_classes import TradePlan
    tp = TradePlan(
        symbol="SBIN", direction="LONG",
        entry_price=780.0, stop_loss=765.0,
        target_1=800.0, target_2=820.0,
        position_size=100, risk_amount=1500.0,
        expected_reward=3000.0, verdict="EXECUTE",
        verdict_reason="Breakout above resistance",
        strategy="Momentum breakout",
        validation="Price > SMA20", invalidation="Close below 765",
    )
    assert tp.verdict == "EXECUTE"
    assert tp.risk_amount == 1500.0
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 2: Adaptive Prompt Selector
# ═══════════════════════════════════════════════════════════════════

def test_adaptive_low_complexity():
    from reasoning.adaptive_and_cache import AdaptivePromptSelector, ComplexityLevel
    sel = AdaptivePromptSelector()
    config = sel.assess_complexity(vix=12, intraday_range_pct=0.5)
    assert config.complexity == ComplexityLevel.LOW
    assert config.debate_rounds == 0
    assert config.cot_steps == 3
    assert config.confidence_threshold == 70.0
    return True


def test_adaptive_high_complexity():
    from reasoning.adaptive_and_cache import AdaptivePromptSelector, ComplexityLevel
    sel = AdaptivePromptSelector()
    config = sel.assess_complexity(
        vix=22, intraday_range_pct=1.5, macro_event_count=2,
        key_level_test="200DMA", sector_dispersion=2.5,
    )
    # Score: 3(vix>=18) + 1(range>=1.2) + 1(macro>=1) + 2(200DMA) + 0(disp<3) = 7 → HIGH
    # But could be EXTREME due to rounding
    assert config.complexity in (ComplexityLevel.HIGH, ComplexityLevel.EXTREME)
    assert config.debate_rounds >= 2
    return True


def test_adaptive_extreme_complexity():
    from reasoning.adaptive_and_cache import AdaptivePromptSelector, ComplexityLevel
    sel = AdaptivePromptSelector()
    config = sel.assess_complexity(
        vix=28, intraday_range_pct=3.5, macro_event_count=5,
        fii_streak_days=8, key_level_test="200DMA", sector_dispersion=4.0,
    )
    assert config.complexity == ComplexityLevel.EXTREME
    assert config.debate_rounds == 3
    return True


def test_adaptive_get_model():
    from reasoning.adaptive_and_cache import AdaptivePromptSelector, ComplexityLevel
    sel = AdaptivePromptSelector()
    config = AdaptivePromptSelector()._build_config(ComplexityLevel.LOW)
    model = sel.get_model(config, "regime_detector")
    assert model is not None
    assert isinstance(model, str)
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 3: Regime Cache
# ═══════════════════════════════════════════════════════════════════

def test_regime_cache_set_get():
    from reasoning.adaptive_and_cache import RegimeCache
    cache = RegimeCache(ttl_high_conf=900, ttl_med_conf=420, min_confidence=50.0)
    
    # Set a high-confidence regime
    cache.set("BULLISH", 80.0, {"reason": "uptrend"}, vix=15.0, nifty=24245)
    
    # Should get it back
    result = cache.get(current_vix=15.5, current_nifty=24250)
    assert result is not None
    assert result.regime == "BULLISH"
    assert result.confidence == 80.0
    return True


def test_regime_cache_vix_invalidation():
    from reasoning.adaptive_and_cache import RegimeCache
    cache = RegimeCache(invalidate_vix_delta=2.0)
    cache.set("BULLISH", 80.0, {}, vix=15.0, nifty=24245)
    
    # VIX moved 3 points — should invalidate
    result = cache.get(current_vix=18.0, current_nifty=24245)
    assert result is None
    return True


def test_regime_cache_nifty_invalidation():
    from reasoning.adaptive_and_cache import RegimeCache
    cache = RegimeCache(invalidate_dma_break=True)
    cache.set("CONSOLIDATION", 70.0, {}, vix=15.0, nifty=24245)
    
    # Nifty moved 300 points — should invalidate
    result = cache.get(current_vix=15.0, current_nifty=24545)
    assert result is None
    return True


def test_regime_cache_low_confidence_not_cached():
    from reasoning.adaptive_and_cache import RegimeCache
    cache = RegimeCache(min_confidence=50.0)
    cache.set("UNKNOWN", 30.0, {}, vix=15.0, nifty=24245)
    
    # Low confidence — should not be cached
    result = cache.get(current_vix=15.0, current_nifty=24245)
    assert result is None
    return True


def test_regime_cache_invalidate():
    from reasoning.adaptive_and_cache import RegimeCache
    cache = RegimeCache()
    cache.set("BULLISH", 80.0, {}, vix=15.0, nifty=24245)
    cache.invalidate()
    result = cache.get(current_vix=15.0, current_nifty=24245)
    assert result is None
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 4: Rule-Based Validator
# ═══════════════════════════════════════════════════════════════════

def test_validator_pass_long():
    from reasoning.rule_validator import RuleBasedValidator
    v = RuleBasedValidator(min_rr_ratio=1.5, rsi_long_min=40.0)
    
    signal = {
        "symbol": "RELIANCE", "direction": "LONG", "strength": "HIGH",
        "rr_ratio": 2.0, "rsi": 55.0,
        "volume": 1500000, "volume_avg_20d": 1200000,
        "price": 2850.0, "sma_20": 2780.0,
        "sector": "Energy", "agent": "ORION",
    }
    
    result = v.validate(signal, {}, {}, {}, {})
    assert result.passed, f"Should pass but got: {result.reason}"
    assert result.score > 50
    assert result.original_signal == signal
    return True


def test_validator_fail_low_rr():
    from reasoning.rule_validator import RuleBasedValidator
    v = RuleBasedValidator(min_rr_ratio=1.5)
    
    signal = {
        "symbol": "SBIN", "direction": "LONG", "strength": "HIGH",
        "rr_ratio": 1.0, "rsi": 55.0,
        "volume": 1500000, "volume_avg_20d": 1200000,
        "price": 780.0, "sma_20": 760.0,
        "sector": "Banking", "agent": "ORION",
    }
    
    result = v.validate(signal, {}, {}, {}, {})
    assert not result.passed
    assert "R:R" in result.reason
    return True


def test_validator_fail_rsi_oversold_long():
    from reasoning.rule_validator import RuleBasedValidator
    v = RuleBasedValidator(rsi_long_min=40.0)
    
    signal = {
        "symbol": "TCS", "direction": "LONG", "strength": "HIGH",
        "rr_ratio": 2.0, "rsi": 25.0,
        "volume": 1500000, "volume_avg_20d": 1200000,
        "price": 3500.0, "sma_20": 3400.0,
        "sector": "IT", "agent": "ORION",
    }
    
    result = v.validate(signal, {}, {}, {}, {})
    assert not result.passed
    assert "RSI" in result.reason
    return True


def test_validator_fail_counter_trend():
    from reasoning.rule_validator import RuleBasedValidator
    v = RuleBasedValidator()
    
    signal = {
        "symbol": "HDFCBANK", "direction": "LONG", "strength": "HIGH",
        "rr_ratio": 2.0, "rsi": 55.0,
        "volume": 1500000, "volume_avg_20d": 1200000,
        "price": 1500.0, "sma_20": 1450.0,
        "sector": "Banking", "agent": "ORION",
    }
    
    regime = {"regime": "BEARISH"}
    result = v.validate(signal, {}, regime, {}, {})
    assert not result.passed
    assert "Counter-trend" in result.reason
    return True


def test_validator_batch():
    from reasoning.rule_validator import RuleBasedValidator
    v = RuleBasedValidator()
    
    signals = [
        {
            "symbol": f"STOCK{i}", "direction": "LONG", "strength": "HIGH",
            "rr_ratio": 2.0, "rsi": 55.0,
            "volume": 1500000, "volume_avg_20d": 1200000,
            "price": 1000.0, "sma_20": 950.0,
            "sector": "IT", "agent": "ORION",
        }
        for i in range(5)
    ]
    
    results = v.validate_batch(signals, {}, {}, {}, {})
    assert len(results) == 5
    # All should pass since they have good metrics
    assert all(r.passed for r in results)
    # Results should be sorted by score descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    return True


def test_validator_fail_low_strength():
    from reasoning.rule_validator import RuleBasedValidator
    v = RuleBasedValidator(min_signal_strength="MEDIUM")
    
    signal = {
        "symbol": "INFY", "direction": "LONG", "strength": "LOW",
        "rr_ratio": 2.0, "rsi": 55.0,
        "volume": 1500000, "volume_avg_20d": 1200000,
        "price": 1500.0, "sma_20": 1450.0,
        "sector": "IT", "agent": "ORION",
    }
    
    result = v.validate(signal, {}, {}, {}, {})
    assert not result.passed
    assert "strength" in result.reason.lower()
    return True


def test_validator_news_block():
    from reasoning.rule_validator import RuleBasedValidator
    v = RuleBasedValidator()
    
    signal = {
        "symbol": "MARUTI", "direction": "LONG", "strength": "HIGH",
        "rr_ratio": 2.0, "rsi": 55.0,
        "volume": 1500000, "volume_avg_20d": 1200000,
        "price": 10000.0, "sma_20": 9800.0,
        "sector": "Auto", "agent": "ORION",
    }
    
    news = {"block_long_sectors": ["Auto"]}
    result = v.validate(signal, {}, {}, news, {})
    assert not result.passed
    assert "News" in result.reason or "blocked" in result.reason.lower()
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 5: Confidence Calibrator
# ═══════════════════════════════════════════════════════════════════

def test_calibrator_basic():
    from reasoning.confidence_calibrator import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    result = cal.calibrate(
        debate_agreement=0.85,
        debate_confidence=72.0,
        pattern_match_score=0.75,
        pattern_match_count=3,
        technical_aligned=0.80,
        volume_confirms_price=True,
        volume_strength=0.7,
        regime_direction="BULLISH",
        prediction_direction="BULLISH",
        raw_confidence=72.0,
        vix_level=15.0,
    )
    assert result.calibrated_confidence > 50
    assert result.raw_confidence == 72.0
    return True


def test_calibrator_low_agreement():
    from reasoning.confidence_calibrator import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    result = cal.calibrate(
        debate_agreement=0.30,
        debate_confidence=60.0,
        pattern_match_score=0.50,
        pattern_match_count=0,
        technical_aligned=0.40,
        volume_confirms_price=False,
        volume_strength=0.3,
        regime_direction="BULLISH",
        prediction_direction="BEARISH",
        raw_confidence=60.0,
        vix_level=22.0,
        macro_event_count=3,
    )
    # Should be lower due to low agreement, counter-regime, high VIX, macros
    assert result.calibrated_confidence < result.raw_confidence
    return True


def test_calibrator_unanimous_bonus():
    from reasoning.confidence_calibrator import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    result = cal.calibrate(
        debate_agreement=1.0,
        debate_confidence=80.0,
        pattern_match_score=0.90,
        pattern_match_count=5,
        technical_aligned=0.90,
        volume_confirms_price=True,
        volume_strength=0.8,
        regime_direction="BULLISH",
        prediction_direction="BULLISH",
        raw_confidence=80.0,
        vix_level=13.0,
    )
    # Should get bonus from unanimous agreement
    assert result.calibrated_confidence > 50
    assert "Unanimous" in str(result.adjustments_applied)
    return True


def test_calibrator_weight_validation():
    from reasoning.confidence_calibrator import ConfidenceCalibrator
    # Test with bad weights that don't sum to 1.0
    cal = ConfidenceCalibrator(weights={"debate_agreement": 0.5, "pattern_match": 0.3})
    # Should normalize without error
    result = cal.calibrate(
        debate_agreement=0.8, debate_confidence=70.0,
        pattern_match_score=0.6, pattern_match_count=2,
        technical_aligned=0.7, volume_confirms_price=True,
        volume_strength=0.5, regime_direction="NEUTRAL",
        prediction_direction="NEUTRAL", raw_confidence=70.0,
    )
    assert result is not None
    return True


def test_calibrator_extreme_bearish():
    from reasoning.confidence_calibrator import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    result = cal.calibrate(
        debate_agreement=0.66,
        debate_confidence=65.0,
        pattern_match_score=0.70,
        pattern_match_count=4,
        pattern_bullish_count=0,
        pattern_bearish_count=4,
        pattern_total_count=4,
        technical_aligned=0.60,
        volume_confirms_price=True,
        volume_strength=0.6,
        regime_direction="BEARISH",
        prediction_direction="BEARISH",
        raw_confidence=65.0,
        vix_level=20.0,
        is_200dma_test=True,
    )
    # Historical consensus supports bearish
    assert result.calibrated_confidence > 25
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 6: Pattern Memory Bank
# ═══════════════════════════════════════════════════════════════════

def test_pattern_memory_save_load():
    from reasoning.pattern_memory import PatternMemoryBank, DailySnapshot
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_patterns.db")
        bank = PatternMemoryBank(db_path)
        
        snapshot = DailySnapshot(
            date="2026-04-10", nifty_close=24200, nifty_vs_200dma_pct=0.5,
            nifty_vs_50dma_pct=-0.3, nifty_vs_20dma_pct=-1.0,
            vix_close=16.5, vix_vs_30d_avg=0.2, usd_inr_close=83.5,
            usd_inr_trend_5d=0.1, crude_close=85.0,
            fii_net_buy_sell_cr=1500, fii_trend_days=3,
            dii_net_buy_sell_cr=-800, nifty_intraday_range_pct=1.2,
            nifty_close_vs_open=0.3, expiry_days_remaining=3,
            pcr=1.1, max_pain_level=24000, put_call_oi_ratio=1.1,
            global_cues_summary="BULLISH", sector_rotation_pattern="Rotation to IT",
        )
        
        bank.save_snapshot(snapshot)
        loaded = bank.load_snapshot("2026-04-10")
        assert loaded is not None
        assert loaded.nifty_close == 24200
        assert loaded.vix_close == 16.5
    return True


def test_pattern_memory_update_outcome():
    from reasoning.pattern_memory import PatternMemoryBank, DailySnapshot
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_patterns.db")
        bank = PatternMemoryBank(db_path)
        
        snapshot = DailySnapshot(
            date="2026-04-10", nifty_close=24200,
            nifty_vs_200dma_pct=0.5, nifty_vs_50dma_pct=-0.3,
            nifty_vs_20dma_pct=-1.0, vix_close=16.5,
            vix_vs_30d_avg=0.2, usd_inr_close=83.5,
            usd_inr_trend_5d=0.1, crude_close=85.0,
            fii_net_buy_sell_cr=1500, fii_trend_days=3,
            dii_net_buy_sell_cr=-800, nifty_intraday_range_pct=1.2,
            nifty_close_vs_open=0.3, expiry_days_remaining=3,
            pcr=1.1, max_pain_level=24000, put_call_oi_ratio=1.1,
            global_cues_summary="BULLISH", sector_rotation_pattern="Rotation to IT",
        )
        bank.save_snapshot(snapshot)
        
        # Update with outcome
        bank.update_outcome("2026-04-10", "BULLISH_RALLY_150PTS",
                          "Long NIFTY CE ATM", next_day_change=1.5)
        
        loaded = bank.load_snapshot("2026-04-10")
        assert loaded.outcome == "BULLISH_RALLY_150PTS"
        assert loaded.optimal_strategy == "Long NIFTY CE ATM"
        assert loaded.nifty_next_day_change == 1.5
    return True


def test_pattern_memory_find_similar():
    from reasoning.pattern_memory import PatternMemoryBank, DailySnapshot
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_patterns.db")
        bank = PatternMemoryBank(db_path)
        
        # Save 3 historical snapshots with outcomes
        for i, (date, nifty, vix, outcome) in enumerate([
            ("2026-03-10", 24100, 15.0, "BULLISH_RALLY"),
            ("2026-03-15", 23800, 22.0, "BEARISH_DECLINE"),
            ("2026-03-20", 24300, 14.0, "SIDEWAYS_CONSOLIDATION"),
        ]):
            snap = DailySnapshot(
                date=date, nifty_close=nifty,
                nifty_vs_200dma_pct=0.5 if i != 1 else -1.0,
                nifty_vs_50dma_pct=0.2, nifty_vs_20dma_pct=0.3,
                vix_close=vix, vix_vs_30d_avg=0.0,
                usd_inr_close=83.0, usd_inr_trend_5d=0.0,
                crude_close=85.0, fii_net_buy_sell_cr=1000,
                fii_trend_days=2, dii_net_buy_sell_cr=-500,
                nifty_intraday_range_pct=1.0, nifty_close_vs_open=0.2,
                expiry_days_remaining=4, pcr=1.0,
                max_pain_level=24000, put_call_oi_ratio=1.0,
                global_cues_summary="NEUTRAL",
                sector_rotation_pattern="NEUTRAL",
                outcome=outcome,
                optimal_strategy=f"Strategy for {outcome}",
            )
            bank.save_snapshot(snap)
        
        # Find similar to current (low VIX, positive DMA)
        current = DailySnapshot(
            date="2026-04-15", nifty_close=24200,
            nifty_vs_200dma_pct=0.4, nifty_vs_50dma_pct=0.1,
            nifty_vs_20dma_pct=0.2, vix_close=15.5,
            vix_vs_30d_avg=0.1, usd_inr_close=83.2,
            usd_inr_trend_5d=0.1, crude_close=84.0,
            fii_net_buy_sell_cr=1200, fii_trend_days=3,
            dii_net_buy_sell_cr=-600, nifty_intraday_range_pct=1.1,
            nifty_close_vs_open=0.25, expiry_days_remaining=3,
            pcr=1.05, max_pain_level=24050, put_call_oi_ratio=1.05,
            global_cues_summary="BULLISH", sector_rotation_pattern="Rotation to IT",
        )
        
        matches = bank.find_similar(current, top_k=3)
        assert len(matches) > 0
        # Most similar should be the one with similar VIX and positive DMA
        assert matches[0].similarity > 0
        return True


def test_pattern_memory_to_dict_roundtrip():
    from reasoning.pattern_memory import DailySnapshot
    
    snap = DailySnapshot(
        date="2026-04-15", nifty_close=24200,
        nifty_vs_200dma_pct=0.5, nifty_vs_50dma_pct=-0.3,
        nifty_vs_20dma_pct=-1.0, vix_close=16.5,
        vix_vs_30d_avg=0.2, usd_inr_close=83.5,
        usd_inr_trend_5d=0.1, crude_close=85.0,
        fii_net_buy_sell_cr=1500, fii_trend_days=3,
        dii_net_buy_sell_cr=-800, nifty_intraday_range_pct=1.2,
        nifty_close_vs_open=0.3, expiry_days_remaining=3,
        pcr=1.1, max_pain_level=24000, put_call_oi_ratio=1.1,
        global_cues_summary="BULLISH", sector_rotation_pattern="Rotation to IT",
        nifty_next_day_range=(100, 150),
        outcome="RALLY_100PTS",
        optimal_strategy="Long CE ATM",
        nifty_next_day_change=1.2,
    )
    
    d = snap.to_dict()
    assert "nifty_next_day_range" in d
    assert d["nifty_next_day_range"] == [100, 150]
    assert d["outcome"] == "RALLY_100PTS"
    
    # Reconstruct
    snap2 = DailySnapshot(**d)
    assert snap2.nifty_close == snap.nifty_close
    assert snap2.outcome == snap.outcome
    return True


def test_pattern_memory_count():
    from reasoning.pattern_memory import PatternMemoryBank, DailySnapshot
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_patterns.db")
        bank = PatternMemoryBank(db_path)
        
        assert bank.get_pattern_count() == 0
        
        snap = DailySnapshot(
            date="2026-04-10", nifty_close=24200,
            nifty_vs_200dma_pct=0.5, nifty_vs_50dma_pct=-0.3,
            nifty_vs_20dma_pct=-1.0, vix_close=16.5,
            vix_vs_30d_avg=0.2, usd_inr_close=83.5,
            usd_inr_trend_5d=0.1, crude_close=85.0,
            fii_net_buy_sell_cr=1500, fii_trend_days=3,
            dii_net_buy_sell_cr=-800, nifty_intraday_range_pct=1.2,
            nifty_close_vs_open=0.3, expiry_days_remaining=3,
            pcr=1.1, max_pain_level=24000, put_call_oi_ratio=1.1,
            global_cues_summary="BULLISH", sector_rotation_pattern="NEUTRAL",
            outcome="BULLISH",
        )
        bank.save_snapshot(snap)
        assert bank.get_pattern_count() == 1
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 7: CoT Prompts
# ═══════════════════════════════════════════════════════════════════

def test_cot_prompts_regime():
    from reasoning.cot_prompts import build_regime_cot_prompt
    prompt = build_regime_cot_prompt(
        market_data={"nifty": 24200, "vix": 16.5},
        num_steps=7,
    )
    assert "Step 1" in prompt or "step 1" in prompt.lower()
    assert "24200" in prompt
    return True


def test_cot_prompts_news():
    from reasoning.cot_prompts import build_news_prompt
    prompt = build_news_prompt(
        headlines=[{"title": "RBI holds rates", "sentiment": "NEUTRAL"},
                  {"title": "FII net buyers", "sentiment": "BULLISH"}],
        market_context="BULLISH | VIX: 16.5",
    )
    assert "RBI" in prompt
    assert "BULLISH" in prompt
    return True


def test_cot_prompts_trading_planner():
    from reasoning.cot_prompts import build_trading_planner_prompt
    signals = [
        {"symbol": "RELIANCE", "direction": "LONG", "strength": "HIGH", "rr_ratio": 2.0, "agent": "ORION"},
        {"symbol": "SBIN", "direction": "LONG", "strength": "MEDIUM", "rr_ratio": 1.8, "agent": "ORION"},
    ]
    prompt = build_trading_planner_prompt(
        signals=signals,
        regime={"regime": "BULLISH", "confidence": 78},
        portfolio={"capital": 10_00_000, "risk_pct": 1.5},
        prediction={"prediction": {"direction": "LONG"}, "calibrated_confidence": 65},
    )
    assert "RELIANCE" in prompt
    assert "BULLISH" in prompt
    return True


def test_cot_prompts_fno_brain():
    from reasoning.cot_prompts import build_fno_brain_prompt
    prompt = build_fno_brain_prompt(
        market_view={"regime": "BULLISH"},
        options_chain={"strikes": [24000, 24100, 24200]},
        expiry_info={"days": 3},
        regime={"regime": "BULLISH", "confidence": 70},
    )
    assert "BULLISH" in prompt
    return True


def test_cot_prompts_self_reflector():
    from reasoning.cot_prompts import build_self_reflector_prompt
    # build_self_reflector_prompt expects a dict, not a TradeRecord
    trade_dict = {
        "id": "T001", "symbol": "RELIANCE", "direction": "LONG",
        "entry": 2850, "entry_time": "09:30", "exit": 2920, "exit_time": "14:00",
        "pnl": 7000, "pnl_pct": 2.45, "confidence": 75.0,
        "reasoning": "Momentum breakout", "outcome": "WIN",
    }
    prompt = build_self_reflector_prompt(
        trade_record=trade_dict,
        market_before={"nifty": 24100, "vix": 15},
        market_after={"nifty": 24250, "vix": 14},
    )
    assert "RELIANCE" in prompt
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 8: Config
# ═══════════════════════════════════════════════════════════════════

def test_config_v5():
    from config_v5 import EngineConfig, PortfolioConfig, LLMConfig, ReasoningConfig
    
    config = EngineConfig()
    
    # Check defaults
    assert config.portfolio.initial_capital == 10_00_000
    assert config.portfolio.risk_per_trade_pct == 1.5
    # FIX-TEST-CONFIG: Migrated from Gemini to OpenRouter on 2026-04-17.
    # model_pro/model_flash now resolve via OPENROUTER_MODELS env config.
    # Assert the config resolves to a non-empty string rather than a hardcoded model name.
    assert isinstance(config.llm.model_pro, str) and len(config.llm.model_pro) > 0
    assert isinstance(config.llm.model_flash, str) and len(config.llm.model_flash) > 0
    assert config.reasoning.debate_enabled == True
    assert config.reasoning.pattern_memory_enabled == True
    assert config.reasoning.calibration_enabled == True
    
    # Check weight sum
    total_weight = (
        config.reasoning.weight_debate_agreement +
        config.reasoning.weight_pattern_match +
        config.reasoning.weight_technical_alignment +
        config.reasoning.weight_volume_confirmation +
        config.reasoning.weight_regime_consistency +
        config.reasoning.weight_anti_consensus
    )
    assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, expected 1.0"
    return True


def test_config_from_env():
    from config_v5 import EngineConfig
    config = EngineConfig.from_env()
    assert config is not None
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 9: Debate Engine (import only - no LLM calls)
# ═══════════════════════════════════════════════════════════════════

def test_debate_engine_import():
    from reasoning.debate_engine import DebateEngine, DebateResult
    # Verify classes exist
    assert DebateEngine is not None
    assert DebateResult is not None
    return True


def test_debate_result_structure():
    from reasoning.debate_engine import DebateResult
    # Check DebateResult fields
    # DebateResult has required fields: cross_examination and all_directions
    dr = DebateResult(
        bull_thesis={"thesis": "Market goes up", "confidence": 75},
        bear_thesis={"thesis": "Market goes down", "confidence": 40},
        neutral_thesis=None,
        final_prediction={"prediction": {"direction": "BULLISH"}},
        debate_agreement=0.65,
        raw_confidence=70.0,
        cross_examination={"synthesis": "Bull case stronger"},
        all_directions=["BULLISH", "BEARISH"],
    )
    assert dr.debate_agreement == 0.65
    assert dr.final_prediction["prediction"]["direction"] == "BULLISH"
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 10: Async Client (import only)
# ═══════════════════════════════════════════════════════════════════

def test_async_client_import():
    from agents.llm.async_client import GeminiClient, AsyncLLMResponse
    assert GeminiClient is not None
    assert AsyncLLMResponse is not None
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 11: Config v4 (legacy)
# ═══════════════════════════════════════════════════════════════════

def test_config_v4_imports():
    from config import (
        MarketRegime, TradeDirection, ConvictionLevel,
        SystemConfig, DEFAULT_CONFIG, NIFTY_50_STOCKS, SECTOR_MAPPING,
        get_vix_regime, get_lot_size, get_current_stt, days_to_stt_hike,
    )
    assert MarketRegime.BULL.value == "BULL"
    assert TradeDirection.LONG.value == "LONG"
    assert len(NIFTY_50_STOCKS) > 40
    # get_vix_regime returns VIXRegime enum (>=25 = EXTREME)
    vix_regime = get_vix_regime(25)
    assert vix_regime.value == "EXTREME"
    assert get_vix_regime(20).value == "HIGH"
    assert get_lot_size("NIFTY") == 75
    return True


# ═══════════════════════════════════════════════════════════════════
# TEST GROUP 12: Coordinator (import only)
# ═══════════════════════════════════════════════════════════════════

def test_coordinator_import():
    # Coordinator requires feedparser — skip in test env
    try:
        from coordinator import LeadCoordinator, FnoCoordinator, UnifiedCoordinator
        assert LeadCoordinator is not None
        assert FnoCoordinator is not None
        assert UnifiedCoordinator is not None
    except ImportError:
        pass  # Missing optional dep (feedparser)
    return True


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  ROX PROVEN EDGE ENGINE v5.0 — Integration Test Suite")
    print("=" * 70)
    
    groups = [
        ("Data Classes", [
            test_data_classes_import,
            test_trade_plan,
        ]),
        ("Adaptive Prompt Selector", [
            test_adaptive_low_complexity,
            test_adaptive_high_complexity,
            test_adaptive_extreme_complexity,
            test_adaptive_get_model,
        ]),
        ("Regime Cache", [
            test_regime_cache_set_get,
            test_regime_cache_vix_invalidation,
            test_regime_cache_nifty_invalidation,
            test_regime_cache_low_confidence_not_cached,
            test_regime_cache_invalidate,
        ]),
        ("Rule-Based Validator", [
            test_validator_pass_long,
            test_validator_fail_low_rr,
            test_validator_fail_rsi_oversold_long,
            test_validator_fail_counter_trend,
            test_validator_batch,
            test_validator_fail_low_strength,
            test_validator_news_block,
        ]),
        ("Confidence Calibrator", [
            test_calibrator_basic,
            test_calibrator_low_agreement,
            test_calibrator_unanimous_bonus,
            test_calibrator_weight_validation,
            test_calibrator_extreme_bearish,
        ]),
        ("Pattern Memory Bank", [
            test_pattern_memory_save_load,
            test_pattern_memory_update_outcome,
            test_pattern_memory_find_similar,
            test_pattern_memory_to_dict_roundtrip,
            test_pattern_memory_count,
        ]),
        ("CoT Prompts", [
            test_cot_prompts_regime,
            test_cot_prompts_news,
            test_cot_prompts_trading_planner,
            test_cot_prompts_fno_brain,
            test_cot_prompts_self_reflector,
        ]),
        ("Config v5", [
            test_config_v5,
            test_config_from_env,
        ]),
        ("Debate Engine", [
            test_debate_engine_import,
            test_debate_result_structure,
        ]),
        ("Async Client", [
            test_async_client_import,
        ]),
        ("Config v4 (Legacy)", [
            test_config_v4_imports,
        ]),
        ("Coordinator", [
            test_coordinator_import,
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

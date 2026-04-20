"""
Tests for OPTIMUS - F&O Weekly Expiry Agent
============================================
Run with: python -m pytest tests/test_optimus.py -v
"""

import sys
import os

# Ensure the engine root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import date, timedelta

from agents.optimus import (
    OptimusAgent,
    WeeklyOptionsData,
    OptionsStrategy,
    OptionType,
)
from config import MarketRegime, TradeDirection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return OptimusAgent()


def _make_data(
    pcr: float = 1.0,
    pcr_trend: str = "stable",
    max_pain: float = 22000,
    current_price: float = 22000,
    india_vix: float = 15,
    iv_rank: float = 50,
    iv_skew: float = 0.0,
    futures_premium: float = 30,
    oi_signal: str = "NEUTRAL",
    price_change: float = 0.0,
    ce_oi_change_pct: float = 0.0,
    pe_oi_change_pct: float = 0.0,
    symbol: str = "NIFTY",
) -> dict:
    """Helper to build a minimal data dict for OPTIMUS."""
    expiry = (date.today() + timedelta(days=(3 - date.today().weekday()) % 7 or 7)).isoformat()
    return {
        "symbol": symbol,
        "current_price": current_price,
        "weekly_expiry": expiry,
        "pcr": pcr,
        "pcr_trend": pcr_trend,
        "max_pain": max_pain,
        "call_oi_walls": [{"strike": current_price + 200, "oi": 500_000}],
        "put_oi_walls": [{"strike": current_price - 200, "oi": 500_000}],
        "ce_oi_change_pct": ce_oi_change_pct,
        "pe_oi_change_pct": pe_oi_change_pct,
        "india_vix": india_vix,
        "iv_rank": iv_rank,
        "iv_skew": iv_skew,
        "futures_premium": futures_premium,
        "oi_signal": oi_signal,
        "price_change": price_change,
        "support_level": current_price - 300,
        "resistance_level": current_price + 300,
    }


# ---------------------------------------------------------------------------
# Basic sanity tests
# ---------------------------------------------------------------------------

class TestOptimusInit:
    def test_name(self, agent):
        assert agent.name == "OPTIMUS"

    def test_domain(self, agent):
        assert "F&O" in agent.domain or "Expiry" in agent.domain

    def test_baseline_weight(self, agent):
        assert agent.baseline_weight == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# analyze() output structure
# ---------------------------------------------------------------------------

class TestAnalyzeOutput:
    def test_returns_agent_report(self, agent):
        from agents.base_agent import AgentReport
        data = _make_data()
        report = agent.analyze(data, MarketRegime.CONSOLIDATION)
        assert isinstance(report, AgentReport)

    def test_verdict_direction_is_valid(self, agent):
        data = _make_data()
        report = agent.analyze(data, MarketRegime.CONSOLIDATION)
        assert report.verdict.direction in TradeDirection

    def test_conviction_in_range(self, agent):
        data = _make_data()
        report = agent.analyze(data, MarketRegime.CONSOLIDATION)
        assert 0 <= report.verdict.conviction <= 100

    def test_analysis_details_contain_signal(self, agent):
        data = _make_data()
        report = agent.analyze(data, MarketRegime.BULL)
        assert "options_signal" in report.analysis_details

    def test_signal_has_required_fields(self, agent):
        data = _make_data()
        report = agent.analyze(data, MarketRegime.BULL)
        sig = report.analysis_details["options_signal"]
        for key in ("symbol", "expiry_date", "option_type", "strategy",
                    "strike", "entry_range", "stop_loss", "target_1", "target_2",
                    "conviction", "rationale"):
            assert key in sig, f"Missing key: {key}"

    def test_key_observations_non_empty(self, agent):
        data = _make_data()
        report = agent.analyze(data, MarketRegime.BULL)
        assert len(report.key_observations) > 0

    def test_metrics_populated(self, agent):
        data = _make_data()
        report = agent.analyze(data, MarketRegime.BULL)
        assert "pcr" in report.metrics
        assert "india_vix" in report.metrics


# ---------------------------------------------------------------------------
# Directional bias tests
# ---------------------------------------------------------------------------

class TestDirectionalBias:
    def test_high_pcr_bullish(self, agent):
        """PCR > 1.3 should produce bullish / LONG signal"""
        data = _make_data(pcr=1.4, price_change=0.5, futures_premium=40)
        report = agent.analyze(data, MarketRegime.BULL)
        assert report.verdict.direction == TradeDirection.LONG

    def test_low_pcr_bearish(self, agent):
        """PCR < 0.6 should produce bearish / SHORT signal"""
        data = _make_data(pcr=0.5, price_change=-0.5, futures_premium=-40,
                          oi_signal="SHORT_BUILDUP")
        report = agent.analyze(data, MarketRegime.BEAR)
        assert report.verdict.direction == TradeDirection.SHORT

    def test_neutral_pcr_consolidation(self, agent):
        """PCR in neutral zone + stable regime → neutral or low conviction"""
        data = _make_data(pcr=0.95, oi_signal="NEUTRAL", futures_premium=5)
        report = agent.analyze(data, MarketRegime.CONSOLIDATION)
        # Either neutral or low conviction
        assert (
            report.verdict.direction == TradeDirection.NEUTRAL
            or report.verdict.conviction <= 50
        )

    def test_short_buildup_bearish(self, agent):
        """SHORT_BUILDUP OI signal + bear regime → bearish"""
        data = _make_data(
            pcr=0.8,
            oi_signal="SHORT_BUILDUP",
            price_change=-1.5,
            futures_premium=-50,
        )
        report = agent.analyze(data, MarketRegime.BEAR)
        assert report.verdict.direction in (TradeDirection.SHORT, TradeDirection.NEUTRAL)

    def test_long_buildup_bullish(self, agent):
        """LONG_BUILDUP OI signal + bull regime → bullish"""
        data = _make_data(
            pcr=1.2,
            oi_signal="LONG_BUILDUP",
            price_change=1.5,
            futures_premium=60,
        )
        report = agent.analyze(data, MarketRegime.BULL)
        assert report.verdict.direction == TradeDirection.LONG


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

class TestStrategySelection:
    def test_low_iv_prefers_buy(self, agent):
        """IV rank below 30 should favour long options."""
        data = _make_data(pcr=1.35, iv_rank=20, futures_premium=50, price_change=1)
        report = agent.analyze(data, MarketRegime.BULL)
        sig = report.analysis_details["options_signal"]
        assert sig["strategy"] in ("LONG_CALL", "LONG_PUT")

    def test_high_iv_considers_sell(self, agent):
        """IV rank above 70 and strong bullish → SHORT_PUT or LONG_CALL."""
        data = _make_data(pcr=1.4, iv_rank=80, futures_premium=60, price_change=1.5)
        report = agent.analyze(data, MarketRegime.BULL)
        sig = report.analysis_details["options_signal"]
        assert sig["strategy"] in ("SHORT_PUT", "LONG_CALL")

    def test_option_type_matches_strategy(self, agent):
        """CE strategies should have option_type == CE, PE strategies PE."""
        for pcr_val, expected_type in [(1.5, "CE"), (0.5, "PE")]:
            data = _make_data(
                pcr=pcr_val,
                price_change=(1.0 if pcr_val > 1 else -1.0),
                futures_premium=(50 if pcr_val > 1 else -50),
                oi_signal=("LONG_BUILDUP" if pcr_val > 1 else "SHORT_BUILDUP"),
            )
            report = agent.analyze(data, MarketRegime.BULL if pcr_val > 1 else MarketRegime.BEAR)
            sig = report.analysis_details["options_signal"]
            strategy = sig["strategy"]
            opt_type = sig["option_type"]
            if "CALL" in strategy:
                assert opt_type == "CE", f"Expected CE for {strategy}, got {opt_type}"
            elif "PUT" in strategy:
                assert opt_type == "PE", f"Expected PE for {strategy}, got {opt_type}"


# ---------------------------------------------------------------------------
# Strike & price sanity
# ---------------------------------------------------------------------------

class TestPriceSanity:
    def test_stop_loss_below_entry(self, agent):
        data = _make_data()
        report = agent.analyze(data, MarketRegime.BULL)
        sig = report.analysis_details["options_signal"]
        entry_mid = (sig["entry_range"][0] + sig["entry_range"][1]) / 2
        assert sig["stop_loss"] < entry_mid

    def test_target_1_above_entry(self, agent):
        data = _make_data()
        report = agent.analyze(data, MarketRegime.BULL)
        sig = report.analysis_details["options_signal"]
        entry_mid = (sig["entry_range"][0] + sig["entry_range"][1]) / 2
        assert sig["target_1"] > entry_mid

    def test_target_2_above_target_1(self, agent):
        data = _make_data()
        report = agent.analyze(data, MarketRegime.BULL)
        sig = report.analysis_details["options_signal"]
        assert sig["target_2"] > sig["target_1"]

    def test_strike_is_round_number(self, agent):
        """Strike should be a multiple of the index's strike gap."""
        for symbol, gap in [("NIFTY", 50), ("BANKNIFTY", 100)]:
            data = _make_data(symbol=symbol, current_price=22000)
            report = agent.analyze(data, MarketRegime.CONSOLIDATION)
            strike = report.analysis_details["options_signal"]["strike"]
            assert strike % gap == 0, f"Strike {strike} not multiple of {gap} for {symbol}"


# ---------------------------------------------------------------------------
# Graceful fallback (empty data)
# ---------------------------------------------------------------------------

class TestGracefulFallback:
    def test_empty_data_does_not_raise(self, agent):
        """OPTIMUS must handle completely empty data without raising."""
        report = agent.analyze({}, MarketRegime.CONSOLIDATION)
        assert report is not None
        assert report.verdict is not None

    def test_missing_price_does_not_crash(self, agent):
        report = agent.analyze({"pcr": 1.2}, MarketRegime.BULL)
        assert 0 <= report.verdict.conviction <= 100


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_next_thursday_is_thursday(self, agent):
        exp = agent._next_thursday()
        d = date.fromisoformat(exp)
        assert d.weekday() == 3  # Thursday

    def test_days_to_expiry_future(self, agent):
        future = (date.today() + timedelta(days=5)).isoformat()
        assert agent._days_to_expiry(future) == 5

    def test_days_to_expiry_past(self, agent):
        past = (date.today() - timedelta(days=2)).isoformat()
        assert agent._days_to_expiry(past) == 0

    def test_lot_size_nifty(self, agent):
        assert agent._default_lot_size("NIFTY") == 75

    def test_lot_size_banknifty(self, agent):
        assert agent._default_lot_size("BANKNIFTY") == 15

    def test_lot_size_generic(self, agent):
        assert agent._default_lot_size("RELIANCE") == 50


# ---------------------------------------------------------------------------
# Integration: coordinator picks up OPTIMUS
# ---------------------------------------------------------------------------

class TestCoordinatorIntegration:
    def test_optimus_in_coordinator_agents(self):
        """Coordinator must include OPTIMUS in its agent registry."""
        from coordinator import LeadCoordinator
        coord = LeadCoordinator()
        assert "OPTIMUS" in coord.agents

    def test_optimus_weight_configured(self):
        from coordinator import LeadCoordinator
        coord = LeadCoordinator()
        agent = coord.agents["OPTIMUS"]
        assert agent.baseline_weight > 0


# ---------------------------------------------------------------------------
# Trade logger: options logging
# ---------------------------------------------------------------------------

class TestOptionsTradeLogger:
    def test_log_options_trade_returns_id(self, tmp_path, monkeypatch):
        """log_options_trade should return an OT-prefixed trade ID."""
        # Patch DATA_DIR to tmp_path to avoid polluting real data
        import data.trade_logger as tl_module
        from data.data_manager import DataManager

        # Use a fresh logger with a mock data manager
        class MockDataManager:
            def get_trade_history(self, limit=100):
                return []
            def log_trade(self, record):
                pass

        monkeypatch.setattr("core.config.DATA_DIR", tmp_path)

        from data.trade_logger import TradeLogger
        logger = TradeLogger(data_manager=MockDataManager())

        signal = {
            "symbol": "NIFTY",
            "expiry_date": "2026-02-27",
            "option_type": "CE",
            "strategy": "LONG_CALL",
            "strike": 22050,
            "entry_range": [120.0, 132.0],
            "stop_loss": 75.0,
            "target_1": 180.0,
            "target_2": 260.0,
            "risk_per_lot": 3375.0,
            "suggested_lots": 1,
            "rationale": "Bullish PCR, long buildup.",
        }

        trade_id = logger.log_options_trade(signal, regime="BULL", conviction=72)
        assert trade_id.startswith("OT")

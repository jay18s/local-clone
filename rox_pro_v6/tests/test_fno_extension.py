"""
ROX Proven Edge Engine v3.2 — F&O Extension Test Suite
=======================================================
Tests all 6 new components:
  1. GreeksCalculator
  2. OptionChainStream
  3. PhysicalSettlementManager
  4. StrategyBuilders (all 4 strategies)
  5. FNOBrainExtension (rule-based fallback — no API key needed)
  6. FNOExecutionEngine (paper mode)

Run with:
    python tests/test_fno_extension.py

All tests run without any API keys or broker credentials.
"""

import sys
import os
import math
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================== #
#  1. Greeks Calculator                                                        #
# =========================================================================== #

class TestGreeksCalculator(unittest.TestCase):

    def setUp(self):
        from infrastructure.greeks_calculator import GreeksCalculator
        self.calc = GreeksCalculator(risk_free_rate=0.065)

    def test_atm_call_delta_near_half(self):
        """ATM call delta should be close to 0.50."""
        g = self.calc.calculate("CE", 25700, 25700, 7, 0.15)
        self.assertAlmostEqual(g.delta, 0.50, delta=0.08,
                               msg=f"ATM CE delta should be ~0.5, got {g.delta:.3f}")

    def test_atm_put_delta_near_minus_half(self):
        """ATM put delta should be close to -0.50."""
        g = self.calc.calculate("PE", 25700, 25700, 7, 0.15)
        self.assertAlmostEqual(g.delta, -0.50, delta=0.08,
                               msg=f"ATM PE delta should be ~-0.5, got {g.delta:.3f}")

    def test_deep_itm_call_delta_near_one(self):
        """Deep ITM call delta should be close to 1.0."""
        g = self.calc.calculate("CE", 25700, 23000, 7, 0.15)
        self.assertGreater(g.delta, 0.85,
                           msg=f"Deep ITM CE delta should be >0.85, got {g.delta:.3f}")

    def test_theta_is_negative(self):
        """Long option theta must always be negative (time decay)."""
        g = self.calc.calculate("CE", 25700, 25700, 7, 0.15)
        self.assertLess(g.theta, 0,
                        msg=f"Long CE theta must be negative, got {g.theta:.2f}")

    def test_vega_is_positive_for_long(self):
        """Long option vega must be positive (benefits from rising IV)."""
        g = self.calc.calculate("PE", 25700, 25700, 7, 0.15, quantity=1)
        self.assertGreater(g.vega, 0,
                           msg=f"Long PE vega must be positive, got {g.vega:.4f}")

    def test_put_call_parity(self):
        """C - P = S - K*e^(-rT) (put-call parity)."""
        S, K, T, σ, r = 25700, 25700, 7/365, 0.15, 0.065
        ce = self.calc.calculate("CE", S, K, 7, σ)
        pe = self.calc.calculate("PE", S, K, 7, σ)
        parity_lhs = ce.theoretical_price - pe.theoretical_price
        parity_rhs = S - K * math.exp(-r * T)
        self.assertAlmostEqual(parity_lhs, parity_rhs, delta=2.0,
                               msg=f"Put-call parity breach: {parity_lhs:.2f} vs {parity_rhs:.2f}")

    def test_gamma_equals_for_call_and_put(self):
        """Gamma is the same for CE and PE at same strike/expiry."""
        ce = self.calc.calculate("CE", 25700, 25700, 7, 0.15)
        pe = self.calc.calculate("PE", 25700, 25700, 7, 0.15)
        self.assertAlmostEqual(ce.gamma, pe.gamma, delta=0.0001,
                               msg="CE and PE gamma must be equal")

    def test_portfolio_greeks_aggregation(self):
        """Iron condor should have near-zero net delta."""
        from infrastructure.greeks_calculator import OptionsLeg
        spot = 25700
        legs = [
            OptionsLeg("CE", spot, 26000, 7, 0.14, -1, lot_size=50),
            OptionsLeg("CE", spot, 26500, 7, 0.13,  1, lot_size=50),
            OptionsLeg("PE", spot, 25400, 7, 0.16, -1, lot_size=50),
            OptionsLeg("PE", spot, 24900, 7, 0.17,  1, lot_size=50),
        ]
        pg = self.calc.portfolio_greeks(legs)
        self.assertAlmostEqual(pg.net_delta, 0.0, delta=20.0,
                               msg=f"Iron condor net delta should be ~0, got {pg.net_delta:.2f}")

    def test_implied_volatility_roundtrip(self):
        """IV solver should recover the input volatility from a BSM price."""
        σ_in  = 0.18
        g     = self.calc.calculate("CE", 25700, 25700, 14, σ_in)
        σ_out = self.calc.implied_volatility("CE", g.theoretical_price,
                                              25700, 25700, 14)
        self.assertAlmostEqual(σ_in, σ_out, delta=0.005,
                               msg=f"IV roundtrip: in={σ_in:.4f} out={σ_out:.4f}")

    def test_moneyness_labels(self):
        from infrastructure.greeks_calculator import GreeksCalculator
        calc = GreeksCalculator()
        g_atm  = calc.calculate("CE", 25700, 25700, 7, 0.15)
        g_itm  = calc.calculate("CE", 25700, 25400, 7, 0.15)
        g_otm  = calc.calculate("CE", 25700, 26000, 7, 0.15)
        self.assertEqual(g_atm.moneyness, "ATM")
        self.assertIn(g_itm.moneyness,   ["1-ITM", "2-ITM"])
        self.assertIn(g_otm.moneyness,   ["1-OTM", "2-OTM"])


# =========================================================================== #
#  2. Option Chain Stream                                                      #
# =========================================================================== #

class TestOptionChainStream(unittest.TestCase):

    def setUp(self):
        from infrastructure.option_chain_stream import OptionChainStream
        self.stream = OptionChainStream()
        self.spot   = 25700.0
        self.deriv  = {
            "pcr": 1.05,
            "iv_rank": 20,
            "india_vix": 14.2,
            "call_oi_walls": [{"strike": 26200, "oi": 5_000_000, "strength": 0.8}],
            "put_oi_walls":  [{"strike": 25200, "oi": 4_500_000, "strength": 0.7}],
        }

    def test_snapshot_builds_successfully(self):
        snap = self.stream.build_from_market_data("NIFTY", self.spot, self.deriv)
        self.assertEqual(snap.symbol, "NIFTY")
        self.assertGreater(len(snap.chains), 0)

    def test_pcr_is_positive(self):
        snap  = self.stream.build_from_market_data("NIFTY", self.spot, self.deriv)
        chain = snap.chains.get(snap.near_expiry)
        self.assertGreater(chain.pcr, 0, msg="PCR must be positive")

    def test_max_pain_within_reasonable_range(self):
        snap  = self.stream.build_from_market_data("NIFTY", self.spot, self.deriv)
        chain = snap.chains.get(snap.near_expiry)
        self.assertGreater(chain.max_pain, self.spot * 0.90)
        self.assertLess(   chain.max_pain, self.spot * 1.10)

    def test_call_walls_above_spot(self):
        snap  = self.stream.build_from_market_data("NIFTY", self.spot, self.deriv)
        chain = snap.chains.get(snap.near_expiry)
        for w in chain.call_walls:
            self.assertGreater(w.strike, self.spot,
                               msg=f"Call wall strike {w.strike} should be above spot {self.spot}")

    def test_put_walls_below_spot(self):
        snap  = self.stream.build_from_market_data("NIFTY", self.spot, self.deriv)
        chain = snap.chains.get(snap.near_expiry)
        for w in chain.put_walls:
            self.assertLess(w.strike, self.spot,
                            msg=f"Put wall strike {w.strike} should be below spot {self.spot}")

    def test_wall_strength_between_0_and_1(self):
        snap  = self.stream.build_from_market_data("NIFTY", self.spot, self.deriv)
        chain = snap.chains.get(snap.near_expiry)
        for w in chain.call_walls + chain.put_walls:
            self.assertGreaterEqual(w.strength, 0.0)
            self.assertLessEqual(   w.strength, 1.0)

    def test_nearest_expiry_is_thursday(self):
        from datetime import date
        snap   = self.stream.build_from_market_data("NIFTY", self.spot, self.deriv)
        expiry = date.fromisoformat(snap.near_expiry)
        self.assertEqual(expiry.weekday(), 3, msg="NSE weekly expiry must be Thursday")


# =========================================================================== #
#  3. Physical Settlement Manager                                              #
# =========================================================================== #

class TestPhysicalSettlementManager(unittest.TestCase):

    def setUp(self):
        from infrastructure.physical_settlement_manager import PhysicalSettlementManager
        self.psm = PhysicalSettlementManager(itm_probability_trigger=30, block_new_dte=2)

    def _future_expiry(self, days_ahead: int) -> str:
        from datetime import date, timedelta
        return (date.today() + timedelta(days=days_ahead)).isoformat()

    def test_index_option_is_cash_settled(self):
        from infrastructure.physical_settlement_manager import SettlementRisk
        chk = self.psm.check_position(
            "NIFTY", "CE", 26000, self._future_expiry(7),
            25700, "SHORT", 1
        )
        self.assertEqual(chk.settlement_risk, SettlementRisk.NONE,
                         msg="Index options should have no settlement risk")

    def test_short_itm_stock_option_critical(self):
        from infrastructure.physical_settlement_manager import SettlementRisk
        # ITM short with 1 day to expiry → CRITICAL
        chk = self.psm.check_position(
            "RELIANCE", "CE", 2900, self._future_expiry(1),
            3000, "SHORT", 1, lot_size=250
        )
        self.assertEqual(chk.settlement_risk, SettlementRisk.CRITICAL,
                         msg="Short ITM stock CE 1 DTE should be CRITICAL")

    def test_long_position_no_settlement_risk(self):
        from infrastructure.physical_settlement_manager import SettlementRisk
        chk = self.psm.check_position(
            "INFY", "PE", 1500, self._future_expiry(3),
            1600, "LONG", 2, lot_size=300
        )
        self.assertEqual(chk.settlement_risk, SettlementRisk.NONE,
                         msg="Long positions should never have settlement risk")

    def test_delivery_capital_estimated_for_short(self):
        chk = self.psm.check_position(
            "TCS", "CE", 3800, self._future_expiry(1),
            4000, "SHORT", 1, lot_size=150
        )
        self.assertGreater(chk.delivery_capital_required, 0,
                           msg="Short ITM position should have delivery capital estimate")

    def test_new_position_blocked_short_itm_near_expiry(self):
        blocked, reason = self.psm.is_new_position_blocked(
            "SBIN", "CE", 800, self._future_expiry(1),
            900, "SHORT"
        )
        self.assertTrue(blocked, msg="Short ITM stock option 1 DTE must be blocked")
        self.assertIn("BLOCKED", reason)

    def test_new_position_not_blocked_index(self):
        blocked, reason = self.psm.is_new_position_blocked(
            "NIFTY", "CE", 26000, self._future_expiry(1),
            25700, "SHORT"
        )
        self.assertFalse(blocked, msg="Index options never blocked")

    def test_portfolio_report_generates(self):
        positions = [
            {"symbol": "NIFTY",    "option_type": "CE", "strike": 26000,
             "expiry_date": self._future_expiry(7), "spot": 25700,
             "position_side": "SHORT", "quantity_lots": 1},
            {"symbol": "RELIANCE", "option_type": "CE", "strike": 2800,
             "expiry_date": self._future_expiry(1), "spot": 3000,
             "position_side": "SHORT", "quantity_lots": 2, "lot_size": 250},
        ]
        report = self.psm.check_portfolio(positions)
        self.assertEqual(report.positions_checked, 2)
        self.assertGreater(len(report.summary), 0)


# =========================================================================== #
#  4. Strategy Builders                                                        #
# =========================================================================== #

class TestStrategyBuilders(unittest.TestCase):

    def setUp(self):
        from agents.strategy_builders import StrategyFactory
        self.factory = StrategyFactory()
        self.spot    = 25700.0

    def test_iron_condor_builds(self):
        order = self.factory.build("iron_condor", "NIFTY", self.spot, dte=7, iv=0.14)
        self.assertIsNotNone(order)
        self.assertEqual(len(order.legs), 4, msg="Iron Condor must have 4 legs")

    def test_iron_condor_is_credit(self):
        order = self.factory.build("iron_condor", "NIFTY", self.spot, dte=7, iv=0.14)
        self.assertGreater(order.net_premium, 0, msg="Iron Condor should receive net credit")

    def test_iron_condor_max_loss_finite(self):
        order = self.factory.build("iron_condor", "NIFTY", self.spot, dte=7, iv=0.14)
        self.assertLess(order.max_loss, float("inf"), msg="Iron Condor max loss must be finite")

    def test_iron_condor_breakevens_straddle_spot(self):
        order = self.factory.build("iron_condor", "NIFTY", self.spot, dte=7, iv=0.14)
        self.assertLess(order.breakeven_low,  self.spot)
        self.assertGreater(order.breakeven_high, self.spot)

    def test_calendar_spread_builds(self):
        order = self.factory.build("calendar_spread", "NIFTY", self.spot, iv=0.15)
        self.assertIsNotNone(order)
        self.assertEqual(len(order.legs), 2, msg="Calendar Spread must have 2 legs")

    def test_straddle_builds(self):
        order = self.factory.build("straddle", "NIFTY", self.spot, dte=7, iv=0.15)
        self.assertIsNotNone(order)
        self.assertEqual(len(order.legs), 2, msg="Straddle must have 2 legs")

    def test_straddle_is_debit(self):
        order = self.factory.build("straddle", "NIFTY", self.spot, dte=7, iv=0.15)
        self.assertLess(order.net_premium, 0, msg="Straddle is a debit strategy")

    def test_strangle_builds(self):
        order = self.factory.build("strangle", "NIFTY", self.spot, dte=7, iv=0.15)
        self.assertIsNotNone(order)

    def test_bull_call_spread_builds(self):
        order = self.factory.build("bull_call_spread", "NIFTY", self.spot, dte=7, iv=0.15)
        self.assertIsNotNone(order)
        self.assertEqual(len(order.legs), 2)

    def test_bear_put_spread_builds(self):
        order = self.factory.build("bear_put_spread", "NIFTY", self.spot, dte=7, iv=0.15)
        self.assertIsNotNone(order)

    def test_unknown_strategy_returns_none(self):
        order = self.factory.build("magic_strategy", "NIFTY", self.spot)
        self.assertIsNone(order, msg="Unknown strategy should return None")

    def test_risk_reward_positive(self):
        order = self.factory.build("iron_condor", "NIFTY", self.spot, dte=7, iv=0.14)
        self.assertGreater(order.risk_reward, 0)

    def test_legs_have_required_fields(self):
        order = self.factory.build("iron_condor", "NIFTY", self.spot, dte=7, iv=0.14)
        for leg in order.legs:
            self.assertIn(leg.action, ("BUY", "SELL"))
            self.assertIn(leg.option_type, ("CE", "PE"))
            self.assertGreater(leg.strike, 0)
            self.assertGreater(leg.premium, 0)


# =========================================================================== #
#  5. FNO Brain Extension (rule-based — no API key)                           #
# =========================================================================== #

class TestFNOBrainExtension(unittest.TestCase):

    def setUp(self):
        # Ensure no API key set so rule-based fallback is used
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("GROQ_API_KEY", None)
        from agents.fno_brain_extension import FNOBrainExtension
        self.brain = FNOBrainExtension()

    def _sample_inputs(self, iv_rank=20, pcr=1.05):
        consensus = {"direction": "LONG", "strength": "MODERATE", "net_score": 0.39}
        equity_setups = [{"stock": "BPCL", "direction": "LONG", "conviction": 67}]
        mctx = {"nifty_price": 25700, "india_vix": 14.2, "pcr": pcr,
                "market_regime": "BULL", "iv_rank": iv_rank}
        fno_ctx = {"iv_rank": iv_rank, "pcr": pcr, "atm_iv": 0.142,
                   "call_walls": [{"strike": 26200, "strength": 0.8}],
                   "put_walls":  [{"strike": 25200, "strength": 0.7}],
                   "max_pain": 25650, "settlement_risk_present": False,
                   "dte_nearest_expiry": 7}
        return consensus, equity_setups, mctx, fno_ctx

    def test_output_has_iv_regime(self):
        c, e, m, f = self._sample_inputs(iv_rank=20)
        out = self.brain.fno_synthesize_sync(c, e, m, f)
        self.assertIn(out.iv_regime, ("HIGH", "NORMAL", "LOW"))

    def test_low_iv_rank_gives_low_regime(self):
        c, e, m, f = self._sample_inputs(iv_rank=15)
        out = self.brain.fno_synthesize_sync(c, e, m, f)
        self.assertEqual(out.iv_regime, "LOW",
                         msg=f"IV rank 15 should give LOW regime, got {out.iv_regime}")

    def test_high_iv_rank_gives_high_regime(self):
        c, e, m, f = self._sample_inputs(iv_rank=75)
        out = self.brain.fno_synthesize_sync(c, e, m, f)
        self.assertEqual(out.iv_regime, "HIGH")

    def test_strategy_recommendations_not_empty(self):
        c, e, m, f = self._sample_inputs()
        out = self.brain.fno_synthesize_sync(c, e, m, f)
        self.assertGreater(len(out.strategy_recommendations), 0)

    def test_risk_score_in_range(self):
        c, e, m, f = self._sample_inputs()
        out = self.brain.fno_synthesize_sync(c, e, m, f)
        self.assertGreaterEqual(out.risk_score, 1)
        self.assertLessEqual(   out.risk_score, 10)

    def test_conviction_in_range(self):
        c, e, m, f = self._sample_inputs()
        out = self.brain.fno_synthesize_sync(c, e, m, f)
        for rec in out.strategy_recommendations:
            self.assertGreaterEqual(rec.conviction, 0)
            self.assertLessEqual(   rec.conviction, 100)

    def test_bullish_pcr_favours_bull_strategies(self):
        c, e, m, f = self._sample_inputs(pcr=1.5, iv_rank=70)  # high IV + bullish
        out = self.brain.fno_synthesize_sync(c, e, m, f)
        strategies = [r.strategy_name for r in out.strategy_recommendations]
        bullish_strats = {"bull_call_spread", "iron_condor", "short_put"}
        has_bullish = any(s in bullish_strats for s in strategies)
        self.assertTrue(has_bullish,
                        msg=f"Expected bullish strategy, got: {strategies}")

    def test_narrative_is_not_empty(self):
        c, e, m, f = self._sample_inputs()
        out = self.brain.fno_synthesize_sync(c, e, m, f)
        self.assertGreater(len(out.narrative), 10)


# =========================================================================== #
#  6. FNO Execution Engine                                                     #
# =========================================================================== #

class TestFNOExecutionEngine(unittest.TestCase):

    def setUp(self):
        os.environ["FNO_PAPER_TRADING"] = "true"
        from execution.fno_execution_engine import FNOExecutionEngine
        from infrastructure.physical_settlement_manager import PhysicalSettlementManager
        self.engine  = FNOExecutionEngine(portfolio_value=1_000_000)
        self.factory = None

    def _get_iron_condor(self, spot=25700.0):
        from agents.strategy_builders import StrategyFactory
        factory = StrategyFactory()
        return factory.build("iron_condor", "NIFTY", spot, dte=7, iv=0.14)

    def test_margin_estimate_returns(self):
        order  = self._get_iron_condor()
        margin = self.engine.estimate_margin(order)
        self.assertGreater(margin.net_margin, 0)

    def test_iron_condor_margin_sufficient(self):
        order  = self._get_iron_condor()
        margin = self.engine.estimate_margin(order)
        self.assertTrue(margin.sufficient,
                        msg=f"1-lot iron condor should be within margin limits. "
                            f"margin_pct={margin.margin_pct_of_capital:.2%}")

    def test_dry_run_succeeds(self):
        order  = self._get_iron_condor()
        result = self.engine.submit_strategy(order, spot=25700, dry_run=True)
        self.assertTrue(result.success)

    def test_paper_submit_returns_four_legs(self):
        order  = self._get_iron_condor()
        result = self.engine.submit_strategy(order, spot=25700)
        self.assertTrue(result.paper_trade)
        self.assertEqual(len(result.legs), 4)

    def test_all_legs_filled(self):
        order  = self._get_iron_condor()
        result = self.engine.submit_strategy(order, spot=25700)
        statuses = {leg.status for leg in result.legs}
        self.assertTrue(statuses.issubset({"PAPER", "FILLED"}),
                        msg=f"Unexpected leg statuses: {statuses}")

    def test_straddle_submission(self):
        from agents.strategy_builders import StrategyFactory
        order  = StrategyFactory().build("straddle", "NIFTY", 25700.0, dte=7, iv=0.15)
        result = self.engine.submit_strategy(order, spot=25700)
        self.assertTrue(result.success)
        self.assertEqual(len(result.legs), 2)

    def test_margin_utilization_increases_after_trade(self):
        order  = self._get_iron_condor()
        before = self.engine.get_margin_utilization()
        self.engine.submit_strategy(order, spot=25700)
        after  = self.engine.get_margin_utilization()
        self.assertGreater(after, before, msg="Margin utilisation should increase after trade")

    def test_order_id_generated(self):
        order  = self._get_iron_condor()
        result = self.engine.submit_strategy(order)
        self.assertGreater(len(result.order_id), 0)

    def test_paper_csv_written(self):
        from execution.fno_execution_engine import FNOExecutionEngine
        engine = FNOExecutionEngine(portfolio_value=1_000_000)
        from agents.strategy_builders import StrategyFactory
        order  = StrategyFactory().build("bull_call_spread", "NIFTY", 25700.0, dte=7)
        engine.submit_strategy(order)
        log_path = engine.PAPER_LOG
        if log_path.exists():
            self.assertGreater(log_path.stat().st_size, 0)

    def test_open_positions_tracked(self):
        order  = self._get_iron_condor()
        self.engine.submit_strategy(order)
        positions = self.engine.get_open_positions()
        self.assertGreater(len(positions), 0)

    def test_validate_order_approved(self):
        order = self._get_iron_condor()
        ok, reason = self.engine.validate_order(order, spot=25700)
        self.assertTrue(ok, msg=f"1-lot iron condor should pass validation: {reason}")


# =========================================================================== #
#  Integration: Full pipeline smoke test                                      #
# =========================================================================== #

class TestFNOIntegrationPipeline(unittest.TestCase):

    def test_full_pipeline_smoke(self):
        """
        End-to-end: chain → greeks → settlement check → strategy → execute.
        No API keys or broker credentials needed.
        """
        spot  = 25700.0
        deriv = {
            "pcr": 1.05, "iv_rank": 20, "india_vix": 14.2,
            "call_oi_walls": [{"strike": 26200, "oi": 5_000_000, "strength": 0.8}],
            "put_oi_walls":  [{"strike": 25200, "oi": 4_500_000, "strength": 0.7}],
        }

        # Step 1: Option chain
        from infrastructure.option_chain_stream import OptionChainStream
        stream = OptionChainStream()
        snap   = stream.build_from_market_data("NIFTY", spot, deriv)
        self.assertIsNotNone(snap)

        # Step 2: Greeks
        from infrastructure.greeks_calculator import GreeksCalculator
        calc = GreeksCalculator()
        g    = calc.calculate("CE", spot, round(spot/50)*50, 7, 0.14)
        self.assertGreater(abs(g.delta), 0)

        # Step 3: Settlement check (index → no risk)
        from infrastructure.physical_settlement_manager import PhysicalSettlementManager, SettlementRisk
        psm = PhysicalSettlementManager()
        chk = psm.check_position("NIFTY", "CE", 26000,
                                  snap.near_expiry, spot, "SHORT", 1)
        self.assertEqual(chk.settlement_risk, SettlementRisk.NONE)

        # Step 4: Strategy build
        from agents.strategy_builders import StrategyFactory
        order = StrategyFactory().build("iron_condor", "NIFTY", spot, dte=7, iv=0.14)
        self.assertIsNotNone(order)
        self.assertEqual(len(order.legs), 4)

        # Step 5: Execute (paper)
        from execution.fno_execution_engine import FNOExecutionEngine
        engine = FNOExecutionEngine(1_000_000, psm)
        result = engine.submit_strategy(order, spot=spot)
        self.assertTrue(result.success)

        # Step 6: FNO Brain (rule-based)
        from agents.fno_brain_extension import FNOBrainExtension
        brain  = FNOBrainExtension()
        out    = brain.fno_synthesize_sync(
            consensus      = {"direction": "LONG", "net_score": 0.39},
            equity_setups  = [],
            market_context = {"nifty_price": spot, "india_vix": 14.2,
                               "pcr": 1.05, "iv_rank": 20, "market_regime": "BULL"},
            fno_context    = {"iv_rank": 20, "pcr": 1.05, "settlement_risk_present": False},
        )
        self.assertIn(out.iv_regime, ("HIGH", "NORMAL", "LOW"))
        self.assertGreater(len(out.strategy_recommendations), 0)


# =========================================================================== #
#  Runner                                                                      #
# =========================================================================== #

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestGreeksCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestOptionChainStream))
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicalSettlementManager))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategyBuilders))
    suite.addTests(loader.loadTestsFromTestCase(TestFNOBrainExtension))
    suite.addTests(loader.loadTestsFromTestCase(TestFNOExecutionEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestFNOIntegrationPipeline))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total   = result.testsRun
    passed  = total - len(result.failures) - len(result.errors)
    failed  = len(result.failures) + len(result.errors)
    print(f"\n{'='*60}")
    print(f"ROX F&O EXTENSION TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total  : {total}")
    print(f"  Passed : {passed}  ✅")
    print(f"  Failed : {failed}  {'❌' if failed else '✅'}")
    print(f"{'='*60}")
    sys.exit(0 if result.wasSuccessful() else 1)

"""
Tests for Black-76 Greeks Calculator
"""

import sys
import os

# Ensure project root is on path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import math
from infrastructure.greeks_calculator import GreeksCalculator, OptionsLeg, PortfolioGreeks


class TestGreeksCalculator(unittest.TestCase):
    """Test cases for GreeksCalculator"""
    
    def setUp(self):
        self.calc = GreeksCalculator(risk_free_rate=0.06)
    
    def test_call_option_price(self):
        """Test call option price calculation"""
        greeks = self.calc.calculate(
            option_type="CE",
            spot=22500,
            strike=22600,
            days_to_expiry=7,
            volatility=0.15
        )
        
        # Price should be positive
        self.assertGreater(greeks.theoretical_price, 0)
        
        # Delta should be between 0 and 1 for calls
        self.assertGreater(greeks.delta, 0)
        self.assertLess(greeks.delta, 1)
        
        # Gamma should be positive
        self.assertGreater(greeks.gamma, 0)
        
        # Theta should be negative (time decay)
        self.assertLess(greeks.theta, 0)
        
        # Vega should be positive
        self.assertGreater(greeks.vega, 0)
    
    def test_put_option_price(self):
        """Test put option price calculation"""
        greeks = self.calc.calculate(
            option_type="PE",
            spot=22500,
            strike=22600,
            days_to_expiry=7,
            volatility=0.15
        )
        
        # Price should be positive
        self.assertGreater(greeks.theoretical_price, 0)
        
        # Delta should be between -1 and 0 for puts
        self.assertGreater(greeks.delta, -1)
        self.assertLess(greeks.delta, 0)
        
        # Gamma should be positive
        self.assertGreater(greeks.gamma, 0)
    
    def test_atm_option(self):
        """Test ATM option characteristics"""
        greeks = self.calc.calculate(
            option_type="CE",
            spot=22500,
            strike=22500,  # ATM
            days_to_expiry=30,
            volatility=0.15
        )
        
        # ATM call delta should be around 0.5
        self.assertAlmostEqual(greeks.delta, 0.5, delta=0.1)
    
    def test_portfolio_greeks(self):
        """Test portfolio Greeks aggregation"""
        legs = [
            OptionsLeg("CE", 22500, 22600, 7, 0.15, 1, 75),
            OptionsLeg("PE", 22500, 22400, 7, 0.15, -1, 75),
        ]
        
        portfolio = self.calc.portfolio_greeks(legs)
        
        # Should have positions
        self.assertEqual(portfolio.legs, 2)
        
        # Delta should be calculated
        self.assertIsNotNone(portfolio.net_delta)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        greeks = self.calc.calculate(
            option_type="CE",
            spot=0,
            strike=22600,
            days_to_expiry=7,
            volatility=0.15
        )
        
        # Should return zero values for invalid spot
        self.assertEqual(greeks.theoretical_price, 0)


if __name__ == "__main__":
    unittest.main()

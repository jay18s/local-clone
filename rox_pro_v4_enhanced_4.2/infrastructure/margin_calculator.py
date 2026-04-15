"""
ROX Proven Edge Engine v4.0 - Margin Calculator
===============================================
Calculates SPAN and Exposure margins for NSE F&O positions.

NSE uses SPAN (Standard Portfolio Analysis of Risk) margin system:
- SPAN Margin: Risk-based margin calculated using 16 scenarios
- Exposure Margin: Additional margin for extreme moves
- Total Margin = SPAN + Exposure

For options:
- Short options: Full margin required
- Long options: Premium paid (no additional margin)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class PositionType(Enum):
    """Position type"""
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class MarginResult:
    """Margin calculation result"""
    symbol: str
    position_type: str
    quantity: int
    lot_size: int
    
    # Margin components
    span_margin: float = 0.0
    exposure_margin: float = 0.0
    premium: float = 0.0
    
    # Totals
    total_margin: float = 0.0
    margin_per_lot: float = 0.0
    
    # Breakdown
    scenario_worst: float = 0.0
    scenario_best: float = 0.0
    
    def __post_init__(self):
        self.total_margin = self.span_margin + self.exposure_margin + self.premium
        if self.quantity > 0:
            self.margin_per_lot = self.total_margin / self.quantity


@dataclass
class PortfolioMargin:
    """Aggregate portfolio margin"""
    total_span: float = 0.0
    total_exposure: float = 0.0
    total_premium: float = 0.0
    total_margin: float = 0.0
    
    # Benefit details
    spread_benefit: float = 0.0
    hedge_benefit: float = 0.0
    
    # Position counts
    num_positions: int = 0
    
    # Margin utilization
    available_funds: float = 0.0
    margin_utilization_pct: float = 0.0


class MarginCalculator:
    """
    SPAN + Exposure Margin Calculator for NSE F&O.
    
    Implements simplified SPAN methodology:
    - 16 scenario analysis for SPAN margin
    - Exposure margin based on contract value
    - Spread benefits for hedged positions
    
    Note: This is a simplified implementation. Production systems
    should use NSE's official SPAN files or broker APIs for accurate
    margin calculations.
    """
    
    # SPAN parameters (simplified)
    SPAN_PRICE_SCAN_RANGE = 0.035  # 3.5% price scan range
    SPAN_VOL_SCAN_RANGE = 0.25     # 25% volatility scan range
    
    # Exposure margin rates
    EXPOSURE_FUTURE_PCT = 0.03     # 3% of contract value for futures
    EXPOSURE_OPTION_PCT = 0.015    # 1.5% for short options
    
    # Minimum margins
    MIN_FUTURE_MARGIN_PCT = 0.05   # Minimum 5% for futures
    MIN_OPTION_MARGIN_PCT = 0.025  # Minimum 2.5% for short options
    
    def __init__(self, risk_free_rate: float = 0.06):
        """
        Initialize margin calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        self._span_scenarios = self._generate_span_scenarios()
    
    def _generate_span_scenarios(self) -> List[Tuple[float, float]]:
        """
        Generate 16 SPAN scenarios.
        
        Each scenario is (price_change_pct, vol_change_pct)
        """
        scenarios = []
        price_ranges = [-1.0, -0.67, -0.33, 0, 0.33, 0.67, 1.0]
        vol_ranges = [-1.0, 0, 1.0]
        
        for price_mult in price_ranges:
            for vol_mult in vol_ranges:
                scenarios.append((
                    price_mult * self.SPAN_PRICE_SCAN_RANGE,
                    vol_mult * self.SPAN_VOL_SCAN_RANGE
                ))
        
        return scenarios
    
    def calculate_future_margin(
        self,
        symbol: str,
        futures_price: float,
        quantity: int,
        lot_size: int,
        position_type: str = "LONG"
    ) -> MarginResult:
        """
        Calculate margin for a futures position.
        
        Args:
            symbol: Contract symbol
            futures_price: Current futures price
            quantity: Number of lots
            lot_size: Lot size
            position_type: LONG or SHORT
            
        Returns:
            MarginResult with margin details
        """
        contract_value = futures_price * lot_size
        
        # SPAN margin (simplified - using price scan range)
        span_margin = contract_value * self.SPAN_PRICE_SCAN_RANGE * quantity
        
        # Exposure margin
        exposure_margin = contract_value * self.EXPOSURE_FUTURE_PCT * quantity
        
        # Ensure minimum margin
        min_margin = contract_value * self.MIN_FUTURE_MARGIN_PCT * quantity
        if span_margin + exposure_margin < min_margin:
            span_margin = min_margin * 0.7
            exposure_margin = min_margin * 0.3
        
        return MarginResult(
            symbol=symbol,
            position_type=position_type,
            quantity=quantity,
            lot_size=lot_size,
            span_margin=round(span_margin, 2),
            exposure_margin=round(exposure_margin, 2),
            premium=0.0,
            scenario_worst=-span_margin,
            scenario_best=span_margin
        )
    
    def calculate_option_margin(
        self,
        symbol: str,
        spot_price: float,
        strike_price: float,
        option_type: str,  # CE or PE
        iv: float,
        dte: int,
        quantity: int,
        lot_size: int,
        option_price: float,
        position_type: str = "LONG"
    ) -> MarginResult:
        """
        Calculate margin for an option position.
        
        Args:
            symbol: Contract symbol
            spot_price: Current spot price
            strike_price: Option strike
            option_type: CE or PE
            iv: Implied volatility
            dte: Days to expiry
            quantity: Number of lots
            lot_size: Lot size
            option_price: Current option price
            position_type: LONG or SHORT
            
        Returns:
            MarginResult with margin details
        """
        contract_value = spot_price * lot_size
        premium_paid = option_price * lot_size * quantity
        
        # Long options: only premium required
        if position_type == "LONG":
            return MarginResult(
                symbol=symbol,
                position_type=position_type,
                quantity=quantity,
                lot_size=lot_size,
                span_margin=0.0,
                exposure_margin=0.0,
                premium=premium_paid,
                scenario_worst=-premium_paid,
                scenario_best=float('inf')
            )
        
        # Short options: full margin required
        # SPAN margin (simplified)
        # For short options, margin is based on max loss in scenarios
        span_margin = self._calculate_short_option_span(
            spot_price, strike_price, option_type, iv, dte, lot_size
        ) * quantity
        
        # Exposure margin
        exposure_margin = contract_value * self.EXPOSURE_OPTION_PCT * quantity
        
        # Ensure minimum margin
        min_margin = contract_value * self.MIN_OPTION_MARGIN_PCT * quantity
        if span_margin + exposure_margin < min_margin:
            span_margin = min_margin * 0.6
            exposure_margin = min_margin * 0.4
        
        # Premium received (negative as it reduces margin requirement)
        premium_received = -premium_paid
        
        return MarginResult(
            symbol=symbol,
            position_type=position_type,
            quantity=quantity,
            lot_size=lot_size,
            span_margin=round(span_margin, 2),
            exposure_margin=round(exposure_margin, 2),
            premium=premium_received,
            scenario_worst=-span_margin,
            scenario_best=premium_paid
        )
    
    def _calculate_short_option_span(
        self,
        spot: float,
        strike: float,
        option_type: str,
        iv: float,
        dte: int,
        lot_size: int
    ) -> float:
        """
        Calculate SPAN margin for short option.
        
        Simplified calculation using Black-Scholes scenarios.
        """
        from infrastructure.greeks_calculator import GreeksCalculator
        
        calc = GreeksCalculator()
        
        # Calculate current option price
        current_greeks = calc.calculate(option_type, spot, strike, dte, iv)
        current_price = current_greeks.theoretical_price
        
        # Find worst scenario
        max_loss = 0
        
        for price_change, vol_change in self._span_scenarios:
            new_spot = spot * (1 + price_change)
            new_iv = max(0.01, iv * (1 + vol_change))
            
            scenario_greeks = calc.calculate(option_type, new_spot, strike, dte, new_iv)
            scenario_price = scenario_greeks.theoretical_price
            
            # For short option, loss = price increase
            loss = max(0, scenario_price - current_price)
            max_loss = max(max_loss, loss)
        
        return max_loss * lot_size
    
    def calculate_portfolio_margin(
        self,
        margins: List[MarginResult],
        available_funds: float = 0.0
    ) -> PortfolioMargin:
        """
        Calculate aggregate portfolio margin.
        
        Args:
            margins: List of individual margin results
            available_funds: Available trading funds
            
        Returns:
            PortfolioMargin with aggregated values
        """
        portfolio = PortfolioMargin()
        portfolio.available_funds = available_funds
        
        for margin in margins:
            portfolio.total_span += margin.span_margin
            portfolio.total_exposure += margin.exposure_margin
            portfolio.total_premium += margin.premium
            portfolio.num_positions += 1
        
        portfolio.total_margin = (
            portfolio.total_span + 
            portfolio.total_exposure + 
            portfolio.total_premium
        )
        
        # Calculate utilization
        if available_funds > 0:
            portfolio.margin_utilization_pct = (
                portfolio.total_margin / available_funds * 100
            )
        
        return portfolio
    
    def check_margin_sufficiency(
        self,
        required_margin: float,
        available_margin: float,
        buffer_pct: float = 1.2
    ) -> Dict:
        """
        Check if available margin is sufficient.
        
        Args:
            required_margin: Required margin for position
            available_margin: Available margin
            buffer_pct: Safety buffer percentage (default 20%)
            
        Returns:
            Dict with check results
        """
        required_with_buffer = required_margin * buffer_pct
        
        return {
            "required_margin": required_margin,
            "required_with_buffer": required_with_buffer,
            "available_margin": available_margin,
            "sufficient": available_margin >= required_with_buffer,
            "shortfall": max(0, required_with_buffer - available_margin),
            "utilization_pct": (required_margin / available_margin * 100) 
                              if available_margin > 0 else float('inf')
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def calculate_order_margin(
    symbol: str,
    transaction_type: str,  # BUY or SELL
    product_type: str,      # NRML, MIS, CNC
    quantity: int,
    price: float,
    trigger_price: Optional[float] = None
) -> Dict:
    """
    Calculate margin required for an order.
    
    This is a simplified wrapper for common order scenarios.
    
    Args:
        symbol: Trading symbol
        transaction_type: BUY or SELL
        product_type: NRML, MIS, CNC
        quantity: Order quantity
        price: Order price
        trigger_price: Trigger price for SL orders
        
    Returns:
        Dict with margin details
    """
    # This would integrate with broker API in production
    # For now, return a placeholder
    return {
        "symbol": symbol,
        "transaction_type": transaction_type,
        "product_type": product_type,
        "quantity": quantity,
        "estimated_margin": quantity * price * 0.15,  # Rough estimate
        "note": "Use broker API for accurate margin calculation"
    }

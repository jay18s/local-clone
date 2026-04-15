"""
ROX Proven Edge Engine v4.0 - THETA Agent (Greeks Management)
=============================================================
F&O Greeks Management Agent - Portfolio-level Greeks tracking and hedging.

THETA manages portfolio Greeks exposure:
- Real-time Greeks calculation
- Portfolio-level risk tracking
- Greeks-based hedging recommendations
- Risk limit monitoring
- Rebalancing alerts
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import threading

from infrastructure.greeks_calculator import (
    GreeksCalculator, Greeks as GreeksResult, OptionsLeg
)
from config import FnoRiskLimits


@dataclass
class PortfolioGreeks:
    """Portfolio-level aggregated Greeks (internal to ThetaAgent)."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    delta_exposure_inr: float = 0.0
    gamma_exposure_inr: float = 0.0
    theta_exposure_inr: float = 0.0
    vega_exposure_inr: float = 0.0
    num_positions: int = 0


class GreeksAlertType(Enum):
    """Greeks alert types"""
    DELTA_LIMIT = "DELTA_LIMIT"
    GAMMA_LIMIT = "GAMMA_LIMIT"
    THETA_LIMIT = "THETA_LIMIT"
    VEGA_LIMIT = "VEGA_LIMIT"
    REBALANCE_NEEDED = "REBALANCE_NEEDED"


@dataclass
class GreeksAlert:
    """Greeks alert"""
    alert_type: GreeksAlertType
    symbol: Optional[str]
    current_value: float
    limit_value: float
    breach_pct: float
    timestamp: datetime
    recommendation: str


@dataclass
class PositionGreeks:
    """Greeks for a single position"""
    symbol: str
    position_type: str  # LONG or SHORT
    quantity: int
    lot_size: int
    
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    # INR exposures
    delta_exposure: float = 0.0
    gamma_exposure: float = 0.0
    theta_exposure: float = 0.0
    vega_exposure: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0  # 95% VaR
    max_loss_scenario: float = 0.0


class ThetaAgent:
    """
    THETA - F&O Greeks Management Agent.
    
    Responsibilities:
    - Portfolio Greeks calculation and tracking
    - Risk limit monitoring
    - Hedging recommendations
    - Rebalancing alerts
    - Greeks-based position sizing
    
    Monitors:
    - Delta exposure (directional risk)
    - Gamma exposure (convexity risk)
    - Theta decay (time decay)
    - Vega exposure (volatility risk)
    """
    
    def __init__(
        self,
        risk_limits: Optional[FnoRiskLimits] = None,
        greeks_calculator: Optional[GreeksCalculator] = None
    ):
        """
        Initialize THETA agent.
        
        Args:
            risk_limits: F&O risk limits
            greeks_calculator: Greeks calculator instance
        """
        self.risk_limits = risk_limits or FnoRiskLimits()
        self.greeks_calc = greeks_calculator or GreeksCalculator()
        
        self._positions: Dict[str, PositionGreeks] = {}
        self._portfolio_greeks: PortfolioGreeks = PortfolioGreeks()
        self._alerts: List[GreeksAlert] = []
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()
        self._running = False
    
    def register_callback(self, callback: Callable):
        """Register alert callback"""
        self._callbacks.append(callback)
    
    def add_position(
        self,
        symbol: str,
        option_type: str,
        spot: float,
        strike: float,
        dte: int,
        iv: float,
        quantity: int,
        lot_size: int = 50
    ) -> PositionGreeks:
        """
        Add an options position and calculate its Greeks.
        
        Args:
            symbol: Position symbol
            option_type: CE or PE
            spot: Current spot price
            strike: Strike price
            dte: Days to expiry
            iv: Implied volatility
            quantity: Position quantity (positive=long, negative=short)
            lot_size: Lot size
            
        Returns:
            PositionGreeks for the position
        """
        # Calculate individual Greeks
        greeks = self.greeks_calc.calculate(option_type, spot, strike, dte, iv)
        
        # Create position Greeks
        position = PositionGreeks(
            symbol=symbol,
            position_type="LONG" if quantity > 0 else "SHORT",
            quantity=abs(quantity),
            lot_size=lot_size
        )
        
        # Scale Greeks by position size
        multiplier = quantity * lot_size
        
        position.delta = greeks.delta * multiplier
        position.gamma = greeks.gamma * multiplier
        position.theta = greeks.theta * multiplier
        position.vega = greeks.vega * multiplier
        position.rho = greeks.rho * multiplier
        
        # Calculate INR exposures
        position.delta_exposure = position.delta * spot
        position.gamma_exposure = position.gamma * spot * 100  # Per 1% move
        position.theta_exposure = position.theta
        position.vega_exposure = position.vega * 100  # Per 1 vol point
        
        # Store position
        with self._lock:
            self._positions[symbol] = position
        
        # Recalculate portfolio Greeks
        self._recalculate_portfolio()
        
        # Check limits
        self._check_limits()
        
        return position
    
    def remove_position(self, symbol: str) -> bool:
        """
        Remove a position from tracking.
        
        Args:
            symbol: Position symbol
            
        Returns:
            True if removed successfully
        """
        with self._lock:
            if symbol in self._positions:
                del self._positions[symbol]
                self._recalculate_portfolio()
                return True
            return False
    
    def update_position_price(
        self,
        symbol: str,
        new_spot: float,
        new_iv: Optional[float] = None
    ) -> Optional[PositionGreeks]:
        """
        Update position Greeks with new price/volatility.
        
        Args:
            symbol: Position symbol
            new_spot: New spot price
            new_iv: New implied volatility (optional)
            
        Returns:
            Updated PositionGreeks
        """
        with self._lock:
            position = self._positions.get(symbol)
            if not position:
                return None
            
            # Would need original strike, dte, option_type to recalculate
            # For now, approximate delta change
            price_change_pct = (new_spot - position.delta_exposure / position.delta) / (position.delta_exposure / position.delta) if position.delta != 0 else 0
            position.delta *= (1 + price_change_pct * position.gamma / position.delta) if position.delta != 0 else 1
            position.delta_exposure = position.delta * new_spot
            
            self._recalculate_portfolio()
            self._check_limits()
            
            return position
    
    def get_portfolio_greeks(self) -> PortfolioGreeks:
        """Get current portfolio Greeks"""
        with self._lock:
            return PortfolioGreeks(
                delta=self._portfolio_greeks.delta,
                gamma=self._portfolio_greeks.gamma,
                theta=self._portfolio_greeks.theta,
                vega=self._portfolio_greeks.vega,
                rho=self._portfolio_greeks.rho,
                delta_exposure_inr=self._portfolio_greeks.delta_exposure_inr,
                gamma_exposure_inr=self._portfolio_greeks.gamma_exposure_inr,
                theta_exposure_inr=self._portfolio_greeks.theta_exposure_inr,
                vega_exposure_inr=self._portfolio_greeks.vega_exposure_inr,
                num_positions=self._portfolio_greeks.num_positions
            )
    
    def get_position_greeks(self, symbol: str) -> Optional[PositionGreeks]:
        """Get Greeks for a specific position"""
        with self._lock:
            return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionGreeks]:
        """Get all tracked positions"""
        with self._lock:
            return self._positions.copy()
    
    def _recalculate_portfolio(self):
        """Recalculate aggregate portfolio Greeks"""
        portfolio = PortfolioGreeks()
        
        for position in self._positions.values():
            portfolio.delta += position.delta
            portfolio.gamma += position.gamma
            portfolio.theta += position.theta
            portfolio.vega += position.vega
            portfolio.rho += position.rho
            portfolio.num_positions += 1
        
        # Calculate INR exposures
        if self._positions:
            # Use average spot as reference
            avg_spot = sum(p.delta_exposure / p.delta for p in self._positions.values() if p.delta != 0) / len(self._positions) if self._positions else 0
            portfolio.delta_exposure_inr = portfolio.delta * avg_spot
            portfolio.gamma_exposure_inr = portfolio.gamma * avg_spot * 100
            portfolio.theta_exposure_inr = portfolio.theta
            portfolio.vega_exposure_inr = portfolio.vega * 100
        
        self._portfolio_greeks = portfolio
    
    def _check_limits(self):
        """Check if portfolio Greeks are within limits"""
        alerts = []
        
        # Check delta limit
        max_delta = self.risk_limits.max_portfolio_delta
        if abs(self._portfolio_greeks.delta) > max_delta:
            alerts.append(GreeksAlert(
                alert_type=GreeksAlertType.DELTA_LIMIT,
                symbol=None,
                current_value=abs(self._portfolio_greeks.delta),
                limit_value=max_delta,
                breach_pct=(abs(self._portfolio_greeks.delta) - max_delta) / max_delta * 100,
                timestamp=datetime.now(),
                recommendation=self._get_delta_hedge_recommendation()
            ))
        
        # Check gamma limit
        max_gamma = self.risk_limits.max_portfolio_gamma
        if abs(self._portfolio_greeks.gamma) > max_gamma:
            alerts.append(GreeksAlert(
                alert_type=GreeksAlertType.GAMMA_LIMIT,
                symbol=None,
                current_value=abs(self._portfolio_greeks.gamma),
                limit_value=max_gamma,
                breach_pct=(abs(self._portfolio_greeks.gamma) - max_gamma) / max_gamma * 100,
                timestamp=datetime.now(),
                recommendation="Consider reducing gamma exposure through closer strikes"
            ))
        
        # Check theta limit
        max_theta = self.risk_limits.max_portfolio_theta
        if self._portfolio_greeks.theta < max_theta:  # Theta is negative for short options
            alerts.append(GreeksAlert(
                alert_type=GreeksAlertType.THETA_LIMIT,
                symbol=None,
                current_value=self._portfolio_greeks.theta,
                limit_value=max_theta,
                breach_pct=(max_theta - self._portfolio_greeks.theta) / abs(max_theta) * 100,
                timestamp=datetime.now(),
                recommendation="High theta decay - consider rolling positions or reducing size"
            ))
        
        # Check vega limit
        max_vega = self.risk_limits.max_portfolio_vega
        if abs(self._portfolio_greeks.vega) > max_vega:
            alerts.append(GreeksAlert(
                alert_type=GreeksAlertType.VEGA_LIMIT,
                symbol=None,
                current_value=abs(self._portfolio_greeks.vega),
                limit_value=max_vega,
                breach_pct=(abs(self._portfolio_greeks.vega) - max_vega) / max_vega * 100,
                timestamp=datetime.now(),
                recommendation="High vega exposure - consider hedging with VIX futures or options"
            ))
        
        # Store and notify
        self._alerts = alerts
        for alert in alerts:
            self._notify_alert(alert)
    
    def _get_delta_hedge_recommendation(self) -> str:
        """Generate delta hedge recommendation"""
        portfolio_delta = self._portfolio_greeks.delta
        
        if portfolio_delta > 0:
            return f"Sell {portfolio_delta:.0f} deltas via futures or ATM puts to neutralize"
        else:
            return f"Buy {abs(portfolio_delta):.0f} deltas via futures or ATM calls to neutralize"
    
    def _notify_alert(self, alert: GreeksAlert):
        """Notify callbacks of alert"""
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception:
                pass
    
    def get_hedge_recommendation(self, target_delta: float = 0.0) -> Dict:
        """
        Get delta hedging recommendation.
        
        Args:
            target_delta: Target portfolio delta (default 0 for neutral)
            
        Returns:
            Dict with hedge recommendation
        """
        current_delta = self._portfolio_greeks.delta
        delta_diff = target_delta - current_delta
        
        if abs(delta_diff) < 0.01:
            return {
                "needed": False,
                "message": "Portfolio already delta-neutral"
            }
        
        # Recommend hedge instrument
        if abs(delta_diff) > 100:
            # Large hedge needed - use futures
            return {
                "needed": True,
                "action": "BUY" if delta_diff > 0 else "SELL",
                "instrument": "FUTURES",
                "quantity": int(abs(delta_diff)),
                "message": f"{'Buy' if delta_diff > 0 else 'Sell'} {int(abs(delta_diff))} futures to hedge"
            }
        else:
            # Smaller hedge - use options
            option_type = "CE" if delta_diff > 0 else "PE"
            return {
                "needed": True,
                "action": "BUY",
                "instrument": f"ATM_{option_type}",
                "quantity": int(abs(delta_diff) / 0.5),  # Assuming ~0.5 delta for ATM
                "message": f"Buy {int(abs(delta_diff) / 0.5)} ATM {option_type} to hedge"
            }
    
    def calculate_position_size_for_delta(
        self,
        target_delta: float,
        option_delta: float,
        lot_size: int
    ) -> int:
        """
        Calculate position size to achieve target delta.
        
        Args:
            target_delta: Target delta contribution
            option_delta: Delta of the option
            lot_size: Lot size
            
        Returns:
            Number of lots needed
        """
        if option_delta == 0:
            return 0
        
        lots_needed = int(target_delta / (option_delta * lot_size))
        return max(1, lots_needed)
    
    def generate_greeks_report(self) -> Dict:
        """Generate comprehensive Greeks report"""
        portfolio = self.get_portfolio_greeks()
        
        return {
            "timestamp": datetime.now(),
            "portfolio_greeks": {
                "delta": round(portfolio.delta, 4),
                "gamma": round(portfolio.gamma, 6),
                "theta": round(portfolio.theta, 2),
                "vega": round(portfolio.vega, 2),
                "rho": round(portfolio.rho, 2),
            },
            "exposure_inr": {
                "delta": round(portfolio.delta_exposure_inr, 2),
                "gamma": round(portfolio.gamma_exposure_inr, 2),
                "theta": round(portfolio.theta_exposure_inr, 2),
                "vega": round(portfolio.vega_exposure_inr, 2),
            },
            "risk_limits": {
                "max_delta": self.risk_limits.max_portfolio_delta,
                "max_gamma": self.risk_limits.max_portfolio_gamma,
                "max_theta": self.risk_limits.max_portfolio_theta,
                "max_vega": self.risk_limits.max_portfolio_vega,
            },
            "limit_status": {
                "delta_ok": abs(portfolio.delta) <= self.risk_limits.max_portfolio_delta,
                "gamma_ok": abs(portfolio.gamma) <= self.risk_limits.max_portfolio_gamma,
                "theta_ok": portfolio.theta >= self.risk_limits.max_portfolio_theta,
                "vega_ok": abs(portfolio.vega) <= self.risk_limits.max_portfolio_vega,
            },
            "active_alerts": len(self._alerts),
            "num_positions": portfolio.num_positions,
            "hedge_recommendation": self.get_hedge_recommendation()
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def calculate_portfolio_var(
    positions: List[PositionGreeks],
    confidence: float = 0.95,
    days: int = 1
) -> float:
    """
    Calculate portfolio Value at Risk (VaR).
    
    Simplified parametric VaR using Greeks.
    
    Args:
        positions: List of positions
        confidence: Confidence level (default 95%)
        days: Time horizon in days
        
    Returns:
        VaR in INR
    """
    # Aggregate Greeks
    total_delta = sum(p.delta_exposure for p in positions)
    total_gamma = sum(p.gamma_exposure for p in positions)
    total_theta = sum(p.theta_exposure for p in positions)
    total_vega = sum(p.vega_exposure for p in positions)
    
    # Assume 1% daily move, 1 vol point change
    price_move = 0.01
    vol_move = 0.01
    
    # Delta-gamma P&L
    delta_pl = total_delta * price_move
    gamma_pl = 0.5 * total_gamma * (price_move ** 2)
    
    # Theta decay
    theta_pl = total_theta * days
    
    # Vega P&L
    vega_pl = total_vega * vol_move
    
    # Total P&L distribution
    expected_pl = delta_pl + gamma_pl + theta_pl + vega_pl
    
    # Simplified VaR (assuming normal distribution)
    z_score = 1.645 if confidence == 0.95 else 2.33  # 95% or 99%
    
    # This is a simplified calculation
    var = abs(expected_pl) * z_score
    
    return var

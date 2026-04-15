"""
ROX Proven Edge Engine v3.0 - Risk Dashboard
===========================================
Real-time risk metrics and monitoring.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    risk_amount: float  # Amount at risk based on stop loss
    risk_pct: float  # Risk as % of portfolio
    beta: float = 1.0
    var_contribution: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "risk_amount": self.risk_amount,
            "risk_pct": self.risk_pct,
            "beta": self.beta
        }


@dataclass
class RiskMetrics:
    """Complete risk metrics snapshot"""
    timestamp: datetime
    
    # Portfolio level
    portfolio_value: float
    cash: float
    deployed_capital: float
    deployment_pct: float
    
    # P&L
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    mtd_pnl: float
    
    # Drawdown
    current_drawdown: float
    current_drawdown_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    peak_value: float
    
    # Risk metrics
    portfolio_heat: float  # Total risk %
    daily_var: float  # Value at Risk
    weekly_var: float
    beta_weighted: float
    
    # Position metrics
    position_count: int
    largest_position_pct: float
    sector_concentration: Dict[str, float]
    
    # Greeks (for options)
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    
    # Risk level
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "deployed_capital": self.deployed_capital,
            "deployment_pct": self.deployment_pct,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "portfolio_heat": self.portfolio_heat,
            "current_drawdown_pct": self.current_drawdown_pct,
            "risk_level": self.risk_level.value
        }


class RiskDashboard:
    """
    Real-time risk dashboard for portfolio monitoring.
    
    Features:
    - Live P&L tracking
    - Portfolio heat calculation
    - VaR computation
    - Drawdown monitoring
    - Greeks tracking
    - Correlation matrix
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("RiskDashboard")
        
        # Portfolio state
        self.portfolio_value = config.get("initial_capital", 1000000)
        self.cash = self.portfolio_value
        self.peak_value = self.portfolio_value
        
        # Positions
        self.positions: Dict[str, PositionRisk] = {}
        
        # P&L tracking
        self.daily_start_value = self.portfolio_value
        self.weekly_start_value = self.portfolio_value
        self.mtd_start_value = self.portfolio_value
        
        # Drawdown tracking
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Risk limits
        self.max_heat = config.get("max_portfolio_heat", 0.08)
        self.max_drawdown_limit = config.get("max_drawdown_limit", 0.20)
        self.max_daily_loss = config.get("max_daily_loss", 0.05)
        
        # History
        self.pnl_history: List[Dict] = []
        self.metrics_history: List[RiskMetrics] = []
    
    def update_position(self, symbol: str, quantity: int, entry_price: float,
                       current_price: float, stop_loss: float, beta: float = 1.0):
        """Update or add a position"""
        market_value = quantity * current_price
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        
        risk_amount = abs(entry_price - stop_loss) * quantity
        risk_pct = risk_amount / self.portfolio_value if self.portfolio_value > 0 else 0
        
        self.positions[symbol] = PositionRisk(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            beta=beta
        )
        
        # Recalculate portfolio
        self._recalculate_portfolio()
    
    def remove_position(self, symbol: str):
        """Remove a position"""
        if symbol in self.positions:
            del self.positions[symbol]
            self._recalculate_portfolio()
    
    def update_prices(self, prices: Dict[str, float]):
        """Update all position prices"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.current_price = price
                pos.market_value = pos.quantity * price
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
                pos.unrealized_pnl_pct = (price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0
        
        self._recalculate_portfolio()
    
    def _recalculate_portfolio(self):
        """Recalculate portfolio metrics"""
        # Total market value
        total_market_value = sum(p.market_value for p in self.positions.values())
        
        # Update portfolio value
        self.cash = self.portfolio_value - sum(
            p.quantity * p.entry_price for p in self.positions.values()
        )
        
        # Portfolio value includes cash and position values
        new_portfolio_value = self.cash + total_market_value
        
        # Update peak
        if new_portfolio_value > self.peak_value:
            self.peak_value = new_portfolio_value
        
        # Calculate drawdown
        self.current_drawdown = self.peak_value - new_portfolio_value
        current_drawdown_pct = self.current_drawdown / self.peak_value if self.peak_value > 0 else 0
        
        if current_drawdown_pct > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
        
        self.portfolio_value = new_portfolio_value
    
    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        # Calculate deployed capital
        deployed = sum(p.market_value for p in self.positions.values())
        deployment_pct = deployed / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # P&L
        daily_pnl = self.portfolio_value - self.daily_start_value
        daily_pnl_pct = daily_pnl / self.daily_start_value if self.daily_start_value > 0 else 0
        
        weekly_pnl = self.portfolio_value - self.weekly_start_value
        mtd_pnl = self.portfolio_value - self.mtd_start_value
        
        # Drawdown
        current_dd = self.peak_value - self.portfolio_value
        current_dd_pct = current_dd / self.peak_value if self.peak_value > 0 else 0
        max_dd_pct = self.max_drawdown / self.peak_value if self.peak_value > 0 else 0
        
        # Portfolio heat
        portfolio_heat = sum(p.risk_pct for p in self.positions.values())
        
        # VaR (simplified - using portfolio volatility)
        daily_var = self._calculate_var(timeframe="daily")
        weekly_var = self._calculate_var(timeframe="weekly")
        
        # Beta weighted
        beta_weighted = self._calculate_beta_weighted()
        
        # Position metrics
        position_count = len(self.positions)
        largest_pct = max((p.market_value / self.portfolio_value for p in self.positions.values()), default=0)
        
        # Sector concentration
        sector_conc = self._calculate_sector_concentration()
        
        # Determine risk level
        risk_level = self._determine_risk_level(
            current_dd_pct, portfolio_heat, daily_pnl_pct
        )
        
        metrics = RiskMetrics(
            timestamp=datetime.now(),
            portfolio_value=self.portfolio_value,
            cash=self.cash,
            deployed_capital=deployed,
            deployment_pct=deployment_pct,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl=weekly_pnl,
            mtd_pnl=mtd_pnl,
            current_drawdown=current_dd,
            current_drawdown_pct=current_dd_pct,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=max_dd_pct,
            peak_value=self.peak_value,
            portfolio_heat=portfolio_heat,
            daily_var=daily_var,
            weekly_var=weekly_var,
            beta_weighted=beta_weighted,
            position_count=position_count,
            largest_position_pct=largest_pct,
            sector_concentration=sector_conc,
            risk_level=risk_level
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def _calculate_var(self, timeframe: str = "daily", confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if not self.positions:
            return 0.0
        
        # Simplified VaR using position values and beta
        total_value = sum(p.market_value for p in self.positions.values())
        
        # Estimate portfolio volatility
        weighted_beta = self._calculate_beta_weighted()
        base_volatility = 0.015  # ~1.5% daily for Nifty
        
        portfolio_volatility = base_volatility * weighted_beta
        
        # Time adjustment
        if timeframe == "weekly":
            portfolio_volatility *= np.sqrt(5)
        
        # VaR calculation (normal distribution)
        z_score = 1.65 if confidence == 0.95 else 2.33  # 95% or 99%
        
        return total_value * portfolio_volatility * z_score
    
    def _calculate_beta_weighted(self) -> float:
        """Calculate beta-weighted portfolio beta"""
        if not self.positions:
            return 1.0
        
        total_value = sum(p.market_value for p in self.positions.values())
        if total_value == 0:
            return 1.0
        
        return sum(
            p.beta * (p.market_value / total_value)
            for p in self.positions.values()
        )
    
    def _calculate_sector_concentration(self) -> Dict[str, float]:
        """Calculate sector exposure"""
        # Placeholder - would need sector mapping
        sectors = {}
        for pos in self.positions.values():
            sector = "UNKNOWN"  # Would map symbol to sector
            sectors[sector] = sectors.get(sector, 0) + pos.market_value
        
        total = sum(sectors.values())
        if total > 0:
            return {s: v / total for s, v in sectors.items()}
        return {}
    
    def _determine_risk_level(self, drawdown_pct: float, heat: float,
                             daily_pnl_pct: float) -> RiskLevel:
        """Determine overall risk level"""
        if drawdown_pct > 0.15 or heat > 0.10 or daily_pnl_pct < -0.05:
            return RiskLevel.CRITICAL
        elif drawdown_pct > 0.10 or heat > 0.08 or daily_pnl_pct < -0.03:
            return RiskLevel.HIGH
        elif drawdown_pct > 0.05 or heat > 0.06:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def new_day(self):
        """Reset daily metrics"""
        self.daily_start_value = self.portfolio_value
    
    def new_week(self):
        """Reset weekly metrics"""
        self.weekly_start_value = self.portfolio_value
    
    def new_month(self):
        """Reset monthly metrics"""
        self.mtd_start_value = self.portfolio_value
    
    def check_limits(self) -> List[Dict]:
        """Check if any risk limits are breached"""
        metrics = self.get_metrics()
        alerts = []
        
        if metrics.portfolio_heat > self.max_heat:
            alerts.append({
                "type": "HEAT_LIMIT",
                "severity": "HIGH",
                "message": f"Portfolio heat {metrics.portfolio_heat:.1%} exceeds limit {self.max_heat:.1%}",
                "action": "Reduce positions"
            })
        
        if metrics.current_drawdown_pct > self.max_drawdown_limit:
            alerts.append({
                "type": "DRAWDOWN_LIMIT",
                "severity": "CRITICAL",
                "message": f"Drawdown {metrics.current_drawdown_pct:.1%} exceeds limit",
                "action": "Exit positions immediately"
            })
        
        if metrics.daily_pnl_pct < -self.max_daily_loss:
            alerts.append({
                "type": "DAILY_LOSS_LIMIT",
                "severity": "HIGH",
                "message": f"Daily loss {metrics.daily_pnl_pct:.1%} exceeds limit",
                "action": "Stop trading for the day"
            })
        
        return alerts
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get position correlation matrix"""
        # Placeholder - would calculate from price history
        n = len(self.positions)
        if n == 0:
            return np.array([])
        
        # Default correlation matrix (placeholder)
        return np.eye(n)
    
    def get_summary(self) -> Dict:
        """Get dashboard summary"""
        metrics = self.get_metrics()
        
        return {
            "portfolio_value": f"₹{self.portfolio_value:,.0f}",
            "cash": f"₹{self.cash:,.0f}",
            "deployed": f"{metrics.deployment_pct:.1%}",
            "daily_pnl": f"₹{metrics.daily_pnl:,.0f} ({metrics.daily_pnl_pct:.2%})",
            "portfolio_heat": f"{metrics.portfolio_heat:.2%}",
            "drawdown": f"{metrics.current_drawdown_pct:.2%}",
            "risk_level": metrics.risk_level.value,
            "positions": len(self.positions)
        }

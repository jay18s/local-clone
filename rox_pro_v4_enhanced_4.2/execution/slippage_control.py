"""
ROX Proven Edge Engine v3.0 - Slippage Control
=============================================
Slippage monitoring and control.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np


@dataclass
class SlippageRecord:
    """Record of slippage for an order"""
    order_id: str
    symbol: str
    side: str
    expected_price: float
    actual_price: float
    quantity: int
    slippage_bps: float  # Basis points
    slippage_amount: float
    market_conditions: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "expected_price": self.expected_price,
            "actual_price": self.actual_price,
            "quantity": self.quantity,
            "slippage_bps": self.slippage_bps,
            "slippage_amount": self.slippage_amount,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MarketImpactModel:
    """Model for estimating market impact"""
    symbol: str
    avg_daily_volume: float
    avg_spread_bps: float
    volatility: float
    
    # Impact coefficients
    temporary_impact_coeff: float = 0.1
    permanent_impact_coeff: float = 0.05
    
    def estimate_impact(self, order_size: float, duration_minutes: int = 30) -> Tuple[float, float]:
        """
        Estimate market impact for an order.
        
        Args:
            order_size: Order size in shares
            duration_minutes: Execution duration
            
        Returns:
            Tuple of (temporary_impact_bps, permanent_impact_bps)
        """
        # Participation rate
        participation = order_size / (self.avg_daily_volume * duration_minutes / 390)
        
        # Temporary impact (resolves after order completes)
        temp_impact = self.temporary_impact_coeff * np.sqrt(participation) * 100
        
        # Permanent impact (lasts)
        perm_impact = self.permanent_impact_coeff * participation * 100
        
        return temp_impact, perm_impact


class SlippageController:
    """
    Monitors and controls execution slippage.
    
    Features:
    - Pre-trade slippage estimation
    - Real-time slippage monitoring
    - Market impact modeling
    - Adaptive execution adjustment
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("SlippageController")
        
        # Slippage history
        self.slippage_history: List[SlippageRecord] = []
        self.slippage_by_symbol: Dict[str, deque] = {}
        
        # Thresholds
        self.warning_threshold_bps = config.get("warning_threshold_bps", 20)  # 20 bps
        self.critical_threshold_bps = config.get("critical_threshold_bps", 50)  # 50 bps
        
        # Market impact models
        self.impact_models: Dict[str, MarketImpactModel] = {}
        
        # Statistics
        self.total_slippage_bps = 0.0
        self.total_orders = 0
        self.total_slippage_amount = 0.0
    
    def register_symbol(self, symbol: str, avg_daily_volume: float,
                        avg_spread_bps: float, volatility: float):
        """Register a symbol for slippage tracking"""
        self.impact_models[symbol] = MarketImpactModel(
            symbol=symbol,
            avg_daily_volume=avg_daily_volume,
            avg_spread_bps=avg_spread_bps,
            volatility=volatility
        )
        
        self.slippage_by_symbol[symbol] = deque(maxlen=100)
    
    def estimate_slippage(self, symbol: str, order_size: int,
                         side: str, current_price: float,
                         duration_minutes: int = 30) -> Dict:
        """
        Estimate expected slippage before placing order.
        
        Args:
            symbol: Stock symbol
            order_size: Order quantity
            side: BUY or SELL
            current_price: Current market price
            duration_minutes: Planned execution duration
            
        Returns:
            Dictionary with slippage estimates
        """
        model = self.impact_models.get(symbol)
        
        if not model:
            # Default estimate
            return {
                "expected_slippage_bps": 10,
                "expected_slippage_amount": current_price * order_size * 0.001,
                "confidence": 0.3
            }
        
        # Get impact estimates
        temp_impact, perm_impact = model.estimate_impact(order_size, duration_minutes)
        
        # Total slippage estimate
        total_impact_bps = temp_impact + perm_impact
        
        # For sells, slippage is negative
        if side == "SELL":
            total_impact_bps = -total_impact_bps
        
        # Calculate amount
        slippage_amount = abs(current_price * order_size * total_impact_bps / 10000)
        
        return {
            "expected_slippage_bps": total_impact_bps,
            "expected_slippage_amount": slippage_amount,
            "temporary_impact_bps": temp_impact,
            "permanent_impact_bps": perm_impact,
            "participation_rate": order_size / (model.avg_daily_volume * duration_minutes / 390),
            "confidence": 0.7
        }
    
    def record_slippage(self, order_id: str, symbol: str, side: str,
                       expected_price: float, actual_price: float,
                       quantity: int, market_conditions: Dict = None) -> SlippageRecord:
        """
        Record actual slippage after execution.
        
        Args:
            order_id: Order identifier
            symbol: Stock symbol
            side: BUY or SELL
            expected_price: Expected execution price
            actual_price: Actual fill price
            quantity: Fill quantity
            market_conditions: Market state at execution
            
        Returns:
            SlippageRecord with calculated slippage
        """
        # Calculate slippage
        if side == "BUY":
            slippage = actual_price - expected_price
        else:
            slippage = expected_price - actual_price
        
        slippage_bps = (slippage / expected_price) * 10000 if expected_price > 0 else 0
        slippage_amount = abs(slippage * quantity)
        
        record = SlippageRecord(
            order_id=order_id,
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            actual_price=actual_price,
            quantity=quantity,
            slippage_bps=slippage_bps,
            slippage_amount=slippage_amount,
            market_conditions=market_conditions or {}
        )
        
        # Update history
        self.slippage_history.append(record)
        if symbol in self.slippage_by_symbol:
            self.slippage_by_symbol[symbol].append(record)
        
        # Update statistics
        self.total_slippage_bps += slippage_bps
        self.total_orders += 1
        self.total_slippage_amount += slippage_amount
        
        # Check thresholds
        self._check_thresholds(record)
        
        return record
    
    def _check_thresholds(self, record: SlippageRecord):
        """Check if slippage exceeds thresholds"""
        abs_slippage = abs(record.slippage_bps)
        
        if abs_slippage >= self.critical_threshold_bps:
            self.logger.warning(
                f"CRITICAL slippage: {record.order_id} | "
                f"{record.slippage_bps:.1f} bps | "
                f"₹{record.slippage_amount:.2f}"
            )
        elif abs_slippage >= self.warning_threshold_bps:
            self.logger.warning(
                f"HIGH slippage: {record.order_id} | "
                f"{record.slippage_bps:.1f} bps"
            )
    
    def get_average_slippage(self, symbol: str = None,
                            lookback: int = 20) -> float:
        """Get average slippage in basis points"""
        if symbol and symbol in self.slippage_by_symbol:
            records = list(self.slippage_by_symbol[symbol])[-lookback:]
        else:
            records = self.slippage_history[-lookback:]
        
        if not records:
            return 0.0
        
        return np.mean([abs(r.slippage_bps) for r in records])
    
    def get_slippage_by_side(self, side: str, 
                            lookback: int = 50) -> Dict:
        """Get slippage statistics by order side"""
        records = [r for r in self.slippage_history[-lookback:] 
                  if r.side == side]
        
        if not records:
            return {"avg_bps": 0, "max_bps": 0, "count": 0}
        
        return {
            "avg_bps": np.mean([r.slippage_bps for r in records]),
            "max_bps": max(r.slippage_bps for r in records),
            "count": len(records)
        }
    
    def suggest_improvements(self, symbol: str, recent_slippage_bps: float) -> List[str]:
        """Suggest execution improvements based on slippage"""
        suggestions = []
        
        if recent_slippage_bps > 30:
            suggestions.append("Consider increasing execution duration")
            suggestions.append("Use limit orders instead of market orders")
        
        if recent_slippage_bps > 50:
            suggestions.append("Switch to VWAP algorithm for better volume following")
            suggestions.append("Reduce order size or split across sessions")
        
        model = self.impact_models.get(symbol)
        if model:
            suggestions.append(
                f"Avg spread for {symbol}: {model.avg_spread_bps:.1f} bps — "
                f"aim for fills within this range"
            )
        
        return suggestions
    
    def get_statistics(self) -> Dict:
        """Get overall slippage statistics"""
        return {
            "total_orders": self.total_orders,
            "average_slippage_bps": self.total_slippage_bps / self.total_orders if self.total_orders > 0 else 0,
            "total_slippage_amount": self.total_slippage_amount,
            "warning_threshold_bps": self.warning_threshold_bps,
            "critical_threshold_bps": self.critical_threshold_bps,
            "symbols_tracked": len(self.impact_models)
        }
    
    def should_pause_execution(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if execution should be paused due to adverse conditions.
        
        Returns:
            Tuple of (should_pause, reason)
        """
        if symbol not in self.slippage_by_symbol:
            return False, ""
        
        recent = list(self.slippage_by_symbol[symbol])[-5:]
        
        if len(recent) < 3:
            return False, ""
        
        avg_recent = np.mean([abs(r.slippage_bps) for r in recent])
        
        if avg_recent > self.critical_threshold_bps:
            return True, f"Recent slippage too high: {avg_recent:.1f} bps average"
        
        # Check for increasing trend
        if len(recent) >= 3:
            trend = recent[-1].slippage_bps - recent[0].slippage_bps
            if trend > 20:
                return True, "Slippage trending worse"
        
        return False, ""

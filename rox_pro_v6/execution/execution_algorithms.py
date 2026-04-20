"""
ROX Proven Edge Engine v3.0 - Execution Algorithms
=================================================
Smart execution algorithms for optimal fills.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import numpy as np


class AlgorithmType(Enum):
    """Execution algorithm types"""
    TWAP = "TWAP"
    VWAP = "VWAP"
    IMPLEMENTATION_SHORTFALL = "IMPLEMENTATION_SHORTFALL"
    POV = "POV"  # Percentage of Volume
    ADAPTIVE = "ADAPTIVE"


@dataclass
class ExecutionSlice:
    """Single execution slice"""
    slice_id: str
    timestamp: datetime
    target_quantity: int
    target_price: float
    actual_quantity: int = 0
    actual_price: float = 0.0
    status: str = "PENDING"
    
    def to_dict(self) -> Dict:
        return {
            "slice_id": self.slice_id,
            "timestamp": self.timestamp.isoformat(),
            "target_quantity": self.target_quantity,
            "target_price": self.target_price,
            "actual_quantity": self.actual_quantity,
            "actual_price": self.actual_price,
            "status": self.status
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan"""
    plan_id: str
    algorithm: AlgorithmType
    symbol: str
    side: str
    total_quantity: int
    start_time: datetime
    end_time: datetime
    slices: List[ExecutionSlice] = field(default_factory=list)
    
    @property
    def executed_quantity(self) -> int:
        return sum(s.actual_quantity for s in self.slices)
    
    @property
    def remaining_quantity(self) -> int:
        return self.total_quantity - self.executed_quantity
    
    @property
    def average_price(self) -> float:
        total = sum(s.actual_quantity * s.actual_price for s in self.slices)
        qty = self.executed_quantity
        return total / qty if qty > 0 else 0
    
    @property
    def completion_pct(self) -> float:
        return self.executed_quantity / self.total_quantity if self.total_quantity > 0 else 0


class ExecutionAlgorithm(ABC):
    """Abstract base class for execution algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"ExecutionAlgorithm.{name}")
        self._active_plans: Dict[str, ExecutionPlan] = {}
    
    @abstractmethod
    def create_plan(self, symbol: str, side: str, quantity: int,
                   duration_minutes: int, **kwargs) -> ExecutionPlan:
        """Create execution plan"""
        pass
    
    @abstractmethod
    async def execute_slice(self, plan: ExecutionPlan, 
                           slice_idx: int) -> bool:
        """Execute a single slice"""
        pass
    
    async def start_execution(self, plan: ExecutionPlan,
                              order_callback: Callable = None):
        """Start executing the plan"""
        self._active_plans[plan.plan_id] = plan
        
        for i, slice_info in enumerate(plan.slices):
            # Wait until slice time
            now = datetime.now()
            if slice_info.timestamp > now:
                wait_seconds = (slice_info.timestamp - now).total_seconds()
                await asyncio.sleep(wait_seconds)
            
            # Execute slice
            success = await self.execute_slice(plan, i)
            
            if order_callback:
                await order_callback(plan, slice_info, success)
    
    def get_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Get execution plan by ID"""
        return self._active_plans.get(plan_id)


class TWAP(ExecutionAlgorithm):
    """
    Time-Weighted Average Price execution algorithm.
    
    Splits order evenly across time period.
    """
    
    def __init__(self):
        super().__init__("TWAP")
    
    def create_plan(self, symbol: str, side: str, quantity: int,
                   duration_minutes: int, slice_interval_seconds: int = 30,
                   randomize: bool = True, **kwargs) -> ExecutionPlan:
        """
        Create TWAP execution plan.
        
        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Total quantity to execute
            duration_minutes: Total execution duration
            slice_interval_seconds: Time between slices
            randomize: Add randomness to slice timing
            
        Returns:
            ExecutionPlan with slices
        """
        plan_id = f"TWAP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Calculate number of slices
        total_seconds = duration_minutes * 60
        num_slices = max(1, total_seconds // slice_interval_seconds)
        
        # Calculate quantity per slice
        base_qty = quantity // num_slices
        remainder = quantity % num_slices
        
        slices = []
        for i in range(num_slices):
            # Calculate slice time
            base_time = start_time + timedelta(seconds=i * slice_interval_seconds)
            
            # Add randomization if enabled
            if randomize:
                random_offset = np.random.uniform(-slice_interval_seconds * 0.2,
                                                  slice_interval_seconds * 0.2)
                slice_time = base_time + timedelta(seconds=random_offset)
            else:
                slice_time = base_time
            
            # Distribute remainder
            slice_qty = base_qty + (1 if i < remainder else 0)
            
            slices.append(ExecutionSlice(
                slice_id=f"{plan_id}-S{i+1:03d}",
                timestamp=slice_time,
                target_quantity=slice_qty,
                target_price=0  # Market order
            ))
        
        return ExecutionPlan(
            plan_id=plan_id,
            algorithm=AlgorithmType.TWAP,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            start_time=start_time,
            end_time=end_time,
            slices=slices
        )
    
    async def execute_slice(self, plan: ExecutionPlan, 
                           slice_idx: int) -> bool:
        """Execute a TWAP slice (market order)"""
        if slice_idx >= len(plan.slices):
            return False
        
        slice_info = plan.slices[slice_idx]
        self.logger.info(f"Executing TWAP slice: {slice_info.slice_id} | "
                        f"Qty: {slice_info.target_quantity}")
        
        # In production, this would submit market order to exchange
        slice_info.status = "EXECUTED"
        slice_info.actual_quantity = slice_info.target_quantity
        
        return True


class VWAP(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price execution algorithm.
    
    Splits order based on historical volume profile.
    """
    
    def __init__(self):
        super().__init__("VWAP")
        
        # Default volume profile (percentage of daily volume by hour)
        # Indian market hours: 9:15 AM to 3:30 PM
        self.default_volume_profile = {
            9: 0.15,   # 9:00-10:00
            10: 0.12,  # 10:00-11:00
            11: 0.10,  # 11:00-12:00
            12: 0.08,  # 12:00-13:00
            13: 0.10,  # 13:00-14:00
            14: 0.18,  # 14:00-15:00
            15: 0.27   # 15:00-15:30
        }
    
    def create_plan(self, symbol: str, side: str, quantity: int,
                   duration_minutes: int, volume_profile: Dict = None,
                   avg_daily_volume: int = 100000, **kwargs) -> ExecutionPlan:
        """
        Create VWAP execution plan.
        
        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Total quantity
            duration_minutes: Duration
            volume_profile: Custom volume profile by hour
            avg_daily_volume: Average daily volume for participation rate
            
        Returns:
            ExecutionPlan
        """
        plan_id = f"VWAP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        profile = volume_profile or self.default_volume_profile
        
        # Create slices based on volume profile
        slices = []
        slice_id = 0
        
        current_time = start_time
        remaining_qty = quantity
        
        while current_time < end_time and remaining_qty > 0:
            hour = current_time.hour
            hour_pct = profile.get(hour, 0.10)
            
            # Calculate slice quantity based on volume profile
            slice_qty = min(
                int(quantity * hour_pct * (duration_minutes / 390)),  # 390 mins = 6.5 hours
                remaining_qty
            )
            
            if slice_qty > 0:
                slice_id += 1
                slices.append(ExecutionSlice(
                    slice_id=f"{plan_id}-S{slice_id:03d}",
                    timestamp=current_time,
                    target_quantity=slice_qty,
                    target_price=0
                ))
                remaining_qty -= slice_qty
            
            current_time += timedelta(minutes=5)  # Check every 5 minutes
        
        return ExecutionPlan(
            plan_id=plan_id,
            algorithm=AlgorithmType.VWAP,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            start_time=start_time,
            end_time=end_time,
            slices=slices
        )
    
    async def execute_slice(self, plan: ExecutionPlan,
                           slice_idx: int) -> bool:
        """Execute a VWAP slice"""
        if slice_idx >= len(plan.slices):
            return False
        
        slice_info = plan.slices[slice_idx]
        self.logger.info(f"Executing VWAP slice: {slice_info.slice_id}")
        
        slice_info.status = "EXECUTED"
        slice_info.actual_quantity = slice_info.target_quantity
        
        return True


class ImplementationShortfall(ExecutionAlgorithm):
    """
    Implementation Shortfall / Arrival Price algorithm.
    
    Balances market impact vs timing risk to minimize total cost.
    """
    
    def __init__(self):
        super().__init__("ImplementationShortfall")
    
    def create_plan(self, symbol: str, side: str, quantity: int,
                   duration_minutes: int, urgency: str = "MEDIUM",
                   risk_aversion: float = 0.5, **kwargs) -> ExecutionPlan:
        """
        Create Implementation Shortfall plan.
        
        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Total quantity
            duration_minutes: Available time
            urgency: LOW, MEDIUM, HIGH
            risk_aversion: 0-1, higher = more aggressive
            
        Returns:
            ExecutionPlan
        """
        plan_id = f"IS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Adjust aggression based on urgency
        urgency_multipliers = {
            "LOW": 0.6,
            "MEDIUM": 1.0,
            "HIGH": 1.5
        }
        
        aggression = urgency_multipliers.get(urgency, 1.0) * risk_aversion
        
        # Create adaptive slices
        # More aggressive = front-load more
        num_slices = max(4, duration_minutes // 5)
        slices = []
        
        for i in range(num_slices):
            # Front-loaded distribution based on aggression
            weight = (1 + aggression) * (1 - (i / num_slices) ** aggression)
            slice_pct = weight / sum(
                (1 + aggression) * (1 - (j / num_slices) ** aggression)
                for j in range(num_slices)
            )
            
            slice_qty = max(1, int(quantity * slice_pct))
            
            slice_time = start_time + timedelta(
                minutes=duration_minutes * i / num_slices
            )
            
            slices.append(ExecutionSlice(
                slice_id=f"{plan_id}-S{i+1:03d}",
                timestamp=slice_time,
                target_quantity=slice_qty,
                target_price=0
            ))
        
        # Adjust last slice for rounding
        if slices:
            total_target = sum(s.target_quantity for s in slices)
            if total_target != quantity:
                slices[-1].target_quantity += (quantity - total_target)
        
        return ExecutionPlan(
            plan_id=plan_id,
            algorithm=AlgorithmType.IMPLEMENTATION_SHORTFALL,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            start_time=start_time,
            end_time=end_time,
            slices=slices
        )
    
    async def execute_slice(self, plan: ExecutionPlan,
                           slice_idx: int) -> bool:
        """Execute IS slice"""
        if slice_idx >= len(plan.slices):
            return False
        
        slice_info = plan.slices[slice_idx]
        self.logger.info(f"Executing IS slice: {slice_info.slice_id}")
        
        slice_info.status = "EXECUTED"
        slice_info.actual_quantity = slice_info.target_quantity
        
        return True


class AdaptiveAlgorithm(ExecutionAlgorithm):
    """
    Adaptive execution algorithm.
    
    Switches between strategies based on market conditions.
    """
    
    def __init__(self):
        super().__init__("Adaptive")
        self.twap = TWAP()
        self.vwap = VWAP()
        self.is_alg = ImplementationShortfall()
    
    def create_plan(self, symbol: str, side: str, quantity: int,
                   duration_minutes: int, market_conditions: Dict = None,
                   **kwargs) -> ExecutionPlan:
        """
        Create adaptive execution plan.
        
        Selects best algorithm based on conditions.
        """
        conditions = market_conditions or {}
        
        # Determine best algorithm
        volatility = conditions.get("volatility", "NORMAL")
        volume = conditions.get("volume", "NORMAL")
        spread = conditions.get("spread", "NORMAL")
        
        if volatility == "HIGH" or spread == "WIDE":
            # Use TWAP for high volatility
            return self.twap.create_plan(symbol, side, quantity, duration_minutes, **kwargs)
        
        elif volume == "HIGH":
            # Use VWAP for high volume
            return self.vwap.create_plan(symbol, side, quantity, duration_minutes, **kwargs)
        
        else:
            # Use Implementation Shortfall as default
            return self.is_alg.create_plan(symbol, side, quantity, duration_minutes, **kwargs)
    
    async def execute_slice(self, plan: ExecutionPlan,
                           slice_idx: int) -> bool:
        """Execute slice"""
        if slice_idx >= len(plan.slices):
            return False
        
        slice_info = plan.slices[slice_idx]
        slice_info.status = "EXECUTED"
        slice_info.actual_quantity = slice_info.target_quantity
        
        return True

"""
ROX Proven Edge Engine v3.0 - Order Manager
==========================================
Order Management System for trade execution.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import uuid


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"
    STOP_MARKET = "STOP_MARKET"
    ICEBERG = "ICEBERG"
    TRAILING_STOP = "TRAILING_STOP"


class OrderStatus(Enum):
    """Order lifecycle states"""
    NEW = "NEW"
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Time in force"""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


@dataclass
class Fill:
    """Order fill information"""
    fill_id: str
    order_id: str
    price: float
    quantity: int
    timestamp: datetime
    exchange: str = "NSE"
    fees: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat(),
            "exchange": self.exchange,
            "fees": self.fees
        }


@dataclass
class Order:
    """Order data structure"""
    # Core fields
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    
    # Price fields
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trigger_price: Optional[float] = None
    
    # Status fields
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expiry: Optional[datetime] = None
    
    # Execution details
    fills: List[Fill] = field(default_factory=list)
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Additional fields
    strategy: str = ""
    notes: str = ""
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    
    # Iceberg specific
    display_quantity: Optional[int] = None  # For iceberg orders
    hidden_quantity: Optional[int] = None
    
    # Trailing stop specific
    trail_amount: Optional[float] = None
    trail_percent: Optional[float] = None
    
    # Metadata
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = self._generate_order_id()
    
    @staticmethod
    def _generate_order_id() -> str:
        return f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"
    
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                              OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.NEW, OrderStatus.PENDING, 
                              OrderStatus.PARTIALLY_FILLED]
    
    def add_fill(self, price: float, quantity: int, exchange: str = "NSE") -> Fill:
        """Add a fill to the order"""
        fill = Fill(
            fill_id=f"FIL-{uuid.uuid4().hex[:8].upper()}",
            order_id=self.order_id,
            price=price,
            quantity=quantity,
            timestamp=datetime.now(),
            exchange=exchange
        )
        
        self.fills.append(fill)
        self.filled_quantity += quantity
        
        # Update average price
        total_value = sum(f.price * f.quantity for f in self.fills)
        self.average_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = datetime.now()
        return fill
    
    def cancel(self, reason: str = "") -> bool:
        """Cancel the order"""
        if self.is_complete:
            return False
        
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now()
        self.notes = reason
        return True
    
    def reject(self, reason: str = "") -> bool:
        """Reject the order"""
        if self.status != OrderStatus.NEW:
            return False
        
        self.status = OrderStatus.REJECTED
        self.updated_at = datetime.now()
        self.notes = reason
        return True
    
    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "remaining_quantity": self.remaining_quantity,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "time_in_force": self.time_in_force.value,
            "strategy": self.strategy,
            "fills": [f.to_dict() for f in self.fills]
        }


class OrderManager:
    """
    Central Order Management System.
    
    Features:
    - Order lifecycle management
    - Smart order routing
    - Fill tracking
    - Order state machine
    - Multi-exchange support
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("OrderManager")
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, List[str]] = {}
        
        # Callbacks
        self._order_callbacks: List[Callable] = []
        self._fill_callbacks: List[Callable] = []
        
        # Exchange adapters
        self.exchanges: Dict[str, Any] = {}
        
        # State
        self._running = False
    
    async def start(self):
        """Start order manager"""
        self._running = True
        self.logger.info("Order Manager started")
    
    async def stop(self):
        """Stop order manager"""
        self._running = False
        
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            await self.cancel_order(order_id, "System shutdown")
        
        self.logger.info("Order Manager stopped")
    
    def register_exchange(self, name: str, adapter: Any):
        """Register exchange adapter"""
        self.exchanges[name] = adapter
        self.logger.info(f"Registered exchange: {name}")
    
    def create_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                     quantity: int, limit_price: float = None,
                     stop_price: float = None, **kwargs) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Stock symbol
            side: BUY or SELL
            order_type: Type of order
            quantity: Number of shares
            limit_price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            **kwargs: Additional order parameters
            
        Returns:
            Created Order object
        """
        order = Order(
            order_id=Order._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            **kwargs
        )
        
        # Validate order
        if not self._validate_order(order):
            order.reject("Validation failed")
            return order
        
        # Store order
        self.orders[order.order_id] = order
        self.active_orders[order.order_id] = order
        
        if symbol not in self.orders_by_symbol:
            self.orders_by_symbol[symbol] = []
        self.orders_by_symbol[symbol].append(order.order_id)
        
        # Notify callbacks
        self._notify_order_created(order)
        
        self.logger.info(f"Created order: {order.order_id} | {symbol} {side.value} {quantity}")
        return order
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters"""
        # Check quantity
        if order.quantity <= 0:
            self.logger.error(f"Invalid quantity: {order.quantity}")
            return False
        
        # Check limit price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.limit_price is None or order.limit_price <= 0:
                self.logger.error("Limit order requires valid limit price")
                return False
        
        # Check stop price for stop orders
        if order.order_type in [OrderType.STOP_LIMIT, OrderType.STOP_MARKET]:
            if order.stop_price is None or order.stop_price <= 0:
                self.logger.error("Stop order requires valid stop price")
                return False
        
        return True
    
    async def submit_order(self, order: Order, exchange: str = "NSE") -> bool:
        """Submit order to exchange"""
        if not self._running:
            self.logger.warning("Order manager not running")
            return False
        
        order.status = OrderStatus.PENDING
        order.updated_at = datetime.now()
        
        # Get exchange adapter
        adapter = self.exchanges.get(exchange)
        
        if adapter:
            # Submit through adapter
            try:
                result = await adapter.submit_order(order)
                return result
            except Exception as e:
                self.logger.error(f"Order submission error: {e}")
                order.reject(str(e))
                return False
        else:
            # Simulate submission
            self.logger.info(f"Simulated submission: {order.order_id}")
            return True
    
    async def cancel_order(self, order_id: str, reason: str = "") -> bool:
        """Cancel an order"""
        order = self.orders.get(order_id)
        if not order:
            self.logger.warning(f"Order not found: {order_id}")
            return False
        
        if order.is_complete:
            self.logger.warning(f"Cannot cancel completed order: {order_id}")
            return False
        
        # Cancel through exchange
        if order.exchange and order.exchange in self.exchanges:
            adapter = self.exchanges[order.exchange]
            try:
                await adapter.cancel_order(order)
            except Exception as e:
                self.logger.error(f"Cancel error: {e}")
        
        # Update status
        order.cancel(reason)
        
        # Remove from active
        if order_id in self.active_orders:
            del self.active_orders[order_id]
        
        self._notify_order_cancelled(order)
        return True
    
    async def modify_order(self, order_id: str, new_quantity: int = None,
                          new_price: float = None) -> bool:
        """Modify an existing order"""
        order = self.orders.get(order_id)
        if not order or not order.is_active:
            return False
        
        # Update order
        if new_quantity is not None:
            order.quantity = new_quantity
        if new_price is not None:
            order.limit_price = new_price
        
        order.updated_at = datetime.now()
        
        # Notify exchange
        self._notify_order_modified(order)
        return True
    
    def process_fill(self, order_id: str, price: float, quantity: int,
                    exchange: str = "NSE") -> Optional[Fill]:
        """Process an order fill"""
        order = self.orders.get(order_id)
        if not order:
            self.logger.warning(f"Order not found for fill: {order_id}")
            return None
        
        # Add fill
        fill = order.add_fill(price, quantity, exchange)
        
        # Remove from active if complete
        if order.is_complete:
            if order_id in self.active_orders:
                del self.active_orders[order_id]
        
        # Notify callbacks
        self._notify_fill(fill, order)
        
        self.logger.info(f"Fill processed: {order_id} | {quantity}@{price}")
        return fill
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        order_ids = self.orders_by_symbol.get(symbol, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return list(self.active_orders.values())
    
    def get_open_quantity(self, symbol: str, side: OrderSide) -> int:
        """Get total open quantity for symbol and side"""
        total = 0
        for order in self.get_orders_by_symbol(symbol):
            if order.is_active and order.side == side:
                total += order.remaining_quantity
        return total
    
    def register_order_callback(self, callback: Callable):
        """Register callback for order events"""
        self._order_callbacks.append(callback)
    
    def register_fill_callback(self, callback: Callable):
        """Register callback for fill events"""
        self._fill_callbacks.append(callback)
    
    def _notify_order_created(self, order: Order):
        """Notify order creation"""
        for callback in self._order_callbacks:
            try:
                callback("CREATED", order)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def _notify_order_cancelled(self, order: Order):
        """Notify order cancellation"""
        for callback in self._order_callbacks:
            try:
                callback("CANCELLED", order)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def _notify_order_modified(self, order: Order):
        """Notify order modification"""
        for callback in self._order_callbacks:
            try:
                callback("MODIFIED", order)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def _notify_fill(self, fill: Fill, order: Order):
        """Notify fill"""
        for callback in self._fill_callbacks:
            try:
                callback(fill, order)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def get_statistics(self) -> Dict:
        """Get order statistics"""
        total_orders = len(self.orders)
        active = len(self.active_orders)
        
        filled = sum(1 for o in self.orders.values() if o.status == OrderStatus.FILLED)
        cancelled = sum(1 for o in self.orders.values() if o.status == OrderStatus.CANCELLED)
        
        return {
            "total_orders": total_orders,
            "active_orders": active,
            "filled_orders": filled,
            "cancelled_orders": cancelled,
            "fill_rate": filled / total_orders if total_orders > 0 else 0
        }

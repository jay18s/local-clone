"""
ROX Proven Edge Engine v4.0 - HERMES Agent (Execution)
======================================================
F&O Execution Agent - Handles order execution, slippage, and fill analysis.

HERMES is the execution specialist for F&O trading:
- Order placement and tracking
- Slippage analysis
- Fill rate monitoring
- Execution quality reporting
- Retry logic for failed orders
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import time
import threading


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL_FILL = "PARTIAL_FILL"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_MARKET = "SL-M"


@dataclass
class Order:
    """Order details"""
    order_id: str
    symbol: str
    transaction_type: str      # BUY or SELL
    quantity: int
    order_type: OrderType
    price: float = 0.0
    trigger_price: float = 0.0
    product_type: str = "NRML"
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_price: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class ExecutionMetrics:
    """Execution quality metrics"""
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    cancelled_orders: int = 0
    
    total_slippage: float = 0.0
    avg_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    
    avg_fill_time_ms: float = 0.0
    fill_rate_pct: float = 0.0
    
    # Per-symbol stats
    symbol_stats: Dict[str, Dict] = field(default_factory=dict)


class HermesAgent:
    """
    HERMES - F&O Execution Agent.
    
    Responsibilities:
    - Order lifecycle management
    - Slippage tracking and analysis
    - Execution quality monitoring
    - Failed order retry logic
    - Fill rate reporting
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1
    
    def __init__(self, broker_api=None):
        """
        Initialize HERMES agent.
        
        Args:
            broker_api: Broker API instance for order placement
        """
        self.broker_api = broker_api
        self._orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        self._metrics = ExecutionMetrics()
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def register_callback(self, callback: Callable):
        """Register order status callback"""
        self._callbacks.append(callback)
    
    def place_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0,
        trigger_price: float = 0.0,
        product_type: str = "NRML",
        strategy_id: Optional[str] = None
    ) -> Order:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol
            transaction_type: BUY or SELL
            quantity: Order quantity
            order_type: MARKET, LIMIT, SL, etc.
            price: Limit price (for limit orders)
            trigger_price: Trigger price (for SL orders)
            product_type: NRML, MIS, CNC
            strategy_id: Associated strategy ID
            
        Returns:
            Order object
        """
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self._orders)}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            price=price,
            trigger_price=trigger_price,
            product_type=product_type,
            strategy_id=strategy_id
        )
        
        with self._lock:
            self._orders[order_id] = order
            self._metrics.total_orders += 1
        
        # Attempt to place order
        self._execute_order(order)
        
        return order
    
    def _execute_order(self, order: Order):
        """Execute order via broker API"""
        if not self.broker_api:
            # Simulate order execution for testing
            self._simulate_execution(order)
            return
        
        try:
            # Place order via broker API
            result = self.broker_api.place_order(
                symbol=order.symbol,
                transaction_type=order.transaction_type,
                quantity=order.quantity,
                order_type=order.order_type.value,
                price=order.price,
                trigger_price=order.trigger_price,
                product_type=order.product_type
            )
            
            if result.get("status") == "success":
                order.status = OrderStatus.OPEN
                order.updated_at = datetime.now()
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = result.get("message", "Unknown error")
                self._handle_failed_order(order)
                
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self._handle_failed_order(order)
    
    def _simulate_execution(self, order: Order):
        """Simulate order execution for testing"""
        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            order.status = OrderStatus.COMPLETE
            order.filled_qty = order.quantity
            order.avg_price = order.price if order.price > 0 else 100.0
            order.completed_at = datetime.now()
            order.updated_at = datetime.now()
            
            self._update_metrics(order)
            self._notify_callbacks(order)
        else:
            order.status = OrderStatus.OPEN
            order.updated_at = datetime.now()
    
    def _handle_failed_order(self, order: Order):
        """Handle failed order with retry logic"""
        with self._lock:
            self._metrics.failed_orders += 1
        
        if order.retry_count < self.MAX_RETRIES:
            order.retry_count += 1
            time.sleep(self.RETRY_DELAY_SECONDS * order.retry_count)
            self._execute_order(order)
        else:
            order.status = OrderStatus.REJECTED
            self._notify_callbacks(order)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                return False
            
            if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.PARTIAL_FILL]:
                return False
            
            if self.broker_api:
                try:
                    self.broker_api.cancel_order(order_id)
                except Exception:
                    pass
            
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            self._metrics.cancelled_orders += 1
            
            return True
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        with self._lock:
            return self._orders.get(order_id)
    
    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """
        Get orders with optional filtering.
        
        Args:
            status: Filter by status
            symbol: Filter by symbol
            
        Returns:
            List of matching orders
        """
        with self._lock:
            orders = list(self._orders.values())
            
            if status:
                orders = [o for o in orders if o.status == status]
            
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            
            return orders
    
    def calculate_slippage(
        self,
        order: Order,
        expected_price: float
    ) -> float:
        """
        Calculate slippage for an order.
        
        Args:
            order: Executed order
            expected_price: Expected fill price
            
        Returns:
            Slippage in basis points
        """
        if order.filled_qty == 0 or order.avg_price == 0:
            return 0.0
        
        if order.transaction_type == "BUY":
            slippage = ((order.avg_price - expected_price) / expected_price) * 10000
        else:
            slippage = ((expected_price - order.avg_price) / expected_price) * 10000
        
        return max(0, slippage)
    
    def get_execution_metrics(self) -> ExecutionMetrics:
        """Get current execution metrics"""
        with self._lock:
            return ExecutionMetrics(
                total_orders=self._metrics.total_orders,
                successful_orders=self._metrics.successful_orders,
                failed_orders=self._metrics.failed_orders,
                cancelled_orders=self._metrics.cancelled_orders,
                total_slippage=self._metrics.total_slippage,
                avg_slippage_bps=self._metrics.avg_slippage_bps,
                max_slippage_bps=self._metrics.max_slippage_bps,
                avg_fill_time_ms=self._metrics.avg_fill_time_ms,
                fill_rate_pct=self._calculate_fill_rate()
            )
    
    def _calculate_fill_rate(self) -> float:
        """Calculate fill rate percentage"""
        if self._metrics.total_orders == 0:
            return 0.0
        
        filled = self._metrics.successful_orders
        return (filled / self._metrics.total_orders) * 100
    
    def _update_metrics(self, order: Order):
        """Update execution metrics"""
        with self._lock:
            if order.status == OrderStatus.COMPLETE:
                self._metrics.successful_orders += 1
            
            # Update symbol stats
            if order.symbol not in self._metrics.symbol_stats:
                self._metrics.symbol_stats[order.symbol] = {
                    "total_orders": 0,
                    "successful": 0,
                    "avg_slippage": 0.0
                }
            
            self._metrics.symbol_stats[order.symbol]["total_orders"] += 1
            if order.status == OrderStatus.COMPLETE:
                self._metrics.symbol_stats[order.symbol]["successful"] += 1
    
    def _notify_callbacks(self, order: Order):
        """Notify registered callbacks of order update"""
        for callback in self._callbacks:
            try:
                callback(order)
            except Exception:
                pass
    
    def generate_execution_report(self) -> Dict:
        """Generate execution quality report"""
        metrics = self.get_execution_metrics()
        
        return {
            "timestamp": datetime.now(),
            "total_orders": metrics.total_orders,
            "successful_orders": metrics.successful_orders,
            "failed_orders": metrics.failed_orders,
            "cancelled_orders": metrics.cancelled_orders,
            "fill_rate_pct": round(metrics.fill_rate_pct, 2),
            "avg_slippage_bps": round(metrics.avg_slippage_bps, 2),
            "max_slippage_bps": round(metrics.max_slippage_bps, 2),
            "avg_fill_time_ms": round(metrics.avg_fill_time_ms, 2),
            "symbol_breakdown": metrics.symbol_stats
        }
    
    def start_monitoring(self):
        """Start order monitoring thread"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop order monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            # Poll open orders for updates
            open_orders = self.get_orders(status=OrderStatus.OPEN)
            
            for order in open_orders:
                if self.broker_api:
                    try:
                        status = self.broker_api.get_order_status(order.order_id)
                        self._update_order_status(order, status)
                    except Exception:
                        pass
            
            time.sleep(1)
    
    def _update_order_status(self, order: Order, status: Dict):
        """Update order status from broker"""
        with self._lock:
            order.filled_qty = status.get("filled_qty", order.filled_qty)
            order.avg_price = status.get("avg_price", order.avg_price)
            
            broker_status = status.get("status", "")
            if broker_status == "COMPLETE":
                order.status = OrderStatus.COMPLETE
                order.completed_at = datetime.now()
                self._update_metrics(order)
                self._notify_callbacks(order)
            elif broker_status == "REJECTED":
                order.status = OrderStatus.REJECTED
                order.error_message = status.get("message", "")
                self._handle_failed_order(order)
            elif broker_status == "CANCELLED":
                order.status = OrderStatus.CANCELLED
                self._notify_callbacks(order)
            
            order.updated_at = datetime.now()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_market_order(
    symbol: str,
    transaction_type: str,
    quantity: int,
    product_type: str = "NRML"
) -> Dict:
    """Create a market order specification"""
    return {
        "symbol": symbol,
        "transaction_type": transaction_type,
        "quantity": quantity,
        "order_type": "MARKET",
        "product_type": product_type
    }


def create_limit_order(
    symbol: str,
    transaction_type: str,
    quantity: int,
    price: float,
    product_type: str = "NRML"
) -> Dict:
    """Create a limit order specification"""
    return {
        "symbol": symbol,
        "transaction_type": transaction_type,
        "quantity": quantity,
        "order_type": "LIMIT",
        "price": price,
        "product_type": product_type
    }


def create_sl_order(
    symbol: str,
    transaction_type: str,
    quantity: int,
    trigger_price: float,
    price: float = 0.0,
    product_type: str = "NRML"
) -> Dict:
    """Create a stop-loss order specification"""
    return {
        "symbol": symbol,
        "transaction_type": transaction_type,
        "quantity": quantity,
        "order_type": "SL" if price > 0 else "SL-M",
        "price": price,
        "trigger_price": trigger_price,
        "product_type": product_type
    }

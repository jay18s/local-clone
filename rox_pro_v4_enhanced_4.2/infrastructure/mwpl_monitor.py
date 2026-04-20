"""
ROX Proven Edge Engine v4.0 - MWPL Monitor
==========================================
Market-Wide Position Limits (MWPL) monitoring for SEBI compliance.

SEBI mandates position limits for F&O trading:
- 85% threshold: Enhanced disclosure required
- 95% threshold: Fresh positions blocked
- MWPL = Market Wide Position Limit
- OI = Open Interest

This module tracks OI against MWPL and generates alerts when
approaching limits to prevent position blocking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, date
from enum import Enum
import threading
import time


class MWPLAlertLevel(Enum):
    """MWPL alert levels"""
    NORMAL = "NORMAL"           # < 70%
    ELEVATED = "ELEVATED"       # 70-85%
    WARNING = "WARNING"         # 85-95%
    CRITICAL = "CRITICAL"       # >= 95%


@dataclass
class MWPLData:
    """MWPL data for a symbol"""
    symbol: str
    mwpl: int                      # Market Wide Position Limit
    current_oi: int                # Current Open Interest
    oi_pct: float                  # OI as % of MWPL
    alert_level: MWPLAlertLevel
    last_updated: datetime
    
    # Historical data
    oi_history: List[Dict] = field(default_factory=list)
    
    @property
    def is_blocked(self) -> bool:
        """Check if fresh positions are blocked"""
        return self.oi_pct >= 95.0
    
    @property
    def requires_disclosure(self) -> bool:
        """Check if enhanced disclosure required"""
        return self.oi_pct >= 85.0


@dataclass
class ClientPositionLimit:
    """Client-level position limits"""
    symbol: str
    client_id: str
    gross_open_position: int
    gross_limit: int
    net_position: int
    net_limit: int
    
    @property
    def gross_utilization_pct(self) -> float:
        return (self.gross_open_position / self.gross_limit * 100) if self.gross_limit > 0 else 0
    
    @property
    def net_utilization_pct(self) -> float:
        return (abs(self.net_position) / self.net_limit * 100) if self.net_limit > 0 else 0


class MWPLMonitor:
    """
    MWPL Monitor for SEBI compliance.
    
    Features:
    - Real-time OI tracking against MWPL
    - Alert generation at 85% and 95% thresholds
    - Client position limit tracking
    - Automatic position blocking near limits
    - Historical OI data for analysis
    """
    
    # SEBI thresholds
    DISCLOSURE_THRESHOLD = 85.0
    BLOCKING_THRESHOLD = 95.0
    ELEVATED_THRESHOLD = 70.0
    
    def __init__(self, update_interval_seconds: int = 300):
        """
        Initialize MWPL monitor.
        
        Args:
            update_interval_seconds: Data refresh interval (default 5 min)
        """
        self._mwpl_data: Dict[str, MWPLData] = {}
        self._client_limits: Dict[str, Dict[str, ClientPositionLimit]] = {}
        self._update_interval = update_interval_seconds
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()
    
    def register_callback(self, callback: Callable):
        """Register alert callback"""
        self._callbacks.append(callback)
    
    def update_mwpl_data(
        self,
        symbol: str,
        mwpl: int,
        current_oi: int
    ):
        """
        Update MWPL data for a symbol.
        
        Args:
            symbol: Symbol name
            mwpl: Market Wide Position Limit
            current_oi: Current Open Interest
        """
        with self._lock:
            oi_pct = (current_oi / mwpl * 100) if mwpl > 0 else 0
            
            # Determine alert level
            if oi_pct >= self.BLOCKING_THRESHOLD:
                alert_level = MWPLAlertLevel.CRITICAL
            elif oi_pct >= self.DISCLOSURE_THRESHOLD:
                alert_level = MWPLAlertLevel.WARNING
            elif oi_pct >= self.ELEVATED_THRESHOLD:
                alert_level = MWPLAlertLevel.ELEVATED
            else:
                alert_level = MWPLAlertLevel.NORMAL
            
            # Get existing data for history
            existing = self._mwpl_data.get(symbol)
            history = existing.oi_history.copy() if existing else []
            
            # Add to history
            history.append({
                "timestamp": datetime.now(),
                "oi": current_oi,
                "oi_pct": oi_pct
            })
            
            # Keep last 100 entries
            history = history[-100:]
            
            # Create/update data
            self._mwpl_data[symbol] = MWPLData(
                symbol=symbol,
                mwpl=mwpl,
                current_oi=current_oi,
                oi_pct=oi_pct,
                alert_level=alert_level,
                last_updated=datetime.now(),
                oi_history=history
            )
            
            # Check for alerts
            if alert_level in [MWPLAlertLevel.WARNING, MWPLAlertLevel.CRITICAL]:
                self._trigger_alert(symbol, alert_level, oi_pct)
    
    def update_client_position(
        self,
        symbol: str,
        client_id: str,
        gross_position: int,
        net_position: int,
        gross_limit: Optional[int] = None,
        net_limit: Optional[int] = None
    ):
        """
        Update client position limits.
        
        Args:
            symbol: Symbol name
            client_id: Client identifier
            gross_position: Gross open position
            net_position: Net position (long - short)
            gross_limit: Gross position limit (default: 5% of MWPL)
            net_limit: Net position limit (default: 5% of MWPL)
        """
        with self._lock:
            mwpl_data = self._mwpl_data.get(symbol)
            default_limit = int(mwpl_data.mwpl * 0.05) if mwpl_data else 1000000
            
            if symbol not in self._client_limits:
                self._client_limits[symbol] = {}
            
            self._client_limits[symbol][client_id] = ClientPositionLimit(
                symbol=symbol,
                client_id=client_id,
                gross_open_position=gross_position,
                gross_limit=gross_limit or default_limit,
                net_position=net_position,
                net_limit=net_limit or default_limit
            )
    
    def get_mwpl_data(self, symbol: str) -> Optional[MWPLData]:
        """Get MWPL data for a symbol"""
        with self._lock:
            return self._mwpl_data.get(symbol)
    
    def get_all_mwpl_data(self) -> Dict[str, MWPLData]:
        """Get all MWPL data"""
        with self._lock:
            return self._mwpl_data.copy()
    
    def get_client_position(
        self,
        symbol: str,
        client_id: str
    ) -> Optional[ClientPositionLimit]:
        """Get client position limit"""
        with self._lock:
            symbol_clients = self._client_limits.get(symbol, {})
            return symbol_clients.get(client_id)
    
    def can_open_position(
        self,
        symbol: str,
        client_id: str,
        quantity: int,
        is_buy: bool = True
    ) -> Dict:
        """
        Check if a new position can be opened.
        
        Args:
            symbol: Symbol name
            client_id: Client identifier
            quantity: Position quantity
            is_buy: True for buy, False for sell
            
        Returns:
            Dict with check results
        """
        result = {
            "can_open": True,
            "reason": None,
            "mwpl_pct": 0.0,
            "client_gross_utilization": 0.0,
            "client_net_utilization": 0.0
        }
        
        with self._lock:
            # Check MWPL limit
            mwpl_data = self._mwpl_data.get(symbol)
            if mwpl_data:
                result["mwpl_pct"] = mwpl_data.oi_pct
                
                if mwpl_data.is_blocked:
                    result["can_open"] = False
                    result["reason"] = f"MWPL limit exceeded ({mwpl_data.oi_pct:.1f}% >= 95%)"
                    return result
            
            # Check client limits
            client_pos = self.get_client_position(symbol, client_id)
            if client_pos:
                result["client_gross_utilization"] = client_pos.gross_utilization_pct
                result["client_net_utilization"] = client_pos.net_utilization_pct
                
                new_gross = client_pos.gross_open_position + quantity
                new_net = client_pos.net_position + (quantity if is_buy else -quantity)
                
                if new_gross > client_pos.gross_limit:
                    result["can_open"] = False
                    result["reason"] = f"Client gross limit exceeded ({new_gross} > {client_pos.gross_limit})"
                    return result
                
                if abs(new_net) > client_pos.net_limit:
                    result["can_open"] = False
                    result["reason"] = f"Client net limit exceeded ({abs(new_net)} > {client_pos.net_limit})"
                    return result
        
        return result
    
    def get_symbols_near_limit(
        self,
        threshold: float = 85.0
    ) -> List[str]:
        """
        Get symbols approaching MWPL limit.
        
        Args:
            threshold: OI percentage threshold
            
        Returns:
            List of symbol names
        """
        with self._lock:
            return [
                symbol for symbol, data in self._mwpl_data.items()
                if data.oi_pct >= threshold
            ]
    
    def _trigger_alert(self, symbol: str, level: MWPLAlertLevel, oi_pct: float):
        """Trigger alert callbacks"""
        alert = {
            "type": "MWPL_ALERT",
            "symbol": symbol,
            "level": level.value,
            "oi_pct": oi_pct,
            "timestamp": datetime.now(),
            "message": self._get_alert_message(symbol, level, oi_pct)
        }
        
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception:
                pass
    
    def _get_alert_message(self, symbol: str, level: MWPLAlertLevel, oi_pct: float) -> str:
        """Generate alert message"""
        if level == MWPLAlertLevel.CRITICAL:
            return f"CRITICAL: {symbol} MWPL at {oi_pct:.1f}% - FRESH POSITIONS BLOCKED"
        elif level == MWPLAlertLevel.WARNING:
            return f"WARNING: {symbol} MWPL at {oi_pct:.1f}% - Enhanced disclosure required"
        elif level == MWPLAlertLevel.ELEVATED:
            return f"ELEVATED: {symbol} MWPL at {oi_pct:.1f}% - Monitor closely"
        return f"NORMAL: {symbol} MWPL at {oi_pct:.1f}%"
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            # In production, this would fetch live data from NSE
            # For now, just trigger any pending alerts
            time.sleep(self._update_interval)
    
    def generate_report(self) -> Dict:
        """Generate MWPL monitoring report"""
        with self._lock:
            total_symbols = len(self._mwpl_data)
            critical_count = sum(1 for d in self._mwpl_data.values() if d.alert_level == MWPLAlertLevel.CRITICAL)
            warning_count = sum(1 for d in self._mwpl_data.values() if d.alert_level == MWPLAlertLevel.WARNING)
            elevated_count = sum(1 for d in self._mwpl_data.values() if d.alert_level == MWPLAlertLevel.ELEVATED)
            
            return {
                "timestamp": datetime.now(),
                "total_symbols_tracked": total_symbols,
                "critical_symbols": critical_count,
                "warning_symbols": warning_count,
                "elevated_symbols": elevated_count,
                "normal_symbols": total_symbols - critical_count - warning_count - elevated_count,
                "symbols_near_limit": self.get_symbols_near_limit(85.0)
            }


# ============================================================================
# Singleton Instance
# ============================================================================

# Global MWPL monitor instance
_mwpl_monitor: Optional[MWPLMonitor] = None


def get_mwpl_monitor() -> MWPLMonitor:
    """Get or create the global MWPL monitor instance"""
    global _mwpl_monitor
    if _mwpl_monitor is None:
        _mwpl_monitor = MWPLMonitor()
    return _mwpl_monitor

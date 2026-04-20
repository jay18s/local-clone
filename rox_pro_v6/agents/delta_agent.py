"""
ROX Proven Edge Engine v4.0 - DELTA Agent (Settlement & Compliance)
===================================================================
Physical Settlement & Compliance Agent - Manages F&O settlement obligations.

DELTA handles physical settlement and compliance:
- Physical settlement obligation tracking
- Auto-exit logic for near-expiry positions
- SEBI compliance monitoring
- Delivery obligation calculations
- Settlement calendar management
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, date, timedelta
from enum import Enum
import threading


class SettlementType(Enum):
    """Settlement type"""
    CASH = "CASH"
    PHYSICAL = "PHYSICAL"


class SettlementStatus(Enum):
    """Settlement status"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ComplianceStatus(Enum):
    """Compliance status"""
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"


@dataclass
class SettlementObligation:
    """Physical settlement obligation"""
    symbol: str
    position_type: str          # LONG or SHORT
    option_type: Optional[str]  # CE, PE, or None for futures
    quantity_lots: int
    quantity_shares: int
    strike: float
    
    obligation_type: str        # BUY_SHARES, SELL_SHARES, DELIVER_SHARES, ACCEPT_SHARES
    obligation_value: float
    
    expiry_date: date
    days_to_expiry: int
    
    status: SettlementStatus = SettlementStatus.PENDING
    requires_action: bool = False
    auto_exit_triggered: bool = False
    
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


@dataclass
class ComplianceCheck:
    """Compliance check result"""
    check_type: str
    status: ComplianceStatus
    message: str
    details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class DeltaAgent:
    """
    DELTA - Physical Settlement & Compliance Agent.
    
    Responsibilities:
    - Track physical settlement obligations
    - Monitor expiry dates and auto-exit positions
    - Ensure SEBI compliance
    - Calculate delivery obligations
    - Manage settlement calendar
    
    Physical Settlement Rules (NSE):
    - ITM stock options: Physical delivery
    - ITM stock futures: Physical delivery
    - Index options: Cash settlement
    - Index futures: Cash settlement
    """
    
    # Auto-exit configuration
    AUTO_EXIT_DAYS_BEFORE_EXPIRY = 5
    CRITICAL_DAYS_BEFORE_EXPIRY = 2
    
    def __init__(
        self,
        instrument_manager=None,
        mwpl_monitor=None,
        auto_exit_enabled: bool = True
    ):
        """
        Initialize DELTA agent.
        
        Args:
            instrument_manager: F&O instrument manager
            mwpl_monitor: MWPL monitor for position limits
            auto_exit_enabled: Enable auto-exit for physical settlement
        """
        self.instrument_manager = instrument_manager
        self.mwpl_monitor = mwpl_monitor
        self.auto_exit_enabled = auto_exit_enabled
        
        self._obligations: Dict[str, SettlementObligation] = {}
        self._compliance_checks: List[ComplianceCheck] = []
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()
        self._running = False
    
    def register_callback(self, callback: Callable):
        """Register settlement alert callback"""
        self._callbacks.append(callback)
    
    def check_physical_settlement_required(
        self,
        symbol: str,
        position_type: str,
        option_type: Optional[str],
        quantity: int,
        strike: float,
        spot_at_expiry: float
    ) -> SettlementObligation:
        """
        Check if a position requires physical settlement.
        
        Args:
            symbol: Contract symbol
            position_type: LONG or SHORT
            option_type: CE, PE, or None for futures
            quantity: Position quantity in lots
            strike: Strike price
            spot_at_expiry: Spot price at expiry
            
        Returns:
            SettlementObligation if physical settlement required
        """
        # Check if index (cash settled)
        if self.instrument_manager:
            contract = self.instrument_manager.parse_symbol(symbol)
            if contract and not self.instrument_manager.is_physical_settlement(symbol):
                # Cash settlement - no physical delivery
                return SettlementObligation(
                    symbol=symbol,
                    position_type=position_type,
                    option_type=option_type,
                    quantity_lots=quantity,
                    quantity_shares=0,
                    strike=strike,
                    obligation_type="CASH_SETTLED",
                    obligation_value=0,
                    expiry_date=date.today(),
                    days_to_expiry=0,
                    requires_action=False
                )
        
        # Get lot size
        lot_size = 50
        if self.instrument_manager:
            lot_size = self.instrument_manager.get_lot_size(symbol.split("2")[0] if "2" in symbol else symbol)
        
        total_shares = quantity * lot_size
        
        obligation = SettlementObligation(
            symbol=symbol,
            position_type=position_type,
            option_type=option_type,
            quantity_lots=quantity,
            quantity_shares=total_shares,
            strike=strike,
            obligation_type=None,
            obligation_value=0,
            expiry_date=date.today(),
            days_to_expiry=0
        )
        
        # Determine obligation
        if option_type == "CE":  # Call option
            if position_type == "LONG":
                if spot_at_expiry > strike:
                    obligation.obligation_type = "BUY_SHARES"
                    obligation.obligation_value = total_shares * strike
                    obligation.requires_action = True
            else:  # SHORT
                if spot_at_expiry > strike:
                    obligation.obligation_type = "DELIVER_SHARES"
                    obligation.obligation_value = total_shares * strike
                    obligation.requires_action = True
                    
        elif option_type == "PE":  # Put option
            if position_type == "LONG":
                if spot_at_expiry < strike:
                    obligation.obligation_type = "SELL_SHARES"
                    obligation.obligation_value = total_shares * strike
                    obligation.requires_action = True
            else:  # SHORT
                if spot_at_expiry < strike:
                    obligation.obligation_type = "ACCEPT_SHARES"
                    obligation.obligation_value = total_shares * strike
                    obligation.requires_action = True
        
        else:  # Futures
            if position_type == "LONG":
                obligation.obligation_type = "ACCEPT_SHARES"
                obligation.obligation_value = total_shares * spot_at_expiry
                obligation.requires_action = True
            else:  # SHORT
                obligation.obligation_type = "DELIVER_SHARES"
                obligation.obligation_value = total_shares * spot_at_expiry
                obligation.requires_action = True
        
        # Store obligation
        if obligation.requires_action:
            with self._lock:
                self._obligations[symbol] = obligation
            self._notify_obligation(obligation)
        
        return obligation
    
    def monitor_expiry_positions(
        self,
        positions: List[Dict],
        current_date: Optional[date] = None
    ) -> List[Dict]:
        """
        Monitor positions approaching expiry.
        
        Args:
            positions: List of position dicts with symbol, expiry, etc.
            current_date: Current date (default: today)
            
        Returns:
            List of positions requiring action
        """
        if current_date is None:
            current_date = date.today()
        
        actions_needed = []
        
        for position in positions:
            expiry = position.get("expiry")
            if not expiry:
                continue
            
            if isinstance(expiry, str):
                expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
            
            days_to_expiry = (expiry - current_date).days
            
            # Check if auto-exit needed
            if days_to_expiry <= self.AUTO_EXIT_DAYS_BEFORE_EXPIRY:
                action = {
                    "symbol": position["symbol"],
                    "days_to_expiry": days_to_expiry,
                    "action_required": None,
                    "urgency": "NORMAL"
                }
                
                # Determine action
                if self.auto_exit_enabled and days_to_expiry <= self.AUTO_EXIT_DAYS_BEFORE_EXPIRY:
                    if days_to_expiry <= self.CRITICAL_DAYS_BEFORE_EXPIRY:
                        action["action_required"] = "IMMEDIATE_EXIT"
                        action["urgency"] = "CRITICAL"
                        action["message"] = f"Exit position immediately - {days_to_expiry} days to expiry"
                    else:
                        action["action_required"] = "PLAN_EXIT"
                        action["urgency"] = "HIGH"
                        action["message"] = f"Plan position exit - {days_to_expiry} days to expiry"
                
                # Check if physical settlement applies
                if self.instrument_manager:
                    if self.instrument_manager.is_physical_settlement(position["symbol"]):
                        action["physical_settlement"] = True
                        action["message"] += " (Physical settlement applies)"
                
                actions_needed.append(action)
        
        return actions_needed
    
    def check_compliance(
        self,
        client_id: str,
        positions: List[Dict]
    ) -> List[ComplianceCheck]:
        """
        Run compliance checks on positions.
        
        Args:
            client_id: Client identifier
            positions: List of positions
            
        Returns:
            List of compliance check results
        """
        checks = []
        
        # Check MWPL limits
        if self.mwpl_monitor:
            for position in positions:
                symbol = position["symbol"]
                can_open = self.mwpl_monitor.can_open_position(
                    symbol, client_id, position.get("quantity", 0)
                )
                
                if not can_open["can_open"]:
                    checks.append(ComplianceCheck(
                        check_type="MWPL_LIMIT",
                        status=ComplianceStatus.VIOLATION,
                        message=can_open["reason"],
                        details={"symbol": symbol, "mwpl_pct": can_open["mwpl_pct"]}
                    ))
        
        # Check position concentration
        total_value = sum(p.get("value", 0) for p in positions)
        for position in positions:
            position_value = position.get("value", 0)
            if total_value > 0:
                concentration = position_value / total_value
                if concentration > 0.25:  # 25% limit
                    checks.append(ComplianceCheck(
                        check_type="CONCENTRATION_LIMIT",
                        status=ComplianceStatus.WARNING,
                        message=f"Position concentration {concentration*100:.1f}% exceeds 25%",
                        details={"symbol": position["symbol"], "concentration": concentration}
                    ))
        
        # Check for pending settlements
        pending_settlements = [o for o in self._obligations.values() 
                              if o.status == SettlementStatus.PENDING]
        if pending_settlements:
            checks.append(ComplianceCheck(
                check_type="PENDING_SETTLEMENTS",
                status=ComplianceStatus.WARNING,
                message=f"{len(pending_settlements)} pending settlement obligations",
                details={"count": len(pending_settlements)}
            ))
        
        self._compliance_checks = checks
        return checks
    
    def get_settlement_obligations(
        self,
        status: Optional[SettlementStatus] = None
    ) -> List[SettlementObligation]:
        """
        Get settlement obligations.
        
        Args:
            status: Filter by status
            
        Returns:
            List of settlement obligations
        """
        with self._lock:
            obligations = list(self._obligations.values())
            
            if status:
                obligations = [o for o in obligations if o.status == status]
            
            return obligations
    
    def resolve_obligation(
        self,
        symbol: str,
        resolution: str
    ) -> bool:
        """
        Mark a settlement obligation as resolved.
        
        Args:
            symbol: Symbol of the obligation
            resolution: Resolution type (EXITED, SETTLED, etc.)
            
        Returns:
            True if resolved successfully
        """
        with self._lock:
            obligation = self._obligations.get(symbol)
            if not obligation:
                return False
            
            obligation.status = SettlementStatus.COMPLETED
            obligation.resolved_at = datetime.now()
            
            return True
    
    def generate_settlement_report(self) -> Dict:
        """Generate settlement status report"""
        with self._lock:
            pending = [o for o in self._obligations.values() if o.status == SettlementStatus.PENDING]
            completed = [o for o in self._obligations.values() if o.status == SettlementStatus.COMPLETED]
            
            total_value = sum(o.obligation_value for o in pending)
            
            return {
                "timestamp": datetime.now(),
                "pending_obligations": len(pending),
                "completed_obligations": len(completed),
                "total_pending_value": round(total_value, 2),
                "auto_exit_enabled": self.auto_exit_enabled,
                "critical_positions": [
                    {
                        "symbol": o.symbol,
                        "obligation_type": o.obligation_type,
                        "shares": o.quantity_shares,
                        "value": o.obligation_value,
                        "days_to_expiry": o.days_to_expiry
                    }
                    for o in pending if o.days_to_expiry <= self.CRITICAL_DAYS_BEFORE_EXPIRY
                ],
                "compliance_status": "COMPLIANT" if not self._compliance_checks else "REVIEW_REQUIRED"
            }
    
    def _notify_obligation(self, obligation: SettlementObligation):
        """Notify callbacks of new obligation"""
        for callback in self._callbacks:
            try:
                callback(obligation)
            except Exception:
                pass
    
    def get_auto_exit_recommendations(self) -> List[Dict]:
        """
        Get positions that should be auto-exited.
        
        Returns:
            List of exit recommendations
        """
        recommendations = []
        
        with self._lock:
            for obligation in self._obligations.values():
                if obligation.requires_action and not obligation.auto_exit_triggered:
                    if obligation.days_to_expiry <= self.AUTO_EXIT_DAYS_BEFORE_EXPIRY:
                        recommendations.append({
                            "symbol": obligation.symbol,
                            "action": "EXIT",
                            "reason": f"Physical settlement in {obligation.days_to_expiry} days",
                            "obligation_type": obligation.obligation_type,
                            "obligation_value": obligation.obligation_value
                        })
        
        return recommendations


# ============================================================================
# Convenience Functions
# ============================================================================

def is_physical_settlement_required(
    symbol: str,
    option_type: Optional[str],
    strike: float,
    spot_at_expiry: float,
    position_type: str
) -> bool:
    """
    Quick check if physical settlement is required.
    
    Args:
        symbol: Contract symbol
        option_type: CE, PE, or None
        strike: Strike price
        spot_at_expiry: Spot at expiry
        position_type: LONG or SHORT
        
    Returns:
        True if physical settlement required
    """
    # Index options are cash settled
    index_symbols = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"}
    underlying = symbol.split("2")[0] if "2" in symbol else symbol
    
    if underlying in index_symbols:
        return False
    
    # Check ITM condition
    if option_type == "CE":
        return spot_at_expiry > strike
    elif option_type == "PE":
        return spot_at_expiry < strike
    
    # Futures always require physical settlement for stocks
    return True


def calculate_settlement_value(
    quantity_shares: int,
    price: float,
    obligation_type: str
) -> Dict:
    """
    Calculate settlement value and margin requirements.
    
    Args:
        quantity_shares: Number of shares
        price: Settlement price
        obligation_type: Type of obligation
        
    Returns:
        Dict with settlement details
    """
    gross_value = quantity_shares * price
    
    # Margin requirements for physical settlement
    margin_pct = 0.20  # 20% margin
    margin_required = gross_value * margin_pct
    
    return {
        "gross_value": gross_value,
        "margin_required": margin_required,
        "net_value": gross_value - margin_required,
        "obligation_type": obligation_type
    }

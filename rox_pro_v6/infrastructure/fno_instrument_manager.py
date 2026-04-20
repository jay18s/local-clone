"""
ROX Proven Edge Engine v4.0 - F&O Instrument Manager
=====================================================
Manages F&O instrument contracts for NSE trading.

NSE uses complex naming conventions (e.g., NIFTY26MAR22600CE) that require
systematic parsing and resolution. This module maintains master contracts
for all F&O instruments, tracks contract lifecycles, and provides lot size
and strike interval data for order construction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, date, timedelta
from enum import Enum
import re


class InstrumentType(Enum):
    """F&O instrument types"""
    INDEX_FUTURE = "INDEX_FUTURE"
    INDEX_OPTION = "INDEX_OPTION"
    STOCK_FUTURE = "STOCK_FUTURE"
    STOCK_OPTION = "STOCK_OPTION"


@dataclass
class FnoContract:
    """F&O contract details"""
    symbol: str                    # Trading symbol (e.g., "NIFTY26MAR22600CE")
    underlying: str                # Underlying symbol (e.g., "NIFTY")
    instrument_type: InstrumentType
    expiry_date: date
    strike_price: float            # 0 for futures
    option_type: Optional[str]     # "CE" or "PE", None for futures
    lot_size: int
    tick_size: float = 0.05
    freeze_quantity: int = 0       # Maximum order quantity
    
    # Derived properties
    @property
    def is_option(self) -> bool:
        return self.instrument_type in [InstrumentType.INDEX_OPTION, InstrumentType.STOCK_OPTION]
    
    @property
    def is_future(self) -> bool:
        return self.instrument_type in [InstrumentType.INDEX_FUTURE, InstrumentType.STOCK_FUTURE]
    
    @property
    def days_to_expiry(self) -> int:
        return max(0, (self.expiry_date - date.today()).days)


@dataclass
class StrikeInfo:
    """Strike price information"""
    strike: float
    call_oi: int = 0
    put_oi: int = 0
    call_volume: int = 0
    put_volume: int = 0
    call_iv: float = 0.0
    put_iv: float = 0.0


@dataclass
class OptionChain:
    """Complete option chain for an underlying"""
    underlying: str
    spot_price: float
    expiry_date: date
    strikes: List[StrikeInfo] = field(default_factory=list)
    pcr: float = 1.0
    max_pain: float = 0.0
    iv_rank: float = 50.0
    
    def get_strikes_near_spot(self, num_strikes: int = 5) -> List[StrikeInfo]:
        """Get strikes closest to spot price"""
        sorted_strikes = sorted(self.strikes, key=lambda s: abs(s.strike - self.spot_price))
        return sorted_strikes[:num_strikes]
    
    def get_call_oi_wall(self) -> Optional[StrikeInfo]:
        """Get strike with highest call OI (resistance)"""
        if not self.strikes:
            return None
        return max(self.strikes, key=lambda s: s.call_oi)
    
    def get_put_oi_wall(self) -> Optional[StrikeInfo]:
        """Get strike with highest put OI (support)"""
        if not self.strikes:
            return None
        return max(self.strikes, key=lambda s: s.put_oi)


class FnoInstrumentManager:
    """
    F&O Instrument Manager for NSE contract resolution.
    
    Handles:
    - Contract symbol parsing (e.g., NIFTY26MAR22600CE)
    - Master contract database
    - Lot size and strike interval management
    - Option chain construction
    - Expiry tracking
    """
    
    # NSE Index symbols
    INDEX_SYMBOLS = {
        "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY",
        "SENSEX", "BANKEX"
    }
    
    # Month codes used by NSE
    MONTH_CODES = {
        1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
        5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG",
        9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
    }
    
    # Reverse month code lookup
    MONTH_CODE_TO_NUM = {v: k for k, v in MONTH_CODES.items()}
    
    # Default strike intervals
    DEFAULT_STRIKE_INTERVALS = {
        "NIFTY": 50,
        "BANKNIFTY": 100,
        "FINNIFTY": 50,
        "MIDCPNIFTY": 50,
        "SENSEX": 100,
        "BANKEX": 100,
    }
    
    def __init__(self):
        """Initialize the instrument manager"""
        self._contracts: Dict[str, FnoContract] = {}
        self._underlying_contracts: Dict[str, List[str]] = {}
        self._expiry_calendar: Dict[str, List[date]] = {}
        self._lot_sizes: Dict[str, int] = {}
        self._strike_intervals: Dict[str, int] = {}
        
        # Initialize with default data
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize with default NSE F&O data"""
        # Default lot sizes for major indices
        self._lot_sizes = {
            "NIFTY": 75,
            "BANKNIFTY": 15,
            "FINNIFTY": 40,
            "MIDCPNIFTY": 50,
            "SENSEX": 10,
            "BANKEX": 15,
            "RELIANCE": 250,
            "TCS": 175,
            "HDFCBANK": 550,
            "ICICIBANK": 700,
            "INFY": 400,
            "ITC": 1600,
            "SBIN": 750,
            "BHARTIARTL": 425,
        }
        
        # Default strike intervals
        self._strike_intervals = dict(self.DEFAULT_STRIKE_INTERVALS)
    
    def parse_symbol(self, symbol: str) -> Optional[FnoContract]:
        """
        Parse NSE F&O symbol into contract details.
        
        NSE symbol format examples:
        - NIFTY26MAR22FUT (Index Future)
        - NIFTY26MAR226000CE (Index Option)
        - RELIANCE26MAR22FUT (Stock Future)
        - RELIANCE26MAR222500CE (Stock Option)
        
        Args:
            symbol: NSE trading symbol
            
        Returns:
            FnoContract if parsing successful, None otherwise
        """
        try:
            # Try to match option symbol
            option_pattern = r'^([A-Z]+)(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$'
            option_match = re.match(option_pattern, symbol.upper())
            
            if option_match:
                underlying = option_match.group(1)
                day = int(option_match.group(2))
                month_str = option_match.group(3)
                year = 2000 + int(option_match.group(4))
                strike = float(option_match.group(5))
                opt_type = option_match.group(6)
                
                month = self.MONTH_CODE_TO_NUM.get(month_str)
                if not month:
                    return None
                
                expiry = date(year, month, day)
                
                # Determine instrument type
                if underlying in self.INDEX_SYMBOLS:
                    inst_type = InstrumentType.INDEX_OPTION
                else:
                    inst_type = InstrumentType.STOCK_OPTION
                
                return FnoContract(
                    symbol=symbol,
                    underlying=underlying,
                    instrument_type=inst_type,
                    expiry_date=expiry,
                    strike_price=strike,
                    option_type=opt_type,
                    lot_size=self.get_lot_size(underlying)
                )
            
            # Try to match future symbol
            future_pattern = r'^([A-Z]+)(\d{2})([A-Z]{3})(\d{2})FUT$'
            future_match = re.match(future_pattern, symbol.upper())
            
            if future_match:
                underlying = future_match.group(1)
                day = int(future_match.group(2))
                month_str = future_match.group(3)
                year = 2000 + int(future_match.group(4))
                
                month = self.MONTH_CODE_TO_NUM.get(month_str)
                if not month:
                    return None
                
                expiry = date(year, month, day)
                
                # Determine instrument type
                if underlying in self.INDEX_SYMBOLS:
                    inst_type = InstrumentType.INDEX_FUTURE
                else:
                    inst_type = InstrumentType.STOCK_FUTURE
                
                return FnoContract(
                    symbol=symbol,
                    underlying=underlying,
                    instrument_type=inst_type,
                    expiry_date=expiry,
                    strike_price=0,
                    option_type=None,
                    lot_size=self.get_lot_size(underlying)
                )
            
            return None
            
        except Exception:
            return None
    
    def build_symbol(
        self,
        underlying: str,
        expiry: date,
        strike: Optional[float] = None,
        option_type: Optional[str] = None
    ) -> str:
        """
        Build NSE F&O symbol from components.
        
        Args:
            underlying: Underlying symbol (e.g., "NIFTY")
            expiry: Expiry date
            strike: Strike price (None for futures)
            option_type: "CE" or "PE" (None for futures)
            
        Returns:
            NSE trading symbol
        """
        day = expiry.day
        month = self.MONTH_CODES.get(expiry.month, "JAN")
        year = expiry.year % 100
        
        if strike is not None and option_type:
            # Option symbol
            return f"{underlying}{day:02d}{month}{year:02d}{int(strike)}{option_type.upper()}"
        else:
            # Future symbol
            return f"{underlying}{day:02d}{month}{year:02d}FUT"
    
    def get_lot_size(self, underlying: str) -> int:
        """Get lot size for an underlying"""
        return self._lot_sizes.get(underlying.upper(), 50)
    
    def set_lot_size(self, underlying: str, lot_size: int):
        """Set lot size for an underlying"""
        self._lot_sizes[underlying.upper()] = lot_size
    
    def get_strike_interval(self, underlying: str) -> int:
        """Get strike interval for an underlying"""
        return self._strike_intervals.get(underlying.upper(), 50)
    
    def set_strike_interval(self, underlying: str, interval: int):
        """Set strike interval for an underlying"""
        self._strike_intervals[underlying.upper()] = interval
    
    def get_atm_strike(self, underlying: str, spot_price: float) -> float:
        """
        Get at-the-money strike for an underlying.
        
        Args:
            underlying: Underlying symbol
            spot_price: Current spot price
            
        Returns:
            ATM strike price
        """
        interval = self.get_strike_interval(underlying)
        return round(spot_price / interval) * interval
    
    def get_strikes_around_spot(
        self,
        underlying: str,
        spot_price: float,
        num_strikes: int = 10
    ) -> List[float]:
        """
        Get strikes around spot price.
        
        Args:
            underlying: Underlying symbol
            spot_price: Current spot price
            num_strikes: Number of strikes on each side
            
        Returns:
            List of strike prices
        """
        interval = self.get_strike_interval(underlying)
        atm = self.get_atm_strike(underlying, spot_price)
        
        strikes = []
        for i in range(-num_strikes, num_strikes + 1):
            strikes.append(atm + i * interval)
        
        return sorted(strikes)
    
    def get_next_expiry(
        self,
        underlying: str,
        option_type: Optional[str] = None,
        weekly: bool = True
    ) -> Optional[date]:
        """
        Get next expiry date for an underlying.
        
        For indices, returns next Thursday (weekly expiry).
        For stocks, returns last Thursday of month (monthly expiry).
        
        Args:
            underlying: Underlying symbol
            option_type: "CE", "PE", or None for futures
            weekly: True for weekly expiry (indices), False for monthly
            
        Returns:
            Next expiry date
        """
        today = date.today()
        
        if weekly and underlying.upper() in self.INDEX_SYMBOLS:
            # Weekly expiry (next Thursday)
            days_to_thursday = (3 - today.weekday()) % 7
            if days_to_thursday == 0:
                days_to_thursday = 7  # If today is Thursday, next week
            return today + timedelta(days=days_to_thursday)
        else:
            # Monthly expiry (last Thursday of month)
            next_month = today.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            
            # Find last Thursday
            while last_day.weekday() != 3:  # 3 = Thursday
                last_day -= timedelta(days=1)
            
            # If last Thursday has passed, go to next month
            if last_day < today:
                next_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
                last_day = next_month - timedelta(days=1)
                while last_day.weekday() != 3:
                    last_day -= timedelta(days=1)
            
            return last_day
    
    def register_contract(self, contract: FnoContract):
        """Register a contract in the manager"""
        self._contracts[contract.symbol] = contract
        
        # Add to underlying index
        if contract.underlying not in self._underlying_contracts:
            self._underlying_contracts[contract.underlying] = []
        if contract.symbol not in self._underlying_contracts[contract.underlying]:
            self._underlying_contracts[contract.underlying].append(contract.symbol)
    
    def get_contract(self, symbol: str) -> Optional[FnoContract]:
        """Get contract by symbol"""
        return self._contracts.get(symbol)
    
    def get_contracts_for_underlying(self, underlying: str) -> List[FnoContract]:
        """Get all contracts for an underlying"""
        symbols = self._underlying_contracts.get(underlying.upper(), [])
        return [self._contracts[s] for s in symbols if s in self._contracts]
    
    def is_physical_settlement(self, symbol: str) -> bool:
        """
        Check if a contract requires physical settlement.
        
        Since 2019, all in-the-money stock F&O contracts must be physically settled.
        Index options are cash-settled.
        
        Args:
            symbol: Contract symbol
            
        Returns:
            True if physical settlement required
        """
        contract = self.parse_symbol(symbol)
        if not contract:
            return False
        
        # Index options are cash-settled
        if contract.instrument_type == InstrumentType.INDEX_OPTION:
            return False
        
        # Index futures are cash-settled
        if contract.instrument_type == InstrumentType.INDEX_FUTURE:
            return False
        
        # Stock options and futures are physically settled
        return True
    
    def calculate_delivery_obligation(
        self,
        symbol: str,
        position_type: str,  # LONG or SHORT
        option_type: Optional[str],  # CE, PE, or None for futures
        quantity: int,
        spot_at_expiry: float,
        strike: float
    ) -> Dict:
        """
        Calculate delivery obligation for physical settlement.
        
        Args:
            symbol: Contract symbol
            position_type: LONG or SHORT
            option_type: CE, PE, or None for futures
            quantity: Number of lots
            spot_at_expiry: Spot price at expiry
            strike: Strike price
            
        Returns:
            Dict with obligation details
        """
        contract = self.parse_symbol(symbol)
        if not contract:
            return {"error": "Invalid symbol"}
        
        lot_size = contract.lot_size
        total_shares = quantity * lot_size
        
        obligation = {
            "symbol": symbol,
            "position_type": position_type,
            "option_type": option_type,
            "quantity_lots": quantity,
            "quantity_shares": total_shares,
            "obligation_type": None,
            "obligation_value": 0.0,
            "requires_action": False
        }
        
        if option_type == "CE":  # Call option
            if position_type == "LONG":
                # Long call: ITM means right to buy
                if spot_at_expiry > strike:
                    obligation["obligation_type"] = "BUY_SHARES"
                    obligation["obligation_value"] = total_shares * strike
                    obligation["requires_action"] = True
            else:  # SHORT
                # Short call: ITM means obligation to deliver
                if spot_at_expiry > strike:
                    obligation["obligation_type"] = "DELIVER_SHARES"
                    obligation["obligation_value"] = total_shares * strike
                    obligation["requires_action"] = True
                    
        elif option_type == "PE":  # Put option
            if position_type == "LONG":
                # Long put: ITM means right to sell
                if spot_at_expiry < strike:
                    obligation["obligation_type"] = "SELL_SHARES"
                    obligation["obligation_value"] = total_shares * strike
                    obligation["requires_action"] = True
            else:  # SHORT
                # Short put: ITM means obligation to accept delivery
                if spot_at_expiry < strike:
                    obligation["obligation_type"] = "ACCEPT_SHARES"
                    obligation["obligation_value"] = total_shares * strike
                    obligation["requires_action"] = True
        
        else:  # Futures
            if position_type == "LONG":
                obligation["obligation_type"] = "ACCEPT_SHARES"
                obligation["obligation_value"] = total_shares * spot_at_expiry
                obligation["requires_action"] = True
            else:  # SHORT
                obligation["obligation_type"] = "DELIVER_SHARES"
                obligation["obligation_value"] = total_shares * spot_at_expiry
                obligation["requires_action"] = True
        
        return obligation


# ============================================================================
# Singleton Instance
# ============================================================================

# Global instrument manager instance
_instrument_manager: Optional[FnoInstrumentManager] = None


def get_instrument_manager() -> FnoInstrumentManager:
    """Get or create the global instrument manager instance"""
    global _instrument_manager
    if _instrument_manager is None:
        _instrument_manager = FnoInstrumentManager()
    return _instrument_manager

"""
ROX Proven Edge Engine v4.0 - Strategy Builders
===============================================
Implements 5 F&O strategy builders for different market conditions.

Strategies:
1. Iron Condor Builder - Range-bound, theta collection
2. Calendar Spread Builder - Low IV, time decay
3. Bull Spread Builder - Bullish directional
4. Bear Spread Builder - Bearish directional
5. Collar Builder - Hedging existing positions

Each builder includes:
- Strike selection logic
- Greeks-aware position sizing
- Margin requirement calculation
- Risk/reward analysis
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import date, timedelta


class StrategyType(Enum):
    """F&O strategy types"""
    IRON_CONDOR = "IRON_CONDOR"
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    BULL_SPREAD = "BULL_SPREAD"
    BEAR_SPREAD = "BEAR_SPREAD"
    COLLAR = "COLLAR"


class MarketBias(Enum):
    """Market bias for strategy selection"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    VOLATILE = "VOLATILE"


@dataclass
class StrategyLeg:
    """Single leg of an options strategy"""
    option_type: str          # CE or PE
    strike: float
    expiry: date
    position: str             # LONG or SHORT
    quantity: int
    lot_size: int = 50
    premium: float = 0.0
    iv: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.position == "LONG"
    
    @property
    def is_short(self) -> bool:
        return self.position == "SHORT"


@dataclass
class StrategyResult:
    """Complete strategy construction result"""
    strategy_type: StrategyType
    underlying: str
    spot_price: float
    
    # Legs
    legs: List[StrategyLeg] = field(default_factory=list)
    
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # P&L
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven_low: float = 0.0
    breakeven_high: float = 0.0
    
    # Margin
    margin_required: float = 0.0
    
    # Risk metrics
    risk_reward_ratio: float = 0.0
    probability_of_profit: float = 0.0
    
    # Metadata
    vix_regime: str = ""
    conviction: int = 0
    notes: List[str] = field(default_factory=list)
    
    def add_note(self, note: str):
        """Add a note to the strategy"""
        self.notes.append(note)


class IronCondorBuilder:
    """
    Iron Condor Strategy Builder.
    
    Strategy: Sell OTM Call + Sell OTM Put + Buy further OTM Call + Buy further OTM Put
    
    Best for:
    - Range-bound markets
    - High IV (VIX > 18)
    - Theta collection
    
    Risk/Reward: Limited profit, limited risk
    """
    
    # Default parameters
    DEFAULT_CALL_DELTA = 0.16
    DEFAULT_PUT_DELTA = -0.16
    DEFAULT_WING_WIDTH = 0.05  # 5% of spot
    MIN_VIX = 18.0
    MAX_VIX = 25.0
    
    def __init__(self, instrument_manager=None, greeks_calculator=None):
        self.instrument_manager = instrument_manager
        self.greeks_calculator = greeks_calculator
    
    def build(
        self,
        underlying: str,
        spot_price: float,
        vix: float,
        expiry: Optional[date] = None,
        call_delta: float = None,
        put_delta: float = None,
        wing_width_pct: float = None,
        quantity: int = 1
    ) -> StrategyResult:
        """
        Build an Iron Condor strategy.
        
        Args:
            underlying: Underlying symbol
            spot_price: Current spot price
            vix: Current VIX level
            expiry: Expiry date (default: next weekly)
            call_delta: Target call delta (default: 0.16)
            put_delta: Target put delta (default: -0.16)
            wing_width_pct: Wing width as % of spot (default: 5%)
            quantity: Number of lots
            
        Returns:
            StrategyResult with complete strategy details
        """
        result = StrategyResult(
            strategy_type=StrategyType.IRON_CONDOR,
            underlying=underlying,
            spot_price=spot_price,
            vix_regime=self._get_vix_regime(vix)
        )
        
        # Check VIX suitability
        if vix < self.MIN_VIX:
            result.add_note(f"VIX {vix:.1f} below minimum {self.MIN_VIX} - consider waiting")
            result.conviction = 50
        elif vix > self.MAX_VIX:
            result.add_note(f"VIX {vix:.1f} above maximum {self.MAX_VIX} - high risk")
            result.conviction = 60
        else:
            result.conviction = 80
        
        # Get expiry
        if expiry is None and self.instrument_manager:
            expiry = self.instrument_manager.get_next_expiry(underlying, weekly=True)
        if expiry is None:
            expiry = date.today() + timedelta(days=7)
        
        dte = max(1, (expiry - date.today()).days)
        
        # Calculate strikes
        wing_width = (wing_width_pct or self.DEFAULT_WING_WIDTH) * spot_price
        
        # Short strikes (based on delta)
        short_call_strike = spot_price * (1 + 0.05)  # Approx 0.16 delta
        short_put_strike = spot_price * (1 - 0.05)   # Approx -0.16 delta
        
        # Long strikes (wings)
        long_call_strike = short_call_strike + wing_width
        long_put_strike = short_put_strike - wing_width
        
        # Round to nearest strike interval
        if self.instrument_manager:
            interval = self.instrument_manager.get_strike_interval(underlying)
            short_call_strike = round(short_call_strike / interval) * interval
            short_put_strike = round(short_put_strike / interval) * interval
            long_call_strike = round(long_call_strike / interval) * interval
            long_put_strike = round(long_put_strike / interval) * interval
        
        # Create legs
        lot_size = self.instrument_manager.get_lot_size(underlying) if self.instrument_manager else 50
        
        legs = [
            StrategyLeg("CE", short_call_strike, expiry, "SHORT", quantity, lot_size),
            StrategyLeg("PE", short_put_strike, expiry, "SHORT", quantity, lot_size),
            StrategyLeg("CE", long_call_strike, expiry, "LONG", quantity, lot_size),
            StrategyLeg("PE", long_put_strike, expiry, "LONG", quantity, lot_size),
        ]
        
        result.legs = legs
        
        # Calculate P&L
        wing_width_value = wing_width
        credit_received = wing_width_value * 0.3 * lot_size * quantity  # Approximate
        max_risk = (wing_width_value - credit_received / (lot_size * quantity)) * lot_size * quantity
        
        result.max_profit = credit_received
        result.max_loss = max_risk
        result.breakeven_low = short_put_strike - credit_received / (lot_size * quantity)
        result.breakeven_high = short_call_strike + credit_received / (lot_size * quantity)
        result.risk_reward_ratio = credit_received / max_risk if max_risk > 0 else 0
        
        # Estimate Greeks (simplified)
        result.theta = credit_received * 0.01  # Positive theta
        result.vega = -credit_received * 0.1   # Negative vega
        result.delta = 0.0                      # Delta neutral
        result.gamma = -0.001                   # Negative gamma
        
        # Margin (rough estimate)
        result.margin_required = max_risk * 1.2
        
        result.add_note(f"Range: {short_put_strike:.0f} - {short_call_strike:.0f}")
        result.add_note(f"Credit: ₹{credit_received:,.0f}, Max Risk: ₹{max_risk:,.0f}")
        
        return result
    
    def _get_vix_regime(self, vix: float) -> str:
        """Classify VIX regime"""
        if vix < 12:
            return "LOW"
        elif vix < 18:
            return "MODERATE"
        elif vix < 25:
            return "HIGH"
        return "EXTREME"


class CalendarSpreadBuilder:
    """
    Calendar Spread Strategy Builder.
    
    Strategy: Sell near-term option + Buy longer-term option at same strike
    
    Best for:
    - Low IV environment
    - Expecting term structure steepening
    - Time decay differential
    
    Risk/Reward: Limited profit, limited risk
    """
    
    MAX_VIX = 15.0
    MIN_TERM_STRUCTURE = 2.0  # Minimum IV difference between months
    
    def __init__(self, instrument_manager=None, greeks_calculator=None):
        self.instrument_manager = instrument_manager
        self.greeks_calculator = greeks_calculator
    
    def build(
        self,
        underlying: str,
        spot_price: float,
        vix: float,
        near_iv: float,
        far_iv: float,
        strike: Optional[float] = None,
        near_expiry: Optional[date] = None,
        far_expiry: Optional[date] = None,
        option_type: str = "CE",
        quantity: int = 1
    ) -> StrategyResult:
        """
        Build a Calendar Spread strategy.
        
        Args:
            underlying: Underlying symbol
            spot_price: Current spot price
            vix: Current VIX level
            near_iv: IV for near-term option
            far_iv: IV for far-term option
            strike: Strike price (default: ATM)
            near_expiry: Near-term expiry
            far_expiry: Far-term expiry
            option_type: CE or PE
            quantity: Number of lots
            
        Returns:
            StrategyResult with complete strategy details
        """
        result = StrategyResult(
            strategy_type=StrategyType.CALENDAR_SPREAD,
            underlying=underlying,
            spot_price=spot_price,
            vix_regime=self._get_vix_regime(vix)
        )
        
        # Check VIX suitability
        if vix > self.MAX_VIX:
            result.add_note(f"VIX {vix:.1f} above maximum {self.MAX_VIX} - not suitable")
            result.conviction = 40
        else:
            result.conviction = 75
        
        # Check term structure
        term_structure = far_iv - near_iv
        if term_structure < self.MIN_TERM_STRUCTURE:
            result.add_note(f"Term structure {term_structure:.1f}% below minimum {self.MIN_TERM_STRUCTURE}%")
            result.conviction = max(40, result.conviction - 20)
        
        # Get expiries
        if near_expiry is None:
            near_expiry = date.today() + timedelta(days=7)
        if far_expiry is None:
            far_expiry = date.today() + timedelta(days=35)
        
        # Get strike
        if strike is None:
            strike = spot_price
        
        lot_size = self.instrument_manager.get_lot_size(underlying) if self.instrument_manager else 50
        
        # Create legs
        legs = [
            StrategyLeg(option_type, strike, near_expiry, "SHORT", quantity, lot_size, iv=near_iv),
            StrategyLeg(option_type, strike, far_expiry, "LONG", quantity, lot_size, iv=far_iv),
        ]
        
        result.legs = legs
        
        # Calculate P&L (simplified)
        debit_paid = (far_iv - near_iv) * spot_price * 0.01 * lot_size * quantity
        max_profit = debit_paid * 2  # Theoretical max at near expiry
        
        result.max_profit = max_profit
        result.max_loss = debit_paid
        result.breakeven_low = strike * 0.95
        result.breakeven_high = strike * 1.05
        result.risk_reward_ratio = max_profit / debit_paid if debit_paid > 0 else 0
        
        # Greeks
        result.theta = debit_paid * 0.02  # Positive theta
        result.vega = debit_paid * 0.5    # Positive vega
        result.delta = 0.0
        result.gamma = -0.0005
        
        # Margin
        result.margin_required = debit_paid * 1.5
        
        result.add_note(f"Term structure: {term_structure:.1f}%")
        result.add_note(f"Debit: ₹{debit_paid:,.0f}, Max Profit: ₹{max_profit:,.0f}")
        
        return result
    
    def _get_vix_regime(self, vix: float) -> str:
        if vix < 12:
            return "LOW"
        elif vix < 18:
            return "MODERATE"
        elif vix < 25:
            return "HIGH"
        return "EXTREME"


class BullSpreadBuilder:
    """
    Bull Spread Strategy Builder.
    
    Strategy: Buy lower strike Call + Sell higher strike Call
    
    Best for:
    - Bullish directional view
    - Limited risk alternative to long call
    - Reducing cost basis
    
    Risk/Reward: Limited profit, limited risk
    """
    
    def __init__(self, instrument_manager=None, greeks_calculator=None):
        self.instrument_manager = instrument_manager
        self.greeks_calculator = greeks_calculator
    
    def build(
        self,
        underlying: str,
        spot_price: float,
        vix: float,
        lower_strike: Optional[float] = None,
        upper_strike: Optional[float] = None,
        expiry: Optional[date] = None,
        quantity: int = 1
    ) -> StrategyResult:
        """
        Build a Bull Call Spread strategy.
        
        Args:
            underlying: Underlying symbol
            spot_price: Current spot price
            vix: Current VIX level
            lower_strike: Lower strike (default: ATM)
            upper_strike: Upper strike (default: 5% OTM)
            expiry: Expiry date
            quantity: Number of lots
            
        Returns:
            StrategyResult with complete strategy details
        """
        result = StrategyResult(
            strategy_type=StrategyType.BULL_SPREAD,
            underlying=underlying,
            spot_price=spot_price,
            vix_regime=self._get_vix_regime(vix),
            conviction=70
        )
        
        # Get expiry
        if expiry is None:
            expiry = date.today() + timedelta(days=30)
        
        # Get strikes
        if lower_strike is None:
            lower_strike = spot_price
        if upper_strike is None:
            upper_strike = spot_price * 1.05
        
        # Round strikes
        if self.instrument_manager:
            interval = self.instrument_manager.get_strike_interval(underlying)
            lower_strike = round(lower_strike / interval) * interval
            upper_strike = round(upper_strike / interval) * interval
        
        lot_size = self.instrument_manager.get_lot_size(underlying) if self.instrument_manager else 50
        
        # Create legs
        legs = [
            StrategyLeg("CE", lower_strike, expiry, "LONG", quantity, lot_size),
            StrategyLeg("CE", upper_strike, expiry, "SHORT", quantity, lot_size),
        ]
        
        result.legs = legs
        
        # Calculate P&L
        spread_width = upper_strike - lower_strike
        debit_paid = spread_width * 0.3 * lot_size * quantity  # Approximate
        max_profit = (spread_width * lot_size * quantity) - debit_paid
        
        result.max_profit = max_profit
        result.max_loss = debit_paid
        result.breakeven_low = lower_strike + debit_paid / (lot_size * quantity)
        result.breakeven_high = upper_strike
        result.risk_reward_ratio = max_profit / debit_paid if debit_paid > 0 else 0
        
        # Greeks
        result.delta = 0.3 * lot_size * quantity
        result.gamma = 0.001
        result.theta = -debit_paid * 0.01
        result.vega = debit_paid * 0.2
        
        # Margin
        result.margin_required = debit_paid + max_profit * 0.2
        
        result.add_note(f"Spread: {lower_strike:.0f}/{upper_strike:.0f}")
        result.add_note(f"Debit: ₹{debit_paid:,.0f}, Max Profit: ₹{max_profit:,.0f}")
        
        return result
    
    def _get_vix_regime(self, vix: float) -> str:
        if vix < 12:
            return "LOW"
        elif vix < 18:
            return "MODERATE"
        elif vix < 25:
            return "HIGH"
        return "EXTREME"


class BearSpreadBuilder:
    """
    Bear Spread Strategy Builder.
    
    Strategy: Buy higher strike Put + Sell lower strike Put
    
    Best for:
    - Bearish directional view
    - Limited risk alternative to long put
    - Reducing cost basis
    
    Risk/Reward: Limited profit, limited risk
    """
    
    def __init__(self, instrument_manager=None, greeks_calculator=None):
        self.instrument_manager = instrument_manager
        self.greeks_calculator = greeks_calculator
    
    def build(
        self,
        underlying: str,
        spot_price: float,
        vix: float,
        upper_strike: Optional[float] = None,
        lower_strike: Optional[float] = None,
        expiry: Optional[date] = None,
        quantity: int = 1
    ) -> StrategyResult:
        """
        Build a Bear Put Spread strategy.
        
        Args:
            underlying: Underlying symbol
            spot_price: Current spot price
            vix: Current VIX level
            upper_strike: Upper strike (default: ATM)
            lower_strike: Lower strike (default: 5% ITM)
            expiry: Expiry date
            quantity: Number of lots
            
        Returns:
            StrategyResult with complete strategy details
        """
        result = StrategyResult(
            strategy_type=StrategyType.BEAR_SPREAD,
            underlying=underlying,
            spot_price=spot_price,
            vix_regime=self._get_vix_regime(vix),
            conviction=70
        )
        
        # Get expiry
        if expiry is None:
            expiry = date.today() + timedelta(days=30)
        
        # Get strikes
        if upper_strike is None:
            upper_strike = spot_price
        if lower_strike is None:
            lower_strike = spot_price * 0.95
        
        # Round strikes
        if self.instrument_manager:
            interval = self.instrument_manager.get_strike_interval(underlying)
            upper_strike = round(upper_strike / interval) * interval
            lower_strike = round(lower_strike / interval) * interval
        
        lot_size = self.instrument_manager.get_lot_size(underlying) if self.instrument_manager else 50
        
        # Create legs
        legs = [
            StrategyLeg("PE", upper_strike, expiry, "LONG", quantity, lot_size),
            StrategyLeg("PE", lower_strike, expiry, "SHORT", quantity, lot_size),
        ]
        
        result.legs = legs
        
        # Calculate P&L
        spread_width = upper_strike - lower_strike
        debit_paid = spread_width * 0.3 * lot_size * quantity  # Approximate
        max_profit = (spread_width * lot_size * quantity) - debit_paid
        
        result.max_profit = max_profit
        result.max_loss = debit_paid
        result.breakeven_low = lower_strike
        result.breakeven_high = upper_strike - debit_paid / (lot_size * quantity)
        result.risk_reward_ratio = max_profit / debit_paid if debit_paid > 0 else 0
        
        # Greeks
        result.delta = -0.3 * lot_size * quantity
        result.gamma = 0.001
        result.theta = -debit_paid * 0.01
        result.vega = debit_paid * 0.2
        
        # Margin
        result.margin_required = debit_paid + max_profit * 0.2
        
        result.add_note(f"Spread: {lower_strike:.0f}/{upper_strike:.0f}")
        result.add_note(f"Debit: ₹{debit_paid:,.0f}, Max Profit: ₹{max_profit:,.0f}")
        
        return result
    
    def _get_vix_regime(self, vix: float) -> str:
        if vix < 12:
            return "LOW"
        elif vix < 18:
            return "MODERATE"
        elif vix < 25:
            return "HIGH"
        return "EXTREME"


class CollarBuilder:
    """
    Collar Strategy Builder.
    
    Strategy: Long Stock + Long Put + Short Call
    
    Best for:
    - Hedging existing long positions
    - Reducing cost of protection
    - Capping upside for downside protection
    
    Risk/Reward: Limited profit, limited risk
    """
    
    def __init__(self, instrument_manager=None, greeks_calculator=None):
        self.instrument_manager = instrument_manager
        self.greeks_calculator = greeks_calculator
    
    def build(
        self,
        underlying: str,
        spot_price: float,
        vix: float,
        stock_quantity: int,
        put_strike: Optional[float] = None,
        call_strike: Optional[float] = None,
        expiry: Optional[date] = None
    ) -> StrategyResult:
        """
        Build a Collar strategy.
        
        Args:
            underlying: Underlying symbol
            spot_price: Current spot price
            vix: Current VIX level
            stock_quantity: Number of shares held
            put_strike: Put strike (default: 5% OTM)
            call_strike: Call strike (default: 5% OTM)
            expiry: Expiry date
            
        Returns:
            StrategyResult with complete strategy details
        """
        result = StrategyResult(
            strategy_type=StrategyType.COLLAR,
            underlying=underlying,
            spot_price=spot_price,
            vix_regime=self._get_vix_regime(vix),
            conviction=80
        )
        
        # Get expiry
        if expiry is None:
            expiry = date.today() + timedelta(days=30)
        
        # Get strikes
        if put_strike is None:
            put_strike = spot_price * 0.95
        if call_strike is None:
            call_strike = spot_price * 1.05
        
        # Round strikes
        if self.instrument_manager:
            interval = self.instrument_manager.get_strike_interval(underlying)
            put_strike = round(put_strike / interval) * interval
            call_strike = round(call_strike / interval) * interval
        
        # Convert stock quantity to option lots
        lot_size = self.instrument_manager.get_lot_size(underlying) if self.instrument_manager else 50
        option_quantity = stock_quantity // lot_size
        
        # Create legs
        legs = [
            StrategyLeg("PE", put_strike, expiry, "LONG", option_quantity, lot_size),
            StrategyLeg("CE", call_strike, expiry, "SHORT", option_quantity, lot_size),
        ]
        
        result.legs = legs
        
        # Calculate P&L
        put_cost = (spot_price - put_strike) * 0.5 * lot_size * option_quantity
        call_credit = (call_strike - spot_price) * 0.3 * lot_size * option_quantity
        net_cost = put_cost - call_credit
        
        stock_value = spot_price * stock_quantity
        protected_value = put_strike * stock_quantity
        capped_value = call_strike * stock_quantity
        
        result.max_profit = capped_value - stock_value - net_cost
        result.max_loss = stock_value - protected_value + net_cost
        result.breakeven_low = spot_price + net_cost / stock_quantity
        result.breakeven_high = call_strike
        result.risk_reward_ratio = abs(result.max_profit / result.max_loss) if result.max_loss != 0 else 0
        
        # Greeks
        result.delta = stock_quantity - option_quantity * lot_size * 0.5
        result.gamma = -0.001
        result.theta = call_credit * 0.01
        result.vega = -put_cost * 0.1
        
        # Margin (collateral for short call)
        result.margin_required = stock_value * 0.2
        
        result.add_note(f"Protection at: ₹{put_strike:.0f}, Cap at: ₹{call_strike:.0f}")
        result.add_note(f"Net cost: ₹{net_cost:,.0f}, Protected value: ₹{protected_value:,.0f}")
        
        return result
    
    def _get_vix_regime(self, vix: float) -> str:
        if vix < 12:
            return "LOW"
        elif vix < 18:
            return "MODERATE"
        elif vix < 25:
            return "HIGH"
        return "EXTREME"


# ============================================================================
# Strategy Factory
# ============================================================================

class StrategyFactory:
    """Factory for creating strategy builders"""
    
    def __init__(self, instrument_manager=None, greeks_calculator=None):
        self.instrument_manager = instrument_manager
        self.greeks_calculator = greeks_calculator
        
        self._builders = {
            StrategyType.IRON_CONDOR: IronCondorBuilder(instrument_manager, greeks_calculator),
            StrategyType.CALENDAR_SPREAD: CalendarSpreadBuilder(instrument_manager, greeks_calculator),
            StrategyType.BULL_SPREAD: BullSpreadBuilder(instrument_manager, greeks_calculator),
            StrategyType.BEAR_SPREAD: BearSpreadBuilder(instrument_manager, greeks_calculator),
            StrategyType.COLLAR: CollarBuilder(instrument_manager, greeks_calculator),
        }
    
    def get_builder(self, strategy_type: StrategyType):
        """Get builder for a strategy type"""
        return self._builders.get(strategy_type)
    
    def recommend_strategy(
        self,
        market_bias: MarketBias,
        vix: float,
        term_structure: float = 0.0
    ) -> List[StrategyType]:
        """
        Recommend strategies based on market conditions.
        
        Args:
            market_bias: Bullish, Bearish, Neutral, or Volatile
            vix: Current VIX level
            term_structure: IV term structure difference
            
        Returns:
            List of recommended strategy types
        """
        recommendations = []
        
        if market_bias == MarketBias.NEUTRAL:
            if 18 <= vix <= 25:
                recommendations.append(StrategyType.IRON_CONDOR)
            if vix < 15 and term_structure >= 2.0:
                recommendations.append(StrategyType.CALENDAR_SPREAD)
                
        elif market_bias == MarketBias.BULLISH:
            recommendations.append(StrategyType.BULL_SPREAD)
            
        elif market_bias == MarketBias.BEARISH:
            recommendations.append(StrategyType.BEAR_SPREAD)
            
        elif market_bias == MarketBias.VOLATILE:
            if vix > 25:
                recommendations.append(StrategyType.IRON_CONDOR)
        
        # Collar is always available for hedging
        recommendations.append(StrategyType.COLLAR)
        
        return recommendations

"""
Short Executor — Converts SHORT equity signals to F&O execution.
NSE equity short-selling is intraday only; swing SHORT requires F&O.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger("rox.execution.short")


class ShortStrategy(Enum):
    """Available short execution strategies using F&O instruments."""
    BUY_ATM_PUT = "BUY_ATM_PUT"
    SELL_ATM_CALL = "SELL_ATM_CALL"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"


@dataclass
class ShortOrder:
    """A constructed short order via F&O instruments."""
    symbol: str
    strategy: ShortStrategy
    transaction_type: str      # "BUY" or "SELL"
    quantity: int
    strike: float
    option_type: str           # "CE" or "PE"
    premium: float
    max_loss: float
    max_profit: float
    stop_loss_premium: float
    target_premium: float
    lot_size: int
    lots: int
    underlying: str
    conviction: float
    regime: str


# NSE lot sizes for major stocks (as of 2024-2025)
DEFAULT_LOT_SIZES = {
    "NSE:SBIN": 1500, "NSE:ICICIBANK": 700, "NSE:HDFCBANK": 550,
    "NSE:AXISBANK": 600, "NSE:KOTAKBANK": 400, "NSE:BAJFINANCE": 125,
    "NSE:RELIANCE": 250, "NSE:TCS": 150, "NSE:INFY": 300,
    "NSE:HCLTECH": 350, "NSE:WIPRO": 1500, "NSE:TATASTEEL": 1063,
    "NSE:JSWSTEEL": 1350, "NSE:HINDALCO": 1400, "NSE:TATAMOTORS": 1125,
    "NSE:LT": 150, "NSE:MARUTI": 100, "NSE:SUNPHARMA": 700,
    "NSE:ITC": 1600, "NSE:ULTRACEMCO": 200, "NSE:TITAN": 375,
    "NSE:NIFTY": 25, "NSE:BANKNIFTY": 15,
}


class ShortExecutor:
    """
    Converts SHORT equity signals to F&O execution plans.

    Strategy selection:
    - conviction < 70% or volatile regime → BUY_ATM_PUT (defined risk)
    - conviction > 70% and range-bound    → SELL_ATM_CALL (credit)
    - conviction > 80% and bearish        → BEAR_PUT_SPREAD (reduced premium)

    Risk per SHORT: 1.5% of portfolio (same as LONG)
    """

    def __init__(self, lot_sizes: Optional[Dict[str, int]] = None):
        self.lot_sizes = lot_sizes or DEFAULT_LOT_SIZES

    def get_lot_size(self, symbol: str) -> int:
        """Get the F&O lot size for a given symbol."""
        return self.lot_sizes.get(symbol, 1)

    def get_atm_strike(self, price: float, symbol: str) -> float:
        """
        Round to nearest strike. Nifty: 50-point strikes, Stocks: 10 or 50.

        Args:
            price: Spot price of the underlying.
            symbol: Symbol string to determine strike intervals.

        Returns:
            ATM strike price rounded to the nearest valid interval.
        """
        if "NIFTY" in symbol or "SENSEX" in symbol:
            return round(price / 50) * 50
        elif price > 2000:
            return round(price / 50) * 50
        elif price > 500:
            return round(price / 20) * 20
        else:
            return round(price / 10) * 10

    def prepare_short_order(
        self,
        symbol: str,
        spot_price: float,
        conviction: float,
        regime: str,
        portfolio_capital: float,
        option_chain: Optional[Dict] = None,
    ) -> Optional[ShortOrder]:
        """
        Build a SHORT order using F&O.

        Args:
            symbol: e.g. "NSE:SBIN"
            spot_price: current underlying price
            conviction: 0-100
            regime: current regime label
            portfolio_capital: current portfolio value
            option_chain: optional dict with option data

        Returns:
            ShortOrder or None if unable to construct
        """
        lot_size = self.get_lot_size(symbol)
        atm_strike = self.get_atm_strike(spot_price, symbol)

        # Get premium (from option chain if available, else estimate)
        premium = self._estimate_premium(symbol, spot_price, atm_strike, option_chain)
        if premium <= 0:
            logger.warning(f"Cannot construct SHORT for {symbol}: premium={premium}")
            return None

        # Risk budget
        risk_amount = portfolio_capital * 0.015  # 1.5%
        max_lots = max(1, int(risk_amount / (premium * lot_size)))

        # Strategy selection
        if conviction > 80 and regime in ("BEARISH", "CAUTIOUS"):
            return self._bear_put_spread(symbol, spot_price, atm_strike,
                                          lot_size, max_lots, premium, conviction, regime)
        elif conviction > 70 and regime == "RANGE_BOUND":
            return self._sell_atm_call(symbol, spot_price, atm_strike,
                                        lot_size, max_lots, premium, conviction, regime)
        else:
            return self._buy_atm_put(symbol, spot_price, atm_strike,
                                      lot_size, max_lots, premium, conviction, regime)

    def _buy_atm_put(self, symbol, spot, strike, lot_size, lots,
                      premium, conviction, regime) -> ShortOrder:
        """BUY ATM PUT — defined risk, max loss = premium paid."""
        quantity = lots * lot_size
        total_premium = premium * quantity
        return ShortOrder(
            symbol=f"{symbol}_{int(strike)}PE",
            strategy=ShortStrategy.BUY_ATM_PUT,
            transaction_type="BUY",
            quantity=quantity,
            strike=strike,
            option_type="PE",
            premium=premium,
            max_loss=total_premium,
            max_profit=float("inf"),  # Theoretical unlimited
            stop_loss_premium=premium * 1.5,   # Exit if premium rises 50%
            target_premium=premium * 0.5,       # Exit if premium drops 50%
            lot_size=lot_size,
            lots=lots,
            underlying=symbol,
            conviction=conviction,
            regime=regime,
        )

    def _sell_atm_call(self, symbol, spot, strike, lot_size, lots,
                        premium, conviction, regime) -> ShortOrder:
        """SELL ATM CALL — credit received, margin-intensive."""
        quantity = lots * lot_size
        total_credit = premium * quantity
        return ShortOrder(
            symbol=f"{symbol}_{int(strike)}CE",
            strategy=ShortStrategy.SELL_ATM_CALL,
            transaction_type="SELL",
            quantity=quantity,
            strike=strike,
            option_type="CE",
            premium=premium,
            max_loss=float("inf"),  # Naked call — theoretical unlimited
            max_profit=total_credit,
            stop_loss_premium=premium * 2.0,   # Exit if premium doubles
            target_premium=premium * 0.2,       # Exit at 80% profit
            lot_size=lot_size,
            lots=lots,
            underlying=symbol,
            conviction=conviction,
            regime=regime,
        )

    def _bear_put_spread(self, symbol, spot, strike, lot_size, lots,
                          premium, conviction, regime) -> ShortOrder:
        """BUY ATM PUT + SELL OTM PUT — reduced premium cost."""
        otm_strike = self.get_atm_strike(spot * 0.97, symbol)  # ~3% OTM
        quantity = lots * lot_size
        # Net debit = ATM premium - OTM premium (estimate OTM at 50% of ATM)
        net_debit = (premium - premium * 0.5) * quantity
        return ShortOrder(
            symbol=f"{symbol}_{int(strike)}PE_{int(otm_strike)}PE",
            strategy=ShortStrategy.BEAR_PUT_SPREAD,
            transaction_type="BUY",  # Primary leg
            quantity=quantity,
            strike=strike,
            option_type="PE",
            premium=premium,
            max_loss=net_debit,
            max_profit=(strike - otm_strike) * quantity - net_debit,
            stop_loss_premium=net_debit * 1.5 / quantity if quantity > 0 else 0,
            target_premium=net_debit * 0.3 / quantity if quantity > 0 else 0,
            lot_size=lot_size,
            lots=lots,
            underlying=symbol,
            conviction=conviction,
            regime=regime,
        )

    def _estimate_premium(self, symbol, spot, strike, option_chain) -> float:
        """
        Get premium from option chain or estimate using simple intrinsic + time value.

        Args:
            symbol: Underlying symbol.
            spot: Spot price.
            strike: Strike price.
            option_chain: Optional dict with live option data.

        Returns:
            Estimated ATM premium as a float.
        """
        if option_chain and symbol in option_chain:
            chain = option_chain[symbol]
            puts = chain.get("puts", {})
            if strike in puts:
                return puts[strike].get("ltp", 0.0)

        # Rough estimate: 2-4% of spot for ATM options
        return spot * 0.03

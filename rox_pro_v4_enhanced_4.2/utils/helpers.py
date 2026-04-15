"""
ROX Proven Edge Engine v3.0 - Helper Functions
=============================================
Utility functions for calculations and formatting.
"""

from typing import Optional, Tuple


def format_currency(amount: float, symbol: str = "₹") -> str:
    """
    Format a number as Indian currency.
    
    Args:
        amount: The amount to format
        symbol: Currency symbol (default: ₹)
        
    Returns:
        Formatted currency string
    """
    if abs(amount) >= 10000000:  # 1 crore
        return f"{symbol}{amount/10000000:.2f} Cr"
    elif abs(amount) >= 100000:  # 1 lakh
        return f"{symbol}{amount/100000:.2f} L"
    else:
        return f"{symbol}{amount:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal as percentage.
    
    Args:
        value: The decimal value (0.15 = 15%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_risk_reward(entry: float, stop_loss: float, 
                         target: float, direction: str = "LONG") -> float:
    """
    Calculate risk-reward ratio.
    
    Args:
        entry: Entry price
        stop_loss: Stop loss price
        target: Target price
        direction: "LONG" or "SHORT"
        
    Returns:
        Risk-reward ratio
    """
    if direction.upper() == "LONG":
        risk = entry - stop_loss
        reward = target - entry
    else:
        risk = stop_loss - entry
        reward = entry - target
    
    if risk <= 0:
        return 0
    
    return reward / risk


def calculate_shares(portfolio_value: float, risk_percent: float,
                    entry_price: float, stop_loss: float) -> int:
    """
    Calculate number of shares based on risk parameters.
    
    Args:
        portfolio_value: Total portfolio value
        risk_percent: Risk percentage (e.g., 0.02 for 2%)
        entry_price: Entry price per share
        stop_loss: Stop loss price per share
        
    Returns:
        Number of shares to buy
    """
    risk_amount = portfolio_value * risk_percent
    risk_per_share = abs(entry_price - stop_loss)
    
    if risk_per_share <= 0:
        return 0
    
    return int(risk_amount / risk_per_share)


def validate_price(price: float) -> bool:
    """
    Validate that a price is reasonable.
    
    Args:
        price: Price to validate
        
    Returns:
        True if valid, False otherwise
    """
    return price is not None and price > 0


def normalize_score(score: float, min_val: float = 0, 
                   max_val: float = 100) -> float:
    """
    Normalize a score to 0-100 range.
    
    Args:
        score: The score to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
        
    Returns:
        Normalized score (0-100)
    """
    if max_val == min_val:
        return 50
    
    normalized = (score - min_val) / (max_val - min_val) * 100
    return max(0, min(100, normalized))


def calculate_atr_stop(entry_price: float, atr: float, 
                      multiplier: float = 1.5, direction: str = "LONG") -> float:
    """
    Calculate ATR-based stop loss.
    
    Args:
        entry_price: Entry price
        atr: Average True Range value
        multiplier: ATR multiplier (default 1.5)
        direction: "LONG" or "SHORT"
        
    Returns:
        Stop loss price
    """
    stop_distance = atr * multiplier
    
    if direction.upper() == "LONG":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance


def calculate_position_value(shares: int, price: float) -> float:
    """
    Calculate total position value.
    
    Args:
        shares: Number of shares
        price: Price per share
        
    Returns:
        Total position value
    """
    return shares * price


def calculate_portfolio_heat(positions: list, portfolio_value: float) -> float:
    """
    Calculate total portfolio heat (risk exposure).
    
    Args:
        positions: List of position dicts with 'risk_amount' key
        portfolio_value: Total portfolio value
        
    Returns:
        Portfolio heat as percentage
    """
    total_risk = sum(p.get('risk_amount', 0) for p in positions)
    return (total_risk / portfolio_value) * 100 if portfolio_value > 0 else 0


def check_sector_exposure(sector_positions: dict, 
                         new_position_sector: str,
                         new_position_value: float,
                         portfolio_value: float,
                         max_exposure: float = 0.25) -> Tuple[bool, float]:
    """
    Check if adding a position would exceed sector exposure limit.
    
    Args:
        sector_positions: Dict of sector -> current exposure
        new_position_sector: Sector of new position
        new_position_value: Value of new position
        portfolio_value: Total portfolio value
        max_exposure: Maximum allowed exposure (default 25%)
        
    Returns:
        Tuple of (is_allowed, new_exposure_percent)
    """
    current_exposure = sector_positions.get(new_position_sector, 0)
    new_exposure = current_exposure + new_position_value
    new_exposure_percent = new_exposure / portfolio_value
    
    return new_exposure_percent <= max_exposure, new_exposure_percent


def get_conviction_multiplier(conviction: int) -> float:
    """
    Get position size multiplier based on conviction level.
    
    Args:
        conviction: Conviction score (0-100)
        
    Returns:
        Position size multiplier
    """
    if conviction >= 85:
        return 1.0  # Full size for very high conviction
    elif conviction >= 75:
        return 0.8  # 80% size for high conviction
    elif conviction >= 65:
        return 0.6  # 60% size for medium conviction
    else:
        return 0.0  # No trade for low conviction

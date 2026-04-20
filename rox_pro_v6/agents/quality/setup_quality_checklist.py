"""
Setup Quality Checklist - Pre-trade quality filtering (Enhancement F2)
========================================================================

Pre-trade quality filtering for swing setups:
- Liquidity checks
- Spread validation
- News impact assessment
- Technical condition validation
- Correlation and position limit checks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

# Import from parent config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TradeDirection, MarketRegime, SECTOR_MAPPING


class CheckSeverity(Enum):
    """Severity level for checklist items."""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class CheckItem:
    """A single checklist item result."""
    name: str
    passed: bool
    details: str
    severity: CheckSeverity
    weight: float = 1.0  # Weight for overall score calculation

    @property
    def score_contribution(self) -> float:
        """Score contribution for this check."""
        if self.severity == CheckSeverity.PASS:
            return self.weight
        elif self.severity == CheckSeverity.WARNING:
            return self.weight * 0.5
        return 0.0


@dataclass
class ChecklistResult:
    """Complete checklist result for a trade setup."""
    setup: Any  # TradeSetup
    checks: List[CheckItem]
    quality_score: int  # 0-100
    recommendation: str  # "PROCEED", "CAUTION", "REJECT"
    rejection_reason: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_checks(self) -> int:
        return len(self.checks)


class SetupQualityChecklist:
    """
    Quality checklist for trade setups.

    Runs before displaying setups to user.
    Filters out low-quality setups based on multiple criteria.
    """

    # Thresholds
    MIN_VOLUME = 500_000
    MIN_MARKET_CAP_CR = 10_000
    MAX_SPREAD_PCT = 0.5
    MAX_RSI_LONG = 75
    MIN_RSI_SHORT = 25
    MAX_CORRELATION = 0.7
    MAX_SECTOR_EXPOSURE = 0.25
    MIN_RISK_REWARD = 1.5

    def __init__(self, config: Any = None):
        self.config = config
        self.logger = logging.getLogger("SetupQualityChecklist")

    def evaluate(
        self,
        setup: Any,
        market_data: Dict[str, Any],
        news_context: Dict[str, Any],
        other_setups: List[Any] = None,
        portfolio: Dict[str, Any] = None
    ) -> ChecklistResult:
        """
        Evaluate a trade setup against quality criteria.

        Args:
            setup: The trade setup to evaluate
            market_data: Current market data
            news_context: News context from NOCTURNAL
            other_setups: Other setups being considered
            portfolio: Current portfolio state

        Returns:
            ChecklistResult with pass/fail and quality score
        """
        checks = []
        total_weight = 0.0
        total_score = 0.0

        # 1. Liquidity check
        check = self._check_liquidity(setup, market_data)
        checks.append(check)
        total_weight += check.weight
        total_score += check.score_contribution

        # 2. Spread check
        check = self._check_spread(setup, market_data)
        checks.append(check)
        total_weight += check.weight
        total_score += check.score_contribution

        # 3. News impact check
        check = self._check_news(setup, news_context)
        checks.append(check)
        total_weight += check.weight
        total_score += check.score_contribution

        # 4. Technical conditions check
        check = self._check_technical(setup, market_data)
        checks.append(check)
        total_weight += check.weight
        total_score += check.score_contribution

        # 5. Risk-reward check
        check = self._check_risk_reward(setup)
        checks.append(check)
        total_weight += check.weight
        total_score += check.score_contribution

        # 6. Correlation check (if other setups exist)
        if other_setups:
            check = self._check_correlation(setup, other_setups)
            checks.append(check)
            total_weight += check.weight
            total_score += check.score_contribution

        # 7. Position limits check (if portfolio exists)
        if portfolio:
            check = self._check_position_limits(setup, portfolio)
            checks.append(check)
            total_weight += check.weight
            total_score += check.score_contribution

        # Calculate quality score
        quality_score = int((total_score / total_weight) * 100) if total_weight > 0 else 0

        # Determine recommendation
        failures = [c for c in checks if c.severity == CheckSeverity.FAIL]
        warnings = [c for c in checks if c.severity == CheckSeverity.WARNING]

        if failures:
            recommendation = "REJECT"
            rejection_reason = failures[0].details
        elif len(warnings) >= 2:
            recommendation = "CAUTION"
            rejection_reason = None
        elif quality_score >= 70:
            recommendation = "PROCEED"
            rejection_reason = None
        elif quality_score >= 50:
            recommendation = "CAUTION"
            rejection_reason = None
        else:
            recommendation = "REJECT"
            rejection_reason = "Quality score below threshold"

        return ChecklistResult(
            setup=setup,
            checks=checks,
            quality_score=quality_score,
            recommendation=recommendation,
            rejection_reason=rejection_reason
        )

    def _check_liquidity(self, setup: Any, market_data: Dict) -> CheckItem:
        """Check if stock has adequate liquidity."""
        stock = setup.stock if hasattr(setup, 'stock') else setup.get('stock', '')

        # Get volume data
        price_data = market_data.get('price_data', {}).get(stock, {})
        volume = price_data.get('volume', 0)
        avg_volume = price_data.get('avg_volume', volume or 1)

        # Get market cap if available
        fundamentals = market_data.get('fundamental_data', {}).get(stock, {})
        market_cap = fundamentals.get('market_cap_cr', 0)

        if volume >= self.MIN_VOLUME and avg_volume >= self.MIN_VOLUME:
            return CheckItem(
                name="Liquidity",
                passed=True,
                details=f"Volume: {volume:,}, Avg: {avg_volume:,}",
                severity=CheckSeverity.PASS,
                weight=1.5
            )
        elif volume >= self.MIN_VOLUME * 0.5:
            return CheckItem(
                name="Liquidity",
                passed=True,
                details=f"Moderate volume: {volume:,}",
                severity=CheckSeverity.WARNING,
                weight=1.5
            )
        else:
            return CheckItem(
                name="Liquidity",
                passed=False,
                details=f"Low volume: {volume:,} (min: {self.MIN_VOLUME:,})",
                severity=CheckSeverity.FAIL,
                weight=1.5
            )

    def _check_spread(self, setup: Any, market_data: Dict) -> CheckItem:
        """Check bid-ask spread is acceptable."""
        stock = setup.stock if hasattr(setup, 'stock') else setup.get('stock', '')

        price_data = market_data.get('price_data', {}).get(stock, {})
        bid = price_data.get('bid', 0)
        ask = price_data.get('ask', 0)

        if bid > 0 and ask > 0:
            spread_pct = ((ask - bid) / bid) * 100

            if spread_pct <= self.MAX_SPREAD_PCT:
                return CheckItem(
                    name="Spread",
                    passed=True,
                    details=f"Spread: {spread_pct:.2f}%",
                    severity=CheckSeverity.PASS,
                    weight=1.0
                )
            elif spread_pct <= self.MAX_SPREAD_PCT * 2:
                return CheckItem(
                    name="Spread",
                    passed=True,
                    details=f"Wide spread: {spread_pct:.2f}%",
                    severity=CheckSeverity.WARNING,
                    weight=1.0
                )
            else:
                return CheckItem(
                    name="Spread",
                    passed=False,
                    details=f"Excessive spread: {spread_pct:.2f}%",
                    severity=CheckSeverity.FAIL,
                    weight=1.0
                )

        # No spread data available - assume liquid
        return CheckItem(
            name="Spread",
            passed=True,
            details="Spread data unavailable",
            severity=CheckSeverity.WARNING,
            weight=0.5
        )

    def _check_news(self, setup: Any, news_context: Dict) -> CheckItem:
        """Check for adverse news."""
        stock = setup.stock if hasattr(setup, 'stock') else setup.get('stock', '')
        direction = setup.direction if hasattr(setup, 'direction') else setup.get('direction')
        if hasattr(direction, 'value'):
            direction = direction.value

        # Check for stock-specific negative news
        stock_news = news_context.get('stock_news', {}).get(stock, [])

        adverse_headlines = []
        for item in stock_news[:5]:
            sentiment = item.get('sentiment', 'NEUTRAL')
            if direction == 'LONG' and sentiment in ['NEGATIVE', 'BEARISH']:
                adverse_headlines.append(item.get('title', ''))
            elif direction == 'SHORT' and sentiment in ['POSITIVE', 'BULLISH']:
                adverse_headlines.append(item.get('title', ''))

        # Check trade restrictions
        restrictions = news_context.get('trade_restrictions', [])

        # Check for sector restrictions
        sector = self._get_sector(stock)
        for restriction in restrictions:
            if sector and sector.upper() in restriction.upper():
                return CheckItem(
                    name="News/Events",
                    passed=False,
                    details=f"Trade restriction: {restriction}",
                    severity=CheckSeverity.FAIL,
                    weight=2.0
                )

        if adverse_headlines:
            return CheckItem(
                name="News/Events",
                passed=True,
                details=f"Adverse news detected: {adverse_headlines[0][:50]}...",
                severity=CheckSeverity.WARNING,
                weight=2.0
            )

        return CheckItem(
            name="News/Events",
            passed=True,
            details="No adverse news",
            severity=CheckSeverity.PASS,
            weight=2.0
        )

    def _check_technical(self, setup: Any, market_data: Dict) -> CheckItem:
        """Check technical conditions (RSI, resistance, etc.)."""
        stock = setup.stock if hasattr(setup, 'stock') else setup.get('stock', '')
        direction = setup.direction if hasattr(setup, 'direction') else setup.get('direction')
        if hasattr(direction, 'value'):
            direction = direction.value

        indicators = market_data.get('indicators', {}).get(stock, {})
        rsi = indicators.get('rsi', 50)

        issues = []

        # RSI check
        if direction == 'LONG' and rsi > self.MAX_RSI_LONG:
            issues.append(f"RSI overbought: {rsi:.1f}")
        elif direction == 'SHORT' and rsi < self.MIN_RSI_SHORT:
            issues.append(f"RSI oversold: {rsi:.1f}")

        # Trend check
        sma20 = indicators.get('sma20', 0)
        sma50 = indicators.get('sma50', 0)
        close = getattr(setup, 'entry_price', 0) or setup.get('entry_price', 0)

        if sma20 > 0 and sma50 > 0:
            if direction == 'LONG' and close < sma20 < sma50:
                issues.append("Price below declining MAs")
            elif direction == 'SHORT' and close > sma20 > sma50:
                issues.append("Price above rising MAs")

        if issues:
            return CheckItem(
                name="Technical",
                passed=True,
                details="; ".join(issues),
                severity=CheckSeverity.WARNING,
                weight=1.5
            )

        return CheckItem(
            name="Technical",
            passed=True,
            details="Technical conditions favorable",
            severity=CheckSeverity.PASS,
            weight=1.5
        )

    def _check_risk_reward(self, setup: Any) -> CheckItem:
        """Check risk-reward ratio."""
        rr = getattr(setup, 'risk_reward', 0) or setup.get('risk_reward', 0)

        if rr >= self.MIN_RISK_REWARD * 1.5:
            return CheckItem(
                name="Risk-Reward",
                passed=True,
                details=f"Excellent R:R: {rr:.2f}",
                severity=CheckSeverity.PASS,
                weight=2.0
            )
        elif rr >= self.MIN_RISK_REWARD:
            return CheckItem(
                name="Risk-Reward",
                passed=True,
                details=f"Good R:R: {rr:.2f}",
                severity=CheckSeverity.PASS,
                weight=2.0
            )
        elif rr >= self.MIN_RISK_REWARD * 0.8:
            return CheckItem(
                name="Risk-Reward",
                passed=True,
                details=f"Marginal R:R: {rr:.2f}",
                severity=CheckSeverity.WARNING,
                weight=2.0
            )
        else:
            return CheckItem(
                name="Risk-Reward",
                passed=False,
                details=f"Poor R:R: {rr:.2f} (min: {self.MIN_RISK_REWARD})",
                severity=CheckSeverity.FAIL,
                weight=2.0
            )

    def _check_correlation(self, setup: Any, other_setups: List[Any]) -> CheckItem:
        """Check correlation with other setups."""
        stock = setup.stock if hasattr(setup, 'stock') else setup.get('stock', '')
        direction = setup.direction if hasattr(setup, 'direction') else setup.get('direction')
        if hasattr(direction, 'value'):
            direction = direction.value

        same_sector_count = 0
        same_direction_count = 0

        stock_sector = self._get_sector(stock)

        for other in other_setups:
            other_stock = other.stock if hasattr(other, 'stock') else other.get('stock', '')
            other_direction = other.direction if hasattr(other, 'direction') else other.get('direction')

            if hasattr(other_direction, 'value'):
                other_direction = other_direction.value

            if other_stock != stock:
                # Check same sector
                other_sector = self._get_sector(other_stock)
                if stock_sector and other_sector == stock_sector:
                    same_sector_count += 1

                # Check same direction
                if direction and other_direction == direction:
                    same_direction_count += 1

        if same_sector_count >= 3:
            return CheckItem(
                name="Correlation",
                passed=True,
                details=f"High sector concentration: {same_sector_count + 1} in {stock_sector}",
                severity=CheckSeverity.WARNING,
                weight=1.0
            )

        return CheckItem(
            name="Correlation",
            passed=True,
            details="Diversified setup selection",
            severity=CheckSeverity.PASS,
            weight=1.0
        )

    def _check_position_limits(self, setup: Any, portfolio: Dict) -> CheckItem:
        """Check position and sector limits."""
        stock = setup.stock if hasattr(setup, 'stock') else setup.get('stock', '')
        position_pct = getattr(setup, 'position_percent', 0) or setup.get('position_percent', 0)

        # Check total exposure
        deployed = portfolio.get('deployed_capital', 0)
        total = portfolio.get('total_capital', 1)
        current_exposure = deployed / total if total > 0 else 0

        if current_exposure + position_pct / 100 > 0.8:
            return CheckItem(
                name="Position Limits",
                passed=False,
                details=f"Portfolio would be overexposed ({current_exposure*100:.0f}% + {position_pct:.1f}%)",
                severity=CheckSeverity.FAIL,
                weight=1.5
            )

        # Check sector exposure
        sector = self._get_sector(stock)
        sector_exposure = portfolio.get('sector_exposure', {}).get(sector, 0)

        if sector and sector_exposure + position_pct / 100 > self.MAX_SECTOR_EXPOSURE:
            return CheckItem(
                name="Position Limits",
                passed=True,
                details=f"High {sector} exposure: {sector_exposure*100:.0f}%",
                severity=CheckSeverity.WARNING,
                weight=1.5
            )

        return CheckItem(
            name="Position Limits",
            passed=True,
            details="Within position limits",
            severity=CheckSeverity.PASS,
            weight=1.5
        )

    def _get_sector(self, stock: str) -> Optional[str]:
        """Get sector for a stock."""
        for sector, stocks in SECTOR_MAPPING.items():
            if stock in stocks:
                return sector
        return None

    def batch_evaluate(
        self,
        setups: List[Any],
        market_data: Dict,
        news_context: Dict,
        portfolio: Dict = None
    ) -> List[ChecklistResult]:
        """
        Evaluate multiple setups.

        Args:
            setups: List of trade setups
            market_data: Current market data
            news_context: News context
            portfolio: Current portfolio state

        Returns:
            List of ChecklistResults
        """
        results = []
        for i, setup in enumerate(setups):
            other_setups = setups[:i] + setups[i+1:]
            result = self.evaluate(
                setup=setup,
                market_data=market_data,
                news_context=news_context,
                other_setups=other_setups,
                portfolio=portfolio
            )
            results.append(result)

        return results

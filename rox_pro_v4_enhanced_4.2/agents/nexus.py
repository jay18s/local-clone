"""
ROX Proven Edge Engine v3.0 - NEXUS Agent
========================================
Fundamental Analysis Agent - Valuation and financial health.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .base_agent import BaseAgent, AgentVerdict, AgentReport
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradeDirection, MarketRegime


@dataclass
class QualityScore:
    """Fundamental quality score components"""
    profitability: int = 0  # 0-25
    growth: int = 0  # 0-25
    financial_health: int = 0  # 0-25
    governance: int = 0  # 0-25
    
    @property
    def total(self) -> int:
        return self.profitability + self.growth + self.financial_health + self.governance


@dataclass
class ValuationScore:
    """Valuation score components"""
    pe_vs_industry: int = 0  # 0-25
    pe_vs_history: int = 0  # 0-25
    peg_ratio: int = 0  # 0-25
    ev_ebitda: int = 0  # 0-25
    
    @property
    def total(self) -> int:
        return self.pe_vs_industry + self.pe_vs_history + self.peg_ratio + self.ev_ebitda


@dataclass
class FundamentalData:
    """Stock fundamental data"""
    pe_ratio: float = 0.0
    industry_pe: float = 0.0
    historical_pe_avg: float = 0.0
    peg_ratio: float = 0.0
    ev_ebitda: float = 0.0
    roe: float = 0.0
    roce: float = 0.0
    operating_margin: float = 0.0
    revenue_cagr_3yr: float = 0.0
    pat_cagr_3yr: float = 0.0
    debt_equity: float = 0.0
    interest_coverage: float = 0.0
    promoter_holding: float = 0.0
    promoter_pledging: float = 0.0
    dividend_years: int = 0
    governance_issues: bool = False
    fcf_positive_years: int = 0


class NexusAgent(BaseAgent):
    """
    NEXUS - Fundamental Analysis Agent
    
    Analyzes valuation, earnings quality, financial health, and corporate
    governance to assess the fundamental quality of trades.
    
    Baseline weight: 15%
    """
    
    # Nifty valuation thresholds
    NIFTY_PE_CHEAP = 18
    NIFTY_PE_FAIR_UPPER = 22
    NIFTY_PE_PREMIUM_UPPER = 26
    
    def __init__(self):
        super().__init__(
            name="NEXUS",
            domain="Fundamental Analysis",
            baseline_weight=0.15
        )
    
    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Perform comprehensive fundamental analysis.
        
        Args:
            data: Should contain:
                - 'stock': Stock symbol
                - 'fundamentals': FundamentalData or dict
                Optional:
                - 'nifty_pe': Current Nifty P/E
                - 'gsec_yield': 10-year G-Sec yield
                - 'earnings_beat': Last quarter earnings beat %
                - 'management_tone': Management commentary tone
                
        Returns:
            AgentReport with fundamental analysis verdict
        """
        stock = data.get('stock', 'UNKNOWN')

        # ── No-data guard: if fundamentals are empty, return NEUTRAL ──────
        # Only check STOCK-specific fields (not market-level pe_ratio) 
        raw_fund = data.get('fundamentals', {})
        has_real_data = any([
            raw_fund.get('roe', 0) > 0,
            raw_fund.get('roce', 0) > 0,
            raw_fund.get('revenue_cagr_3yr', 0) > 0,
        ])
        if not has_real_data:
            from .base_agent import AgentVerdict
            nifty_pe   = data.get('nifty_pe', 22.5)
            gsec_yield = data.get('gsec_yield', 7.0)
            eq_premium = (1 / nifty_pe * 100) - gsec_yield
            direction  = TradeDirection.LONG if eq_premium > -1 else TradeDirection.NEUTRAL
            conviction = max(40, min(60, 50 + eq_premium * 3))
            return AgentReport(
                agent_name=self.name,
                verdict=AgentVerdict(
                    direction=direction,
                    conviction=conviction,
                    weight=self.current_weight,
                    reason=f"No stock fundamentals available | Market PE {nifty_pe:.1f} | ERP {eq_premium:.1f}%"
                ),
                analysis_details={"f_score": 50, "no_data": True},
                key_observations=["Fundamental data unavailable — using market-level proxy"],
                metrics={"f_score": 50, "nifty_pe": nifty_pe}
            )

        # Parse fundamental data
        fund_data = self._parse_fundamental_data(raw_fund)
        
        # Index-level valuation context
        index_context = self._analyze_index_valuation(
            data.get('nifty_pe', 20),
            data.get('gsec_yield', 7.0)
        )
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(fund_data)
        
        # Calculate valuation score
        valuation_score = self._calculate_valuation_score(fund_data)
        
        # Calculate combined F-Score
        f_score = self._calculate_f_score(quality_score, valuation_score)
        
        # Analyze earnings quality
        earnings_analysis = self._analyze_earnings(
            data.get('earnings_beat', 0),
            data.get('management_tone', 'neutral')
        )
        
        # Generate verdict
        verdict = self._generate_verdict(
            fund_data, quality_score, valuation_score, f_score, index_context, regime
        )
        
        # Build key observations
        key_observations = self._generate_observations(
            fund_data, quality_score, valuation_score, f_score, index_context
        )
        
        return AgentReport(
            agent_name=self.name,
            verdict=verdict,
            analysis_details={
                "stock": stock,
                "quality_score": quality_score.total,
                "valuation_score": valuation_score.total,
                "f_score": f_score,
                "f_score_rating": self._get_f_score_rating(f_score),
                "index_valuation": index_context.get("valuation_zone", "unknown"),
                "earnings_quality": earnings_analysis.get("quality", "unknown")
            },
            key_observations=key_observations,
            metrics={
                "f_score": f_score,
                "quality_score": quality_score.total,
                "valuation_score": valuation_score.total,
                "pe_ratio": fund_data.pe_ratio,
                "roe": fund_data.roe,
                "debt_equity": fund_data.debt_equity
            },
            raw_data={
                "fundamentals": fund_data.__dict__,
                "quality_breakdown": {
                    "profitability": quality_score.profitability,
                    "growth": quality_score.growth,
                    "financial_health": quality_score.financial_health,
                    "governance": quality_score.governance
                }
            }
        )
    
    def _parse_fundamental_data(self, data: Dict) -> FundamentalData:
        """Parse fundamental data from dict"""
        if isinstance(data, FundamentalData):
            return data
        
        return FundamentalData(
            pe_ratio=data.get('pe_ratio', 0),
            industry_pe=data.get('industry_pe', 0),
            historical_pe_avg=data.get('historical_pe_avg', 0),
            peg_ratio=data.get('peg_ratio', 0),
            ev_ebitda=data.get('ev_ebitda', 0),
            roe=data.get('roe', 0),
            roce=data.get('roce', 0),
            operating_margin=data.get('operating_margin', 0),
            revenue_cagr_3yr=data.get('revenue_cagr_3yr', 0),
            pat_cagr_3yr=data.get('pat_cagr_3yr', 0),
            debt_equity=data.get('debt_equity', 0),
            interest_coverage=data.get('interest_coverage', 0),
            promoter_holding=data.get('promoter_holding', 0),
            promoter_pledging=data.get('promoter_pledging', 0),
            dividend_years=data.get('dividend_years', 0),
            governance_issues=data.get('governance_issues', False),
            fcf_positive_years=data.get('fcf_positive_years', 0)
        )
    
    def _analyze_index_valuation(self, nifty_pe: float, gsec_yield: float) -> Dict:
        """Analyze index-level valuation context"""
        context = {
            "nifty_pe": nifty_pe,
            "valuation_zone": "fair",
            "bond_comparison": "neutral"
        }
        
        # Nifty P/E zone
        if nifty_pe < self.NIFTY_PE_CHEAP:
            context["valuation_zone"] = "cheap"
        elif nifty_pe < self.NIFTY_PE_FAIR_UPPER:
            context["valuation_zone"] = "fair"
        elif nifty_pe < self.NIFTY_PE_PREMIUM_UPPER:
            context["valuation_zone"] = "premium"
        else:
            context["valuation_zone"] = "expensive"
        
        # Bond yield comparison
        if nifty_pe > 0:
            earnings_yield = 100 / nifty_pe  # Earnings yield in %
            if earnings_yield > gsec_yield:
                context["bond_comparison"] = "stocks_attractive"
            else:
                context["bond_comparison"] = "bonds_attractive"
        
        return context
    
    def _calculate_quality_score(self, data: FundamentalData) -> QualityScore:
        """Calculate quality score (0-100)"""
        score = QualityScore()
        
        # Profitability (25 points)
        if data.roe > 20:
            score.profitability = 10
        elif data.roe > 15:
            score.profitability = 7
        else:
            score.profitability = 3
        
        if data.roce > 25:
            score.profitability += 10
        elif data.roce > 15:
            score.profitability += 7
        else:
            score.profitability += 3
        
        if data.operating_margin > 0:  # Assume sector average comparison needed
            score.profitability += 5
        
        # Growth (25 points)
        if data.revenue_cagr_3yr > 15:
            score.growth = 10
        elif data.revenue_cagr_3yr > 8:
            score.growth = 6
        else:
            score.growth = 2
        
        if data.pat_cagr_3yr > 20:
            score.growth += 10
        elif data.pat_cagr_3yr > 10:
            score.growth += 6
        else:
            score.growth += 2
        
        if data.revenue_cagr_3yr > 0 and data.pat_cagr_3yr > 0:
            score.growth += 5  # Consistent growth
        
        # Financial Health (25 points)
        if data.debt_equity < 0.5:
            score.financial_health = 10
        elif data.debt_equity < 1:
            score.financial_health = 6
        else:
            score.financial_health = 0
        
        if data.interest_coverage > 5:
            score.financial_health += 8
        elif data.interest_coverage > 3:
            score.financial_health += 4
        
        if data.fcf_positive_years >= 3:
            score.financial_health += 7
        elif data.fcf_positive_years >= 2:
            score.financial_health += 4
        
        # Governance (25 points)
        if data.promoter_holding > 50:
            score.governance = 8
        elif data.promoter_holding > 30:
            score.governance = 4
        
        if data.promoter_pledging == 0:
            score.governance += 7
        elif data.promoter_pledging < 10:
            score.governance += 3
        
        if data.dividend_years >= 5:
            score.governance += 5
        elif data.dividend_years >= 3:
            score.governance += 3
        
        if not data.governance_issues:
            score.governance += 5
        
        return score
    
    def _calculate_valuation_score(self, data: FundamentalData) -> ValuationScore:
        """Calculate valuation score (0-100)"""
        score = ValuationScore()
        
        # P/E vs Industry (25 points)
        if data.industry_pe > 0 and data.pe_ratio > 0:
            pe_ratio_industry = data.pe_ratio / data.industry_pe
            if pe_ratio_industry < 0.7:
                score.pe_vs_industry = 25
            elif pe_ratio_industry < 0.9:
                score.pe_vs_industry = 15
            elif pe_ratio_industry < 1.1:
                score.pe_vs_industry = 10
        
        # P/E vs History (25 points)
        if data.historical_pe_avg > 0 and data.pe_ratio > 0:
            pe_ratio_history = data.pe_ratio / data.historical_pe_avg
            if pe_ratio_history < 0.8:
                score.pe_vs_history = 25
            elif pe_ratio_history < 0.95:
                score.pe_vs_history = 15
            elif pe_ratio_history < 1.1:
                score.pe_vs_history = 10
        
        # PEG Ratio (25 points)
        if data.peg_ratio > 0:
            if data.peg_ratio < 0.8:
                score.peg_ratio = 25
            elif data.peg_ratio < 1.0:
                score.peg_ratio = 15
            elif data.peg_ratio < 1.5:
                score.peg_ratio = 8
        
        # EV/EBITDA vs Industry (simplified)
        if data.ev_ebitda > 0:
            # Assume industry median is around 12 for Indian markets
            industry_median = 12
            if data.ev_ebitda < industry_median:
                score.ev_ebitda = 25
            elif data.ev_ebitda < industry_median * 1.2:
                score.ev_ebitda = 12
        
        return score
    
    def _calculate_f_score(self, quality: QualityScore, valuation: ValuationScore) -> int:
        """Calculate combined F-Score (0-100)"""
        return int(quality.total * 0.6 + valuation.total * 0.4)
    
    def _get_f_score_rating(self, f_score: int) -> str:
        """Get rating from F-Score"""
        if f_score > 80:
            return "Strong Buy"
        elif f_score > 60:
            return "Good"
        elif f_score > 40:
            return "Mixed"
        else:
            return "Avoid"
    
    def _analyze_earnings(self, beat_pct: float, management_tone: str) -> Dict:
        """Analyze earnings quality"""
        analysis = {
            "quality": "unknown",
            "beat_status": "in-line",
            "management_signal": "neutral"
        }
        
        if beat_pct > 5:
            analysis["beat_status"] = "beat"
            analysis["quality"] = "strong"
        elif beat_pct < -5:
            analysis["beat_status"] = "miss"
            analysis["quality"] = "weak"
        
        if management_tone == "confident":
            analysis["management_signal"] = "positive"
        elif management_tone == "cautious":
            analysis["management_signal"] = "negative"
        
        return analysis
    
    def _generate_verdict(self, fund_data: FundamentalData, quality: QualityScore,
                         valuation: ValuationScore, f_score: int,
                         index_context: Dict, regime: MarketRegime) -> AgentVerdict:
        """Generate the final verdict"""
        # Base direction from F-Score
        if f_score > 70:
            direction = TradeDirection.LONG
            conviction = 50 + (f_score - 70) * 1.5
        elif f_score < 40:
            direction = TradeDirection.SHORT  # Or NEUTRAL with warning
            conviction = 50 + (40 - f_score)
        else:
            direction = TradeDirection.NEUTRAL
            conviction = 50
        
        # Index valuation overlay
        if index_context["valuation_zone"] == "expensive":
            conviction -= 10  # Be more cautious in expensive markets
        elif index_context["valuation_zone"] == "cheap":
            conviction += 10
        
        # Regime adjustments
        if regime == MarketRegime.BEAR and f_score > 70:
            # In bear markets, good fundamentals provide floor
            conviction += 5
        elif regime == MarketRegime.BULL and f_score < 40:
            # In bull markets, weak fundamentals are risky
            conviction -= 5
        
        # Governance red flag
        if fund_data.governance_issues or fund_data.promoter_pledging > 20:
            conviction -= 15
            direction = TradeDirection.NEUTRAL
        
        # High debt warning
        if fund_data.debt_equity > 2:
            conviction -= 10
        
        # Clamp conviction
        conviction = max(0, min(100, conviction))
        
        # Generate reason
        reason = f"F-Score: {f_score} ({self._get_f_score_rating(f_score)})"
        reason += f" | Quality: {quality.total}/100 | Valuation: {valuation.total}/100"
        
        # Generate risks
        risks = []
        if f_score < 50:
            risks.append(f"Low F-Score ({f_score}) - fundamental weakness")
        if fund_data.debt_equity > 1:
            risks.append(f"High debt/equity: {fund_data.debt_equity:.2f}")
        if fund_data.promoter_pledging > 10:
            risks.append(f"Promoter pledging: {fund_data.promoter_pledging:.1f}%")
        if fund_data.governance_issues:
            risks.append("Governance concerns flagged")
        
        return AgentVerdict(
            direction=direction,
            conviction=conviction,
            weight=self.current_weight,
            reason=reason,
            risks=risks
        )
    
    def _generate_observations(self, fund_data: FundamentalData, quality: QualityScore,
                              valuation: ValuationScore, f_score: int,
                              index_context: Dict) -> List[str]:
        """Generate key observations"""
        observations = []
        
        # F-Score summary
        observations.append(f"F-Score: {f_score}/100 ({self._get_f_score_rating(f_score)})")
        
        # Quality breakdown
        observations.append(
            f"Quality: {quality.total}/100 "
            f"(Prof: {quality.profitability}, Growth: {quality.growth}, "
            f"Health: {quality.financial_health}, Gov: {quality.governance})"
        )
        
        # Valuation
        if fund_data.pe_ratio > 0:
            obs = f"P/E: {fund_data.pe_ratio:.1f}x"
            if fund_data.industry_pe > 0:
                discount = (1 - fund_data.pe_ratio / fund_data.industry_pe) * 100
                obs += f" ({discount:+.0f}% vs industry)"
            observations.append(obs)
        
        # Key metrics
        if fund_data.roe > 0:
            observations.append(f"ROE: {fund_data.roe:.1f}% | ROCE: {fund_data.roce:.1f}%")
        
        if fund_data.debt_equity > 0:
            observations.append(f"D/E: {fund_data.debt_equity:.2f}x | Interest coverage: {fund_data.interest_coverage:.1f}x")
        
        # Index context
        observations.append(f"Nifty P/E: {index_context['nifty_pe']:.1f}x ({index_context['valuation_zone']})")
        
        return observations

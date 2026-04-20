"""
ROX Market Impact Monitor
==========================
Gives the Virtual Execution Engine real-time awareness of macro,
news, technical and flow events that cause F&O premium moves.

Called every cycle from VirtualExecutionBridge.  Produces an
ImpactSnapshot that the VEB uses to:
  - Adjust MTM premium estimates when live option prices are absent
  - Attach impact context to each open trade
  - Warn when a position is exposed to an active impact event
  - Feed the self-improvement engine with regime+impact context

Impact categories tracked
─────────────────────────
  VIX_SPIKE       VIX ≥20 or VIX change > +2 in session
  GAP_RISK        GIFT Nifty gap > ±0.5%
  NEWS_BEARISH     LLM news magnitude HIGH/EXTREME with bearish bias
  NEWS_BULLISH     LLM news magnitude HIGH/EXTREME with bullish bias
  FII_SELLING     FII 5d net < -2000 Cr
  FII_BUYING      FII 5d net > +2000 Cr
  DII_SELLING     DII 5d net < -3000 Cr
  PCR_CRASH       NIFTY PCR < 0.70 (heavy put buying → fear)
  PCR_EXTREME     NIFTY PCR > 1.40 (complacency / reversal risk)
  USDINR_STRESS   USD/INR > 85 or move > ±0.5 in session
  YIELD_SPIKE     G-sec yield > 7.5%
  DMA200_BREACH   Nifty > 2% below 200 DMA
  EXPIRY_PINRISK  Within 1 day of expiry AND near max-pain ± 0.3%
  REGIME_CHANGE   Detected regime transition this cycle

Each impact event has:
  name, severity (LOW/MEDIUM/HIGH/EXTREME),
  direction (BULLISH/BEARISH/VOLATILE/NEUTRAL),
  premium_impact_pct (estimated % change in option premium),
  description
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rox.market_impact")


# ── Severity ordering ────────────────────────────────────────────────────────
SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "EXTREME": 4}


@dataclass
class ImpactEvent:
    name:                str
    severity:            str          # LOW / MEDIUM / HIGH / EXTREME
    direction:           str          # BULLISH / BEARISH / VOLATILE / NEUTRAL
    premium_impact_pct:  float        # +ve = premium rises, -ve = falls
    description:         str
    raw_value:           Optional[float] = None   # the metric that triggered this


@dataclass
class ImpactSnapshot:
    timestamp:           str
    cycle:               int
    events:              List[ImpactEvent] = field(default_factory=list)
    overall_direction:   str = "NEUTRAL"    # net of all events
    overall_severity:    str = "LOW"
    vix_premium_mult:    float = 1.0        # multiply raw premium by this
    should_widen_sl:     bool = False       # ≥1 HIGH event present
    should_reduce_size:  bool = False       # ≥1 EXTREME or ≥2 HIGH events
    active_restrictions: List[str] = field(default_factory=list)
    summary:             str = ""

    @property
    def is_elevated(self) -> bool:
        return self.overall_severity in ("HIGH", "EXTREME")

    @property
    def event_names(self) -> List[str]:
        return [e.name for e in self.events]

    def get_premium_adjustment(self, base_premium: float) -> float:
        """Return adjusted premium after applying impact multiplier."""
        total_pct = sum(e.premium_impact_pct for e in self.events)
        capped = max(-60.0, min(total_pct, 120.0))   # cap at ±120%
        return base_premium * (1 + capped / 100.0)

    def to_log_line(self) -> str:
        if not self.events:
            return "[IMPACT] No elevated market impact events"
        ev_str = " | ".join(f"{e.name}({e.severity})" for e in self.events)
        return (
            f"[IMPACT] {self.overall_severity}/{self.overall_direction} | "
            f"{ev_str} | prem_mult={self.vix_premium_mult:.2f} | "
            f"widen_sl={self.should_widen_sl}"
        )


# ─────────────────────────────────────────────────────────────────────────────

class MarketImpactMonitor:
    """
    Scans market_data every cycle and returns an ImpactSnapshot.
    Stateless across cycles — no persistence needed.
    """

    def scan(
        self,
        market_data: Dict,
        regime: str,
        regime_changed: bool = False,
        open_trades: Optional[List] = None,      # VirtualTrade list
    ) -> ImpactSnapshot:
        """
        Full impact scan.  Returns ImpactSnapshot with all active events.
        """
        events: List[ImpactEvent] = []

        # ── 1. VIX ────────────────────────────────────────────────────
        vix     = float(market_data.get("india_vix", market_data.get("vix", 15.0)))
        vix_prev = float(market_data.get("vix_prev_close", vix))
        vix_chg  = vix - vix_prev

        if vix >= 25:
            events.append(ImpactEvent(
                "VIX_EXTREME", "EXTREME", "VOLATILE", +60.0,
                f"VIX={vix:.1f} ≥25 — panic zone, premiums exploding",
                vix,
            ))
        elif vix >= 20:
            events.append(ImpactEvent(
                "VIX_ELEVATED", "HIGH", "VOLATILE", +30.0,
                f"VIX={vix:.1f} ≥20 — elevated fear, straddle premiums up",
                vix,
            ))
        elif vix >= 18:
            events.append(ImpactEvent(
                "VIX_RISING", "MEDIUM", "VOLATILE", +15.0,
                f"VIX={vix:.1f} approaching elevated zone",
                vix,
            ))

        if vix_chg >= 2.0:
            events.append(ImpactEvent(
                "VIX_SPIKE_INTRADAY", "HIGH", "VOLATILE", +25.0,
                f"VIX spiked +{vix_chg:.1f} this session — immediate premium expansion",
                vix_chg,
            ))
        elif vix_chg <= -2.0:
            events.append(ImpactEvent(
                "VIX_CRUSH", "MEDIUM", "NEUTRAL", -20.0,
                f"VIX dropped {vix_chg:.1f} — premium contraction risk for long options",
                vix_chg,
            ))

        # ── 2. GIFT Nifty gap ─────────────────────────────────────────
        gift_gap = float(market_data.get("gift_nifty_gap_pct", 0.0))
        if abs(gift_gap) >= 1.0:
            direction = "BEARISH" if gift_gap < 0 else "BULLISH"
            events.append(ImpactEvent(
                "GAP_LARGE", "HIGH", direction,
                +20.0 if abs(gift_gap) >= 1.5 else +10.0,
                f"GIFT Nifty gap {gift_gap:+.2f}% — gap-{direction.lower()} open expected",
                gift_gap,
            ))
        elif abs(gift_gap) >= 0.5:
            direction = "BEARISH" if gift_gap < 0 else "BULLISH"
            events.append(ImpactEvent(
                "GAP_MODERATE", "MEDIUM", direction, +8.0,
                f"GIFT Nifty gap {gift_gap:+.2f}%",
                gift_gap,
            ))

        # ── 3. FII / DII flows ────────────────────────────────────────
        fii_5d = float(market_data.get("fii_cash_5day", 0.0))
        dii_5d = float(market_data.get("dii_cash_5day", 0.0))

        if fii_5d <= -3000:
            events.append(ImpactEvent(
                "FII_HEAVY_SELLING", "HIGH", "BEARISH", +15.0,
                f"FII 5d net={fii_5d:,.0f} Cr — sustained institutional outflows",
                fii_5d,
            ))
        elif fii_5d <= -1500:
            events.append(ImpactEvent(
                "FII_SELLING", "MEDIUM", "BEARISH", +8.0,
                f"FII 5d net={fii_5d:,.0f} Cr",
                fii_5d,
            ))
        elif fii_5d >= 3000:
            events.append(ImpactEvent(
                "FII_HEAVY_BUYING", "HIGH", "BULLISH", +12.0,
                f"FII 5d net=+{fii_5d:,.0f} Cr — strong institutional inflows",
                fii_5d,
            ))

        if dii_5d <= -4000:
            events.append(ImpactEvent(
                "DII_HEAVY_SELLING", "HIGH", "BEARISH", +12.0,
                f"DII 5d net={dii_5d:,.0f} Cr — domestic institutions offloading",
                dii_5d,
            ))

        # ── 4. PCR ────────────────────────────────────────────────────
        nifty_pcr = float(market_data.get("nifty_pcr",
                    market_data.get("pcr", 1.0)))
        if nifty_pcr < 0.65:
            events.append(ImpactEvent(
                "PCR_FEAR", "HIGH", "BEARISH", +20.0,
                f"NIFTY PCR={nifty_pcr:.2f} — extreme put buying, fear spike",
                nifty_pcr,
            ))
        elif nifty_pcr < 0.80:
            events.append(ImpactEvent(
                "PCR_BEARISH", "MEDIUM", "BEARISH", +8.0,
                f"NIFTY PCR={nifty_pcr:.2f} — put-heavy, bearish bias",
                nifty_pcr,
            ))
        elif nifty_pcr > 1.50:
            events.append(ImpactEvent(
                "PCR_COMPLACENCY", "MEDIUM", "VOLATILE", +5.0,
                f"NIFTY PCR={nifty_pcr:.2f} — extreme call writing, reversal risk",
                nifty_pcr,
            ))

        # ── 5. USD/INR stress ─────────────────────────────────────────
        # FIX-USDINR-EXTREME: threshold raised to 93 (futures trade ~0.3-1.5 INR above spot;
        # old threshold of 90 was routinely tripped by the futures fallback even at normal
        # spot rates ~84-85).  Also suppress EXTREME when source is identified as futures,
        # since futures price is structurally above spot and not a true macro signal.
        usdinr = float(market_data.get("usdinr", market_data.get("usd_inr", 0.0)))
        usdinr_source = str(market_data.get("usd_inr_source", "UNKNOWN")).upper()
        _is_futures_source = usdinr_source in ("FUTURES", "FX_FUTURES", "FYERS_FX")
        if usdinr >= 93 and not _is_futures_source:
            events.append(ImpactEvent(
                "USDINR_EXTREME", "HIGH", "VOLATILE", +20.0,
                f"USD/INR={usdinr:.2f} (src={usdinr_source}) — severe rupee stress, macro volatility elevated",
                usdinr,
            ))
        elif usdinr >= 93 and _is_futures_source:
            # Log but do NOT fire EXTREME — futures price includes forward premium
            logger.warning(
                f"USD/INR futures={usdinr:.2f} (src={usdinr_source}) exceeds 93 but source is "
                f"futures — suppressing USDINR_EXTREME. Verify spot rate independently."
            )
        elif usdinr >= 86:
            events.append(ImpactEvent(
                "USDINR_ELEVATED", "MEDIUM", "VOLATILE", +10.0,
                f"USD/INR={usdinr:.2f} — rupee under pressure",
                usdinr,
            ))

        # ── 6. G-sec yield ────────────────────────────────────────────
        yield_val = float(market_data.get("gsec_yield", 0.0))
        if yield_val >= 7.5:
            events.append(ImpactEvent(
                "YIELD_SPIKE", "HIGH", "BEARISH", +12.0,
                f"G-sec yield={yield_val:.2f}% ≥7.5% — tightening pressure on equities",
                yield_val,
            ))

        # ── 7. 200 DMA breach ─────────────────────────────────────────
        nifty_price  = float(market_data.get("nifty_price", 0.0))
        nifty_200dma = float(market_data.get("nifty_200dma", 0.0))
        if nifty_price > 0 and nifty_200dma > 0:
            dma_pct = (nifty_price - nifty_200dma) / nifty_200dma * 100
            if dma_pct <= -3.0:
                events.append(ImpactEvent(
                    "DMA200_BREACH_SEVERE", "HIGH", "BEARISH", +15.0,
                    f"Nifty {dma_pct:.1f}% below 200DMA — structural downtrend",
                    dma_pct,
                ))
            elif dma_pct <= -1.5:
                events.append(ImpactEvent(
                    "DMA200_BREACH", "MEDIUM", "BEARISH", +8.0,
                    f"Nifty {dma_pct:.1f}% below 200DMA",
                    dma_pct,
                ))

        # ── 8. News impact (from LLM news analyzer) ───────────────────
        news_impact = market_data.get("news_impact")
        if news_impact is not None:
            magnitude   = ""
            mkt_impact  = {}
            restrictions = []
            try:
                mkt_impact   = getattr(news_impact, "overall_market_impact", {}) or {}
                magnitude    = str(mkt_impact.get("magnitude", "")).upper()
                restrictions = list(getattr(news_impact, "trade_restrictions", []) or [])
            except Exception:
                pass

            if magnitude in ("HIGH", "EXTREME"):
                bias      = str(mkt_impact.get("bias", "VOLATILE")).upper()
                direction = "BEARISH" if "BEAR" in bias else \
                            "BULLISH" if "BULL" in bias else "VOLATILE"
                sev       = "EXTREME" if magnitude == "EXTREME" else "HIGH"
                prem_adj  = +35.0 if sev == "EXTREME" else +20.0
                events.append(ImpactEvent(
                    f"NEWS_{direction}", sev, direction, prem_adj,
                    f"LLM news: {magnitude} impact | {bias}",
                ))
            elif magnitude == "MODERATE":
                events.append(ImpactEvent(
                    "NEWS_MODERATE", "MEDIUM", "VOLATILE", +8.0,
                    "LLM news: MODERATE impact",
                ))

        # ── 9. Restrictions from news ─────────────────────────────────
        news_restrictions = list(market_data.get("news_restrictions", []) or [])

        # ── 10. Regime transition ─────────────────────────────────────
        if regime_changed:
            bear_regimes = {"MILD_BEAR", "BEAR", "CORRECTION"}
            direction    = "BEARISH" if regime in bear_regimes else \
                           "BULLISH" if regime in {"MILD_BULL", "BULL"} else "VOLATILE"
            events.append(ImpactEvent(
                "REGIME_TRANSITION", "HIGH", direction, +15.0,
                f"Regime transition detected → {regime}",
            ))

        # ── 11. Expiry pin risk (check open trades) ───────────────────
        if open_trades:
            today = date.today()
            for t in open_trades:
                try:
                    exp    = date.fromisoformat(t.expiry_date)
                    dte    = (exp - today).days
                    spot   = t.current_spot
                    strike = t.legs[0].get("strike", 0) if t.legs else 0
                    if dte <= 1 and strike > 0 and spot > 0:
                        proximity = abs(spot - strike) / strike * 100
                        if proximity <= 0.3:
                            events.append(ImpactEvent(
                                "EXPIRY_PIN_RISK", "HIGH", "VOLATILE", -25.0,
                                f"{t.underlying} {t.strategy} at ATM within {dte}d of expiry "
                                f"(spot={spot:.0f} strike={strike:.0f} proximity={proximity:.2f}%)",
                                proximity,
                            ))
                except Exception:
                    pass

        # ── Compute overall ───────────────────────────────────────────
        if not events:
            snap = ImpactSnapshot(
                timestamp=datetime.now().isoformat(),
                cycle=0,
                events=[],
                overall_direction="NEUTRAL",
                overall_severity="LOW",
                vix_premium_mult=1.0,
                should_widen_sl=False,
                should_reduce_size=False,
                active_restrictions=news_restrictions,
                summary="No elevated market impact events",
            )
            return snap

        max_sev     = max(events, key=lambda e: SEVERITY_RANK.get(e.severity, 0))
        bearish_n   = sum(1 for e in events if e.direction == "BEARISH")
        bullish_n   = sum(1 for e in events if e.direction == "BULLISH")
        volatile_n  = sum(1 for e in events if e.direction == "VOLATILE")
        overall_dir = ("BEARISH"  if bearish_n  > bullish_n + volatile_n else
                       "BULLISH"  if bullish_n  > bearish_n + volatile_n else
                       "VOLATILE")

        high_count    = sum(1 for e in events if SEVERITY_RANK.get(e.severity,0) >= 3)
        extreme_count = sum(1 for e in events if e.severity == "EXTREME")
        widen_sl      = high_count >= 1
        reduce_size   = extreme_count >= 1 or high_count >= 2

        # Premium multiplier: VIX-based
        vix_mult = 1.0
        if vix >= 25: vix_mult = 1.50
        elif vix >= 22: vix_mult = 1.30
        elif vix >= 20: vix_mult = 1.20
        elif vix >= 18: vix_mult = 1.10

        event_names = [e.name for e in events[:5]]  # top 5 for summary
        summary_parts = [f"{e.name}: {e.description}" for e in
                         sorted(events, key=lambda x: SEVERITY_RANK.get(x.severity,0), reverse=True)[:3]]

        snap = ImpactSnapshot(
            timestamp=datetime.now().isoformat(),
            cycle=0,
            events=events,
            overall_direction=overall_dir,
            overall_severity=max_sev.severity,
            vix_premium_mult=vix_mult,
            should_widen_sl=widen_sl,
            should_reduce_size=reduce_size,
            active_restrictions=news_restrictions,
            summary=" | ".join(summary_parts),
        )

        logger.info(snap.to_log_line())
        return snap

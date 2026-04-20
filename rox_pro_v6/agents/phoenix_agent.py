"""
PHOENIX — Pre-Momentum Recovery Radar
======================================
ROX Engine v4.3 — New agent that detects market capitulation and the early
signs of recovery BEFORE price confirms.  Designed specifically for war/macro
shock selloffs where the engine was previously catching the bounce 200-300 pts
late.

Philosophy
----------
Price is the LAST thing to recover. Smart money enters during:
  1. VIX exhaustion (fear index falls while price still near lows)
  2. Institutional put-sellers returning (PCR rising from oversold)
  3. FII selling decelerating (the selling pace slows before it reverses)
  4. Capitulation volume (panic dump on 2x volume, then silence)
  5. 200 DMA defence (key level holding, buyers absorbing supply)

PHOENIX watches all 10 signals, scores 0–100, and feeds back into
the coordinator's regime gate to lower the conviction bar before the
crowd catches the move.

Architecture
------------
- Extends BaseAgent but is NOT a consensus voter.
  It runs as an independent observer after _run_all_agents().
- Maintains rolling cross-cycle state (PCR history, VIX-price divergence,
  GIFT gap streak, capitulation flags) — persisted via the coordinator
  instance across 60-second live cycles.
- Output: PhoenixOutput dataclass stored on DailyTradingPlan.
- Effect: Adjusts CONSOLIDATION/MILD_BEAR regime gate threshold downward
  when pre-momentum score is meaningful (≥ 46).

Signal Battery (10 signals, total 100 pts)
------------------------------------------
S1  VIX-Price Divergence      10 pts  Price falls, VIX flat/falls → fear exhausted
S2  PCR Recovery Trend        10 pts  PCR rising from <0.7 over 3+ days
S3  FII Selling Deceleration  10 pts  Daily FII selling less bad than 3-day avg
S4  DII Absorption Dominance   8 pts  DII net buys > |FII net sells| on recent days
S5  Capitulation Volume        10 pts  High-vol red candle + subsequent quiet
S6  200 DMA Defence            10 pts  Price bouncing near/off 200 DMA support
S7  GIFT Gap Trend Reversal     8 pts  First positive GIFT gap after negative streak
S8  Multi-Index PCR Convergence 8 pts  3+ indices show PCR > 0.85 simultaneously
S9  IV Compression              8 pts  IV/VIX falling while price recovers or holds
S10 Sector Breadth Rotation     8 pts  4+ sectors turning positive after selloff
Total                          100 pts

Score Tiers
-----------
0–25   DORMANT        No signal. Normal operation.
26–45  MONITOR        First flickers. Log, don't act.
46–65  EARLY_WARNING  Meaningful signal. Conviction gate: 65→62.
66–80  PRE_MOMENTUM   Strong signal. Conviction gate: 62→58. PROCEED hint.
81–100 IMMINENT       All cylinders. Conviction gate: 58→55. Aggressive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple

from .base_agent import BaseAgent, AgentVerdict, AgentReport
from config import TradeDirection, MarketRegime

logger = logging.getLogger("rox.phoenix")


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass (stored on DailyTradingPlan.phoenix_analysis)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhoenixSignal:
    """Result of one individual signal check."""
    name: str
    score: float        # actual points awarded (0 → max)
    max_score: float    # maximum possible
    fired: bool         # True when signal meaningfully active
    detail: str         # human-readable explanation
    confidence: float   # 0.0–1.0 internal confidence in this signal

@dataclass
class PhoenixOutput:
    """Full PHOENIX analysis output."""
    timestamp: datetime
    phoenix_score: float          # 0–100
    tier: str                     # DORMANT / MONITOR / EARLY_WARNING / PRE_MOMENTUM / IMMINENT
    signals: List[PhoenixSignal]
    conviction_gate_override: Optional[int]   # None or adjusted threshold (e.g. 62, 58, 55)
    recovery_probability: float   # 0.0–1.0 model estimate of recovery within 5 sessions
    key_observations: List[str]
    cautions: List[str]
    recommended_action: str       # Free-text guidance for the report
    days_since_bottom_signal: int # How many sessions since score first crossed 46

    @property
    def is_active(self) -> bool:
        return self.phoenix_score >= 46

    @property
    def tier_icon(self) -> str:
        return {
            "DORMANT":       "💤",
            "MONITOR":       "👁",
            "EARLY_WARNING": "⚡",
            "PRE_MOMENTUM":  "🔥",
            "IMMINENT":      "🚀",
        }.get(self.tier, "❓")


# ─────────────────────────────────────────────────────────────────────────────
# PHOENIX Agent
# ─────────────────────────────────────────────────────────────────────────────

class PhoenixAgent(BaseAgent):
    """
    Pre-Momentum Recovery Radar.

    Runs as a non-voting observer after the consensus panel.
    Does NOT affect weighted_votes or net_score.
    Feeds back to coordinator via DailyTradingPlan.phoenix_analysis.
    """

    # ── Class-level rolling state (survives across 60-second cycles) ──────────
    # These are stored on the instance, not class, because LeadCoordinator
    # holds a single PhoenixAgent instance for the lifetime of the session.

    def __init__(self):
        super().__init__(
            name="PHOENIX",
            domain="pre_momentum_recovery",
            baseline_weight=0.0,   # Non-voter — weight has no effect on consensus
        )
        # Rolling cross-cycle state
        self._pcr_history:      List[Tuple[str, float]] = []  # (date_str, nifty_pcr)
        self._vix_price_log:    List[Tuple[str, float, float]] = []  # (date, vix, nifty_close)
        self._gift_gap_history: List[float] = []   # last 7 gift_nifty_gap_pct values
        self._cap_vol_flag:     bool  = False      # capitulation candle seen
        self._cap_vol_date:     str   = ""         # date of capitulation candle
        self._first_active_date: Optional[str] = None  # date score first crossed 46

        # FIX-PHOENIX-WARMUP: Track how many market sessions have been observed.
        # S1 (VIX-Price Divergence) and S2 (PCR Recovery) need 3+ sessions of
        # rolling history before they can fire.  On session 1 PHOENIX will always
        # score ~8/100 = DORMANT.  We suppress scorecard-visible reporting for the
        # first WARMUP_SESSIONS calls so the examiner doesn't fire a weight-cut
        # on a non-voter over a structurally unavoidable low score.
        self._sessions_run: int = 0
        self._WARMUP_SESSIONS: int = 5  # days before score is meaningful

        # Reference data (set on first call)
        self._prev_score: float = 0.0
        self.last_output: Optional[PhoenixOutput] = None
        logger.info("PHOENIX agent initialised — pre-momentum recovery radar active")

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Run all 10 signals and return an AgentReport.
        The verdict direction/conviction is set NEUTRAL because PHOENIX is an
        observer — it does not vote in the consensus.  The real output is in
        analysis_details["phoenix_output"].
        """
        output = self._run_full_analysis(data, regime)
        self.last_output = output

        # FIX-PHOENIX-WARMUP: Increment session counter and flag warm-up state.
        self._sessions_run += 1
        _warming_up = self._sessions_run <= self._WARMUP_SESSIONS
        if _warming_up:
            logger.info(
                f"PHOENIX warm-up: session {self._sessions_run}/{self._WARMUP_SESSIONS} — "
                f"score={output.phoenix_score:.0f}/100 not yet meaningful (history accumulating). "
                f"Scorecard penalties suppressed."
            )

        # Update first-active timestamp
        today = date.today().isoformat()
        if output.is_active and self._first_active_date is None:
            self._first_active_date = today
        elif not output.is_active:
            self._first_active_date = None

        return AgentReport(
            agent_name="PHOENIX",
            verdict=AgentVerdict(
                direction=TradeDirection.NEUTRAL,
                conviction=0.0,
                weight=0.0,
                reason=f"PHOENIX score={output.phoenix_score:.0f}/100 | tier={output.tier}",
            ),
            analysis_details={"phoenix_output": output},
            key_observations=output.key_observations[:5],
            metrics={
                "phoenix_score":           output.phoenix_score,
                "recovery_probability":    output.recovery_probability,
                "conviction_gate_override": float(output.conviction_gate_override or 0),
                "signals_fired":           float(sum(1 for s in output.signals if s.fired)),
                "warming_up":              float(_warming_up),      # 1.0 during first 5 sessions
                "sessions_run":            float(self._sessions_run),
            },
        )

    # ── Rolling state update helpers ──────────────────────────────────────────

    def _update_pcr_history(self, today: str, nifty_pcr: float):
        """Keep the last 10 unique trading-day PCR readings."""
        if not self._pcr_history or self._pcr_history[-1][0] != today:
            self._pcr_history.append((today, nifty_pcr))
        else:
            self._pcr_history[-1] = (today, nifty_pcr)   # update intraday
        self._pcr_history = self._pcr_history[-10:]

    def _update_vix_price_log(self, today: str, vix: float, close: float):
        if not self._vix_price_log or self._vix_price_log[-1][0] != today:
            self._vix_price_log.append((today, vix, close))
        else:
            self._vix_price_log[-1] = (today, vix, close)
        self._vix_price_log = self._vix_price_log[-20:]

    def _update_gift_gap(self, today: str, gap_pct: float):
        if gap_pct != 0.0:   # 0.0 = unavailable, skip
            self._gift_gap_history.append(gap_pct)
            self._gift_gap_history = self._gift_gap_history[-10:]

    # ── Signal implementations ────────────────────────────────────────────────

    def _s1_vix_price_divergence(self) -> PhoenixSignal:
        """
        S1: VIX-Price Divergence (10 pts)
        Price making lower lows while VIX is flat or falling =
        smart money is NOT getting more scared → panic exhausted.
        Requires at least 3 data points in the log.
        """
        MAX = 10.0
        if len(self._vix_price_log) < 3:
            return PhoenixSignal("VIX-Price Divergence", 0, MAX, False,
                                 "Insufficient history (<3 sessions)", 0.0)

        recent = self._vix_price_log[-3:]
        price_chg = (recent[-1][2] - recent[0][2]) / max(recent[0][2], 1) * 100  # %
        vix_chg   = recent[-1][1] - recent[0][1]   # absolute VIX points

        # Divergence fires when price fell and VIX didn't rise commensurately
        # Strong: price down ≥1.5% AND VIX down
        # Moderate: price down ≥1% AND VIX up less than 0.5 pt
        if price_chg < -1.5 and vix_chg <= 0:
            score = min(MAX, abs(price_chg) * 2.5)
            fired = True
            conf  = 0.85
            detail = (f"Price {price_chg:.1f}% | VIX {vix_chg:+.1f} pts — "
                      f"smart money not panicking despite price weakness")
        elif price_chg < -1.0 and vix_chg < 0.5:
            score = min(MAX * 0.6, abs(price_chg) * 2.0)
            fired = True
            conf  = 0.60
            detail = (f"Price {price_chg:.1f}% | VIX only {vix_chg:+.1f} pts — "
                      f"fear not amplifying (early signal)")
        elif price_chg >= 0 and vix_chg < -0.5:
            # Price recovering AND VIX falling = confirmation
            score = MIN = min(MAX * 0.8, 8.0)
            fired = True
            conf  = 0.90
            detail = (f"Price recovering {price_chg:+.1f}% with VIX dropping "
                      f"{vix_chg:.1f} pts — recovery underway")
            score = MIN
        else:
            score = 0.0
            fired = False
            conf  = 0.0
            detail = f"No divergence: price {price_chg:+.1f}% | VIX {vix_chg:+.1f}"

        return PhoenixSignal("VIX-Price Divergence", round(score, 1), MAX,
                             fired, detail, conf)

    def _s2_pcr_recovery_trend(self) -> PhoenixSignal:
        """
        S2: PCR Recovery Trend (10 pts)
        PCR rising from below 0.7 over 3+ sessions = institutional put-sellers
        returning to the market, implying confidence that downside is limited.
        """
        MAX = 10.0
        if len(self._pcr_history) < 3:
            return PhoenixSignal("PCR Recovery", 0, MAX, False,
                                 "Insufficient PCR history", 0.0)

        values = [v for _, v in self._pcr_history[-5:]]
        current = values[-1]
        low     = min(values)
        trend   = current - values[0]       # overall direction over window

        # Recovery requires: started oversold (<0.7), now rising
        if low < 0.70 and trend > 0.08 and current > low:
            recovery_pct = (current - low) / max(low, 0.01)
            score = min(MAX, recovery_pct * 40)
            fired = True
            conf  = 0.80
            detail = (f"PCR recovering: low={low:.2f} → current={current:.2f} "
                      f"(+{trend:.2f} over {len(values)} sessions) — "
                      f"put sellers returning, fear dissipating")
        elif low < 0.80 and trend > 0.05:
            score = min(MAX * 0.5, 5.0)
            fired = True
            conf  = 0.55
            detail = (f"PCR slight recovery: {values[0]:.2f} → {current:.2f} "
                      f"— early signs of stabilisation")
        else:
            score = 0.0
            fired = False
            conf  = 0.0
            detail = (f"PCR not recovering: current={current:.2f} trend={trend:+.2f}")

        return PhoenixSignal("PCR Recovery Trend", round(score, 1), MAX,
                             fired, detail, conf)

    def _s3_fii_deceleration(self, flow: Dict) -> PhoenixSignal:
        """
        S3: FII Selling Deceleration (10 pts)
        The magnitude of daily FII selling is shrinking vs the 3-day average.
        FII selling always decelerates BEFORE it reverses — this is the tell.
        """
        MAX = 10.0
        fii_daily = flow.get("fii_cash_daily", 0.0)
        fii_3day  = flow.get("fii_cash_3day",  0.0)
        fii_5day  = flow.get("fii_cash_5day",  0.0)

        # Only relevant when FII is net selling
        if fii_5day >= 0:
            return PhoenixSignal("FII Deceleration", 0, MAX, False,
                                 "FII is net buying — no deceleration signal needed", 0.0)

        avg_daily_3d = fii_3day / 3.0 if fii_3day else fii_daily
        avg_daily_5d = fii_5day / 5.0 if fii_5day else fii_daily

        # Deceleration: today's selling < 3-day average selling pace
        if avg_daily_3d < 0 and fii_daily > avg_daily_3d:
            decel_ratio = abs(fii_daily - avg_daily_3d) / max(abs(avg_daily_3d), 1)
            score = min(MAX, decel_ratio * 15)
            fired = True
            conf  = 0.75
            detail = (f"FII daily={fii_daily:+,.0f}Cr vs 3d avg={avg_daily_3d:+,.0f}Cr — "
                      f"selling pace decelerating by {decel_ratio*100:.0f}%")
        elif fii_daily > 0 and fii_5day < 0:
            # FII actually turned net buyer today after selling streak
            score = MAX
            fired = True
            conf  = 0.95
            detail = (f"FII TURNED BUYER: +{fii_daily:,.0f}Cr today vs "
                      f"5d selling of {fii_5day:,.0f}Cr — potential inflection")
        else:
            score = 0.0
            fired = False
            conf  = 0.0
            detail = f"FII selling steady: daily={fii_daily:+,.0f}Cr avg5d={avg_daily_5d:+,.0f}Cr"

        return PhoenixSignal("FII Deceleration", round(score, 1), MAX,
                             fired, detail, conf)

    def _s4_dii_absorption(self, flow: Dict) -> PhoenixSignal:
        """
        S4: DII Absorption Dominance (8 pts)
        When DII buying exceeds FII selling — domestic institutions are calling
        the bottom.  The ratio determines score strength.
        """
        MAX = 8.0
        fii_1  = flow.get("fii_cash_daily", 0.0)
        dii_1  = flow.get("dii_cash_daily", 0.0)
        fii_5  = flow.get("fii_cash_5day",  0.0)
        dii_5  = flow.get("dii_cash_5day",  0.0)

        # Daily crossover: DII bought more than FII sold
        if dii_1 > 0 and dii_1 > abs(fii_1):
            ratio = dii_1 / max(abs(fii_1), 1)
            score = min(MAX, ratio * 3)
            fired = True
            conf  = 0.85
            detail = (f"DII absorbed all FII selling: DII={dii_1:+,.0f}Cr / "
                      f"FII={fii_1:+,.0f}Cr (ratio={ratio:.1f}x)")
        elif dii_5 > 0 and dii_5 > abs(fii_5):
            # 5-day net positive flow
            score = min(MAX * 0.75, 6.0)
            fired = True
            conf  = 0.70
            detail = (f"5-day DII dominance: DII 5d={dii_5:+,.0f}Cr vs FII={fii_5:+,.0f}Cr")
        elif dii_1 > 0 and fii_1 < 0 and dii_1 > abs(fii_1) * 0.7:
            # DII absorbing 70%+ of FII selling
            score = MAX * 0.5
            fired = True
            conf  = 0.55
            detail = (f"DII absorbing {dii_1/max(abs(fii_1),1)*100:.0f}% of FII selling")
        else:
            score = 0.0
            fired = False
            conf  = 0.0
            detail = f"DII not dominant: DII={dii_1:+,.0f}Cr FII={fii_1:+,.0f}Cr"

        return PhoenixSignal("DII Absorption", round(score, 1), MAX,
                             fired, detail, conf)

    def _s5_capitulation_volume(self, ohlcv: List[Dict]) -> PhoenixSignal:
        """
        S5: Capitulation Volume (10 pts)
        A single panic candle with 2x+ normal volume, followed by lower-volume
        sessions = forced sellers exhausted.  The silence after the panic is
        as important as the panic itself.
        """
        MAX = 10.0
        if len(ohlcv) < 5:
            return PhoenixSignal("Capitulation Volume", 0, MAX, False,
                                 "Insufficient OHLCV history", 0.0)

        closes  = [c["close"]  for c in ohlcv]
        volumes = [c["volume"] for c in ohlcv]
        recent  = ohlcv[-10:]  # look in last 10 sessions

        if not volumes or max(volumes) == 0:
            return PhoenixSignal("Capitulation Volume", 0, MAX, False,
                                 "No volume data available", 0.0)

        avg_vol = sum(volumes[-20:]) / max(len(volumes[-20:]), 1)

        # Find the highest-volume red candle in the last 10 sessions
        cap_idx  = None
        cap_ratio = 0.0
        for i, c in enumerate(recent):
            if c["close"] < c.get("open", c["close"]):  # red candle
                ratio = c["volume"] / max(avg_vol, 1)
                if ratio > cap_ratio:
                    cap_ratio = ratio
                    cap_idx = i

        if cap_idx is None or cap_ratio < 1.5:
            # Check if we previously flagged a capitulation candle
            if not self._cap_vol_flag:
                return PhoenixSignal("Capitulation Volume", 0, MAX, False,
                                     f"No capitulation candle (max vol ratio={cap_ratio:.1f}x)", 0.0)

        if cap_idx is not None and cap_ratio >= 1.5:
            self._cap_vol_flag = True
            self._cap_vol_date = recent[cap_idx].get("date", "")

        # If capitulation candle found, check for volume drying up after
        if self._cap_vol_flag and cap_idx is not None:
            sessions_after = recent[cap_idx+1:]
            if sessions_after:
                post_vol_ratio = [s["volume"] / max(avg_vol, 1) for s in sessions_after]
                silence = all(r < 0.9 for r in post_vol_ratio)
                if cap_ratio >= 2.0 and silence:
                    score = MAX
                    conf  = 0.90
                    detail = (f"Capitulation: {cap_ratio:.1f}x vol red candle "
                              f"+ {len(sessions_after)} quiet session(s) after — "
                              f"sellers exhausted")
                elif cap_ratio >= 1.5:
                    score = min(MAX * 0.7, 7.0)
                    conf  = 0.65
                    detail = (f"High-vol red candle {cap_ratio:.1f}x avg — "
                              f"possible capitulation (waiting for quiet)")
                else:
                    score = MAX * 0.4
                    conf  = 0.40
                    detail = "Prior capitulation candle on record"
            else:
                # Capitulation was today
                score = min(MAX * 0.6, 6.0)
                conf  = 0.60
                detail = f"Possible capitulation candle today: {cap_ratio:.1f}x vol"
            return PhoenixSignal("Capitulation Volume", round(score, 1), MAX,
                                 score > 0, detail, conf)

        return PhoenixSignal("Capitulation Volume", 0, MAX, False,
                             "No capitulation pattern", 0.0)

    def _s6_200dma_defence(self, nifty_price: float, nifty_200dma: float,
                           ohlcv: List[Dict]) -> PhoenixSignal:
        """
        S6: 200 DMA Defence (10 pts)
        The 200 DMA is the single most-watched support level in Indian markets.
        When price tests it and holds, institutions step in.  The closer to the
        200 DMA from above, the more significant the support signal.
        """
        MAX = 10.0
        if nifty_200dma <= 0:
            return PhoenixSignal("200DMA Defence", 0, MAX, False,
                                 "200 DMA not available", 0.0)

        dist_pct = (nifty_price - nifty_200dma) / nifty_200dma * 100

        # Price below 200 DMA but recovering
        if -1.0 <= dist_pct <= 0:
            score = 9.0
            fired = True
            conf  = 0.90
            detail = (f"Price {dist_pct:.4f}% below 200 DMA ({nifty_200dma:.0f}) — "
                      f"testing critical support, buyers holding the line")
        elif 0 < dist_pct <= 1.0:
            score = 8.0
            fired = True
            conf  = 0.85
            detail = (f"Price just {dist_pct:.4f}% above 200 DMA — "
                      f"bounced off key support, defensive floor confirmed")
        elif -2.5 <= dist_pct < -1.0:
            # Slightly below — monitor for bounce
            score = 6.0
            fired = True
            conf  = 0.65
            detail = (f"Price {dist_pct:.4f}% below 200 DMA — "
                      f"below support but bounce zone, watch for recovery candle")
        elif 1.0 < dist_pct <= 2.5:
            # Above but close — still relevant if recently bounced from it
            if len(ohlcv) >= 5:
                recent_lows = [c["low"] for c in ohlcv[-5:]]
                if min(recent_lows) <= nifty_200dma * 1.005:
                    score = 7.0
                    fired = True
                    conf  = 0.75
                    detail = (f"Recent low touched 200 DMA ({nifty_200dma:.0f}) "
                              f"and bounced — support defended")
                else:
                    score = 3.0
                    fired = False
                    conf  = 0.35
                    detail = f"Price {dist_pct:.3f}% above 200 DMA — not directly relevant"
            else:
                score = 2.0
                fired = False
                conf  = 0.25
                detail = f"Price {dist_pct:.3f}% above 200 DMA"
        else:
            score = 0.0
            fired = False
            conf  = 0.0
            detail = f"Price {dist_pct:.3f}% from 200 DMA — not a support/recovery signal"

        return PhoenixSignal("200DMA Defence", round(score, 1), MAX,
                             fired, detail, conf)

    def _s7_gift_gap_reversal(self) -> PhoenixSignal:
        """
        S7: GIFT Nifty Gap Trend Reversal (8 pts)
        After consecutive negative pre-market gaps (war risk priced in),
        the first positive GIFT gap signals global sentiment turning.
        Overnight US/Asia have stopped reacting negatively to the trigger.
        """
        MAX = 8.0
        if len(self._gift_gap_history) < 3:
            return PhoenixSignal("GIFT Gap Reversal", 0, MAX, False,
                                 "Insufficient GIFT gap history", 0.0)

        recent = self._gift_gap_history[-5:]
        current = recent[-1]
        prev    = recent[:-1]

        neg_streak = sum(1 for g in prev if g < -0.2)
        any_prev_positive = any(g > 0.2 for g in prev)

        if neg_streak >= 2 and current > 0.3 and not any_prev_positive:
            # The reversal: negative streak broken by positive gap
            score = min(MAX, current * 6 + neg_streak * 1.5)
            fired = True
            conf  = 0.85
            detail = (f"GIFT gap reversed: {neg_streak} negative days → "
                      f"today +{current:.2f}% — global sentiment shift, "
                      f"overnight supply absorbed")
        elif neg_streak >= 2 and abs(current) < 0.2:
            # Gap going neutral after negatives = deceleration
            score = MAX * 0.4
            fired = True
            conf  = 0.50
            detail = (f"{neg_streak} negative GIFT gaps → neutral gap today — "
                      f"overnight selling slowing")
        elif current > 0.5:
            score = min(MAX * 0.6, 5.0)
            fired = True
            conf  = 0.60
            detail = f"Strong positive GIFT gap: +{current:.2f}% — Asia/US supportive"
        else:
            score = 0.0
            fired = False
            conf  = 0.0
            detail = (f"No gap reversal signal: recent={[f'{g:.2f}' for g in recent]}")

        return PhoenixSignal("GIFT Gap Reversal", round(score, 1), MAX,
                             fired, detail, conf)

    def _s8_multi_index_pcr(self, chains: Dict) -> PhoenixSignal:
        """
        S8: Multi-Index PCR Convergence (8 pts)
        When NIFTY, BANKNIFTY and FINNIFTY all show PCR > 0.85 simultaneously,
        it's a broad-market signal of put-sellers returning — not just one index.
        """
        MAX = 8.0
        if not chains:
            return PhoenixSignal("Multi-Index PCR", 0, MAX, False,
                                 "No chain data available", 0.0)

        target_indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]
        healthy = []
        oversold_recovering = []

        for idx in target_indices:
            ch = chains.get(idx, {})
            pcr = ch.get("pcr", 0.0)
            if pcr >= 0.85:
                healthy.append(f"{idx}={pcr:.2f}")
            elif pcr >= 0.70:
                oversold_recovering.append(f"{idx}={pcr:.2f}")

        n_healthy = len(healthy)

        if n_healthy >= 3:
            score = MAX
            fired = True
            conf  = 0.88
            detail = (f"Broad PCR recovery — {n_healthy} indices healthy: "
                      f"{', '.join(healthy)} — institutional confidence wide")
        elif n_healthy == 2 and len(oversold_recovering) >= 1:
            score = MAX * 0.65
            fired = True
            conf  = 0.65
            detail = (f"2 indices with healthy PCR ({', '.join(healthy)}), "
                      f"1 recovering ({', '.join(oversold_recovering)})")
        elif n_healthy >= 1:
            score = MAX * 0.35
            fired = False
            conf  = 0.35
            detail = f"Only {n_healthy} index healthy: {', '.join(healthy)}"
        else:
            score = 0.0
            fired = False
            conf  = 0.0
            detail = f"No index with PCR ≥ 0.85. Recovering: {', '.join(oversold_recovering) or 'none'}"

        return PhoenixSignal("Multi-Index PCR", round(score, 1), MAX,
                             fired, detail, conf)

    def _s9_iv_compression(self, vix: float, iv_rank: float,
                           nifty_price: float, nifty_200dma: float) -> PhoenixSignal:
        """
        S9: IV Compression (8 pts)
        VIX / IV falling while price is at or recovering from support =
        option premium sellers are back.  Smart money sells puts only when
        they believe downside risk is limited.
        Low IV at low prices = market not afraid at the low = bottoming signal.
        """
        MAX = 8.0
        if len(self._vix_price_log) < 3:
            return PhoenixSignal("IV Compression", 0, MAX, False,
                                 "Insufficient VIX history", 0.0)

        recent_vix = [v for _, v, _ in self._vix_price_log[-5:]]
        vix_trend  = recent_vix[-1] - recent_vix[0]   # negative = falling
        price_near_support = nifty_200dma > 0 and abs(nifty_price - nifty_200dma) / nifty_200dma < 0.025

        # Low IV rank + falling VIX + price at support = premium sellers confident
        if vix < 16 and vix_trend < -0.5 and price_near_support:
            score = MAX
            fired = True
            conf  = 0.88
            detail = (f"VIX={vix:.1f} (low) falling {vix_trend:.1f} pts while price "
                      f"holds near 200 DMA — premium sellers back, no fear at lows")
        elif vix_trend < -1.0 and iv_rank < 40:
            score = MAX * 0.75
            fired = True
            conf  = 0.75
            detail = (f"VIX falling sharply ({vix_trend:.1f} pts), IV rank={iv_rank:.0f} — "
                      f"fear dissipating")
        elif vix_trend < 0 and vix < 18:
            score = MAX * 0.5
            fired = True
            conf  = 0.55
            detail = (f"VIX easing: {recent_vix[0]:.1f} → {vix:.1f} "
                      f"(trend {vix_trend:.1f} pts) — modest fear reduction")
        else:
            score = 0.0
            fired = False
            conf  = 0.0
            detail = f"VIX={vix:.1f} trend={vix_trend:+.1f} — no compression signal"

        return PhoenixSignal("IV Compression", round(score, 1), MAX,
                             fired, detail, conf)

    def _s10_sector_breadth(self, sector_perf: Dict[str, float],
                            regime: MarketRegime) -> PhoenixSignal:
        """
        S10: Sector Breadth Rotation (8 pts)
        After a selloff, sectors start recovering at different speeds.
        When 4+ sectors show positive 1-day performance, it indicates
        broad participation — not a dead-cat bounce in one sector.
        Defensive + cyclical sectors turning together is most bullish.
        """
        MAX = 8.0
        if not sector_perf:
            return PhoenixSignal("Sector Breadth", 0, MAX, False,
                                 "No sector performance data", 0.0)

        positive_sectors = [(s, v) for s, v in sector_perf.items() if v > 0]
        negative_sectors = [(s, v) for s, v in sector_perf.items() if v < 0]

        # Weight: defensive (pharma, FMCG) + cyclical (metals, energy) together
        _defensive = {"Pharma", "FMCG", "Healthcare"}
        _cyclical  = {"Metals", "Energy", "Auto", "Infrastructure", "Realty"}
        def_positive = [s for s, _ in positive_sectors if s in _defensive]
        cyc_positive = [s for s, _ in positive_sectors if s in _cyclical]

        n_positive = len(positive_sectors)
        total      = len(sector_perf)

        if n_positive >= 5 and (def_positive or cyc_positive):
            score = MAX
            fired = True
            conf  = 0.88
            detail = (f"{n_positive}/{total} sectors positive | "
                      f"Defensive: {def_positive or 'none'} | "
                      f"Cyclical: {cyc_positive or 'none'} — broad recovery")
        elif n_positive >= 4:
            score = MAX * 0.75
            fired = True
            conf  = 0.70
            detail = (f"{n_positive}/{total} sectors positive — "
                      f"broad participation, not just defensive rotation")
        elif n_positive >= 3 and regime in (MarketRegime.MILD_BEAR,
                                            MarketRegime.CONSOLIDATION):
            score = MAX * 0.5
            fired = True
            conf  = 0.55
            detail = (f"{n_positive}/{total} sectors positive in "
                      f"{regime.value} regime — early breadth recovery")
        else:
            score = 0.0
            fired = False
            conf  = 0.0
            detail = f"Only {n_positive}/{total} sectors positive — no breadth"

        return PhoenixSignal("Sector Breadth", round(score, 1), MAX,
                             fired, detail, conf)

    # ── Main analysis orchestrator ────────────────────────────────────────────

    def _run_full_analysis(self, data: Dict[str, Any],
                           regime: MarketRegime) -> PhoenixOutput:
        """
        Run all 10 signals, aggregate score, determine tier and gate override.
        """
        today = date.today().isoformat()

        # ── Extract inputs from market_data ──────────────────────────────────
        nifty_price   = data.get("nifty_price",   0.0)
        nifty_200dma  = data.get("nifty_200dma",  0.0)
        india_vix     = data.get("india_vix",     15.0)
        flow_data     = data.get("flow_data",     {})
        derivatives   = data.get("derivatives_data", {})
        chains        = data.get("index_option_chains", {})
        sector_perf   = data.get("sector_performance", {})
        ohlcv         = data.get("ohlcv_history",  {}).get("NIFTY50", [])
        gift_gap      = data.get("gift_nifty_gap_pct",  0.0)
        nifty_intra   = data.get("nifty_intraday",      {})
        nifty_close   = nifty_price  # live price or close
        nifty_pcr     = derivatives.get("pcr", 1.0)
        iv_rank       = derivatives.get("iv_rank", 50)

        # ── Update rolling state ─────────────────────────────────────────────
        self._update_pcr_history(today, nifty_pcr)
        self._update_vix_price_log(today, india_vix, nifty_close)
        self._update_gift_gap(today, gift_gap)

        # ── Run all 10 signals ───────────────────────────────────────────────
        s1  = self._s1_vix_price_divergence()
        s2  = self._s2_pcr_recovery_trend()
        s3  = self._s3_fii_deceleration(flow_data)
        s4  = self._s4_dii_absorption(flow_data)
        s5  = self._s5_capitulation_volume(ohlcv)
        s6  = self._s6_200dma_defence(nifty_price, nifty_200dma, ohlcv)
        s7  = self._s7_gift_gap_reversal()
        s8  = self._s8_multi_index_pcr(chains)
        s9  = self._s9_iv_compression(india_vix, iv_rank, nifty_price, nifty_200dma)
        s10 = self._s10_sector_breadth(sector_perf, regime)

        signals = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
        total_score = sum(s.score for s in signals)

        # ── FIX-PHOENIX-06 (rev 2): Top-4 Key Signals composite override ─────
        # S6 (200DMA Defence), S8 (Multi-Index PCR), S10 (Sector Breadth), and
        # S4 (DII Absorption) are the four highest-weight leading indicators.
        # Original used s.fired which requires a signal to reach its FULL threshold.
        # Revised to use s.score > 0: a partial score on any of these four is still
        # a meaningful market signal — e.g. S10 with 3/10 sectors positive in
        # CONSOLIDATION scores 4/8 but fired=False because it's below the full-fire
        # threshold. That still warrants attention.
        # Guard: total_score must be >= 18 to prevent spurious upgrades on noise.
        _top4_all_contributing = (
            s6.score > 0    # 200 DMA Defence — any price near 200DMA
            and s8.score > 0    # Multi-Index PCR — any broad PCR health
            and s10.score > 0   # Sector Breadth — any positive sector rotation
            and s4.score > 0    # DII Absorption — any institutional buying
        )
        _top4_all_firing = _top4_all_contributing  # alias kept for observation notes
        _top4_upgrade_applied = False
        if _top4_all_contributing and 18 <= total_score < 46:
            total_score_display = total_score  # preserve raw score for logging
            tier = "EARLY_WARNING"
            _top4_upgrade_applied = True
        else:
            # ── Standard tier classification ──────────────────────────────────
            if total_score >= 81:
                tier = "IMMINENT"
            elif total_score >= 66:
                tier = "PRE_MOMENTUM"
            elif total_score >= 46:
                tier = "EARLY_WARNING"
            elif total_score >= 26:
                tier = "MONITOR"
            else:
                tier = "DORMANT"

        # ── Conviction gate override ──────────────────────────────────────────
        # Only applies in CONSOLIDATION / MILD_BEAR / BEAR / CORRECTION
        _bearish_ambiguous = {
            MarketRegime.CONSOLIDATION, MarketRegime.MILD_BEAR,
            MarketRegime.BEAR, MarketRegime.CORRECTION
        }
        if regime in _bearish_ambiguous:
            if total_score >= 81:
                gate_override = 55
            elif total_score >= 66:
                gate_override = 58
            elif total_score >= 46 or _top4_upgrade_applied:
                # Top-4 composite upgrade also earns the EARLY_WARNING gate
                gate_override = 62
            else:
                gate_override = None
        else:
            gate_override = None

        # ── Recovery probability model ────────────────────────────────────────
        # Weighted average of signal confidences for fired signals, adjusted by tier
        fired_signals = [s for s in signals if s.fired]
        if fired_signals:
            raw_prob = sum(s.confidence * s.score for s in fired_signals) / max(total_score, 1)
            tier_mult = {"DORMANT": 0.2, "MONITOR": 0.4, "EARLY_WARNING": 0.55,
                         "PRE_MOMENTUM": 0.72, "IMMINENT": 0.88}.get(tier, 0.3)
            recovery_prob = min(0.92, raw_prob * tier_mult * 1.5)
        else:
            recovery_prob = 0.10

        # ── Key observations ──────────────────────────────────────────────────
        obs: List[str] = []
        top_signals = sorted(fired_signals, key=lambda s: s.score, reverse=True)[:4]
        for sig in top_signals:
            obs.append(f"[{sig.name}] {sig.detail}")

        # Regime-specific notes
        dist_pct = (nifty_price - nifty_200dma) / max(nifty_200dma, 1) * 100
        if abs(dist_pct) < 2:
            obs.append(f"Nifty {dist_pct:+.4f}% from 200 DMA — critical inflection zone")
        if india_vix < 15 and total_score > 40:
            obs.append(f"VIX={india_vix:.1f} low despite selloff — smart money not scared")
        if _top4_upgrade_applied:
            obs.append(
                f"⚡ TOP-4 KEY SIGNALS ALL FIRING (200DMA + MultiPCR + SectorBreadth + DII) — "
                f"structural inflection confirmed; tier upgraded MONITOR → EARLY_WARNING"
            )

        # ── Cautions ─────────────────────────────────────────────────────────
        cautions: List[str] = []
        unfired = [s for s in signals if not s.fired and s.max_score >= 8]
        # FIX-PHOENIX-06: suppress "most signals not firing" caution when Top-4 override is active
        # — the four highest-weight signals are more predictive than the full 10-signal count.
        if len(unfired) >= 5 and total_score < 46 and not _top4_upgrade_applied:
            cautions.append("Most signals not yet firing — premature to act")
        if s3.score == 0 and total_score > 45:
            cautions.append("FII selling not decelerating — recovery may be delayed")
        if s5.score == 0 and total_score > 50:
            cautions.append("No capitulation volume confirmed — may not be the bottom")
        if regime == MarketRegime.BEAR and total_score < 70:
            cautions.append("BEAR regime — high confidence required before entry")

        # ── Recommended action ────────────────────────────────────────────────
        fired_count = len(fired_signals)
        if tier == "IMMINENT":
            action = (f"⚡ PHOENIX IMMINENT ({total_score:.0f}/100, {fired_count} signals) — "
                      f"Pre-momentum signals very strong. Actively watch for breakout entries. "
                      f"Conviction gate lowered to {gate_override}. "
                      f"Recovery probability: {recovery_prob*100:.0f}%.")
        elif tier == "PRE_MOMENTUM":
            action = (f"🔥 PHOENIX PRE-MOMENTUM ({total_score:.0f}/100, {fired_count} signals) — "
                      f"Smart money positioning for recovery. Lower conviction threshold to {gate_override}. "
                      f"Look for LONG setups near 200 DMA support. "
                      f"Recovery probability: {recovery_prob*100:.0f}%.")
        elif tier == "EARLY_WARNING":
            upgrade_note = " [TOP-4 composite upgrade applied]" if _top4_upgrade_applied else ""
            action = (f"⚡ PHOENIX EARLY WARNING ({total_score:.0f}/100, {fired_count} signals){upgrade_note} — "
                      f"First recovery signals firing. Reduce conviction gate to {gate_override}. "
                      f"Monitor closely but don't front-run. "
                      f"Recovery probability: {recovery_prob*100:.0f}%.")
        elif tier == "MONITOR":
            action = (f"👁 PHOENIX MONITORING ({total_score:.0f}/100, {fired_count} signals) — "
                      f"Faint signals. No action yet. Watch VIX trend and PCR recovery.")
        else:
            action = f"💤 PHOENIX DORMANT ({total_score:.0f}/100) — No pre-momentum signals."

        # ── Days since first active ───────────────────────────────────────────
        if self._first_active_date:
            try:
                d0 = date.fromisoformat(self._first_active_date)
                days_since = (date.today() - d0).days
            except Exception:
                days_since = 0
        else:
            days_since = 0

        return PhoenixOutput(
            timestamp=datetime.now(),
            phoenix_score=round(total_score, 1),
            tier=tier,
            signals=signals,
            conviction_gate_override=gate_override,
            recovery_probability=round(recovery_prob, 3),
            key_observations=obs,
            cautions=cautions,
            recommended_action=action,
            days_since_bottom_signal=days_since,
        )

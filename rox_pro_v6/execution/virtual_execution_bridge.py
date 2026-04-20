"""
ROX Virtual Execution Bridge v2.0
===================================
v2.0 additions:
  - SHADOW tracking: WATCH-ONLY → paper-tracked, no capital used
  - Holiday/weekend self-review: forward risk + theta bleed calc
  - Conservatism audit: were WAIT decisions correct?
  - Rich report with full market context every cycle
"""
from __future__ import annotations
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from execution.virtual_broker import VirtualBroker, get_virtual_broker, OrderStatus
from execution.self_improvement import SelfImprovementEngine, get_self_improvement_engine
from execution.market_impact_monitor import MarketImpactMonitor, ImpactSnapshot

logger = logging.getLogger("rox.virtual_execution")


class VirtualExecutionBridge:
    def __init__(self, initial_capital: float = 1_000_000.0):
        self._broker: VirtualBroker = get_virtual_broker(initial_capital)
        self._sie: SelfImprovementEngine = get_self_improvement_engine()
        self._cycle_count = 0
        self._impact_monitor = MarketImpactMonitor()
        self._last_regime: str = ""
        logger.info(f"[VEB] VirtualExecutionBridge ready | capital=₹{initial_capital:,.0f}")

    def process_cycle(
        self,
        fno_suggestions: Any,
        market_data: Dict,
        regime: str,
        regime_confidence: float,
        agent_consensus: str,
        cycle_number: int,
        iv_regime: str = "MODERATE",
        is_market_holiday: bool = False,
        force_execute: bool = False,
    ) -> Dict:
        self._cycle_count = cycle_number
        report = {
            "cycle": cycle_number,
            "timestamp": datetime.now().isoformat(),
            "is_holiday": is_market_holiday,
            "regime": regime,
            "regime_confidence": regime_confidence,
            "iv_regime": iv_regime,
            "new_live_executions": [],
            "new_shadow_executions": [],
            "exits": [],
            "open_live_positions": [],
            "open_shadow_positions": [],
            "portfolio": {},
            "shadow_conservatism": {},
            "self_improvement": [],
            "improvement_signals": [],
            "holiday_review": {},
            "errors": [],
        }

        # 0. Real-time market impact scan
        regime_changed = (regime != self._last_regime and self._last_regime != "")
        self._last_regime = regime
        open_all = self._broker.get_live_trades(OrderStatus.OPEN.value) +                    self._broker.get_shadow_trades(OrderStatus.OPEN.value)
        try:
            impact: ImpactSnapshot = self._impact_monitor.scan(
                market_data=market_data,
                regime=regime,
                regime_changed=regime_changed,
                open_trades=open_all,
            )
            report["impact"] = {
                "overall_severity":   impact.overall_severity,
                "overall_direction":  impact.overall_direction,
                "events":             [(e.name, e.severity, e.direction, e.premium_impact_pct, e.description)
                                       for e in impact.events],
                "vix_premium_mult":   impact.vix_premium_mult,
                "should_widen_sl":    impact.should_widen_sl,
                "should_reduce_size": impact.should_reduce_size,
                "active_restrictions":impact.active_restrictions,
                "summary":            impact.summary,
            }
        except Exception as e:
            impact = ImpactSnapshot(timestamp="", cycle=0, summary="scan error")
            report["errors"].append(f"impact scan: {e}")

        # 1. MTM — use impact-adjusted premium mult if option prices unavailable
        price_updates = self._extract_prices(market_data)
        _theta_decay = 1.0 if is_market_holiday else 0.0
        try:
            # Pass VIX premium multiplier so theta-only MTM reflects vol expansion
            self._broker.mark_to_market(
                price_updates,
                theta_decay_factor=_theta_decay,
                vix_premium_mult=impact.vix_premium_mult,
            )
        except Exception as e:
            report["errors"].append(f"MTM: {e}")

        # 2. Auto-exits
        try:
            exits = self._broker.check_exits()
            for ex in exits:
                report["exits"].append({
                    "trade_id": ex.get("trade_id"),
                    "underlying": ex.get("underlying"),
                    "strategy": ex.get("strategy"),
                    "reason": ex.get("exit_reason", "UNKNOWN"),
                    "pnl": ex.get("realised_pnl", 0),
                    "correct": ex.get("prediction_correct"),
                    "verdict": ex.get("verdict"),
                })
        except Exception as e:
            report["errors"].append(f"exits: {e}")

        # Store impact for use in _execute_suggestion
        self._last_impact_report = report.get("impact", {})

        # 3. Process suggestions (LIVE + SHADOW)
        if fno_suggestions is not None:
            suggestions = getattr(fno_suggestions, "suggestions", []) or []
            watch_only  = getattr(fno_suggestions, "_watch_only", False)

            # Deduplication: build set of already-open (underlying+strategy+expiry)
            # so we don't create duplicate shadow trades across multiple runs
            _open_all = (self._broker.get_live_trades("OPEN") +
                         self._broker.get_shadow_trades("OPEN"))
            _already_open = {
                (t.underlying, t.strategy, t.expiry_date)
                for t in _open_all
            }

            # Read examiner size multiplier (REDUCE_SIZE = 0.5x)
            _examiner_size_mult = float(
                getattr(fno_suggestions, "_examiner_size_mult", 1.0)
            )
            _examiner_verdict = str(
                getattr(fno_suggestions, "_examiner_verdict", "PROCEED")
            )

            for sug in suggestions:
                is_watch  = watch_only or not getattr(sug, "proceed", True)
                exec_mode = "SHADOW" if (is_watch and not force_execute) else "LIVE"

                # Skip if same position already open from a previous cycle/run
                _key = (
                    getattr(sug, "index", "?"),
                    str(getattr(sug, "strategy", "?")).upper(),
                    str(getattr(sug, "expiry_str", None) or getattr(sug, "expiry", "")),
                )
                if _key in _already_open:
                    logger.info(
                        f"[VEB] DEDUP SKIP — {_key[0]} {_key[1]} exp={_key[2]} "
                        f"already {'open/shadow' if exec_mode=='SHADOW' else 'open live'}"
                    )
                    continue

                result = self._execute_suggestion(
                    sug, market_data, regime, regime_confidence,
                    agent_consensus, cycle_number, iv_regime, mode=exec_mode,
                    examiner_size_mult=_examiner_size_mult,
                )
                if result:
                    key = "new_shadow_executions" if exec_mode == "SHADOW" else "new_live_executions"
                    report[key].append(result)
                    # Add to dedup set so subsequent suggestions in same cycle don't double-up
                    _already_open.add(_key)

        # 4. Portfolio snapshot
        try:
            pf = self._broker.portfolio_state()
            open_live   = self._broker.get_live_trades(OrderStatus.OPEN.value)
            open_shadow = self._broker.get_shadow_trades(OrderStatus.OPEN.value)
            report["portfolio"] = {
                "available_capital":    pf.get("available_capital", 0),
                "deployed_capital":     pf.get("deployed_capital", 0),
                "total_unrealised_pnl": pf.get("total_unrealised_pnl", 0),
                "total_realised_pnl":   pf.get("total_realised_pnl", 0),
                "win_rate":             pf.get("win_rate", 0),
                "total_trades":         pf.get("total_trades", 0),
                "open_positions":       len(open_live),
                "shadow_trades":        pf.get("shadow_trades", 0),
                "shadow_realised_pnl":  pf.get("shadow_realised_pnl", 0.0),
            }
            report["open_live_positions"]   = [self._trade_summary(t) for t in open_live]
            report["open_shadow_positions"] = [self._trade_summary(t) for t in open_shadow]
            report["shadow_conservatism"]   = self._broker.shadow_conservatism_report()
        except Exception as e:
            report["errors"].append(f"portfolio: {e}")

        # 5. Holiday self-review
        if is_market_holiday:
            report["holiday_review"] = self._holiday_review(market_data, regime)

        # 6. Self-improvement
        try:
            closed_live = (self._broker.get_live_trades("CLOSED") +
                           self._broker.get_live_trades("EXPIRED"))
            params = self._sie.maybe_calibrate(len(closed_live), self._broker.PERF_FILE)
            if params:
                report["self_improvement"] = params.get("calibration_notes", [])[-8:]
        except Exception as e:
            report["errors"].append(f"self-improve: {e}")

        try:
            report["improvement_signals"] = self._sie.get_improvement_signals()
        except Exception:
            pass

        return report

    # ------------------------------------------------------------------ #
    #  Holiday Review                                                      #
    # ------------------------------------------------------------------ #

    def _holiday_review(self, market_data: Dict, regime: str) -> Dict:
        open_live   = self._broker.get_live_trades(OrderStatus.OPEN.value)
        open_shadow = self._broker.get_shadow_trades(OrderStatus.OPEN.value)
        all_open    = open_live + open_shadow
        today       = date.today()
        vix         = float(market_data.get("india_vix", market_data.get("vix", 17.0)))

        review = {
            "date": today.isoformat(),
            "open_live_count":   len(open_live),
            "open_shadow_count": len(open_shadow),
            "positions_at_risk": [],
            "theta_bleed_today": 0.0,
            "total_unrealised":  sum(t.unrealised_pnl for t in all_open),
            "forward_risks":     [],
            "recommendation":    "",
        }

        theta_bleed = 0.0
        for t in all_open:
            dte            = self._days_to_expiry(t.expiry_date)
            daily_theta    = abs(t.greeks.get("theta", 0.0))
            theta_bleed   += daily_theta * t.lot_size * t.lots
            days_remaining = max(0, t.max_hold_days - t.days_held)
            pnl_pct        = (t.unrealised_pnl / t.total_cost * 100) if t.total_cost > 0 else 0

            alert = ""
            if days_remaining <= 1:
                alert = "⚠ THETA-STOP IMMINENT — exits at Monday open"
            elif dte <= 2:
                alert = "⚠ EXPIRY APPROACHING — review at open"

            review["positions_at_risk"].append({
                "trade_id": t.trade_id, "underlying": t.underlying,
                "strategy": t.strategy, "mode": getattr(t, "mode", "LIVE"),
                "days_to_expiry": dte, "days_held": t.days_held,
                "days_remaining_before_theta_stop": days_remaining,
                "unrealised_pnl": t.unrealised_pnl, "pnl_pct": pnl_pct,
                "theta_bleed_per_day": daily_theta * t.lot_size * t.lots,
                "total_cost": t.total_cost, "alert": alert,
            })

        review["theta_bleed_today"] = theta_bleed

        if vix > 20:
            review["forward_risks"].append(
                f"VIX={vix:.1f} elevated — option premiums may gap at open"
            )
        if regime in ("MILD_BEAR", "BEAR", "CORRECTION"):
            review["forward_risks"].append(
                f"Regime={regime} — gap-down risk on Monday open"
            )

        imminent = [r for r in review["positions_at_risk"] if "IMMINENT" in r.get("alert","")]
        if imminent:
            review["recommendation"] = (
                f"⚠ ACTION REQUIRED MONDAY 9:15 AM: {len(imminent)} position(s) "
                f"will hit theta-stop. Review before new entries."
            )
        elif open_live:
            review["recommendation"] = (
                f"Monitor {len(open_live)} live position(s). "
                f"Weekend theta bleed: ₹{theta_bleed*2:,.0f} (2 days). "
                f"No action until market opens."
            )
        else:
            review["recommendation"] = (
                "No live positions. Capital-safe weekend. "
                "Review shadow trade outcomes for Monday setup quality."
            )

        return review

    # ------------------------------------------------------------------ #
    #  Execution                                                           #
    # ------------------------------------------------------------------ #

    def _execute_suggestion(
        self, sug, market_data, regime, regime_confidence,
        agent_consensus, cycle_number, iv_regime, mode="LIVE",
        examiner_size_mult: float = 1.0,
    ) -> Optional[Dict]:
        try:
            underlying = getattr(sug, "index", "NIFTY")
            strategy   = str(getattr(sug, "strategy", "UNKNOWN")).upper()

            # OptionSuggestion uses 'confidence' not 'conviction',
            # 'estimated_premium' / 'cost_per_lot' not 'entry_price',
            # 'expiry_str' (string) as well as 'expiry' (date obj)
            conviction   = int(getattr(sug, "confidence",
                               getattr(sug, "conviction", 0)))
            cost_per_lot = float(getattr(sug, "cost_per_lot",
                                  getattr(sug, "estimated_premium",
                                  getattr(sug, "entry_price", 0.0))))
            # expiry: prefer expiry_str (ISO string), fall back to expiry (date)
            _expiry_raw  = getattr(sug, "expiry_str", None) or getattr(sug, "expiry", "")
            expiry       = str(_expiry_raw)
            dte          = int(getattr(sug, "dte", 7))
            strike       = float(getattr(sug, "strike", 0))
            iv_rank      = float(getattr(sug, "iv_rank", 0))
            spot         = float(getattr(sug, "spot",
                            self._get_spot(underlying, market_data, sug)))
            vix          = float(market_data.get("india_vix", market_data.get("vix", 17.0)))

            # Guard: skip if cost is zero (bad data)
            if cost_per_lot <= 0:
                logger.warning(
                    f"[VEB] {underlying} {strategy} has zero cost_per_lot — skipping"
                )
                return None

            if mode == "LIVE":
                if not self._sie.is_strategy_enabled(strategy):
                    logger.info(f"[VEB] {strategy} disabled by calibration")
                    return None
                min_c = self._sie.get_min_conviction(strategy)
                if conviction < min_c:
                    logger.info(f"[VEB] {underlying} {strategy} conviction {conviction} < {min_c}")
                    return None

            size_mult = self._sie.get_position_size_multiplier(regime) if mode == "LIVE" else 1.0
            # Apply cross-examiner reduction (REDUCE_SIZE = 0.5x)
            if examiner_size_mult < 1.0 and mode == "LIVE":
                size_mult = size_mult * examiner_size_mult
                logger.info(
                    f"[VEB] {underlying} {strategy}: examiner size_mult={examiner_size_mult:.0%} applied"
                )
            sl_pct    = self._sie.get_sl_pct(iv_regime)
            tgt_pct   = self._sie.get_target_pct(iv_regime)
            max_hold  = self._sie.get_max_hold_days(dte)

            # Apply real-time impact adjustments
            _impact_rep = report.get("impact", {}) if "report" in dir() else {}
            _impact_rep = getattr(self, "_last_impact_report", {})
            if _impact_rep.get("should_reduce_size"):
                size_mult = size_mult * 0.50
                logger.info(f"[VEB] {underlying} size reduced 50% — EXTREME/multi-HIGH impact")
            elif _impact_rep.get("should_widen_sl"):
                sl_pct = min(sl_pct + 0.10, 0.65)
                logger.info(f"[VEB] {underlying} SL widened to {sl_pct:.0%} — HIGH impact event")

            lots = max(1, round(size_mult))
            legs      = self._build_legs(underlying, strategy, strike, expiry, cost_per_lot, lots)
            greeks    = self._extract_greeks(sug)

            tid = self._broker.execute_fno_trade(
                underlying=underlying, strategy=strategy, legs=legs,
                expiry_date=expiry, dte=dte, entry_regime=regime,
                regime_confidence=regime_confidence, conviction=conviction,
                agent_consensus=agent_consensus, vix=vix, spot=spot,
                iv_rank=iv_rank, cost_per_lot=cost_per_lot, lots=lots,
                greeks=greeks, stop_loss_pct=sl_pct, target_pct=tgt_pct,
                max_hold_days=max_hold, cycle_number=cycle_number,
                notes=(f"SHADOW: WAIT gate | regime={regime}"
                       if mode == "SHADOW"
                       else f"LIVE | regime={regime}"),
                mode=mode,
            )
            label = "✅ LIVE" if mode == "LIVE" else "👁 SHADOW"
            logger.info(f"[VEB] {label} | {tid} | {underlying} {strategy} | exp={expiry}")
            return {
                "trade_id": tid, "underlying": underlying, "strategy": strategy,
                "conviction": conviction, "cost_per_lot": cost_per_lot,
                "lots": lots, "total_cost": cost_per_lot * lots,
                "expiry": expiry, "dte": dte, "sl_pct": sl_pct,
                "target_pct": tgt_pct, "max_hold": max_hold, "mode": mode,
            }
        except Exception as e:
            logger.error(f"[VEB] _execute_suggestion: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Report                                                              #
    # ------------------------------------------------------------------ #

    def format_execution_report(self, report: Dict) -> str:
        lines = ["", "=" * 65,
                 "VIRTUAL EXECUTION ENGINE — LIVE-LIKE TRADE TRACKER",
                 "=" * 65]

        if report.get("is_holiday"):
            lines.append("  📅 HOLIDAY/WEEKEND — Theta decay applied | Self-review active")

        # ── Real-time market impact ────────────────────────────────────
        imp = report.get("impact", {})
        if imp and imp.get("overall_severity", "LOW") != "LOW":
            sev  = imp["overall_severity"]
            dirn = imp["overall_direction"]
            icon = "🔴" if sev == "EXTREME" else "🟠" if sev == "HIGH" else "🟡"
            lines.append(f"")
            lines.append(f"REAL-TIME MARKET IMPACT — {icon} {sev} / {dirn}")
            for ev_name, ev_sev, ev_dir, ev_prem, ev_desc in imp.get("events", []):
                ev_icon = "🔴" if ev_sev in ("HIGH","EXTREME") else "🟡"
                lines.append(f"  {ev_icon} [{ev_sev}] {ev_name}: {ev_desc}")
            if imp.get("should_widen_sl"):
                lines.append(f"  ⚠ SL WIDENED +10% on new trades due to active HIGH impact")
            if imp.get("should_reduce_size"):
                lines.append(f"  ⚠ POSITION SIZE CUT 50% — EXTREME/multi-HIGH impact active")
            if imp.get("vix_premium_mult", 1.0) > 1.0:
                lines.append(
                    f"  📈 Vol expansion: premiums adjusted ×{imp['vix_premium_mult']:.2f} "
                    f"(VIX-driven)"
                )
            for restr in imp.get("active_restrictions", []):
                lines.append(f"  🚫 RESTRICTION: {restr}")
        elif imp:
            lines.append("")
            lines.append("REAL-TIME MARKET IMPACT — ✅ LOW / NEUTRAL (no elevated events)")

        pf = report.get("portfolio", {})
        lines += [
            "",
            "PORTFOLIO STATUS (Virtual Live Account)",
            f"  Available : ₹{pf.get('available_capital', 0):>12,.0f}",
            f"  Deployed  : ₹{pf.get('deployed_capital', 0):>12,.0f}",
            f"  Unrealised: ₹{pf.get('total_unrealised_pnl', 0):>+12,.0f}",
            f"  Realised  : ₹{pf.get('total_realised_pnl', 0):>+12,.0f}",
            f"  Win Rate  : {pf.get('win_rate', 0):.1f}%  "
            f"({pf.get('total_trades', 0)} live | "
            f"{pf.get('open_positions', 0)} open | "
            f"{pf.get('shadow_trades', 0)} shadow tracked)",
        ]
        sh_pnl = pf.get("shadow_realised_pnl", 0.0)
        if pf.get("shadow_trades", 0) > 0:
            lines.append(
                f"  Shadow PnL: ₹{sh_pnl:>+12,.0f}  "
                f"← what WATCH trades would have returned"
            )

        # New live
        le = report.get("new_live_executions", [])
        if le:
            lines.append(f"\nNEW LIVE ORDERS EXECUTED ({len(le)})")
            for x in le:
                lines.append(
                    f"  ✅ [{x['trade_id']}] {x['underlying']} {x['strategy']} "
                    f"| lots={x['lots']} | cost=₹{x['total_cost']:,.0f} "
                    f"| exp={x['expiry']} | SL={x['sl_pct']*100:.0f}% "
                    f"| TGT={x['target_pct']*100:.0f}% | hold≤{x['max_hold']}d"
                )

        # Shadow
        se = report.get("new_shadow_executions", [])
        if se:
            lines.append(f"\nSHADOW TRACKED — WAIT gate (zero capital used)")
            for x in se:
                lines.append(
                    f"  👁 [{x['trade_id']}] {x['underlying']} {x['strategy']} "
                    f"| cost=₹{x['total_cost']:,.0f} | exp={x['expiry']} "
                    f"| conviction={x['conviction']}"
                )
            lines.append("  → Running as paper trades. Outcomes feed conservatism audit.")

        if not le and not se:
            lines.append("\nNEW EXECUTIONS: None this cycle")

        # Auto-exits
        exits = report.get("exits", [])
        if exits:
            lines.append(f"\nAUTO-EXITS ({len(exits)})")
            for ex in exits:
                icon = "✅" if ex.get("correct") else "❌"
                lines.append(
                    f"  {icon} [{ex['trade_id']}] {ex['underlying']} "
                    f"{ex['strategy']} | {ex['reason']} "
                    f"| PnL=₹{ex.get('pnl',0):+,.0f} | {ex.get('verdict','')}"
                )

        # Open live
        lop = report.get("open_live_positions", [])
        if lop:
            lines.append(f"\nOPEN LIVE POSITIONS ({len(lop)})")
            for p in lop:
                pct = (p["unrealised_pnl"] / p["total_cost"] * 100) if p["total_cost"] > 0 else 0
                lines.append(
                    f"  [{p['trade_id']}] {p['underlying']} {p['strategy']} "
                    f"| held={p['days_held']:.0f}d/{p['max_hold']}d "
                    f"| MTM=₹{p['unrealised_pnl']:+,.0f} ({pct:+.1f}%) "
                    f"| exp={p['expiry']}"
                )

        # Open shadow
        sop = report.get("open_shadow_positions", [])
        if sop:
            lines.append(f"\nSHADOW POSITIONS BEING TRACKED ({len(sop)})")
            for p in sop:
                pct = (p["unrealised_pnl"] / p["total_cost"] * 100) if p["total_cost"] > 0 else 0
                lines.append(
                    f"  👁 [{p['trade_id']}] {p['underlying']} {p['strategy']} "
                    f"| held={p['days_held']:.0f}d "
                    f"| MTM=₹{p['unrealised_pnl']:+,.0f} ({pct:+.1f}%)"
                )

        # Conservatism audit
        sc = report.get("shadow_conservatism", {})
        if sc.get("trades", 0) > 0:
            v    = sc.get("verdict", "")
            icon = "✅" if "CORRECT" in v else ("⚠" if "NEUTRAL" in v else "📈")
            lines += [
                "",
                "CONSERVATISM AUDIT — Were WAIT decisions correct?",
                f"  {icon} {v}",
                f"  Shadow closed  : {sc.get('trades', 0)}  "
                f"| Shadow WR: {sc.get('shadow_win_rate', 0):.1f}%  "
                f"| Live WR: {sc.get('live_win_rate', 0):.1f}%",
                f"  {sc.get('message', '')}",
            ]

        # Holiday review
        hr = report.get("holiday_review", {})
        if hr:
            lines += [
                "",
                "HOLIDAY / WEEKEND POSITION REVIEW",
                f"  Theta bleed today : ₹{hr.get('theta_bleed_today', 0):,.0f}",
                f"  Total unrealised  : ₹{hr.get('total_unrealised', 0):+,.0f}",
            ]
            for r in hr.get("positions_at_risk", []):
                mode_tag = "[SHADOW] " if r.get("mode") == "SHADOW" else ""
                alert    = f"  {r['alert']}" if r.get("alert") else ""
                lines.append(
                    f"  {mode_tag}[{r['trade_id']}] {r['underlying']} "
                    f"| DTE={r['days_to_expiry']} | held={r['days_held']:.0f}d "
                    f"| MTM=₹{r['unrealised_pnl']:+,.0f} ({r['pnl_pct']:+.1f}%)"
                    + alert
                )
            for risk in hr.get("forward_risks", []):
                lines.append(f"  ⚠ {risk}")
            if hr.get("recommendation"):
                lines.append(f"\n  → {hr['recommendation']}")

        # Self-improvement
        si = report.get("self_improvement", [])
        if si:
            lines.append("\nSELF-IMPROVEMENT CALIBRATION TRIGGERED")
            for n in si:
                lines.append(f"  ⚙  {n}")

        # Active signals
        sigs = report.get("improvement_signals", [])
        if sigs:
            lines.append(f"\nACTIVE IMPROVEMENT SIGNALS ({len(sigs)})")
            for sig in sigs:
                sev  = sig.get("severity", "INFO")
                icon = "🔴" if sev == "HIGH" else "🟡" if sev == "MEDIUM" else "ℹ "
                lines.append(f"  {icon} [{sev}] {sig.get('signal','')}")
                lines.append(f"      → {sig.get('action','')}")

        lines.append("=" * 65)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _trade_summary(self, t) -> Dict:
        return {
            "trade_id": t.trade_id, "underlying": t.underlying,
            "strategy": t.strategy, "mode": getattr(t, "mode", "LIVE"),
            "expiry": t.expiry_date, "days_held": t.days_held,
            "max_hold": t.max_hold_days, "total_cost": t.total_cost,
            "current_premium": t.current_premium,
            "sl_premium": t.cost_per_lot * (1 - t.stop_loss_pct),
            "unrealised_pnl": t.unrealised_pnl, "regime": t.entry_regime,
        }

    def _get_spot(self, underlying, market_data, sug) -> float:
        for k in [f"{underlying.lower()}_price", f"{underlying.lower()}_spot",
                  "nifty_price" if "NIFTY" in underlying.upper() else ""]:
            v = market_data.get(k)
            if v and float(v) > 0:
                return float(v)
        return float(getattr(sug, "spot", 0) or 0)

    def _extract_greeks(self, sug) -> Dict:
        g = getattr(sug, "greeks", None)
        if g is None:
            return {}
        if isinstance(g, dict):
            return g
        # Greeks dataclass from directional_option_advisor
        return {
            "delta": float(getattr(g, "delta", 0) or 0),
            "gamma": float(getattr(g, "gamma", 0) or 0),
            "theta": float(getattr(g, "theta", 0) or 0),
            "vega":  float(getattr(g, "vega",  0) or 0),
        }

    def _build_legs(self, underlying, strategy, strike, expiry, cost_per_lot, lots) -> List[Dict]:
        try:
            from config import get_lot_size
            ls = get_lot_size(underlying)
        except Exception:
            ls = {"NIFTY":75,"BANKNIFTY":15,"SENSEX":10,"FINNIFTY":40,"BANKEX":15}.get(underlying.upper(),50)
        s = strategy.upper()
        base = {"symbol":underlying,"strike":strike,"expiry":expiry,
                "entry_price":cost_per_lot,"current_price":cost_per_lot,
                "lot_size":ls,"lots":lots,"is_buy":True}
        if s in ("LONG_STRADDLE","STRADDLE"):
            h = cost_per_lot/2
            return [{**base,"option_type":"CE","entry_price":h,"current_price":h},
                    {**base,"option_type":"PE","entry_price":h,"current_price":h}]
        elif s in ("BUY_PE","BEAR_SPREAD","BEAR_PUT_SPREAD"):
            return [{**base,"option_type":"PE"}]
        else:
            return [{**base,"option_type":"CE"}]

    def _extract_prices(self, market_data) -> Dict[str,float]:
        prices = {}
        for sym,keys in {"NIFTY":["nifty_price","nifty_spot"],
                         "BANKNIFTY":["banknifty_price","banknifty_spot"],
                         "SENSEX":["sensex_price","sensex_spot"],
                         "FINNIFTY":["finnifty_price","finnifty_spot"],
                         "BANKEX":["bankex_price","bankex_spot"]}.items():
            for k in keys:
                v = market_data.get(k)
                if v and float(v)>0:
                    prices[sym]=float(v); break
        return prices

    def _days_to_expiry(self, expiry_str: str) -> int:
        try:
            from execution.virtual_broker import VirtualBroker
            exp = VirtualBroker._parse_expiry_date(expiry_str)
            if exp:
                return max(0, (exp - date.today()).days)
        except Exception:
            pass
        return 99

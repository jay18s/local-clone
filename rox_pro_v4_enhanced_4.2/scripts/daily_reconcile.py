"""
ROX Proven Edge Engine v4.0 — Daily Reconciler
================================================
Run once after market close each day:

    python daily_reconcile.py

What it does:
  1. Reads all open (unresolved) trades from Recommendation_Accuracy_Log.csv
  2. Fetches the latest closing price for each stock via Fyers API
  3. Classifies each trade as:
       TARGET_HIT  — price reached target_price
       STOP_HIT    — price hit stop_loss
       OPEN        — still within range (no action)
       EXPIRED     — open for >10 trading days, mark as TIMED_OUT
  4. Writes outcome + pnl_pct back to CSV
  5. Updates AgentScorecard (scorecard.json) with WIN/LOSS per agent
  6. Prints a daily accuracy report

Run with --dry-run to preview without writing anything.
Run with --date 2026-02-25 to reconcile a specific date.
"""

import os
import sys
import csv
import json
import argparse
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# __file__ is  .../rox_pro_v4_unified/scripts/daily_reconcile.py
# Project root is one level up from scripts/
SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.data_manager import DataManager
from data.scorecard import AgentScorecard, AGENTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Reconciler")

CSV_PATH  = PROJECT_ROOT / "data" / "Market_Trends" / "Recommendation_Accuracy_Log.csv"
HOLD_DAYS = 10   # max trading days before a trade is marked TIMED_OUT


# ── Price / history fetcher ───────────────────────────────────────────────────

def _load_fyers_client():
    """Return an authenticated Fyers client, or None if unavailable."""
    try:
        from core.config import get_system_config
        cfg = get_system_config()
        api = cfg.api
        from fyers_apiv3 import fyersModel
        fyers = fyersModel.FyersModel(
            client_id=api.fyers_app_id or api.fyers_api_key,
            token=api.fyers_access_token,
            is_async=False,
            log_path="",
        )
        return fyers
    except Exception as e:
        logger.warning(f"Fyers client unavailable: {e}")
        return None


def fetch_price_history(symbols: List[str], days_back: int = 15) -> Dict[str, List[Dict]]:
    """
    Fetch daily OHLCV history for each symbol for the last `days_back` calendar days.
    Returns {symbol: [{date, open, high, low, close}, ...]} sorted oldest→newest.

    For a manual-execution signal engine, we need HIGH and LOW of each day —
    not just a single snapshot price — so we can correctly detect whether a
    swing trade target or stop was touched on any day since recommendation.
    """
    fyers = _load_fyers_client()
    if not fyers:
        logger.warning("No Fyers client — price history unavailable.")
        return {}

    end_date   = date.today()
    start_date = end_date - timedelta(days=days_back + 5)   # buffer for weekends

    history: Dict[str, List[Dict]] = {}
    for sym in symbols:
        fyers_sym = f"NSE:{sym}-EQ"
        try:
            resp = fyers.history({
                "symbol":      fyers_sym,
                "resolution":  "D",
                "date_format": "1",
                "range_from":  start_date.strftime("%Y-%m-%d"),
                "range_to":    end_date.strftime("%Y-%m-%d"),
                "cont_flag":   "1",
            })
            if resp.get("s") != "ok":
                continue
            candles = []
            for bar in resp.get("candles", []):
                ts, o, h, l, c, v = bar
                candles.append({
                    "date":  datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                    "open":  o, "high": h, "low": l, "close": c,
                })
            history[sym] = sorted(candles, key=lambda x: x["date"])
        except Exception as e:
            logger.warning(f"History fetch failed for {sym}: {e}")
        import time as _time; _time.sleep(0.3)

    logger.info(f"Fetched price history for {len(history)}/{len(symbols)} symbols")
    return history


def fetch_current_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch latest LTP for a list of NSE stock symbols.
    Returns {symbol: price}. Used as fallback when history is unavailable.
    """
    fyers = _load_fyers_client()
    if not fyers:
        logger.warning("No Fyers client — prices unavailable. Use --prices flag to supply manually.")
        return {}

    nse_syms = [f"NSE:{s}-EQ" for s in symbols]
    prices   = {}
    batch    = 50

    for i in range(0, len(nse_syms), batch):
        chunk = nse_syms[i:i + batch]
        try:
            resp = fyers.quotes({"symbols": ",".join(chunk)})
            if resp.get("s") != "ok":
                continue
            for item in resp.get("d", []):
                sym  = item.get("n", "").replace("NSE:", "").replace("-EQ", "")
                ltp  = item.get("v", {}).get("lp", 0.0) or 0.0
                if sym and ltp:
                    prices[sym] = float(ltp)
        except Exception as e:
            logger.warning(f"Price fetch error for batch {i}: {e}")

    logger.info(f"Fetched prices for {len(prices)}/{len(symbols)} symbols")
    return prices


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_open_trades() -> List[Dict]:
    """Return all rows from CSV that have no date_closed."""
    if not CSV_PATH.exists():
        logger.error(f"CSV not found: {CSV_PATH}")
        return []

    trades = []
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(dict(row))

    open_trades = [t for t in trades if not t.get("date_closed")]
    logger.info(f"Found {len(open_trades)} open trades (of {len(trades)} total)")
    return open_trades


def rewrite_csv(updated_rows: List[Dict]):
    """Rewrite the full CSV with updated outcome fields."""
    if not updated_rows:
        return

    # Build a canonical fieldname list from the UNION of all row keys,
    # preserving insertion order and ensuring outcome columns are always present.
    # This prevents ValueError when resolved rows have extra fields that
    # unresolved rows don't have yet.
    seen_fields: dict = {}
    for row in updated_rows:
        for k in row.keys():
            seen_fields[k] = None
    # Guarantee outcome columns exist in the header even if no row has them yet
    for col in ("date_closed", "exit_price", "pnl_pct", "exit_reason"):
        seen_fields[col] = None
    fieldnames = list(seen_fields.keys())

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in updated_rows:
            # Fill any missing fields with empty string so every row is complete
            complete = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(complete)


def load_all_trades() -> List[Dict]:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, "r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


# ── Outcome logic ─────────────────────────────────────────────────────────────

def _trading_days_since(date_str: str) -> int:
    """Approximate trading days between date_str and today (Mon–Fri, no holiday check)."""
    try:
        start = date.fromisoformat(date_str)
    except ValueError:
        return 0
    today = date.today()
    days  = 0
    cur   = start + timedelta(days=1)
    while cur <= today:
        if cur.weekday() < 5:   # Mon=0 … Fri=4
            days += 1
        cur += timedelta(days=1)
    return days


def classify_trade(trade: Dict, price_history: List[Dict]) -> Tuple[str, float]:
    """
    Classify a swing trade recommendation using daily OHLCV history.

    For a manual-execution signal engine, correctness means:
      - TARGET_HIT : stock's daily HIGH reached the target on any day since entry date
      - STOP_HIT   : stock's daily LOW touched the stop on any day since entry date
                     (if both hit on same day, target wins — assumes intraday target first)
      - TIMED_OUT  : neither hit after HOLD_DAYS trading days; use latest close for P&L
      - OPEN       : still within hold window, no level touched yet

    Args:
        trade        : CSV row dict
        price_history: list of {date, open, high, low, close} sorted oldest→newest,
                       covering the period from date_recommended onward.
    """
    entry   = float(trade.get("entry_price", 0) or 0)
    sl      = float(trade.get("stop_loss",    0) or 0)
    target  = float(trade.get("target_price", 0) or 0)
    dirn    = trade.get("direction", "LONG").upper()
    rec_date = trade.get("date_recommended", "2000-01-01")

    if entry <= 0 or not price_history:
        return "OPEN", 0.0

    # Only look at bars from the day AFTER recommendation (entry is next-day open)
    relevant = [b for b in price_history if b["date"] > rec_date]

    trading_days = len(relevant)
    latest_close = relevant[-1]["close"] if relevant else entry

    for bar in relevant:
        h = bar["high"]; l = bar["low"]

        if dirn == "LONG":
            target_hit = target > 0 and h >= target
            stop_hit   = sl > 0     and l <= sl
        else:
            target_hit = target > 0 and l <= target
            stop_hit   = sl > 0     and h >= sl

        if target_hit:
            pnl = (target - entry) / entry * 100 if dirn == "LONG" \
                  else (entry - target) / entry * 100
            return "TARGET_HIT", round(pnl, 2)

        if stop_hit:
            pnl = (sl - entry) / entry * 100 if dirn == "LONG" \
                  else (entry - sl) / entry * 100
            return "STOP_HIT", round(pnl, 2)

    # Neither level hit yet
    if trading_days >= HOLD_DAYS:
        pnl = (latest_close - entry) / entry * 100 if dirn == "LONG" \
              else (entry - latest_close) / entry * 100
        return "TIMED_OUT", round(pnl, 2)

    return "OPEN", 0.0


# ── Scorecard update ──────────────────────────────────────────────────────────

def update_scorecard(trade: Dict, outcome: str, pnl_pct: float,
                     scorecard: AgentScorecard, dry_run: bool):
    """Update each agent listed in the trade's scorecard entry."""
    if dry_run:
        return

    agents = [a.strip() for a in
              trade.get("recommending_agents", "").split(",") if a.strip()]

    win_loss = "WIN" if outcome in ("TARGET_HIT",) else \
               "LOSS" if outcome in ("STOP_HIT", "TIMED_OUT") else None

    if not win_loss:
        return

    pred_date = trade.get("date_recommended", "")

    # Check upfront whether this date has any pending entry in the scorecard.
    # Trades recommended before the scorecard was wired up (pre-fix CSV entries)
    # won't have a matching pending row — skip them silently to avoid log spam.
    sc_data = scorecard._load()
    has_pending = any(
        any(p["date"] == pred_date for p in sc_data.get(agent, {}).get("pending", []))
        for agent in agents if agent in AGENTS
    )
    if not has_pending:
        logger.debug(
            f"Skipping scorecard update for {trade.get('stock')} "
            f"on {pred_date} — no pending prediction (pre-wiring CSV entry)"
        )
        return

    for agent in agents:
        if agent not in AGENTS:
            continue
        r_multiple = 0.0
        entry  = float(trade.get("entry_price", 0) or 0)
        sl     = float(trade.get("stop_loss", 0) or 0)
        if entry > 0 and sl > 0:
            risk = abs(entry - sl)
            gain = abs(pnl_pct / 100 * entry)
            r_multiple = round(gain / risk, 2) if risk > 0 else 0.0

        scorecard.resolve_prediction(
            agent_name      = agent,
            prediction_date = pred_date,
            outcome         = win_loss,
            r_multiple      = r_multiple if win_loss == "WIN" else -r_multiple,
        )


def dedup_csv():
    """
    Remove duplicate trade entries from the CSV.
    Keeps only the first occurrence of each (date_recommended, stock, direction) triple.
    This cleans up the bloat caused by the pre-fix duplicate logging bug.
    """
    if not CSV_PATH.exists():
        logger.info("No CSV to deduplicate.")
        return

    all_trades = load_all_trades()
    seen = set()
    deduped = []
    dupes = 0
    for row in all_trades:
        key = (row.get("date_recommended",""), row.get("stock",""), row.get("direction",""))
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
            deduped.append(row)

    if dupes == 0:
        logger.info("CSV is already clean — no duplicates found.")
        return

    rewrite_csv(deduped)
    logger.info(f"Deduplication complete: removed {dupes} duplicate rows, {len(deduped)} unique trades remain.")


# ── Main reconciliation loop ──────────────────────────────────────────────────

def reconcile(dry_run: bool = False, manual_prices: Dict[str, float] = None):
    open_trades = load_open_trades()
    if not open_trades:
        logger.info("No open trades to reconcile.")
        _print_summary()
        return

    symbols = list({t["stock"] for t in open_trades if t.get("stock")})

    # Fetch daily OHLCV history for proper swing trade outcome detection.
    # For a manual-execution signal engine we need daily HIGH/LOW — not just
    # today's snapshot — to know if a target or stop was touched on any day.
    if manual_prices:
        # Manual prices supplied: convert to single-bar history stub so
        # classify_trade can still work (treats it as "today's close only")
        price_histories: Dict[str, List[Dict]] = {
            sym: [{"date": date.today().isoformat(),
                   "open": p, "high": p, "low": p, "close": p}]
            for sym, p in manual_prices.items()
        }
        logger.info(f"Using {len(manual_prices)} manually supplied prices.")
    else:
        price_histories = fetch_price_history(symbols, days_back=HOLD_DAYS + 5)

    if not price_histories:
        logger.warning("No price data available. Skipping outcome classification.")
        logger.warning("Run:  python daily_reconcile.py --prices RELIANCE:2850.5,TCS:4120")
        _print_summary()
        return

    scorecard  = AgentScorecard()

    # FIX 9: Expire stale pending predictions BEFORE resolving today's trades.
    # This purges AVOID-cycle suppressed rows (age > SUPPRESSED_MAX_AGE_DAYS)
    # and counts long-orphaned un-linked predictions as losses so the win rate
    # stays accurate.  Must run before the trade loop below.
    _expired = scorecard.expire_stale_predictions()
    if _expired:
        logger.info(f"[Reconcile] Pre-run stale prediction cleanup: {_expired}")

    all_trades = load_all_trades()

    resolved = {"TARGET_HIT": 0, "STOP_HIT": 0, "TIMED_OUT": 0, "OPEN": 0}

    for trade in all_trades:
        if trade.get("date_closed"):
            continue   # already resolved

        sym     = trade.get("stock", "")
        history = price_histories.get(sym)

        if not history:
            logger.debug(f"No price history for {sym} — leaving open")
            resolved["OPEN"] += 1
            continue

        outcome, pnl = classify_trade(trade, history)
        resolved[outcome] += 1

        if outcome != "OPEN":
            # Use the bar on which the outcome occurred for logging
            latest_price = history[-1]["close"]
            if not dry_run:
                trade["date_closed"]  = date.today().isoformat()
                trade["exit_price"]   = str(round(latest_price, 2))
                trade["pnl_pct"]      = str(round(pnl, 2))
                trade["exit_reason"]  = outcome

            update_scorecard(trade, outcome, pnl, scorecard, dry_run)

            tag = "✅" if pnl > 0 else "❌"
            logger.info(
                f"  {tag} {sym:12s} {outcome:12s}  "
                f"entry={trade.get('entry_price')}  "
                f"latest={latest_price:.2f}  pnl={pnl:+.2f}%"
                + (" [DRY RUN]" if dry_run else "")
            )

    if not dry_run:
        rewrite_csv(all_trades)
        logger.info("CSV updated.")

    logger.info(
        f"Reconciliation complete | "
        f"TARGET={resolved['TARGET_HIT']}  "
        f"STOP={resolved['STOP_HIT']}  "
        f"TIMEOUT={resolved['TIMED_OUT']}  "
        f"OPEN={resolved['OPEN']}"
    )

    _print_summary()


def _print_summary():
    """Print current scorecard to stdout."""
    sc = AgentScorecard()
    print("\n" + sc.get_summary_table())


# ── Accuracy report ───────────────────────────────────────────────────────────

def print_accuracy_report():
    """Print a detailed accuracy report from the CSV + scorecard."""
    trades = load_all_trades()
    closed = [t for t in trades if t.get("date_closed")]

    if not closed:
        print("No closed trades yet.")
        return

    wins   = [t for t in closed if float(t.get("pnl_pct", 0) or 0) > 0]
    losses = [t for t in closed if float(t.get("pnl_pct", 0) or 0) <= 0]
    total_pnl = sum(float(t.get("pnl_pct", 0) or 0) for t in closed)
    avg_win  = sum(float(t.get("pnl_pct", 0) or 0) for t in wins) / len(wins) if wins else 0
    avg_loss = sum(float(t.get("pnl_pct", 0) or 0) for t in losses) / len(losses) if losses else 0

    print("\n" + "=" * 60)
    print("ROX PROVEN EDGE ENGINE v4.0 — ACCURACY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"  Total closed trades : {len(closed)}")
    print(f"  Wins                : {len(wins)}  ({len(wins)/len(closed)*100:.1f}%)")
    print(f"  Losses              : {len(losses)}  ({len(losses)/len(closed)*100:.1f}%)")
    print(f"  Avg win             : +{avg_win:.2f}%")
    print(f"  Avg loss            : {avg_loss:.2f}%")
    print(f"  Total P&L           : {total_pnl:+.2f}%")
    print()

    # Per-stock accuracy
    stock_stats: Dict[str, Dict] = {}
    for t in closed:
        sym = t.get("stock", "UNKNOWN")
        pnl = float(t.get("pnl_pct", 0) or 0)
        if sym not in stock_stats:
            stock_stats[sym] = {"wins": 0, "total": 0, "pnl": 0.0}
        stock_stats[sym]["total"] += 1
        stock_stats[sym]["pnl"]   += pnl
        if pnl > 0:
            stock_stats[sym]["wins"] += 1

    print(f"  {'Stock':<14} {'Trades':>6} {'Win%':>6} {'Total P&L':>10}")
    print("  " + "-" * 40)
    for sym, s in sorted(stock_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr = s["wins"] / s["total"] * 100 if s["total"] else 0
        print(f"  {sym:<14} {s['total']:>6} {wr:>5.0f}%  {s['pnl']:>+9.2f}%")

    print()
    sc = AgentScorecard()
    print(sc.get_summary_table())
    print("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROX Daily Reconciler")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview outcomes without writing to CSV")
    parser.add_argument("--report",   action="store_true",
                        help="Print accuracy report and exit")
    parser.add_argument("--dedup",    action="store_true",
                        help="Remove duplicate rows from Recommendation_Accuracy_Log.csv")
    parser.add_argument("--prices",   type=str, default="",
                        help="Manual prices: SYMBOL:PRICE,SYMBOL:PRICE ...")
    args = parser.parse_args()

    if args.report:
        print_accuracy_report()
        sys.exit(0)

    if args.dedup:
        dedup_csv()
        sys.exit(0)

    # Parse manual prices if provided
    manual = {}
    if args.prices:
        for pair in args.prices.split(","):
            if ":" in pair:
                sym, price = pair.strip().split(":", 1)
                try:
                    manual[sym.strip().upper()] = float(price.strip())
                except ValueError:
                    pass

    reconcile(dry_run=args.dry_run, manual_prices=manual or None)

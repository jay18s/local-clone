#!/usr/bin/env python3
"""
Weekly Meta-Analysis Script
============================

Scheduled job for running LLM-powered meta-learning analysis.

Usage:
    python scripts/weekly_meta_analysis.py [--apply] [--date YYYY-MM-DD]

Features:
- Analyzes weekly performance data
- Generates improvement recommendations via LLM
- Optionally applies approved recommendations
- Stores results for review
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from datetime import datetime, date, timedelta
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_system_config, LLMConfig
from agents.llm.llm_meta_learner import LLMMetaLearner, MetaLearningResult
from data.meta_learning.recommendations_store import RecommendationsStore


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_week_boundaries(target_date: date = None) -> tuple:
    """
    Get the start and end dates for the trading week.

    Args:
        target_date: Target date (defaults to today)

    Returns:
        Tuple of (week_start, week_end)
    """
    if target_date is None:
        target_date = date.today()

    # Get Monday of the current week
    days_since_monday = target_date.weekday()
    week_start = target_date - timedelta(days=days_since_monday)

    # Friday is 4 days after Monday
    week_end = week_start + timedelta(days=4)

    # If we're before Friday, use last week
    if target_date.weekday() > 4:  # Weekend
        week_start = week_start - timedelta(days=7)
        week_end = week_end - timedelta(days=7)

    return week_start, week_end


def collect_performance_data(week_start: date, week_end: date) -> dict:
    """
    Collect performance data for the analysis period.

    Args:
        week_start: Start date of the analysis period
        week_end: End date of the analysis period

    Returns:
        Dictionary of performance metrics
    """
    logger = logging.getLogger("MetaAnalysis")

    # This would normally query the DataManager for actual trade data
    # For now, we return placeholder data that can be populated

    data = {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "total_predictions": 0,
        "win_rate": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "profit_factor": 0.0,
        "agent_performance": {},
        "failure_analysis": [],
        "success_analysis": [],
        "regime_performance": {},
        "pattern_performance": {},
        "sector_performance": {},
    }

    # Try to load actual data from trade log
    try:
        trade_log_path = Path(__file__).parent.parent / "data" / "Market_Trends" / "Recommendation_Accuracy_Log.csv"
        if trade_log_path.exists():
            import csv
            with open(trade_log_path, 'r') as f:
                reader = csv.DictReader(f)
                trades = list(reader)

            # Filter to week
            week_trades = [
                t for t in trades
                if week_start.isoformat() <= t.get('date', '') <= week_end.isoformat()
            ]

            if week_trades:
                wins = [t for t in week_trades if t.get('outcome') == 'WIN']
                losses = [t for t in week_trades if t.get('outcome') == 'LOSS']

                data["total_predictions"] = len(week_trades)
                data["win_rate"] = len(wins) / len(week_trades) * 100 if week_trades else 0

                if wins:
                    data["avg_win_pct"] = sum(float(t.get('return_pct', 0)) for t in wins) / len(wins)
                if losses:
                    data["avg_loss_pct"] = sum(abs(float(t.get('return_pct', 0))) for t in losses) / len(losses)

                logger.info(f"Loaded {len(week_trades)} trades for week {week_start} to {week_end}")

    except Exception as e:
        logger.warning(f"Could not load trade data: {e}")

    return data


def run_meta_analysis(
    week_start: date,
    week_end: date,
    llm_config: LLMConfig,
    apply_recommendations: bool = False
) -> MetaLearningResult:
    """
    Run the meta-learning analysis.

    Args:
        week_start: Start date
        week_end: End date
        llm_config: LLM configuration
        apply_recommendations: Whether to auto-apply recommendations

    Returns:
        MetaLearningResult
    """
    logger = logging.getLogger("MetaAnalysis")

    # Initialize meta-learner
    meta_learner = LLMMetaLearner(llm_config)

    # Collect performance data
    performance_data = collect_performance_data(week_start, week_end)

    # Run analysis
    logger.info(f"Running meta-analysis for {week_start} to {week_end}")
    result = meta_learner.analyze_weekly_performance(week_start, week_end, performance_data)

    # Log results
    logger.info(f"Analysis complete: {result.source}")
    logger.info(f"  Agent adjustments: {len(result.agent_weight_adjustments)}")
    logger.info(f"  Regime rules: {len(result.regime_specific_rules)}")
    logger.info(f"  Confidence: {result.confidence_in_recommendations}%")

    # Store results
    store = RecommendationsStore()
    rec_id = store.store(result)
    logger.info(f"Stored recommendation: {rec_id}")

    # Apply if requested
    if apply_recommendations and result.confidence_in_recommendations >= 70:
        logger.info("Applying recommendations...")
        config = get_system_config()
        success = meta_learner.apply_recommendations(result, config)
        if success:
            store.mark_applied(rec_id, "Auto-applied with high confidence")
            logger.info("Recommendations applied successfully")
        else:
            store.mark_rejected(rec_id, "Failed to apply")
            logger.error("Failed to apply recommendations")
    elif apply_recommendations:
        logger.warning(f"Confidence ({result.confidence_in_recommendations}%) below threshold for auto-apply")
        store.update_status(rec_id, "PENDING", "Requires manual review - confidence below threshold")

    return result


def print_report(result: MetaLearningResult):
    """Print a human-readable report."""
    print("\n" + "=" * 70)
    print("WEEKLY META-LEARNING ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nPeriod: {result.week_start} to {result.week_end}")
    print(f"Analysis Time: {result.analysis_timestamp}")
    print(f"Source: {result.source}")
    print(f"Confidence: {result.confidence_in_recommendations}%")

    if result.agent_weight_adjustments:
        print("\n📊 AGENT WEIGHT ADJUSTMENTS:")
        for adj in result.agent_weight_adjustments:
            print(f"  • {adj.agent_name}: {adj.action} by {adj.amount:.2f}")
            print(f"    Reason: {adj.reason}")

    if result.regime_specific_rules:
        print("\n⚙️ REGIME-SPECIFIC RULES:")
        for rule in result.regime_specific_rules:
            print(f"  • [{rule.regime}] {rule.rule}")
            print(f"    Reason: {rule.reason}")

    if result.pattern_adjustments:
        print("\n📈 PATTERN ADJUSTMENTS:")
        for adj in result.pattern_adjustments:
            print(f"  • {adj}")

    if result.sector_insights:
        print("\n🏢 SECTOR INSIGHTS:")
        for insight in result.sector_insights:
            print(f"  • {insight}")

    if result.systemic_improvements:
        print("\n🔧 SYSTEMIC IMPROVEMENTS:")
        for imp in result.systemic_improvements:
            print(f"  • {imp}")

    print(f"\n🎯 NEXT WEEK FOCUS: {result.next_week_focus}")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run weekly meta-learning analysis for ROX Engine"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Automatically apply high-confidence recommendations"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target date (YYYY-MM-DD), defaults to current week"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis without storing results"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger("MetaAnalysis")

    # Parse target date
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = None

    # Get week boundaries
    week_start, week_end = get_week_boundaries(target_date)
    logger.info(f"Analysis period: {week_start} to {week_end}")

    # Get LLM configuration
    llm_config = LLMConfig.from_env()

    if not llm_config.enabled:
        logger.warning("LLM is disabled - using fallback analysis")
    elif not llm_config.api_key:
        logger.warning("No Gemini API key configured - using fallback analysis")

    # Run analysis
    result = run_meta_analysis(
        week_start=week_start,
        week_end=week_end,
        llm_config=llm_config,
        apply_recommendations=args.apply and not args.dry_run
    )

    # Print report
    print_report(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
ROX Engine Dashboard API Server
================================
FastAPI server that exposes ROX engine data to the Next.js dashboard.

This server runs alongside the ROX engine and provides REST endpoints
for the dashboard to fetch real-time trading data.

Usage:
    # Run the API server (runs on port 8000)
    python api_server.py

    # Or run both engine and API together:
    python main.py --mode live --with-api

Endpoints:
    GET /api/status          - Engine status and health
    GET /api/regime          - Current market regime analysis
    GET /api/consensus       - Agent consensus panel data
    GET /api/positions       - Current F&O positions
    GET /api/suggestions     - Option trading suggestions
    GET /api/news            - Latest news analysis
    GET /api/portfolio       - Portfolio Greeks and P&L
    GET /api/market          - Live market data (NIFTY, VIX, etc.)
    GET /api/dashboard       - Combined dashboard data (all-in-one)
"""

import json
import asyncio
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ============================================================================
# SHARED STATE - This is where ROX engine publishes its data
# ============================================================================

class SharedState:
    """
    Singleton class to hold ROX engine state.
    The ROX engine updates this state, and the API server reads from it.
    
    In production, consider using Redis or a database for persistence.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize empty state"""
        self.engine_status = "idle"
        self.last_update: Optional[datetime] = None
        self.regime: Dict[str, Any] = {}
        self.consensus: Dict[str, Any] = {}
        self.positions: List[Dict] = []
        self.suggestions: List[Dict] = []
        self.news: List[Dict] = []
        self.portfolio: Dict[str, Any] = {}
        self.market: Dict[str, Any] = {}
        self.agents: List[Dict] = []
        self.swing_setups: List[Dict] = []
        self.action_items: List[str] = []
        
        # Performance metrics
        self.performance = {
            "win_rate_7d": 0.0,
            "win_rate_30d": 0.0,
            "total_trades": 0,
            "profit_factor": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
        }
    
    def update_from_engine(self, engine_data: Dict):
        """
        Update state from ROX engine output.
        Call this from main.py after each trading cycle.
        """
        self.last_update = datetime.now()
        self.engine_status = engine_data.get("status", "running")
        self.regime = engine_data.get("regime", {})
        self.consensus = engine_data.get("consensus", {})
        self.positions = engine_data.get("positions", [])
        self.suggestions = engine_data.get("suggestions", [])
        self.news = engine_data.get("news", [])
        self.portfolio = engine_data.get("portfolio", {})
        self.market = engine_data.get("market", {})
        self.agents = engine_data.get("agents", [])
        self.swing_setups = engine_data.get("swing_setups", [])
        self.action_items = engine_data.get("action_items", [])
        self.performance = engine_data.get("performance", self.performance)
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for JSON response"""
        return {
            "engine_status": self.engine_status,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "regime": self.regime,
            "consensus": self.consensus,
            "positions": self.positions,
            "suggestions": self.suggestions,
            "news": self.news,
            "portfolio": self.portfolio,
            "market": self.market,
            "agents": self.agents,
            "swing_setups": self.swing_setups,
            "action_items": self.action_items,
            "performance": self.performance,
        }


# Global shared state instance
shared_state = SharedState()

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="ROX Proven Edge Engine API",
    description="REST API for ROX trading engine dashboard",
    version="4.1.0",
)

# CORS configuration - allow Next.js dashboard to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def serialize_datetime(obj):
    """JSON serializer for datetime objects"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "ROX Proven Edge Engine API",
        "version": "4.1.0",
        "status": shared_state.engine_status,
        "last_update": shared_state.last_update.isoformat() if shared_state.last_update else None,
        "endpoints": [
            "/api/status",
            "/api/regime", 
            "/api/consensus",
            "/api/positions",
            "/api/suggestions",
            "/api/news",
            "/api/portfolio",
            "/api/market",
            "/api/dashboard",
        ]
    }


@app.get("/api/status")
async def get_status():
    """Get engine status and health"""
    return {
        "engine_status": shared_state.engine_status,
        "last_update": shared_state.last_update.isoformat() if shared_state.last_update else None,
        "uptime_seconds": (
            (datetime.now() - shared_state.last_update).total_seconds()
            if shared_state.last_update else 0
        ),
        "agents_loaded": len(shared_state.agents),
        "positions_count": len(shared_state.positions),
        "suggestions_count": len(shared_state.suggestions),
    }


@app.get("/api/regime")
async def get_regime():
    """Get current market regime analysis"""
    if not shared_state.regime:
        raise HTTPException(status_code=503, detail="Regime data not available yet")
    return shared_state.regime


@app.get("/api/consensus")
async def get_consensus():
    """Get agent consensus panel data"""
    if not shared_state.consensus:
        raise HTTPException(status_code=503, detail="Consensus data not available yet")
    return {
        "consensus": shared_state.consensus,
        "agents": shared_state.agents,
    }


@app.get("/api/positions")
async def get_positions():
    """Get current F&O positions"""
    return {
        "positions": shared_state.positions,
        "portfolio": shared_state.portfolio,
        "total_positions": len(shared_state.positions),
    }


@app.get("/api/suggestions")
async def get_suggestions():
    """Get option trading suggestions"""
    return {
        "suggestions": shared_state.suggestions,
        "total_suggestions": len(shared_state.suggestions),
        "market_stance": shared_state.consensus.get("direction", "NEUTRAL"),
        "vix": shared_state.market.get("vix", 0),
    }


@app.get("/api/news")
async def get_news():
    """Get latest news analysis"""
    return {
        "news": shared_state.news,
        "total_items": len(shared_state.news),
        "last_update": shared_state.last_update.isoformat() if shared_state.last_update else None,
    }


@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio Greeks and P&L"""
    return {
        "portfolio": shared_state.portfolio,
        "positions": shared_state.positions,
    }


@app.get("/api/market")
async def get_market():
    """Get live market data"""
    if not shared_state.market:
        raise HTTPException(status_code=503, detail="Market data not available yet")
    return shared_state.market


@app.get("/api/dashboard")
async def get_dashboard():
    """
    Get all dashboard data in one call.
    This is the main endpoint used by the Next.js dashboard.
    """
    return shared_state.to_dict()


@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    return shared_state.performance


# ============================================================================
# DEMO DATA ENDPOINT (for testing dashboard without running engine)
# ============================================================================

@app.post("/api/demo/load")
async def load_demo_data():
    """Load demo data for dashboard testing"""
    demo_data = {
        "status": "running",
        "regime": {
            "regime": "CONSOLIDATION",
            "confidence": 55.0,
            "probability_distribution": {
                "BULL": 0.15,
                "MILD_BULL": 0.12,
                "CONSOLIDATION": 0.55,
                "MILD_BEAR": 0.10,
                "BEAR": 0.05,
                "CORRECTION": 0.03
            },
            "reasoning": "Market showing sideways movement with mixed signals. VIX at 13.5 suggests low volatility expectations. FII flows negative but DII support present.",
            "key_factors": [
                "VIX at moderate levels (13.5)",
                "FII 5-day outflow of ₹3,466 Cr",
                "DII 5-day inflow of ₹5,032 Cr",
                "PCR at 0.57 for NIFTY",
            ],
            "transition_warning": None,
        },
        "consensus": {
            "direction": "NEUTRAL",
            "strength": "NO_CONSENSUS",
            "net_score": 0.004,
            "confidence": 55,
        },
        "agents": [
            {"name": "ORION", "verdict": "NEUTRAL", "conviction": 35, "weight": 0.14, "status": "active"},
            {"name": "VESPER", "verdict": "LONG", "conviction": 65, "weight": 0.12, "status": "active"},
            {"name": "KAIRO", "verdict": "LONG", "conviction": 64, "weight": 0.09, "status": "active"},
            {"name": "SENTINEL", "verdict": "SHORT", "conviction": 65, "weight": 0.13, "status": "active"},
            {"name": "NEXUS", "verdict": "NEUTRAL", "conviction": 42, "weight": 0.12, "status": "active"},
            {"name": "PRUDENCE", "verdict": "LONG", "conviction": 80, "weight": 0.07, "status": "active"},
            {"name": "CATALYST", "verdict": "NEUTRAL", "conviction": 80, "weight": 0.08, "status": "active"},
            {"name": "OPTIMUS", "verdict": "SHORT", "conviction": 66, "weight": 0.15, "status": "active"},
            {"name": "HERMES", "verdict": "NEUTRAL", "conviction": 50, "weight": 0.10, "status": "active"},
            {"name": "THETA", "verdict": "LONG", "conviction": 55, "weight": 0.08, "status": "active"},
            {"name": "DELTA", "verdict": "NEUTRAL", "conviction": 45, "weight": 0.12, "status": "active"},
            {"name": "NOCTURNAL", "verdict": "LONG", "conviction": 70, "weight": 0.06, "status": "active"},
        ],
        "positions": [],
        "portfolio": {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "unrealized_pnl": 0,
            "portfolio_value": 1000000,
        },
        "suggestions": [
            {
                "index": "NIFTY",
                "strategy": "LONG STRADDLE (CE+PE)",
                "strike": 25350,
                "strike_type": "ATM",
                "expiry": "2026-03-10",
                "dte": 6,
                "spot": 25332.3,
                "entry_price": 476,
                "cost_per_lot": 35715,
                "stop_loss": 190,
                "target": 857,
                "breakeven_lower": 24874,
                "breakeven_upper": 25826,
                "prob_profit": 42,
                "delta": 0.038,
                "gamma": 0.001336,
                "theta": -34.63,
                "vega": 29.89,
                "iv_rank": 17,
                "iv_regime": "LOW",
                "conviction": 55,
                "score": 71.0,
                "basis": "NO_CONSENSUS + LOW IV 17 -> buy vol | ATM=25350 | BE 24874-25826",
                "oi": 415609870,
                "volume": 62341480,
                "max_pain": 25350,
                "pcr": 0.57,
            },
            {
                "index": "BANKNIFTY",
                "strategy": "LONG STRADDLE (CE+PE)",
                "strike": 60700,
                "strike_type": "ATM",
                "expiry": "2026-03-11",
                "dte": 7,
                "spot": 60739.8,
                "entry_price": 1281,
                "cost_per_lot": 19212,
                "stop_loss": 512,
                "target": 2305,
                "breakeven_lower": 59419,
                "breakeven_upper": 61981,
                "prob_profit": 42,
                "delta": 0.089,
                "gamma": 0.000496,
                "theta": -75.17,
                "vega": 79.72,
                "iv_rank": 17,
                "iv_regime": "LOW",
                "conviction": 55,
                "score": 71.0,
                "basis": "NO_CONSENSUS + LOW IV 17 -> buy vol | ATM=60700 | BE 59419-61981",
                "oi": 20062700,
                "volume": 3009405,
                "max_pain": 60700,
                "pcr": 1.04,
            },
            {
                "index": "SENSEX",
                "strategy": "LONG STRADDLE (CE+PE)",
                "strike": 81800,
                "strike_type": "ATM",
                "expiry": "2026-03-12",
                "dte": 8,
                "spot": 81790.9,
                "entry_price": 1806,
                "cost_per_lot": 18055,
                "stop_loss": 722,
                "target": 3250,
                "breakeven_lower": 79994,
                "breakeven_upper": 83606,
                "prob_profit": 42,
                "delta": 0.069,
                "gamma": 0.000352,
                "theta": -97.45,
                "vega": 112.86,
                "iv_rank": 17,
                "iv_regime": "LOW",
                "conviction": 55,
                "score": 71.0,
                "basis": "NO_CONSENSUS + LOW IV 17 -> buy vol | ATM=81800 | BE 79994-83606",
                "oi": 20478240,
                "volume": 3071736,
                "max_pain": 81800,
                "pcr": 0.77,
            },
            {
                "index": "FINNIFTY",
                "strategy": "LONG STRADDLE (CE+PE)",
                "strike": 28050,
                "strike_type": "ATM",
                "expiry": "2026-03-10",
                "dte": 6,
                "spot": 28063.8,
                "entry_price": 529,
                "cost_per_lot": 21156,
                "stop_loss": 212,
                "target": 952,
                "breakeven_lower": 27521,
                "breakeven_upper": 28579,
                "prob_profit": 42,
                "delta": 0.079,
                "gamma": 0.001202,
                "theta": -38.24,
                "vega": 32.99,
                "iv_rank": 17,
                "iv_regime": "LOW",
                "conviction": 55,
                "score": 71.0,
                "basis": "NO_CONSENSUS + LOW IV 17 -> buy vol | ATM=28050 | BE 27521-28579",
                "oi": 863640,
                "volume": 129546,
                "max_pain": 28050,
                "pcr": 1.06,
            },
            {
                "index": "BANKEX",
                "strategy": "LONG STRADDLE (CE+PE)",
                "strike": 68300,
                "strike_type": "ATM",
                "expiry": "2026-03-12",
                "dte": 8,
                "spot": 68323.6,
                "entry_price": 1510,
                "cost_per_lot": 22647,
                "stop_loss": 604,
                "target": 2718,
                "breakeven_lower": 66790,
                "breakeven_upper": 69810,
                "prob_profit": 42,
                "delta": 0.082,
                "gamma": 0.000420,
                "theta": -81.29,
                "vega": 94.13,
                "iv_rank": 17,
                "iv_regime": "LOW",
                "conviction": 55,
                "score": 67.5,
                "basis": "NO_CONSENSUS + LOW IV 17 -> buy vol | ATM=68300 | BE 66790-69810",
                "oi": 42720,
                "volume": 6408,
                "max_pain": 68300,
                "pcr": 4.50,
            },
        ],
        "news": [
            {
                "id": 1,
                "headline": "RBI keeps repo rate unchanged at 6.5%",
                "source": "Economic Times",
                "category": "POLICY",
                "sentiment": "NEUTRAL",
                "impact_score": 0.8,
                "timestamp": datetime.now().isoformat(),
                "summary": "Reserve Bank of India maintains status quo on interest rates amid global uncertainty.",
            },
            {
                "id": 2,
                "headline": "FII selling continues for 5th consecutive day",
                "source": "Moneycontrol",
                "category": "FLOWS",
                "sentiment": "BEARISH",
                "impact_score": 0.6,
                "timestamp": datetime.now().isoformat(),
                "summary": "Foreign investors pull out ₹3,466 Cr in last 5 trading sessions.",
            },
            {
                "id": 3,
                "headline": "IT sector sees strong Q3 results",
                "source": "Business Standard",
                "category": "EARNINGS",
                "sentiment": "BULLISH",
                "impact_score": 0.7,
                "timestamp": datetime.now().isoformat(),
                "summary": "Major IT companies report better-than-expected quarterly earnings.",
            },
            {
                "id": 4,
                "headline": "Crude oil prices surge 3%",
                "source": "Reuters",
                "category": "COMMODITY",
                "sentiment": "BEARISH",
                "impact_score": 0.5,
                "timestamp": datetime.now().isoformat(),
                "summary": "Brent crude rises on supply concerns, may impact oil marketing companies.",
            },
            {
                "id": 5,
                "headline": "Banking sector shows mixed performance",
                "source": "Financial Express",
                "category": "SECTOR",
                "sentiment": "NEUTRAL",
                "impact_score": 0.4,
                "timestamp": datetime.now().isoformat(),
                "summary": "PSU banks outperform while private banks see profit booking.",
            },
        ],
        "market": {
            "nifty": 25332.3,
            "nifty_change": 0.15,
            "nifty_change_pct": 0.06,
            "banknifty": 60739.8,
            "banknifty_change": -45.2,
            "banknifty_change_pct": -0.07,
            "sensex": 81790.9,
            "sensex_change": 120.5,
            "sensex_change_pct": 0.15,
            "vix": 13.5,
            "vix_change": -0.3,
            "vix_change_pct": -2.17,
            "pcr_nifty": 0.57,
            "pcr_banknifty": 1.04,
            "pcr_sensex": 0.77,
            "pcr_finnifty": 1.06,
            "pcr_bankex": 4.50,
            "fii_5d_flow": -3466,
            "dii_5d_flow": 5032,
            "pe_ratio": 22.3,
            "gsec_yield": 7.08,
            "advances": 32,
            "declines": 18,
            "unchanged": 2,
        },
        "swing_setups": [],
        "action_items": [
            "No high-conviction setups — wait for better opportunities",
            "Verify all stop losses before market open",
            "Check overnight news affecting positions",
        ],
        "performance": {
            "win_rate_7d": 62.5,
            "win_rate_30d": 58.3,
            "total_trades": 24,
            "profit_factor": 1.85,
            "avg_win_pct": 3.2,
            "avg_loss_pct": -1.8,
        },
    }
    
    shared_state.update_from_engine(demo_data)
    return {"status": "demo_data_loaded", "timestamp": datetime.now().isoformat()}


# ============================================================================
# INTEGRATION HELPER FOR ROX ENGINE
# ============================================================================

def update_state(engine_data: Dict):
    """
    Call this function from your ROX engine (main.py) to update the shared state.
    
    Example usage in main.py:
    
        from api_server import update_state
        
        # After generating trading plan
        engine_data = {
            "status": "running",
            "regime": {"regime": "CONSOLIDATION", "confidence": 55.0, ...},
            "consensus": {"direction": "NEUTRAL", "strength": "NO_CONSENSUS", ...},
            "agents": [...],
            "positions": [...],
            "suggestions": [...],
            "news": [...],
            "portfolio": {...},
            "market": {...},
        }
        update_state(engine_data)
    """
    shared_state.update_from_engine(engine_data)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ROX Proven Edge Engine API Server")
    print("=" * 60)
    print()
    print("Starting API server on http://localhost:8000")
    print()
    print("Endpoints:")
    print("  GET /api/status      - Engine status")
    print("  GET /api/dashboard   - All dashboard data")
    print("  GET /api/regime      - Market regime")
    print("  GET /api/consensus   - Agent consensus")
    print("  GET /api/suggestions - Option suggestions")
    print("  GET /api/news        - News analysis")
    print()
    print("Demo mode:")
    print("  POST /api/demo/load  - Load demo data for testing")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

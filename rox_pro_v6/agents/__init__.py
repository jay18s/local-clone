"""
ROX Proven Edge Engine v5.0 Unified - Agents Package
=====================================================
11-Agent Framework:

Core Swing Agents (v3.2):
  ORION    - Technical Analysis
  VESPER   - Flow Analysis
  KAIRO    - Sentiment Analysis
  SENTINEL - Derivatives Analysis
  NEXUS    - Fundamental Analysis
  PRUDENCE - Risk Management
  CATALYST - Event Calendar
  OPTIMUS  - F&O Weekly Expiry

F&O Specialist Agents (v4.0 additions):
  HERMES   - Execution & Order Management
  THETA    - Greeks Management & Hedging
  DELTA    - Physical Settlement & Compliance
"""

from .base_agent import BaseAgent, AgentVerdict, AgentReport

# Core 8 agents (v3.2)
from .orion    import OrionAgent
from .vesper   import VesperAgent
from .kairo    import KairoAgent
from .sentinel import SentinelAgent
from .nexus    import NexusAgent
from .prudence import PrudenceAgent
from .catalyst import CatalystAgent
from .optimus  import OptimusAgent, OptionsSignal, OptionsStrategy, OptionType

# F&O specialist agents (v4.0)
from .hermes_agent import HermesAgent, Order, OrderStatus, OrderType, ExecutionMetrics
from .theta_agent  import ThetaAgent, PositionGreeks, GreeksAlert, GreeksAlertType
from .delta_agent  import DeltaAgent, SettlementObligation, SettlementType, SettlementStatus

# v4.3 — Pre-Momentum Recovery Radar
from .phoenix_agent import PhoenixAgent, PhoenixOutput, PhoenixSignal

__all__ = [
    # Base
    "BaseAgent", "AgentVerdict", "AgentReport",
    # Core agents
    "OrionAgent", "VesperAgent", "KairoAgent", "SentinelAgent",
    "NexusAgent", "PrudenceAgent", "CatalystAgent",
    "OptimusAgent", "OptionsSignal", "OptionsStrategy", "OptionType",
    # F&O specialists
    "HermesAgent", "Order", "OrderStatus", "OrderType", "ExecutionMetrics",
    "ThetaAgent", "PositionGreeks", "GreeksAlert", "GreeksAlertType",
    "DeltaAgent", "SettlementObligation", "SettlementType", "SettlementStatus",
    # v4.3
    "PhoenixAgent", "PhoenixOutput", "PhoenixSignal",
]

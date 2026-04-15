"""
ROX Proven Edge Engine v5.0 - Base Agent Class
=============================================
Foundation class for all trading agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from enum import Enum
import hashlib
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TradeDirection, MarketRegime, ConvictionLevel


# ---------------------------------------------------------------------------
# Enhancement 1: ReAct Pattern
# ---------------------------------------------------------------------------

class ReasoningStep(Enum):
    OBSERVE = "observe"
    THINK = "think"
    ACT = "act"
    REFLECT = "reflect"


@dataclass
class ReasoningTrace:
    step: ReasoningStep
    content: str
    data_used: List[str]
    confidence_delta: float


class ReActMixin:
    """
    ReAct reasoning mixin for dynamic agent intelligence.
    Adds iterative reasoning without changing agent structure.
    """

    def __init__(self):
        self.reasoning_traces: List[ReasoningTrace] = []
        self.max_reasoning_steps = 3

    def reason_act_loop(self,
                        initial_observation: str,
                        think_fn: Callable[[str, Dict], str],
                        act_fn: Callable[[str], Any],
                        reflect_fn: Callable[[Any, Dict], float],
                        data: Dict[str, Any]) -> Tuple[str, float, List[ReasoningTrace]]:
        """
        Execute ReAct loop: Observe → Think → Act → Reflect
        Returns: (final_reasoning, final_confidence, traces)
        """
        traces = []
        current_thought = initial_observation
        confidence = 50.0

        for step in range(self.max_reasoning_steps):
            thought = think_fn(current_thought, data)
            traces.append(ReasoningTrace(
                step=ReasoningStep.THINK,
                content=thought,
                data_used=self._extract_data_keys(thought, data),
                confidence_delta=0
            ))

            action_result = act_fn(thought)
            traces.append(ReasoningTrace(
                step=ReasoningStep.ACT,
                content=str(action_result),
                data_used=[],
                confidence_delta=0
            ))

            confidence_delta = reflect_fn(action_result, data)
            confidence += confidence_delta
            traces.append(ReasoningTrace(
                step=ReasoningStep.REFLECT,
                content=f"Confidence adjusted by {confidence_delta:+.1f}",
                data_used=[],
                confidence_delta=confidence_delta
            ))

            if abs(confidence_delta) < 5:
                break

            current_thought = f"Based on {action_result}, reconsidering: {thought}"

        self.reasoning_traces = traces
        return current_thought, max(0, min(100, confidence)), traces

    def _extract_data_keys(self, thought: str, data: Dict) -> List[str]:
        """Extract which data keys were referenced in reasoning."""
        return [k for k in data.keys() if k.lower() in thought.lower()]


# ---------------------------------------------------------------------------
# Enhancement 3: Intelligent Caching
# ---------------------------------------------------------------------------

class AgentCache:
    """Per-agent cache with market-regime awareness."""

    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[Any, float, MarketRegime]] = {}
        self.ttl = ttl_seconds
        self.cache_hits = 0
        self.cache_misses = 0

    def get(self, key: str, current_regime: MarketRegime) -> Optional[Any]:
        if key not in self.cache:
            self.cache_misses += 1
            return None
        value, timestamp, cached_regime = self.cache[key]
        if current_regime != cached_regime or time.time() - timestamp > self.ttl:
            del self.cache[key]
            self.cache_misses += 1
            return None
        self.cache_hits += 1
        return value

    def set(self, key: str, value: Any, regime: MarketRegime):
        self.cache[key] = (value, time.time(), regime)

    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0


# ---------------------------------------------------------------------------
# Enhancement 7: Incremental Analysis
# ---------------------------------------------------------------------------

class IncrementalAnalyzer:
    """Tracks data changes and only re-analyzes modified components."""

    def __init__(self):
        self.last_data_hash: Optional[str] = None
        self.last_result: Optional[Any] = None
        self.component_hashes: Dict[str, str] = {}

    def should_reanalyze(self, data: Dict, components: List[str]) -> Tuple[bool, List[str]]:
        """
        Check which components changed since last run.
        Returns: (needs_full_reanalysis, changed_components)
        """
        changed = []
        for component in components:
            current_hash = hashlib.md5(
                str(data.get(component, "")).encode()
            ).hexdigest()[:8]
            if self.component_hashes.get(component) != current_hash:
                changed.append(component)
                self.component_hashes[component] = current_hash
        needs_full = len(changed) > len(components) * 0.5
        return needs_full, changed


@dataclass
class AgentVerdict:
    """Structured output from an agent's analysis"""
    direction: TradeDirection
    conviction: float  # 0-100
    weight: float
    weighted_vote: float = 0.0
    reason: str = ""
    risks: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        # Calculate weighted vote
        direction_value = {
            TradeDirection.LONG: 1,
            TradeDirection.SHORT: -1,
            TradeDirection.NEUTRAL: 0
        }
        self.weighted_vote = direction_value[self.direction] * self.weight * (self.conviction / 100)


@dataclass
class AgentReport:
    """Comprehensive report from an agent"""
    agent_name: str
    verdict: AgentVerdict
    analysis_details: Dict[str, Any] = field(default_factory=dict)
    key_observations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary for serialization"""
        return {
            "agent_name": self.agent_name,
            "verdict": {
                "direction": self.verdict.direction.value,
                "conviction": self.verdict.conviction,
                "weight": self.verdict.weight,
                "weighted_vote": self.verdict.weighted_vote,
                "reason": self.verdict.reason,
                "risks": self.verdict.risks,
                "timestamp": self.verdict.timestamp.isoformat()
            },
            "analysis_details": self.analysis_details,
            "key_observations": self.key_observations,
            "metrics": self.metrics,
            "raw_data": self.raw_data
        }


class BaseAgent(ReActMixin, ABC):
    """
    Abstract base class for all trading agents.

    Each agent specializes in a specific domain and produces
    a structured verdict with direction, conviction, and reasoning.

    Enhanced with:
      - ReAct reasoning loops (Enhancement 1)
      - Intelligent caching (Enhancement 3)
      - Incremental analysis tracking (Enhancement 7)
      - Self-reflection & auto-calibration (Enhancement 8)
    """

    def __init__(self, name: str, domain: str, baseline_weight: float):
        # ReActMixin init
        ReActMixin.__init__(self)

        self.name = name
        self.domain = domain
        self.baseline_weight = baseline_weight
        self.current_weight = baseline_weight
        self.accuracy_by_regime: Dict[MarketRegime, List[bool]] = {
            regime: [] for regime in MarketRegime
        }
        self.recent_predictions: List[Dict] = []

        # Enhancement 3: Caching
        self.cache = AgentCache()
        self.computation_stats = {
            "cached_calls": 0,
            "computed_calls": 0,
            "time_saved_ms": 0
        }

        # Enhancement 7: Incremental analysis
        self.incremental = IncrementalAnalyzer()

        # Enhancement 8: Confidence calibration multiplier
        self.confidence_calibration: float = 1.0
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any], regime: MarketRegime) -> AgentReport:
        """
        Perform domain-specific analysis and return a structured report.
        
        Args:
            data: Dictionary containing all relevant market data
            regime: Current market regime
            
        Returns:
            AgentReport with verdict and analysis details
        """
        pass
    
    def update_weight(self, regime: MarketRegime, performance_bonus: float = 0.0):
        """
        Update agent weight based on recent performance in current regime.
        
        Args:
            regime: Current market regime
            performance_bonus: Additional adjustment from meta-learning
        """
        accuracy = self.get_accuracy(regime)
        
        # Weight adjustment rules
        adjustment = 0.0
        if accuracy > 0.75:
            adjustment = 0.05  # +5% for excellent performance
        elif accuracy > 0.65:
            adjustment = 0.03  # +3% for good performance
        elif accuracy < 0.45:
            adjustment = -0.03  # -3% for poor performance
        
        adjustment += performance_bonus
        
        # Apply adjustment with bounds
        self.current_weight = max(0.05, min(0.30, self.baseline_weight + adjustment))
    
    def get_accuracy(self, regime: Optional[MarketRegime] = None) -> float:
        """
        Calculate agent accuracy, optionally filtered by regime.
        
        Args:
            regime: Optional regime to filter by
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if regime:
            predictions = self.accuracy_by_regime.get(regime, [])
        else:
            predictions = []
            for regime_predictions in self.accuracy_by_regime.values():
                predictions.extend(regime_predictions)
        
        if not predictions:
            return 0.5  # Default to 50% if no data
        
        return sum(predictions) / len(predictions)
    
    def record_prediction(self, prediction: Dict, correct: bool, regime: MarketRegime):
        """
        Record a prediction outcome for learning.
        
        Args:
            prediction: The prediction details
            correct: Whether the prediction was correct
            regime: Market regime at time of prediction
        """
        self.accuracy_by_regime[regime].append(correct)
        self.recent_predictions.append({
            **prediction,
            "correct": correct,
            "regime": regime.value,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 predictions
        if len(self.recent_predictions) > 100:
            self.recent_predictions = self.recent_predictions[-100:]
    
    def get_verdict(self, data: Dict[str, Any], regime: MarketRegime) -> AgentVerdict:
        """
        Get a verdict from this agent.
        
        This method calls analyze() and extracts the verdict.
        
        Args:
            data: Market data for analysis
            regime: Current market regime
            
        Returns:
            AgentVerdict with direction, conviction, and reasoning
        """
        report = self.analyze(data, regime)
        report.verdict.weight = self.current_weight
        report.verdict.__post_init__()  # Recalculate weighted vote
        return report.verdict
    
    def reset_weights(self):
        """Reset agent weight to baseline."""
        self.current_weight = self.baseline_weight

    # ------------------------------------------------------------------
    # Enhancement 3: Intelligent Caching
    # ------------------------------------------------------------------

    def cached_analyze(self, data: Dict[str, Any], regime: MarketRegime) -> "AgentReport":
        """
        Wrapper that caches results when market conditions haven't changed.
        Calls self.analyze() on a cache miss.
        """
        cacheable_data = {k: v for k, v in data.items()
                         if k not in ('timestamp', 'current_time')}
        key = hashlib.md5(
            f"{self.name}:{str(cacheable_data)}".encode()
        ).hexdigest()[:16]

        cached = self.cache.get(key, regime)
        if cached is not None:
            self.computation_stats["cached_calls"] += 1
            cached.verdict.timestamp = datetime.now()
            return cached

        self.computation_stats["computed_calls"] += 1
        start = time.time()
        result = self.analyze(data, regime)
        elapsed = (time.time() - start) * 1000
        self.computation_stats["time_saved_ms"] += elapsed  # track avg compute time

        self.cache.set(key, result, regime)
        return result

    # ------------------------------------------------------------------
    # Enhancement 8: Self-Reflection & Learning
    # ------------------------------------------------------------------

    def reflect_on_prediction(self,
                              prediction: "AgentVerdict",
                              actual_outcome: Dict) -> Dict:
        """
        Post-trade reflection to improve future reasoning.
        Called by coordinator after trade closes.
        """
        reflection = {
            "agent": self.name,
            "prediction_direction": prediction.direction.value,
            "actual_direction": actual_outcome.get("direction"),
            "prediction_conviction": prediction.conviction,
            "actual_return": actual_outcome.get("return_pct", 0),
            "errors": [],
            "lessons": []
        }

        if prediction.direction.value != actual_outcome.get("direction"):
            reflection["errors"].append("direction_wrong")
            if prediction.conviction > 80:
                reflection["lessons"].append(
                    "High conviction did not predict direction – review signals"
                )

        if prediction.conviction > 70 and actual_outcome.get("return_pct", 0) < 0:
            reflection["errors"].append("overconfident")
            reflection["lessons"].append(
                "Conviction >70 but negative return – reduce confidence calibration"
            )

        if prediction.conviction < 50 and actual_outcome.get("return_pct", 0) > 5:
            reflection["errors"].append("underconfident")
            reflection["lessons"].append(
                "Missed opportunity due to low confidence"
            )

        self._update_calibration(reflection)
        return reflection

    def _update_calibration(self, reflection: Dict):
        """Adjust future confidence calculations based on errors."""
        for error in reflection.get("errors", []):
            if error == "overconfident":
                self.confidence_calibration *= 0.95
            elif error == "underconfident":
                self.confidence_calibration = min(1.0, self.confidence_calibration * 1.05)
        # Clamp to reasonable range
        self.confidence_calibration = max(0.5, min(1.5, self.confidence_calibration))

    def __repr__(self) -> str:
        return (f"{self.name}({self.domain}, weight={self.current_weight:.1%}, "
                f"calib={self.confidence_calibration:.2f})")

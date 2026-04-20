"""ROX Engine v5.0 — Adaptive Prompt Selector & Regime Cache"""
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from .data_classes import ComplexityLevel

logger = logging.getLogger("reasoning.adaptive")

# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE PROMPT SELECTOR
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveConfig:
    """Runtime configuration based on market complexity."""
    complexity: ComplexityLevel
    model_tier: str          # "pro" or "flash"
    debate_rounds: int
    cot_steps: int
    pattern_match_count: int
    confidence_threshold: float
    max_position_pct: float
    cross_examination: bool
    meta_learner_enabled: bool
    reasoning_steps: int


class AdaptivePromptSelector:
    """
    Dynamically adjusts analysis depth and model selection
    based on current market complexity.
    """
    
    def assess_complexity(
        self,
        vix: float,
        intraday_range_pct: float = 0,
        macro_event_count: int = 0,
        fii_streak_days: int = 0,
        key_level_test: str = "NONE",  # "NONE", "200DMA", "ATH"
        sector_dispersion: float = 0.0,
    ) -> AdaptiveConfig:
        """Assess market complexity and return optimal configuration."""
        
        complexity_score = 0
        
        # VIX contribution (heaviest weight)
        if vix >= 25:
            complexity_score += 4
        elif vix >= 18:
            complexity_score += 3
        elif vix >= 14:
            complexity_score += 1
        # Below 14 = LOW
        
        # Intraday range
        if intraday_range_pct >= 2.5:
            complexity_score += 2
        elif intraday_range_pct >= 1.2:
            complexity_score += 1
        
        # Macro events
        if macro_event_count >= 3:
            complexity_score += 2
        elif macro_event_count >= 1:
            complexity_score += 1
        
        # FII streak
        if abs(fii_streak_days) >= 7:
            complexity_score += 1
        elif abs(fii_streak_days) >= 5:
            complexity_score += 0.5
        
        # Key level tests
        if key_level_test in ("200DMA", "ATH"):
            complexity_score += 2
        
        # Sector dispersion
        if sector_dispersion >= 3.0:
            complexity_score += 1
        
        # Determine level
        if complexity_score >= 7:
            level = ComplexityLevel.EXTREME
        elif complexity_score >= 4:
            level = ComplexityLevel.HIGH
        elif complexity_score >= 2:
            level = ComplexityLevel.MEDIUM
        else:
            level = ComplexityLevel.LOW
        
        return self._build_config(level)
    
    def _build_config(self, level: ComplexityLevel) -> AdaptiveConfig:
        """Build configuration for a complexity level."""
        configs = {
            ComplexityLevel.LOW: AdaptiveConfig(
                complexity=ComplexityLevel.LOW,
                model_tier="flash",
                debate_rounds=0,
                cot_steps=3,
                pattern_match_count=2,
                confidence_threshold=70.0,
                max_position_pct=1.0,
                cross_examination=False,
                meta_learner_enabled=False,
                reasoning_steps=3,
            ),
            ComplexityLevel.MEDIUM: AdaptiveConfig(
                complexity=ComplexityLevel.MEDIUM,
                model_tier="pro",
                debate_rounds=1,
                cot_steps=5,
                pattern_match_count=3,
                confidence_threshold=60.0,
                max_position_pct=1.0,
                cross_examination=True,
                meta_learner_enabled=False,
                reasoning_steps=5,
            ),
            ComplexityLevel.HIGH: AdaptiveConfig(
                complexity=ComplexityLevel.HIGH,
                model_tier="pro",
                debate_rounds=2,
                cot_steps=7,
                pattern_match_count=5,
                confidence_threshold=55.0,
                max_position_pct=0.75,
                cross_examination=True,
                meta_learner_enabled=False,
                reasoning_steps=7,
            ),
            ComplexityLevel.EXTREME: AdaptiveConfig(
                complexity=ComplexityLevel.EXTREME,
                model_tier="pro",
                debate_rounds=3,
                cot_steps=7,
                pattern_match_count=5,
                confidence_threshold=50.0,
                max_position_pct=0.5,
                cross_examination=True,
                meta_learner_enabled=False,
                reasoning_steps=7,
            ),
        }
        
        return configs.get(level, configs[ComplexityLevel.MEDIUM])
    
    def get_model(self, config: AdaptiveConfig, module: str) -> str:
        """Select the appropriate model for a module."""
        # Pro-model modules
        if config.model_tier == "pro":
            return "gemini-3-flash-preview"  # Default pro
        
        # Flash-model modules (LOW complexity)
        if module in ("regime_detector", "cross_examiner"):
            return "gemini-3-flash-preview"  # These always need quality
        if module in ("trading_planner", "fno_brain"):
            return "gemini-3-flash-preview"  # Important decisions
        
        return "gemini-2.0-flash"


# ═══════════════════════════════════════════════════════════════════
# REGIME CACHE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CachedRegime:
    """Cached regime result with metadata."""
    regime: str
    confidence: float
    reasoning: dict
    cached_at: datetime
    ttl_seconds: int
    vix_at_cache: float
    nifty_at_cache: float


class RegimeCache:
    """
    Cache regime detection results to avoid redundant LLM calls.
    Only serves cached results when confidence is above threshold.
    """
    
    def __init__(
        self,
        ttl_high_conf: int = 900,    # 15 min
        ttl_med_conf: int = 420,     # 7 min
        min_confidence: float = 50.0,
        invalidate_vix_delta: float = 2.0,
        invalidate_dma_break: bool = True,
    ):
        self.ttl_high_conf = ttl_high_conf
        self.ttl_med_conf = ttl_med_conf
        self.min_confidence = min_confidence
        self.invalidate_vix_delta = invalidate_vix_delta
        self.invalidate_dma_break = invalidate_dma_break
        self._cached: Optional[CachedRegime] = None
    
    @property
    def is_cached(self) -> bool:
        """Check if a valid cache exists."""
        if self._cached is None:
            return False
        
        elapsed = (datetime.now() - self._cached.cached_at).total_seconds()
        if elapsed >= self._cached.ttl_seconds:
            return False
        
        return True
    
    def get(self, current_vix: float, current_nifty: float) -> Optional[CachedRegime]:
        """Get cached regime if still valid, otherwise None."""
        if not self.is_cached:
            return None
        
        # Check invalidation conditions
        vix_delta = abs(current_vix - self._cached.vix_at_cache)
        if vix_delta >= self.invalidate_vix_delta:
            logger.info(
                f"Regime cache INVALIDATED: VIX moved {vix_delta:.1f} "
                f"(threshold: {self.invalidate_vix_delta})"
            )
            self._cached = None
            return None
        
        if self.invalidate_dma_break:
            nifty_delta = abs(current_nifty - self._cached.nifty_at_cache)
            if nifty_delta > 200:  # 200-point move invalidates
                logger.info(
                    f"Regime cache INVALIDATED: Nifty moved {nifty_delta:.0f}pts "
                    f"since cache"
                )
                self._cached = None
                return None
        
        elapsed = (datetime.now() - self._cached.cached_at).total_seconds()
        logger.info(
            f"Regime CACHE HIT: {self._cached.regime} "
            f"(conf: {self._cached.confidence}%, "
            f"age: {elapsed/60:.1f}m, "
            f"TTL: {self._cached.ttl_seconds/60:.1f}m)"
        )
        return self._cached
    
    def set(self, regime: str, confidence: float, reasoning: dict,
              vix: float, nifty: float):
        """Cache a regime result."""
        if confidence >= self.min_confidence:
            if confidence >= 75:
                ttl = self.ttl_high_conf
            else:
                ttl = self.ttl_med_conf
        else:
            ttl = 0  # Don't cache low-confidence results
        
        self._cached = CachedRegime(
            regime=regime,
            confidence=confidence,
            reasoning=reasoning,
            cached_at=datetime.now(),
            ttl_seconds=ttl,
            vix_at_cache=vix,
            nifty_at_cache=nifty,
        )
        
        logger.info(
            f"Regime CACHED: {regime} (conf: {confidence}%, TTL: {ttl/60:.1f}m)"
        )
    
    def invalidate(self):
        """Force invalidate the cache."""
        self._cached = None
        logger.info("Regime cache manually invalidated")

"""
ROX Proven Edge Engine v3.0 - Pattern Database
=============================================
Historical pattern storage and matching system.
"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from .data_manager import DataManager


@dataclass
class TradingPattern:
    """Historical trading pattern"""
    trade_id: str
    date: str
    stock: str
    direction: str
    setup_type: str
    technical_score: float
    flow_score: float
    sentiment_score: float
    event_context: str
    regime: str
    entry: float
    stop_loss: float
    target: float
    outcome: str  # TARGET_HIT, STOP_HIT, PARTIAL, MANUAL_EXIT
    days_to_outcome: int
    return_pct: float
    similarity_tags: List[str]


class PatternDatabase:
    """
    Pattern database for historical setup matching.
    
    Features:
    - Store completed trade patterns
    - Search for similar setups
    - Calculate historical win rates
    - Update pattern outcomes
    """
    
    # Similarity weights for matching
    SIMILARITY_WEIGHTS = {
        "setup_type": 0.40,
        "regime": 0.20,
        "flow_direction": 0.15,
        "sentiment_zone": 0.15,
        "event_context": 0.10
    }
    
    def __init__(self, data_manager: DataManager = None):
        self.data_manager = data_manager or DataManager()
        self.logger = logging.getLogger("PatternDatabase")
        self._patterns_cache = None
    
    def _get_patterns_file(self) -> str:
        """Get path to patterns database file"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(
            base_dir, "data", "Market_Trends", 
            "Historical_Patterns", "patterns_database.json"
        )
    
    def load_patterns(self, force_reload: bool = False) -> List[Dict]:
        """Load all patterns from database"""
        if self._patterns_cache and not force_reload:
            return self._patterns_cache
        
        patterns_file = self._get_patterns_file()
        
        if not os.path.exists(patterns_file):
            self._patterns_cache = []
            return []
        
        try:
            with open(patterns_file, 'r') as f:
                self._patterns_cache = json.load(f)
            return self._patterns_cache
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
            return []
    
    def save_pattern(self, pattern: TradingPattern) -> bool:
        """Save a new pattern to the database"""
        patterns_file = self._get_patterns_file()
        
        try:
            # Load existing patterns
            patterns = self.load_patterns()
            
            # Convert to dict and add
            pattern_dict = asdict(pattern)
            if 'trade_id' not in pattern_dict or not pattern_dict['trade_id']:
                pattern_dict['trade_id'] = f"P{len(patterns) + 1:04d}"
            
            pattern_dict['date_added'] = datetime.now().isoformat()
            patterns.append(pattern_dict)
            
            # Save back
            os.makedirs(os.path.dirname(patterns_file), exist_ok=True)
            with open(patterns_file, 'w') as f:
                json.dump(patterns, f, indent=2)
            
            # Clear cache
            self._patterns_cache = None
            
            self.logger.info(f"Saved pattern: {pattern.setup_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving pattern: {e}")
            return False
    
    def search_similar_patterns(self, criteria: Dict, 
                                min_similarity: float = 0.5,
                                limit: int = 10) -> List[Dict]:
        """
        Search for patterns similar to the given criteria.
        
        Args:
            criteria: Dict with setup_type, regime, flow_direction, etc.
            min_similarity: Minimum similarity score (0-1)
            limit: Maximum number of results
            
        Returns:
            List of matching patterns with similarity scores
        """
        patterns = self.load_patterns()
        
        matches = []
        for pattern in patterns:
            similarity = self._calculate_similarity(pattern, criteria)
            
            if similarity >= min_similarity:
                matches.append({
                    **pattern,
                    "similarity_score": similarity
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return matches[:limit]
    
    def _calculate_similarity(self, pattern: Dict, criteria: Dict) -> float:
        """Calculate similarity score between pattern and criteria"""
        total_score = 0.0
        
        # Setup type match (most important)
        if pattern.get('setup_type') == criteria.get('setup_type'):
            total_score += self.SIMILARITY_WEIGHTS['setup_type']
        elif pattern.get('setup_type') and criteria.get('setup_type'):
            # Partial match for similar setup types
            if self._are_similar_setups(pattern['setup_type'], criteria['setup_type']):
                total_score += self.SIMILARITY_WEIGHTS['setup_type'] * 0.5
        
        # Regime match
        if pattern.get('regime') == criteria.get('regime'):
            total_score += self.SIMILARITY_WEIGHTS['regime']
        
        # Flow direction match
        pattern_flow = self._get_flow_direction(pattern.get('flow_score', 0))
        criteria_flow = self._get_flow_direction(criteria.get('flow_score', 0))
        if pattern_flow == criteria_flow:
            total_score += self.SIMILARITY_WEIGHTS['flow_direction']
        
        # Sentiment zone match
        pattern_sentiment = self._get_sentiment_zone(pattern.get('sentiment_score', 0))
        criteria_sentiment = self._get_sentiment_zone(criteria.get('sentiment_score', 0))
        if pattern_sentiment == criteria_sentiment:
            total_score += self.SIMILARITY_WEIGHTS['sentiment_zone']
        
        # Event context match
        if pattern.get('event_context') == criteria.get('event_context'):
            total_score += self.SIMILARITY_WEIGHTS['event_context']
        
        return total_score
    
    def _are_similar_setups(self, setup1: str, setup2: str) -> bool:
        """Check if two setup types are similar"""
        similar_groups = [
            {'pullback_to_50dma', 'pullback_to_200dma', 'pullback_to_support'},
            {'breakout_resistance', 'breakout_consolidation', 'breakout_pattern'},
            {'double_bottom', 'inverse_head_shoulders', 'reversal_pattern'},
            {'double_top', 'head_shoulders', 'reversal_pattern'}
        ]
        
        for group in similar_groups:
            if setup1 in group and setup2 in group:
                return True
        return False
    
    def _get_flow_direction(self, flow_score: float) -> str:
        """Convert flow score to direction"""
        if flow_score > 60:
            return "strong_buying"
        elif flow_score > 40:
            return "moderate_buying"
        elif flow_score < -60:
            return "strong_selling"
        elif flow_score < -40:
            return "moderate_selling"
        else:
            return "neutral"
    
    def _get_sentiment_zone(self, sentiment_score: float) -> str:
        """Convert sentiment score to zone"""
        if sentiment_score > 70:
            return "euphoria"
        elif sentiment_score > 40:
            return "bullish"
        elif sentiment_score < -70:
            return "panic"
        elif sentiment_score < -40:
            return "bearish"
        else:
            return "neutral"
    
    def calculate_historical_win_rate(self, setup_type: str = None,
                                      regime: str = None) -> Dict:
        """
        Calculate historical win rate for a setup type.
        
        Args:
            setup_type: Optional setup type filter
            regime: Optional regime filter
            
        Returns:
            Dict with win_rate, total_trades, avg_return
        """
        patterns = self.load_patterns()
        
        # Filter patterns
        if setup_type:
            patterns = [p for p in patterns if p.get('setup_type') == setup_type]
        if regime:
            patterns = [p for p in patterns if p.get('regime') == regime]
        
        if not patterns:
            return {
                "win_rate": 0.5,
                "total_trades": 0,
                "avg_return": 0,
                "avg_days": 0
            }
        
        # Calculate statistics
        wins = sum(1 for p in patterns if p.get('return_pct', 0) > 0)
        total_return = sum(p.get('return_pct', 0) for p in patterns)
        total_days = sum(p.get('days_to_outcome', 0) for p in patterns)
        
        return {
            "win_rate": wins / len(patterns),
            "total_trades": len(patterns),
            "wins": wins,
            "losses": len(patterns) - wins,
            "avg_return": total_return / len(patterns),
            "total_return": total_return,
            "avg_days": total_days / len(patterns) if patterns else 0
        }
    
    def get_best_setups(self, regime: str = None, limit: int = 5) -> List[Dict]:
        """Get the historically best performing setups"""
        patterns = self.load_patterns()
        
        # Group by setup type
        setup_stats = {}
        for pattern in patterns:
            setup_type = pattern.get('setup_type', 'unknown')
            
            if setup_type not in setup_stats:
                setup_stats[setup_type] = {
                    "setup_type": setup_type,
                    "wins": 0,
                    "total": 0,
                    "total_return": 0
                }
            
            setup_stats[setup_type]["total"] += 1
            if pattern.get('return_pct', 0) > 0:
                setup_stats[setup_type]["wins"] += 1
            setup_stats[setup_type]["total_return"] += pattern.get('return_pct', 0)
        
        # Calculate win rates
        results = []
        for setup_type, stats in setup_stats.items():
            results.append({
                "setup_type": setup_type,
                "win_rate": stats["wins"] / stats["total"] if stats["total"] > 0 else 0,
                "total_trades": stats["total"],
                "avg_return": stats["total_return"] / stats["total"] if stats["total"] > 0 else 0
            })
        
        # Sort by win rate
        results.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return results[:limit]
    
    def get_setup_statistics(self, setup_type: str) -> Dict:
        """Get detailed statistics for a specific setup type"""
        patterns = self.load_patterns()
        
        matching = [p for p in patterns if p.get('setup_type') == setup_type]
        
        if not matching:
            return {
                "setup_type": setup_type,
                "total_trades": 0,
                "win_rate": 0,
                "avg_return": 0,
                "avg_days": 0,
                "max_win": 0,
                "max_loss": 0,
                "consecutive_wins": 0,
                "consecutive_losses": 0
            }
        
        returns = [p.get('return_pct', 0) for p in matching]
        days = [p.get('days_to_outcome', 0) for p in matching]
        
        # Calculate consecutive streaks
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        
        for r in sorted(returns, reverse=True):
            if r > 0:
                current_wins += 1
                max_consec_wins = max(max_consec_wins, current_wins)
                current_losses = 0
            else:
                current_losses += 1
                max_consec_losses = max(max_consec_losses, current_losses)
                current_wins = 0
        
        return {
            "setup_type": setup_type,
            "total_trades": len(matching),
            "win_rate": sum(1 for r in returns if r > 0) / len(returns),
            "avg_return": sum(returns) / len(returns),
            "avg_days": sum(days) / len(days) if days else 0,
            "max_win": max(returns) if returns else 0,
            "max_loss": min(returns) if returns else 0,
            "consecutive_wins": max_consec_wins,
            "consecutive_losses": max_consec_losses
        }
    
    def export_patterns(self, output_path: str) -> bool:
        """Export patterns to a file"""
        patterns = self.load_patterns()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(patterns, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error exporting patterns: {e}")
            return False
    
    def import_patterns(self, input_path: str, merge: bool = True) -> bool:
        """Import patterns from a file"""
        try:
            with open(input_path, 'r') as f:
                new_patterns = json.load(f)
            
            if merge:
                existing = self.load_patterns()
                existing_ids = {p.get('trade_id') for p in existing}
                
                for pattern in new_patterns:
                    if pattern.get('trade_id') not in existing_ids:
                        existing.append(pattern)
                
                # Save merged
                patterns_file = self._get_patterns_file()
                with open(patterns_file, 'w') as f:
                    json.dump(existing, f, indent=2)
            else:
                # Replace
                patterns_file = self._get_patterns_file()
                with open(patterns_file, 'w') as f:
                    json.dump(new_patterns, f, indent=2)
            
            self._patterns_cache = None
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing patterns: {e}")
            return False

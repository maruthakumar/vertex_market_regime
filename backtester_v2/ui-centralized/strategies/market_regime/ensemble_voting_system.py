#!/usr/bin/env python3
"""
Ensemble Voting System for Enhanced Market Regime Framework V2.0

This module implements sophisticated ensemble voting algorithms that combine
multiple regime detection methods to improve accuracy and robustness.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import Sentry configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.sentry_config import capture_exception, add_breadcrumb, set_tag, track_errors, capture_message
except ImportError:
    # Fallback if sentry not available
    def capture_exception(*args, **kwargs): pass
    def add_breadcrumb(*args, **kwargs): pass
    def set_tag(*args, **kwargs): pass
    def track_errors(func): return func
    def capture_message(*args, **kwargs): pass

logger = logging.getLogger(__name__)

class VotingMethod(Enum):
    """Ensemble voting methods"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    BAYESIAN_ENSEMBLE = "bayesian_ensemble"

@dataclass
class VoterResult:
    """Result from individual voter"""
    voter_id: str
    regime_prediction: Any
    confidence: float
    weight: float
    performance_score: float
    metadata: Dict[str, Any]

@dataclass
class EnsembleVotingConfig:
    """Configuration for ensemble voting"""
    voting_method: VotingMethod = VotingMethod.ADAPTIVE_WEIGHTED
    min_voters: int = 3
    confidence_threshold: float = 0.6
    agreement_threshold: float = 0.7
    enable_voter_weighting: bool = True
    enable_performance_tracking: bool = True
    enable_adaptive_weights: bool = True
    weight_decay: float = 0.95
    performance_window: int = 50

class EnsembleVotingSystem:
    """
    Ensemble Voting System for Market Regime Formation
    
    Implements sophisticated ensemble voting algorithms including:
    - Weighted voting based on historical performance
    - Confidence-based voting
    - Adaptive weight adjustment
    - Bayesian ensemble methods
    - Voter performance tracking
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 config: Optional[EnsembleVotingConfig] = None):
        """
        Initialize Ensemble Voting System
        
        Args:
            weights: Initial voter weights
            config: Ensemble voting configuration
        """
        set_tag("component", "ensemble_voting_system")
        
        self.config = config or EnsembleVotingConfig()
        self.voter_weights = weights or {}
        
        # Voter performance tracking
        self.voter_performance = {}
        self.voting_history = []
        self.performance_history = {}
        
        # Adaptive learning
        self.weight_adaptation_history = []
        self.ensemble_performance = []
        
        logger.info(f"Ensemble Voting System initialized with {self.config.voting_method.value}")
        add_breadcrumb(
            message="Ensemble Voting System initialized",
            category="initialization",
            data={"voting_method": self.config.voting_method.value}
        )
    
    @track_errors
    async def vote(self, voting_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform ensemble voting across multiple methods
        
        Args:
            voting_inputs: Inputs from different voting methods
            
        Returns:
            Dict: Ensemble voting result
        """
        set_tag("operation", "ensemble_voting")
        
        try:
            # Prepare voter results
            voter_results = self._prepare_voter_results(voting_inputs)
            
            if len(voter_results) < self.config.min_voters:
                logger.warning(f"Insufficient voters: {len(voter_results)} < {self.config.min_voters}")
                return self._fallback_result(voter_results)
            
            # Perform voting based on method
            if self.config.voting_method == VotingMethod.MAJORITY_VOTE:
                result = await self._majority_vote(voter_results)
            elif self.config.voting_method == VotingMethod.WEIGHTED_VOTE:
                result = await self._weighted_vote(voter_results)
            elif self.config.voting_method == VotingMethod.CONFIDENCE_WEIGHTED:
                result = await self._confidence_weighted_vote(voter_results)
            elif self.config.voting_method == VotingMethod.PERFORMANCE_WEIGHTED:
                result = await self._performance_weighted_vote(voter_results)
            elif self.config.voting_method == VotingMethod.ADAPTIVE_WEIGHTED:
                result = await self._adaptive_weighted_vote(voter_results)
            elif self.config.voting_method == VotingMethod.BAYESIAN_ENSEMBLE:
                result = await self._bayesian_ensemble_vote(voter_results)
            else:
                result = await self._weighted_vote(voter_results)  # Default
            
            # Calculate ensemble metrics
            result['agreement_score'] = self._calculate_agreement_score(voter_results, result)
            result['ensemble_confidence'] = self._calculate_ensemble_confidence(voter_results, result)
            result['voter_count'] = len(voter_results)
            
            # Store voting result
            self._store_voting_result(voter_results, result)
            
            add_breadcrumb(
                message="Ensemble voting completed",
                category="ensemble_voting",
                data={
                    "voters": len(voter_results),
                    "agreement": result.get('agreement_score', 0),
                    "confidence": result.get('ensemble_confidence', 0)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble voting: {e}")
            capture_exception(e, component="ensemble_voting")
            return self._fallback_result([])
    
    @track_errors
    async def _majority_vote(self, voter_results: List[VoterResult]) -> Dict[str, Any]:
        """Perform majority voting"""
        try:
            # Count votes for each regime
            regime_votes = Counter([voter.regime_prediction for voter in voter_results])
            
            # Get majority regime
            majority_regime = regime_votes.most_common(1)[0][0]
            vote_count = regime_votes[majority_regime]
            
            # Calculate confidence based on vote proportion
            confidence = vote_count / len(voter_results)
            
            return {
                'regime_type': majority_regime,
                'confidence_score': confidence,
                'voting_method': 'majority_vote',
                'vote_distribution': dict(regime_votes),
                'total_votes': len(voter_results)
            }
            
        except Exception as e:
            logger.error(f"Error in majority voting: {e}")
            capture_exception(e, component="majority_vote")
            return self._fallback_result(voter_results)
    
    @track_errors
    async def _weighted_vote(self, voter_results: List[VoterResult]) -> Dict[str, Any]:
        """Perform weighted voting"""
        try:
            # Calculate weighted votes
            regime_weights = {}
            total_weight = 0
            
            for voter in voter_results:
                regime = voter.regime_prediction
                weight = voter.weight
                
                if regime not in regime_weights:
                    regime_weights[regime] = 0
                
                regime_weights[regime] += weight
                total_weight += weight
            
            # Normalize weights
            if total_weight > 0:
                for regime in regime_weights:
                    regime_weights[regime] /= total_weight
            
            # Get regime with highest weight
            winning_regime = max(regime_weights, key=regime_weights.get)
            confidence = regime_weights[winning_regime]
            
            return {
                'regime_type': winning_regime,
                'confidence_score': confidence,
                'voting_method': 'weighted_vote',
                'weight_distribution': regime_weights,
                'total_weight': total_weight
            }
            
        except Exception as e:
            logger.error(f"Error in weighted voting: {e}")
            capture_exception(e, component="weighted_vote")
            return self._fallback_result(voter_results)
    
    @track_errors
    async def _confidence_weighted_vote(self, voter_results: List[VoterResult]) -> Dict[str, Any]:
        """Perform confidence-weighted voting"""
        try:
            # Weight votes by confidence
            regime_scores = {}
            total_confidence = 0
            
            for voter in voter_results:
                regime = voter.regime_prediction
                confidence = voter.confidence
                
                if regime not in regime_scores:
                    regime_scores[regime] = 0
                
                regime_scores[regime] += confidence
                total_confidence += confidence
            
            # Normalize scores
            if total_confidence > 0:
                for regime in regime_scores:
                    regime_scores[regime] /= total_confidence
            
            # Get regime with highest score
            winning_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[winning_regime]
            
            return {
                'regime_type': winning_regime,
                'confidence_score': confidence,
                'voting_method': 'confidence_weighted_vote',
                'confidence_distribution': regime_scores,
                'total_confidence': total_confidence
            }
            
        except Exception as e:
            logger.error(f"Error in confidence-weighted voting: {e}")
            capture_exception(e, component="confidence_weighted_vote")
            return self._fallback_result(voter_results)
    
    @track_errors
    async def _performance_weighted_vote(self, voter_results: List[VoterResult]) -> Dict[str, Any]:
        """Perform performance-weighted voting"""
        try:
            # Weight votes by historical performance
            regime_scores = {}
            total_performance = 0
            
            for voter in voter_results:
                regime = voter.regime_prediction
                performance = voter.performance_score
                
                if regime not in regime_scores:
                    regime_scores[regime] = 0
                
                regime_scores[regime] += performance
                total_performance += performance
            
            # Normalize scores
            if total_performance > 0:
                for regime in regime_scores:
                    regime_scores[regime] /= total_performance
            
            # Get regime with highest score
            winning_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[winning_regime]
            
            return {
                'regime_type': winning_regime,
                'confidence_score': confidence,
                'voting_method': 'performance_weighted_vote',
                'performance_distribution': regime_scores,
                'total_performance': total_performance
            }
            
        except Exception as e:
            logger.error(f"Error in performance-weighted voting: {e}")
            capture_exception(e, component="performance_weighted_vote")
            return self._fallback_result(voter_results)
    
    @track_errors
    async def _adaptive_weighted_vote(self, voter_results: List[VoterResult]) -> Dict[str, Any]:
        """Perform adaptive weighted voting"""
        try:
            # Combine confidence and performance weights adaptively
            regime_scores = {}
            total_score = 0
            
            for voter in voter_results:
                regime = voter.regime_prediction
                confidence = voter.confidence
                performance = voter.performance_score
                base_weight = voter.weight
                
                # Adaptive weight calculation
                adaptive_weight = self._calculate_adaptive_weight(
                    base_weight, confidence, performance, voter.voter_id
                )
                
                if regime not in regime_scores:
                    regime_scores[regime] = 0
                
                regime_scores[regime] += adaptive_weight
                total_score += adaptive_weight
            
            # Normalize scores
            if total_score > 0:
                for regime in regime_scores:
                    regime_scores[regime] /= total_score
            
            # Get regime with highest score
            winning_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[winning_regime]
            
            return {
                'regime_type': winning_regime,
                'confidence_score': confidence,
                'voting_method': 'adaptive_weighted_vote',
                'adaptive_distribution': regime_scores,
                'total_adaptive_score': total_score
            }
            
        except Exception as e:
            logger.error(f"Error in adaptive weighted voting: {e}")
            capture_exception(e, component="adaptive_weighted_vote")
            return self._fallback_result(voter_results)
    
    def _prepare_voter_results(self, voting_inputs: Dict[str, Any]) -> List[VoterResult]:
        """Prepare voter results from voting inputs"""
        try:
            voter_results = []
            
            for voter_id, voter_data in voting_inputs.items():
                if not voter_data:
                    continue
                
                # Extract voter information
                regime_prediction = voter_data.get('regime_type')
                confidence = voter_data.get('confidence', 0.5)
                weight = self.voter_weights.get(voter_id, 1.0)
                performance_score = self._get_voter_performance(voter_id)
                
                voter_result = VoterResult(
                    voter_id=voter_id,
                    regime_prediction=regime_prediction,
                    confidence=confidence,
                    weight=weight,
                    performance_score=performance_score,
                    metadata=voter_data.get('metadata', {})
                )
                
                voter_results.append(voter_result)
            
            return voter_results
            
        except Exception as e:
            logger.error(f"Error preparing voter results: {e}")
            return []

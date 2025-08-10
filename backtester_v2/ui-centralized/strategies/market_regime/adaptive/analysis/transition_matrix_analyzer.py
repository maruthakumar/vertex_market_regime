"""
Transition Matrix Analyzer

This module analyzes regime transition patterns, calculates transition probabilities,
and identifies stable transition paths for improved regime prediction.

Key Features:
- Transition probability calculation
- Pattern identification in regime changes
- Stability analysis of transitions
- Markov chain analysis for regime sequences
- Transition clustering for pattern discovery
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
from sklearn.cluster import DBSCAN

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TransitionPattern:
    """Represents a regime transition pattern"""
    from_regime: int
    to_regime: int
    probability: float
    average_duration_before: float
    average_duration_after: float
    confidence: float
    occurrence_count: int
    stability_score: float
    trigger_indicators: List[str]
    metadata: Dict[str, Any]


@dataclass
class TransitionCluster:
    """Represents a cluster of similar transitions"""
    cluster_id: int
    transitions: List[TransitionPattern]
    centroid_features: Dict[str, float]
    cluster_stability: float
    dominant_path: List[int]


@dataclass
class MarkovChainAnalysis:
    """Results from Markov chain analysis"""
    stationary_distribution: np.ndarray
    mean_recurrence_times: Dict[int, float]
    ergodic: bool
    irreducible: bool
    periodic_states: Set[int]
    transient_states: Set[int]


class TransitionMatrixAnalyzer:
    """
    Analyzes regime transition dynamics and patterns
    """
    
    def __init__(self, regime_count: int = 12):
        """
        Initialize transition matrix analyzer
        
        Args:
            regime_count: Number of regimes in the system
        """
        self.regime_count = regime_count
        
        # Core data structures
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_durations = defaultdict(list)
        self.transition_features = defaultdict(list)
        self.regime_sequences = []
        
        # Analysis results
        self.transition_matrix = None
        self.transition_patterns = {}
        self.markov_analysis = None
        self.transition_clusters = []
        
        # Configuration
        self.min_transitions_for_pattern = 3
        self.stability_threshold = 0.7
        self.clustering_eps = 0.3
        
        logger.info(f"TransitionMatrixAnalyzer initialized for {regime_count} regimes")
    
    def analyze_transitions(self, regime_sequence: List[int], 
                          timestamps: Optional[List[datetime]] = None,
                          features: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Analyze regime transitions from sequence data
        
        Args:
            regime_sequence: Sequence of regime IDs
            timestamps: Optional timestamps for each regime
            features: Optional feature data at transition points
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing transitions from {len(regime_sequence)} data points")
        
        try:
            # Store sequence for analysis
            self.regime_sequences.append(regime_sequence)
            
            # Build transition counts and durations
            self._build_transition_data(regime_sequence, timestamps, features)
            
            # Calculate transition matrix
            self.transition_matrix = self._calculate_transition_matrix()
            
            # Identify transition patterns
            self.transition_patterns = self._identify_transition_patterns()
            
            # Perform Markov chain analysis
            self.markov_analysis = self._perform_markov_analysis()
            
            # Cluster similar transitions
            if features:
                self.transition_clusters = self._cluster_transitions()
            
            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics()
            
            return {
                'transition_matrix': self.transition_matrix,
                'transition_patterns': self.transition_patterns,
                'markov_analysis': self.markov_analysis,
                'transition_clusters': self.transition_clusters,
                'stability_metrics': stability_metrics,
                'summary_statistics': self._get_summary_statistics()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transitions: {e}")
            raise
    
    def _build_transition_data(self, regime_sequence: List[int],
                              timestamps: Optional[List[datetime]] = None,
                              features: Optional[List[Dict[str, float]]] = None):
        """
        Build transition counts and metadata
        
        Args:
            regime_sequence: Sequence of regimes
            timestamps: Optional timestamps
            features: Optional features at each point
        """
        current_regime = regime_sequence[0]
        regime_start_idx = 0
        
        for i in range(1, len(regime_sequence)):
            if regime_sequence[i] != current_regime:
                # Transition detected
                from_regime = current_regime
                to_regime = regime_sequence[i]
                
                # Update counts
                self.transition_counts[from_regime][to_regime] += 1
                
                # Calculate duration in current regime
                duration = i - regime_start_idx
                self.transition_durations[(from_regime, to_regime)].append(duration)
                
                # Store features if available
                if features and i < len(features):
                    self.transition_features[(from_regime, to_regime)].append(features[i])
                
                # Update current regime
                current_regime = to_regime
                regime_start_idx = i
    
    def _calculate_transition_matrix(self) -> pd.DataFrame:
        """
        Calculate transition probability matrix
        
        Returns:
            Transition probability matrix as DataFrame
        """
        # Initialize matrix
        matrix = pd.DataFrame(
            index=range(self.regime_count),
            columns=range(self.regime_count),
            data=0.0
        )
        
        # Calculate probabilities
        for from_regime in range(self.regime_count):
            total_transitions = sum(self.transition_counts[from_regime].values())
            
            if total_transitions > 0:
                for to_regime in range(self.regime_count):
                    count = self.transition_counts[from_regime][to_regime]
                    matrix.loc[from_regime, to_regime] = count / total_transitions
            else:
                # No transitions from this regime - set self-transition to 1
                matrix.loc[from_regime, from_regime] = 1.0
        
        return matrix
    
    def _identify_transition_patterns(self) -> Dict[Tuple[int, int], TransitionPattern]:
        """
        Identify significant transition patterns
        
        Returns:
            Dictionary of transition patterns
        """
        patterns = {}
        
        for (from_regime, to_regime), count in self._flatten_transition_counts():
            if count >= self.min_transitions_for_pattern:
                # Calculate pattern statistics
                durations = self.transition_durations[(from_regime, to_regime)]
                
                if durations:
                    avg_duration_before = np.mean(durations)
                    duration_std = np.std(durations)
                    
                    # Calculate confidence based on consistency
                    cv = duration_std / avg_duration_before if avg_duration_before > 0 else 1.0
                    confidence = max(0.0, 1.0 - cv)
                    
                    # Identify trigger indicators
                    triggers = self._identify_triggers(from_regime, to_regime)
                    
                    # Calculate stability score
                    stability = self._calculate_pattern_stability(from_regime, to_regime)
                    
                    pattern = TransitionPattern(
                        from_regime=from_regime,
                        to_regime=to_regime,
                        probability=self.transition_matrix.loc[from_regime, to_regime],
                        average_duration_before=avg_duration_before,
                        average_duration_after=self._get_avg_duration_after(to_regime),
                        confidence=confidence,
                        occurrence_count=count,
                        stability_score=stability,
                        trigger_indicators=triggers,
                        metadata={
                            'duration_std': duration_std,
                            'cv': cv
                        }
                    )
                    
                    patterns[(from_regime, to_regime)] = pattern
        
        return patterns
    
    def _flatten_transition_counts(self) -> List[Tuple[Tuple[int, int], int]]:
        """Flatten transition counts for analysis"""
        flattened = []
        for from_regime, to_counts in self.transition_counts.items():
            for to_regime, count in to_counts.items():
                if count > 0:
                    flattened.append(((from_regime, to_regime), count))
        return flattened
    
    def _identify_triggers(self, from_regime: int, to_regime: int) -> List[str]:
        """
        Identify common triggers for a transition
        
        Args:
            from_regime: Source regime
            to_regime: Target regime
            
        Returns:
            List of trigger indicators
        """
        triggers = []
        
        # Analyze features at transition points
        features_list = self.transition_features.get((from_regime, to_regime), [])
        
        if features_list:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(features_list)
            
            # Identify features with significant changes
            for column in df.columns:
                values = df[column].values
                
                # Check for consistent patterns
                if len(values) >= 3:
                    # Trend analysis
                    if all(values[i] > values[i-1] for i in range(1, min(3, len(values)))):
                        triggers.append(f"{column}_increasing")
                    elif all(values[i] < values[i-1] for i in range(1, min(3, len(values)))):
                        triggers.append(f"{column}_decreasing")
                    
                    # Threshold analysis
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    if std_val > 0:
                        if mean_val > np.percentile(values, 80):
                            triggers.append(f"{column}_high")
                        elif mean_val < np.percentile(values, 20):
                            triggers.append(f"{column}_low")
        
        return triggers[:5]  # Limit to top 5 triggers
    
    def _calculate_pattern_stability(self, from_regime: int, to_regime: int) -> float:
        """
        Calculate stability score for a transition pattern
        
        Args:
            from_regime: Source regime
            to_regime: Target regime
            
        Returns:
            Stability score (0-1)
        """
        durations = self.transition_durations.get((from_regime, to_regime), [])
        
        if len(durations) < 2:
            return 0.0
        
        # Consistency in durations
        cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 1.0
        duration_stability = max(0.0, 1.0 - cv)
        
        # Frequency stability (how regular are the transitions)
        if len(self.regime_sequences) > 0:
            transition_intervals = []
            for seq in self.regime_sequences:
                intervals = self._get_transition_intervals(seq, from_regime, to_regime)
                transition_intervals.extend(intervals)
            
            if len(transition_intervals) > 1:
                interval_cv = np.std(transition_intervals) / np.mean(transition_intervals)
                frequency_stability = max(0.0, 1.0 - interval_cv)
            else:
                frequency_stability = 0.5
        else:
            frequency_stability = 0.5
        
        # Combined stability
        return duration_stability * 0.6 + frequency_stability * 0.4
    
    def _get_transition_intervals(self, sequence: List[int], 
                                from_regime: int, to_regime: int) -> List[int]:
        """Get intervals between specific transitions in a sequence"""
        intervals = []
        last_occurrence = None
        
        for i in range(1, len(sequence)):
            if sequence[i-1] == from_regime and sequence[i] == to_regime:
                if last_occurrence is not None:
                    intervals.append(i - last_occurrence)
                last_occurrence = i
        
        return intervals
    
    def _get_avg_duration_after(self, regime: int) -> float:
        """Get average duration after entering a regime"""
        all_durations = []
        
        for (from_r, to_r), durations in self.transition_durations.items():
            if from_r == regime:
                all_durations.extend(durations)
        
        return np.mean(all_durations) if all_durations else 0.0
    
    def _perform_markov_analysis(self) -> MarkovChainAnalysis:
        """
        Perform Markov chain analysis on transition matrix
        
        Returns:
            Markov chain analysis results
        """
        if self.transition_matrix is None:
            return None
        
        try:
            P = self.transition_matrix.values
            
            # Check properties
            ergodic = self._is_ergodic(P)
            irreducible = self._is_irreducible(P)
            
            # Find stationary distribution
            stationary = self._find_stationary_distribution(P)
            
            # Calculate mean recurrence times
            recurrence_times = self._calculate_recurrence_times(P, stationary)
            
            # Identify periodic and transient states
            periodic_states = self._find_periodic_states(P)
            transient_states = self._find_transient_states(P)
            
            return MarkovChainAnalysis(
                stationary_distribution=stationary,
                mean_recurrence_times=recurrence_times,
                ergodic=ergodic,
                irreducible=irreducible,
                periodic_states=periodic_states,
                transient_states=transient_states
            )
            
        except Exception as e:
            logger.error(f"Error in Markov analysis: {e}")
            return None
    
    def _is_ergodic(self, P: np.ndarray) -> bool:
        """Check if Markov chain is ergodic"""
        return self._is_irreducible(P) and len(self._find_periodic_states(P)) == 0
    
    def _is_irreducible(self, P: np.ndarray) -> bool:
        """Check if Markov chain is irreducible"""
        # Convert to sparse matrix for efficiency
        sparse_P = csr_matrix(P > 0)
        n_components, _ = connected_components(sparse_P, directed=True, connection='strong')
        return n_components == 1
    
    def _find_stationary_distribution(self, P: np.ndarray) -> np.ndarray:
        """Find stationary distribution of Markov chain"""
        try:
            # Solve π = πP by finding left eigenvector for eigenvalue 1
            eigenvalues, eigenvectors = np.linalg.eig(P.T)
            
            # Find eigenvector corresponding to eigenvalue 1
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, idx])
            
            # Normalize
            stationary = stationary / stationary.sum()
            
            return np.abs(stationary)
            
        except Exception as e:
            logger.error(f"Error finding stationary distribution: {e}")
            # Return uniform distribution as fallback
            return np.ones(self.regime_count) / self.regime_count
    
    def _calculate_recurrence_times(self, P: np.ndarray, 
                                  stationary: np.ndarray) -> Dict[int, float]:
        """Calculate mean recurrence times for each state"""
        recurrence_times = {}
        
        for i in range(self.regime_count):
            if stationary[i] > 1e-10:
                recurrence_times[i] = 1.0 / stationary[i]
            else:
                recurrence_times[i] = float('inf')
        
        return recurrence_times
    
    def _find_periodic_states(self, P: np.ndarray) -> Set[int]:
        """Find periodic states in Markov chain"""
        periodic_states = set()
        
        # Create graph representation
        G = nx.DiGraph()
        for i in range(self.regime_count):
            for j in range(self.regime_count):
                if P[i, j] > 0:
                    G.add_edge(i, j)
        
        # Check periodicity for each state
        for state in range(self.regime_count):
            if state in G:
                try:
                    # Find all cycles containing this state
                    cycles = []
                    for cycle in nx.simple_cycles(G):
                        if state in cycle:
                            cycles.append(len(cycle))
                    
                    if cycles:
                        # GCD of cycle lengths gives period
                        period = np.gcd.reduce(cycles)
                        if period > 1:
                            periodic_states.add(state)
                except:
                    pass
        
        return periodic_states
    
    def _find_transient_states(self, P: np.ndarray) -> Set[int]:
        """Find transient states in Markov chain"""
        transient_states = set()
        
        # Power method to identify absorbing states
        P_power = P.copy()
        for _ in range(100):  # Sufficient iterations
            P_power = np.dot(P_power, P)
        
        # States with very low long-term probability are transient
        for i in range(self.regime_count):
            if np.max(P_power[:, i]) < 0.01:
                transient_states.add(i)
        
        return transient_states
    
    def _cluster_transitions(self) -> List[TransitionCluster]:
        """
        Cluster similar transitions based on features
        
        Returns:
            List of transition clusters
        """
        if not self.transition_features:
            return []
        
        try:
            # Prepare data for clustering
            transition_data = []
            transition_labels = []
            
            for (from_regime, to_regime), features_list in self.transition_features.items():
                if len(features_list) >= self.min_transitions_for_pattern:
                    # Average features for this transition type
                    avg_features = pd.DataFrame(features_list).mean().values
                    transition_data.append(avg_features)
                    transition_labels.append((from_regime, to_regime))
            
            if len(transition_data) < 2:
                return []
            
            # Perform clustering
            X = np.array(transition_data)
            clustering = DBSCAN(eps=self.clustering_eps, min_samples=2).fit(X)
            
            # Build clusters
            clusters = []
            unique_labels = set(clustering.labels_)
            unique_labels.discard(-1)  # Remove noise label
            
            for cluster_id in unique_labels:
                mask = clustering.labels_ == cluster_id
                cluster_transitions = [transition_labels[i] for i in range(len(mask)) if mask[i]]
                
                # Get transition patterns for this cluster
                patterns = []
                for from_r, to_r in cluster_transitions:
                    if (from_r, to_r) in self.transition_patterns:
                        patterns.append(self.transition_patterns[(from_r, to_r)])
                
                if patterns:
                    # Calculate centroid features
                    cluster_features = X[mask].mean(axis=0)
                    
                    # Find dominant path
                    dominant_path = self._find_dominant_path(patterns)
                    
                    cluster = TransitionCluster(
                        cluster_id=int(cluster_id),
                        transitions=patterns,
                        centroid_features={f"feature_{i}": float(v) 
                                         for i, v in enumerate(cluster_features)},
                        cluster_stability=np.mean([p.stability_score for p in patterns]),
                        dominant_path=dominant_path
                    )
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering transitions: {e}")
            return []
    
    def _find_dominant_path(self, patterns: List[TransitionPattern]) -> List[int]:
        """Find dominant regime path from transition patterns"""
        # Build transition graph
        G = nx.DiGraph()
        
        for pattern in patterns:
            G.add_edge(pattern.from_regime, pattern.to_regime, 
                      weight=pattern.probability * pattern.occurrence_count)
        
        # Find path with highest weight
        if G.number_of_nodes() >= 2:
            try:
                # Find longest path (approximation)
                all_paths = []
                for source in G.nodes():
                    for target in G.nodes():
                        if source != target:
                            try:
                                paths = list(nx.all_simple_paths(G, source, target, cutoff=5))
                                all_paths.extend(paths)
                            except:
                                pass
                
                if all_paths:
                    # Score paths by total weight
                    best_path = max(all_paths, key=lambda p: sum(
                        G[p[i]][p[i+1]]['weight'] for i in range(len(p)-1)
                    ))
                    return best_path
            except:
                pass
        
        return []
    
    def _calculate_stability_metrics(self) -> Dict[str, float]:
        """Calculate overall stability metrics"""
        metrics = {}
        
        # Transition frequency
        total_data_points = sum(len(seq) for seq in self.regime_sequences)
        total_transitions = sum(sum(counts.values()) for counts in self.transition_counts.values())
        
        metrics['transition_rate'] = total_transitions / total_data_points if total_data_points > 0 else 0
        metrics['avg_regime_duration'] = total_data_points / (total_transitions + len(self.regime_sequences)) if total_transitions > 0 else 0
        
        # Transition predictability (entropy)
        if self.transition_matrix is not None:
            entropies = []
            for i in range(self.regime_count):
                row = self.transition_matrix.loc[i].values
                row = row[row > 0]
                if len(row) > 0:
                    entropy = -np.sum(row * np.log(row))
                    entropies.append(entropy)
            
            metrics['avg_transition_entropy'] = np.mean(entropies) if entropies else 0
            metrics['transition_predictability'] = 1.0 - (metrics['avg_transition_entropy'] / np.log(self.regime_count))
        
        # Pattern stability
        if self.transition_patterns:
            pattern_stabilities = [p.stability_score for p in self.transition_patterns.values()]
            metrics['avg_pattern_stability'] = np.mean(pattern_stabilities)
        else:
            metrics['avg_pattern_stability'] = 0.0
        
        return metrics
    
    def _get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of transition analysis"""
        stats = {
            'total_transitions': sum(sum(counts.values()) for counts in self.transition_counts.values()),
            'unique_transitions': sum(len(counts) for counts in self.transition_counts.values()),
            'regime_coverage': len(set(regime for counts in self.transition_counts.values() for regime in counts)),
            'pattern_count': len(self.transition_patterns),
            'cluster_count': len(self.transition_clusters)
        }
        
        # Most common transitions
        all_transitions = []
        for from_regime, to_counts in self.transition_counts.items():
            for to_regime, count in to_counts.items():
                all_transitions.append(((from_regime, to_regime), count))
        
        all_transitions.sort(key=lambda x: x[1], reverse=True)
        stats['top_transitions'] = all_transitions[:5]
        
        return stats
    
    def predict_next_regime(self, current_regime: int, 
                          features: Optional[Dict[str, float]] = None) -> Dict[int, float]:
        """
        Predict next regime based on current state
        
        Args:
            current_regime: Current regime ID
            features: Optional current features
            
        Returns:
            Probability distribution over next regimes
        """
        if self.transition_matrix is None:
            # Uniform distribution if no data
            return {i: 1.0/self.regime_count for i in range(self.regime_count)}
        
        # Base prediction from transition matrix
        base_probs = self.transition_matrix.loc[current_regime].to_dict()
        
        # Adjust based on features if available
        if features and self.transition_clusters:
            adjusted_probs = self._adjust_probabilities_by_features(
                current_regime, base_probs, features
            )
            return adjusted_probs
        
        return base_probs
    
    def _adjust_probabilities_by_features(self, current_regime: int,
                                        base_probs: Dict[int, float],
                                        features: Dict[str, float]) -> Dict[int, float]:
        """Adjust transition probabilities based on current features"""
        adjusted = base_probs.copy()
        
        # Find relevant patterns
        relevant_patterns = [
            pattern for (from_r, to_r), pattern in self.transition_patterns.items()
            if from_r == current_regime
        ]
        
        if relevant_patterns:
            # Calculate feature similarity to historical transitions
            for pattern in relevant_patterns:
                historical_features = self.transition_features.get(
                    (pattern.from_regime, pattern.to_regime), []
                )
                
                if historical_features:
                    # Simple similarity based on feature distance
                    avg_historical = pd.DataFrame(historical_features).mean()
                    
                    similarity = 0.0
                    common_features = set(features.keys()) & set(avg_historical.index)
                    
                    if common_features:
                        for feat in common_features:
                            diff = abs(features[feat] - avg_historical[feat])
                            similarity += np.exp(-diff)
                        
                        similarity /= len(common_features)
                        
                        # Boost probability based on similarity
                        boost_factor = 1.0 + similarity * 0.5
                        adjusted[pattern.to_regime] *= boost_factor
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}
        
        return adjusted
    
    def get_regime_network(self) -> nx.DiGraph:
        """
        Get regime transition network
        
        Returns:
            NetworkX directed graph of regime transitions
        """
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(self.regime_count):
            G.add_node(i)
        
        # Add edges with transition probabilities
        if self.transition_matrix is not None:
            for i in range(self.regime_count):
                for j in range(self.regime_count):
                    prob = self.transition_matrix.loc[i, j]
                    if prob > 0.01:  # Threshold for visualization
                        G.add_edge(i, j, weight=prob)
        
        return G
    
    def export_analysis(self, filepath: str):
        """Export analysis results to file"""
        results = {
            'regime_count': self.regime_count,
            'transition_matrix': self.transition_matrix.to_dict() if self.transition_matrix is not None else None,
            'patterns': {
                f"{p.from_regime}->{p.to_regime}": {
                    'probability': p.probability,
                    'avg_duration_before': p.average_duration_before,
                    'stability_score': p.stability_score,
                    'occurrence_count': p.occurrence_count
                }
                for p in self.transition_patterns.values()
            },
            'markov_analysis': {
                'stationary_distribution': self.markov_analysis.stationary_distribution.tolist() if self.markov_analysis else None,
                'ergodic': self.markov_analysis.ergodic if self.markov_analysis else None,
                'irreducible': self.markov_analysis.irreducible if self.markov_analysis else None
            } if self.markov_analysis else None
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Transition analysis exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create analyzer
    analyzer = TransitionMatrixAnalyzer(regime_count=8)
    
    # Generate sample regime sequence
    np.random.seed(42)
    regime_sequence = []
    current_regime = 0
    
    for i in range(1000):
        # Simulate regime transitions
        if np.random.random() < 0.1:  # 10% chance of transition
            current_regime = np.random.randint(0, 8)
        regime_sequence.append(current_regime)
    
    # Generate timestamps
    timestamps = [datetime.now() - timedelta(minutes=5*i) for i in range(1000)]
    timestamps.reverse()
    
    # Analyze transitions
    results = analyzer.analyze_transitions(regime_sequence, timestamps)
    
    print("\nTransition Matrix:")
    print(results['transition_matrix'])
    
    print("\nTop Transition Patterns:")
    for (from_r, to_r), pattern in list(results['transition_patterns'].items())[:5]:
        print(f"{from_r} -> {to_r}: prob={pattern.probability:.3f}, "
              f"stability={pattern.stability_score:.3f}, count={pattern.occurrence_count}")
    
    print("\nMarkov Analysis:")
    if results['markov_analysis']:
        print(f"Ergodic: {results['markov_analysis'].ergodic}")
        print(f"Irreducible: {results['markov_analysis'].irreducible}")
        print(f"Stationary distribution: {results['markov_analysis'].stationary_distribution}")
    
    print("\nStability Metrics:")
    for metric, value in results['stability_metrics'].items():
        print(f"{metric}: {value:.3f}")
"""
Test module for Dynamic Boundary Optimizer
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.dynamic_boundary_optimizer import (
    DynamicBoundaryOptimizer, RegimeBoundary, OptimizationResult, BoundaryUpdate
)


class TestDynamicBoundaryOptimizer(unittest.TestCase):
    """Test cases for Dynamic Boundary Optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = DynamicBoundaryOptimizer(
            regime_count=8,
            optimization_window=50,
            update_frequency=10
        )
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.optimizer.regime_count, 8)
        self.assertEqual(self.optimizer.optimization_window, 50)
        self.assertEqual(len(self.optimizer.current_boundaries), 8)
        
        # Check default boundaries
        for i in range(8):
            boundary = self.optimizer.current_boundaries[i]
            self.assertIsInstance(boundary, RegimeBoundary)
            self.assertEqual(boundary.regime_id, i)
            self.assertIsNotNone(boundary.volatility_bounds)
            self.assertIsNotNone(boundary.trend_bounds)
            self.assertIsNotNone(boundary.volume_bounds)
    
    def test_boundary_initialization_properties(self):
        """Test properties of initialized boundaries"""
        # Check that boundaries are properly ordered
        for i in range(self.optimizer.regime_count - 1):
            curr_boundary = self.optimizer.current_boundaries[i]
            next_boundary = self.optimizer.current_boundaries[i + 1]
            
            # Volatility should generally increase with regime ID
            self.assertLessEqual(
                curr_boundary.volatility_bounds[0],
                next_boundary.volatility_bounds[1]
            )
    
    def test_performance_metrics_update(self):
        """Test updating performance metrics"""
        # Generate performance data
        performance_data = []
        for i in range(20):
            performance_data.append({
                'predicted_regime': i % 8,
                'actual_regime': i % 8,  # Perfect predictions
                'timestamp': datetime.now() - timedelta(minutes=i*5)
            })
        
        # Update metrics
        self.optimizer._update_performance_metrics(performance_data)
        
        # Check accuracy improved
        for regime_id in range(8):
            if regime_id < 3:  # Only first 3 regimes have data
                self.assertGreater(
                    self.optimizer.regime_accuracy[regime_id], 0.5
                )
    
    def test_boundary_array_conversion(self):
        """Test conversion between boundaries and optimization array"""
        # Convert to array
        arr = self.optimizer._boundaries_to_array(self.optimizer.current_boundaries)
        self.assertEqual(len(arr), 8 * 6)  # 8 regimes * 6 values each
        
        # Convert back
        boundaries = self.optimizer._array_to_boundaries(arr)
        self.assertEqual(len(boundaries), 8)
        
        # Check values preserved
        for i in range(8):
            orig = self.optimizer.current_boundaries[i]
            conv = boundaries[i]
            
            np.testing.assert_almost_equal(
                orig.volatility_bounds, conv.volatility_bounds
            )
            np.testing.assert_almost_equal(
                orig.trend_bounds, conv.trend_bounds
            )
            np.testing.assert_almost_equal(
                orig.volume_bounds, conv.volume_bounds
            )
    
    def test_range_overlap_calculation(self):
        """Test range overlap calculation"""
        # No overlap
        overlap = self.optimizer._range_overlap((0, 1), (2, 3))
        self.assertEqual(overlap, 0.0)
        
        # Full overlap
        overlap = self.optimizer._range_overlap((0, 2), (0, 2))
        self.assertEqual(overlap, 1.0)
        
        # Partial overlap
        overlap = self.optimizer._range_overlap((0, 2), (1, 3))
        self.assertGreater(overlap, 0)
        self.assertLess(overlap, 1)
    
    def test_boundary_overlap_calculation(self):
        """Test boundary overlap calculation"""
        # Create two boundaries
        boundary1 = RegimeBoundary(
            regime_id=0,
            volatility_bounds=(0.0, 0.1),
            trend_bounds=(-0.01, 0.01),
            volume_bounds=(0.5, 1.0),
            confidence_threshold=0.6,
            hysteresis_factor=0.05,
            last_updated=datetime.now(),
            performance_score=0.5
        )
        
        boundary2 = RegimeBoundary(
            regime_id=1,
            volatility_bounds=(0.05, 0.15),  # Overlaps with boundary1
            trend_bounds=(0.0, 0.02),        # Overlaps with boundary1
            volume_bounds=(0.8, 1.5),        # Overlaps with boundary1
            confidence_threshold=0.6,
            hysteresis_factor=0.05,
            last_updated=datetime.now(),
            performance_score=0.5
        )
        
        overlap = self.optimizer._calculate_overlap(boundary1, boundary2)
        self.assertGreater(overlap, 0)  # Should have some overlap
    
    def test_boundary_change_calculation(self):
        """Test boundary change calculation"""
        old_boundary = self.optimizer.current_boundaries[0]
        
        # Create modified boundary
        new_boundary = RegimeBoundary(
            regime_id=0,
            volatility_bounds=(
                old_boundary.volatility_bounds[0] + 0.01,
                old_boundary.volatility_bounds[1] + 0.01
            ),
            trend_bounds=old_boundary.trend_bounds,
            volume_bounds=old_boundary.volume_bounds,
            confidence_threshold=old_boundary.confidence_threshold,
            hysteresis_factor=old_boundary.hysteresis_factor,
            last_updated=datetime.now(),
            performance_score=old_boundary.performance_score
        )
        
        change = self.optimizer._calculate_boundary_change(old_boundary, new_boundary)
        self.assertGreater(change, 0)  # Should detect change
    
    def test_boundary_validation(self):
        """Test boundary validation"""
        # Create invalid boundaries (reversed bounds)
        invalid_boundaries = {
            0: RegimeBoundary(
                regime_id=0,
                volatility_bounds=(0.2, 0.1),  # Invalid: max < min
                trend_bounds=(0.01, -0.01),     # Invalid: max < min
                volume_bounds=(1.0, 0.5),       # Invalid: max < min
                confidence_threshold=0.6,
                hysteresis_factor=0.05,
                last_updated=datetime.now(),
                performance_score=0.5
            )
        }
        
        # Validate
        validated = self.optimizer._validate_boundaries(invalid_boundaries)
        
        # Check bounds are corrected
        boundary = validated[0]
        self.assertLess(boundary.volatility_bounds[0], boundary.volatility_bounds[1])
        self.assertLess(boundary.trend_bounds[0], boundary.trend_bounds[1])
        self.assertLess(boundary.volume_bounds[0], boundary.volume_bounds[1])
    
    def test_regime_transition_check(self):
        """Test regime transition checking with hysteresis"""
        # Set up market conditions that match regime 3
        market_data = {
            'volatility': 0.15,
            'trend': 0.0,
            'volume_ratio': 1.0
        }
        
        # Check transition from regime 2
        current_regime = 2
        new_regime, confidence = self.optimizer.check_regime_transition(
            current_regime, market_data
        )
        
        # Should suggest a regime (might stay in current due to hysteresis)
        self.assertIsInstance(new_regime, int)
        self.assertGreaterEqual(new_regime, 0)
        self.assertLess(new_regime, 8)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_hysteresis_update(self):
        """Test adaptive hysteresis update"""
        # Enable adaptive hysteresis
        self.optimizer.adaptive_hysteresis = True
        
        # Create transition data with false transitions
        transition_data = [
            {
                'from_regime': 0,
                'to_regime': 1,
                'duration': 5,
                'reversed_quickly': True
            },
            {
                'from_regime': 0,
                'to_regime': 1,
                'duration': 3,
                'reversed_quickly': True
            }
        ]
        
        # Get initial hysteresis
        initial_hysteresis = self.optimizer.current_boundaries[0].hysteresis_factor
        
        # Update hysteresis
        self.optimizer.update_hysteresis(transition_data)
        
        # Check hysteresis increased for regime 0
        updated_hysteresis = self.optimizer.current_boundaries[0].hysteresis_factor
        self.assertGreater(updated_hysteresis, initial_hysteresis)
    
    def test_optimization_with_sample_data(self):
        """Test boundary optimization with sample data"""
        # Generate performance data with patterns
        performance_data = []
        for i in range(100):
            # Create pattern: regime 0 is well predicted, regime 7 is poorly predicted
            if i % 8 == 0:
                predicted = 0
                actual = 0
            elif i % 8 == 7:
                predicted = 7
                actual = 3  # Misclassification
            else:
                predicted = i % 8
                actual = i % 8
            
            performance_data.append({
                'predicted_regime': predicted,
                'actual_regime': actual,
                'timestamp': datetime.now() - timedelta(minutes=i*5)
            })
        
        # Run optimization
        result = self.optimizer.optimize_boundaries(performance_data)
        
        # Check result
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.optimized_boundaries)
        self.assertGreaterEqual(result.iterations, 0)
    
    def test_optimization_metrics(self):
        """Test optimization metrics retrieval"""
        metrics = self.optimizer.get_optimization_metrics()
        
        self.assertIn('total_optimizations', metrics)
        self.assertIn('successful_optimizations', metrics)
        self.assertIn('success_rate', metrics)
        self.assertIn('average_improvement', metrics)
        self.assertIn('regime_accuracy', metrics)
        self.assertIn('current_hysteresis', metrics)
        
        # Check regime accuracy structure
        self.assertEqual(len(metrics['regime_accuracy']), 8)
        
        # Check hysteresis structure
        self.assertEqual(len(metrics['current_hysteresis']), 8)
    
    def test_boundary_export(self):
        """Test boundary export functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Export boundaries
            self.optimizer.export_boundaries(filepath)
            
            # Load and verify
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(data['regime_count'], 8)
            self.assertIn('boundaries', data)
            self.assertEqual(len(data['boundaries']), 8)
            
            # Check boundary structure
            for regime_id in range(8):
                boundary_data = data['boundaries'][str(regime_id)]
                self.assertIn('volatility_bounds', boundary_data)
                self.assertIn('trend_bounds', boundary_data)
                self.assertIn('volume_bounds', boundary_data)
                self.assertIn('hysteresis_factor', boundary_data)
        
        finally:
            # Cleanup
            Path(filepath).unlink(missing_ok=True)
    
    def test_simulate_accuracy(self):
        """Test accuracy simulation for boundaries"""
        # Test with current boundary
        boundary = self.optimizer.current_boundaries[0]
        accuracy = self.optimizer._simulate_accuracy(boundary)
        
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_optimization_bounds(self):
        """Test optimization bounds generation"""
        bounds = self.optimizer._get_optimization_bounds()
        
        # Should have 6 bounds per regime
        self.assertEqual(len(bounds), 8 * 6)
        
        # Check all bounds are tuples
        for bound in bounds:
            self.assertIsInstance(bound, tuple)
            self.assertEqual(len(bound), 2)
            self.assertLess(bound[0], bound[1])


if __name__ == '__main__':
    unittest.main()
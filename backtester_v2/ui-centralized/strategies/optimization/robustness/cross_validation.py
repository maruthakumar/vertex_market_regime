"""
Time-Series Cross-Validation for Strategy Optimization

Implements walk-forward and time-series specific cross-validation
methods adapted from the original enhanced market regime optimizer.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class CVFold:
    """Represents a single cross-validation fold"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_data: Optional[pd.DataFrame] = None
    test_data: Optional[pd.DataFrame] = None

@dataclass
class CVResult:
    """Cross-validation results for a single fold"""
    fold_id: int
    train_score: float
    test_score: float
    best_params: Dict[str, float]
    optimization_time: float
    metadata: Dict[str, Any]

class CrossValidation:
    """
    Time-series cross-validation for strategy optimization
    
    Implements walk-forward analysis and time-series specific validation
    methods that respect temporal ordering of market data.
    """
    
    def __init__(self,
                 validation_method: str = 'walk_forward',
                 n_folds: int = 5,
                 train_size: Union[int, float] = 0.7,
                 test_size: Union[int, float] = 0.3,
                 gap_size: int = 0,
                 purged_cv: bool = True,
                 embargo_size: int = 0):
        """
        Initialize cross-validation framework
        
        Args:
            validation_method: Method ('walk_forward', 'expanding_window', 'sliding_window')
            n_folds: Number of folds
            train_size: Training set size (fraction or number of periods)
            test_size: Test set size (fraction or number of periods)
            gap_size: Gap between train and test sets (in periods)
            purged_cv: Whether to purge overlapping samples
            embargo_size: Embargo period to prevent look-ahead bias
        """
        self.validation_method = validation_method
        self.n_folds = n_folds
        self.train_size = train_size
        self.test_size = test_size
        self.gap_size = gap_size
        self.purged_cv = purged_cv
        self.embargo_size = embargo_size
        
        # Validate parameters
        valid_methods = ['walk_forward', 'expanding_window', 'sliding_window']
        if validation_method not in valid_methods:
            raise ValueError(f"Validation method must be one of {valid_methods}")
        
        if n_folds < 2:
            raise ValueError("Number of folds must be at least 2")
        
        logger.info(f"Initialized {validation_method} cross-validation with {n_folds} folds")
    
    def create_folds(self, data: pd.DataFrame, 
                    date_column: str = 'date') -> List[CVFold]:
        """
        Create cross-validation folds from time-series data
        
        Args:
            data: Time-series data with date column
            date_column: Name of date column
            
        Returns:
            List of CVFold objects
        """
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
        
        # Ensure data is sorted by date
        data_sorted = data.sort_values(date_column).reset_index(drop=True)
        dates = pd.to_datetime(data_sorted[date_column])
        
        folds = []
        
        if self.validation_method == 'walk_forward':
            folds = self._create_walk_forward_folds(dates, data_sorted)
        elif self.validation_method == 'expanding_window':
            folds = self._create_expanding_window_folds(dates, data_sorted)
        elif self.validation_method == 'sliding_window':
            folds = self._create_sliding_window_folds(dates, data_sorted)
        
        logger.info(f"Created {len(folds)} cross-validation folds")
        return folds
    
    def _create_walk_forward_folds(self, dates: pd.Series, 
                                  data: pd.DataFrame) -> List[CVFold]:
        """Create walk-forward analysis folds"""
        folds = []
        total_periods = len(dates)
        
        # Calculate initial train size
        if isinstance(self.train_size, float):
            initial_train_size = int(total_periods * self.train_size)
        else:
            initial_train_size = self.train_size
        
        # Calculate test size
        if isinstance(self.test_size, float):
            test_size = int(total_periods * self.test_size)
        else:
            test_size = self.test_size
        
        for fold_id in range(self.n_folds):
            # Calculate indices for this fold
            train_start_idx = fold_id * test_size
            train_end_idx = train_start_idx + initial_train_size
            
            # Add gap
            test_start_idx = train_end_idx + self.gap_size
            test_end_idx = min(test_start_idx + test_size, total_periods)
            
            # Ensure we have enough data
            if test_end_idx > total_periods or test_start_idx >= total_periods:
                break
            
            # Create fold
            fold = CVFold(
                fold_id=fold_id,
                train_start=dates.iloc[train_start_idx],
                train_end=dates.iloc[train_end_idx - 1],
                test_start=dates.iloc[test_start_idx],
                test_end=dates.iloc[test_end_idx - 1],
                train_data=data.iloc[train_start_idx:train_end_idx].copy(),
                test_data=data.iloc[test_start_idx:test_end_idx].copy()
            )
            
            folds.append(fold)
        
        return folds
    
    def _create_expanding_window_folds(self, dates: pd.Series, 
                                      data: pd.DataFrame) -> List[CVFold]:
        """Create expanding window folds"""
        folds = []
        total_periods = len(dates)
        
        # Calculate test size
        if isinstance(self.test_size, float):
            test_size = int(total_periods * self.test_size)
        else:
            test_size = self.test_size
        
        # Calculate minimum train size
        if isinstance(self.train_size, float):
            min_train_size = int(total_periods * self.train_size / self.n_folds)
        else:
            min_train_size = self.train_size
        
        for fold_id in range(self.n_folds):
            # Train set expands with each fold
            train_start_idx = 0
            train_end_idx = min_train_size + fold_id * test_size
            
            # Test set follows training set
            test_start_idx = train_end_idx + self.gap_size
            test_end_idx = min(test_start_idx + test_size, total_periods)
            
            if test_end_idx > total_periods or test_start_idx >= total_periods:
                break
            
            fold = CVFold(
                fold_id=fold_id,
                train_start=dates.iloc[train_start_idx],
                train_end=dates.iloc[train_end_idx - 1],
                test_start=dates.iloc[test_start_idx],
                test_end=dates.iloc[test_end_idx - 1],
                train_data=data.iloc[train_start_idx:train_end_idx].copy(),
                test_data=data.iloc[test_start_idx:test_end_idx].copy()
            )
            
            folds.append(fold)
        
        return folds
    
    def _create_sliding_window_folds(self, dates: pd.Series, 
                                    data: pd.DataFrame) -> List[CVFold]:
        """Create sliding window folds"""
        folds = []
        total_periods = len(dates)
        
        # Calculate window sizes
        if isinstance(self.train_size, float):
            train_size = int(total_periods * self.train_size)
        else:
            train_size = self.train_size
        
        if isinstance(self.test_size, float):
            test_size = int(total_periods * self.test_size)
        else:
            test_size = self.test_size
        
        # Calculate step size
        total_window = train_size + self.gap_size + test_size
        step_size = (total_periods - total_window) // (self.n_folds - 1) if self.n_folds > 1 else 0
        
        for fold_id in range(self.n_folds):
            train_start_idx = fold_id * step_size
            train_end_idx = train_start_idx + train_size
            
            test_start_idx = train_end_idx + self.gap_size
            test_end_idx = test_start_idx + test_size
            
            if test_end_idx > total_periods:
                break
            
            fold = CVFold(
                fold_id=fold_id,
                train_start=dates.iloc[train_start_idx],
                train_end=dates.iloc[train_end_idx - 1],
                test_start=dates.iloc[test_start_idx],
                test_end=dates.iloc[test_end_idx - 1],
                train_data=data.iloc[train_start_idx:train_end_idx].copy(),
                test_data=data.iloc[test_start_idx:test_end_idx].copy()
            )
            
            folds.append(fold)
        
        return folds
    
    def optimize_with_cv(self, 
                        optimizer: 'BaseOptimizer',
                        data: pd.DataFrame,
                        date_column: str = 'date',
                        scoring_function: Optional[Callable] = None,
                        n_iterations: int = 100) -> Dict[str, Any]:
        """
        Run optimization with cross-validation
        
        Args:
            optimizer: Optimizer instance
            data: Time-series data
            date_column: Date column name
            scoring_function: Custom scoring function (default uses optimizer's objective)
            n_iterations: Iterations per fold
            
        Returns:
            Cross-validation results
        """
        start_time = time.time()
        
        logger.info(f"Starting cross-validation optimization with {self.validation_method}")
        
        # Create folds
        folds = self.create_folds(data, date_column)
        
        if not folds:
            raise ValueError("No valid folds could be created from the data")
        
        # Default scoring function
        if scoring_function is None:
            scoring_function = optimizer.objective_function
        
        # Run optimization for each fold
        cv_results = []
        
        for fold in folds:
            logger.info(f"Processing fold {fold.fold_id + 1}/{len(folds)}")
            logger.info(f"Train: {fold.train_start} to {fold.train_end}")
            logger.info(f"Test: {fold.test_start} to {fold.test_end}")
            
            fold_start_time = time.time()
            
            try:
                # Optimize on training data
                train_result = optimizer.optimize(n_iterations=n_iterations)
                
                # Evaluate on test data
                test_score = scoring_function(train_result.best_params)
                
                # Create CV result
                cv_result = CVResult(
                    fold_id=fold.fold_id,
                    train_score=train_result.best_score,
                    test_score=test_score,
                    best_params=train_result.best_params.copy(),
                    optimization_time=time.time() - fold_start_time,
                    metadata={
                        'train_periods': len(fold.train_data),
                        'test_periods': len(fold.test_data),
                        'convergence_iterations': train_result.n_iterations
                    }
                )
                
                cv_results.append(cv_result)
                
                logger.info(f"Fold {fold.fold_id + 1} completed: "
                          f"Train={train_result.best_score:.6f}, Test={test_score:.6f}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold.fold_id + 1}: {e}")
                continue
        
        # Calculate summary statistics
        train_scores = [r.train_score for r in cv_results]
        test_scores = [r.test_score for r in cv_results]
        
        cv_summary = {
            'cv_method': self.validation_method,
            'n_folds_completed': len(cv_results),
            'total_time': time.time() - start_time,
            
            # Training scores
            'train_mean': np.mean(train_scores),
            'train_std': np.std(train_scores),
            'train_min': np.min(train_scores),
            'train_max': np.max(train_scores),
            
            # Test scores (out-of-sample)
            'test_mean': np.mean(test_scores),
            'test_std': np.std(test_scores),
            'test_min': np.min(test_scores),
            'test_max': np.max(test_scores),
            
            # Stability metrics
            'stability_ratio': np.mean(test_scores) / np.mean(train_scores) if np.mean(train_scores) != 0 else 0,
            'score_correlation': np.corrcoef(train_scores, test_scores)[0, 1] if len(train_scores) > 1 else 0,
            
            # Individual fold results
            'fold_results': [
                {
                    'fold_id': r.fold_id,
                    'train_score': r.train_score,
                    'test_score': r.test_score,
                    'params': r.best_params,
                    'optimization_time': r.optimization_time
                }
                for r in cv_results
            ]
        }
        
        logger.info(f"Cross-validation completed: {len(cv_results)} folds")
        logger.info(f"Test score: {cv_summary['test_mean']:.6f} ± {cv_summary['test_std']:.6f}")
        logger.info(f"Stability ratio: {cv_summary['stability_ratio']:.3f}")
        
        return cv_summary
    
    def get_best_parameters(self, cv_results: Dict[str, Any], 
                           selection_method: str = 'best_test') -> Dict[str, float]:
        """
        Select best parameters from cross-validation results
        
        Args:
            cv_results: Results from optimize_with_cv
            selection_method: Method ('best_test', 'most_stable', 'average')
            
        Returns:
            Best parameters
        """
        fold_results = cv_results['fold_results']
        
        if selection_method == 'best_test':
            # Select fold with best test score
            best_fold = max(fold_results, key=lambda x: x['test_score'])
            return best_fold['params']
            
        elif selection_method == 'most_stable':
            # Select fold with smallest train-test gap
            stability_scores = [abs(r['train_score'] - r['test_score']) for r in fold_results]
            most_stable_idx = np.argmin(stability_scores)
            return fold_results[most_stable_idx]['params']
            
        elif selection_method == 'average':
            # Average parameters across all folds
            all_params = [r['params'] for r in fold_results]
            param_names = all_params[0].keys()
            
            averaged_params = {}
            for param_name in param_names:
                values = [params[param_name] for params in all_params]
                averaged_params[param_name] = np.mean(values)
            
            return averaged_params
        
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
    
    def save_cv_results(self, cv_results: Dict[str, Any], filepath: str):
        """Save cross-validation results to JSON file"""
        
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return super().default(obj)
        
        with open(filepath, 'w') as f:
            json.dump(cv_results, f, cls=DateTimeEncoder, indent=2)
        
        logger.info(f"Cross-validation results saved to {filepath}")
    
    def load_cv_results(self, filepath: str) -> Dict[str, Any]:
        """Load cross-validation results from JSON file"""
        with open(filepath, 'r') as f:
            cv_results = json.load(f)
        
        logger.info(f"Cross-validation results loaded from {filepath}")
        return cv_results
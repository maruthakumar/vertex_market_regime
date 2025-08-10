"""
Benchmark Suite

Standard benchmark problems and test functions for optimization algorithm
evaluation including mathematical test functions, real-world problems,
and strategy optimization scenarios.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class ProblemCategory(Enum):
    """Categories of optimization problems"""
    MATHEMATICAL = "mathematical"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    CONSTRAINED = "constrained"
    MULTI_MODAL = "multi_modal"
    NOISY = "noisy"
    HIGH_DIMENSIONAL = "high_dimensional"

class DifficultyLevel(Enum):
    """Problem difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"

@dataclass
class BenchmarkProblem:
    """Definition of a benchmark optimization problem"""
    name: str
    param_space: Dict[str, Tuple[float, float]]
    objective_function: Callable[[Dict[str, float]], float]
    optimal_value: Optional[float]
    optimal_parameters: Optional[Dict[str, float]]
    category: ProblemCategory
    difficulty: DifficultyLevel
    description: str
    properties: Dict[str, Any]
    
    def evaluate(self, parameters: Dict[str, float]) -> float:
        """Evaluate objective function"""
        return self.objective_function(parameters)
    
    def get_dimensionality(self) -> int:
        """Get problem dimensionality"""
        return len(self.param_space)
    
    def is_solution_optimal(self, parameters: Dict[str, float], tolerance: float = 1e-6) -> bool:
        """Check if solution is optimal within tolerance"""
        if self.optimal_parameters is None:
            return False
        
        for param, optimal_val in self.optimal_parameters.items():
            if param in parameters:
                if abs(parameters[param] - optimal_val) > tolerance:
                    return False
        return True


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for optimization algorithms
    
    Provides a standardized collection of test problems covering different
    categories, difficulties, and dimensionalities for thorough algorithm evaluation.
    """
    
    def __init__(self):
        """Initialize benchmark suite"""
        self.problems: Dict[str, BenchmarkProblem] = {}
        self._initialize_mathematical_problems()
        self._initialize_strategy_problems()
        self._initialize_constrained_problems()
        self._initialize_multimodal_problems()
        self._initialize_noisy_problems()
        self._initialize_high_dimensional_problems()
        
        logger.info(f"BenchmarkSuite initialized with {len(self.problems)} problems")
    
    def get_problem(self, name: str) -> BenchmarkProblem:
        """Get specific benchmark problem by name"""
        if name not in self.problems:
            available = list(self.problems.keys())
            raise ValueError(f"Problem '{name}' not found. Available: {available}")
        return self.problems[name]
    
    def list_problems(self, 
                     category: Optional[ProblemCategory] = None,
                     difficulty: Optional[DifficultyLevel] = None,
                     max_dimensions: Optional[int] = None) -> List[str]:
        """List problems with optional filtering"""
        filtered_problems = []
        
        for name, problem in self.problems.items():
            # Apply filters
            if category and problem.category != category:
                continue
            if difficulty and problem.difficulty != difficulty:
                continue
            if max_dimensions and problem.get_dimensionality() > max_dimensions:
                continue
            
            filtered_problems.append(name)
        
        return filtered_problems
    
    def get_problems_by_category(self, category: ProblemCategory) -> List[BenchmarkProblem]:
        """Get all problems in a specific category"""
        return [p for p in self.problems.values() if p.category == category]
    
    def get_problems_by_difficulty(self, difficulty: DifficultyLevel) -> List[BenchmarkProblem]:
        """Get all problems of specific difficulty"""
        return [p for p in self.problems.values() if p.difficulty == difficulty]
    
    def create_custom_problem(self,
                            name: str,
                            param_space: Dict[str, Tuple[float, float]],
                            objective_function: Callable,
                            category: ProblemCategory = ProblemCategory.MATHEMATICAL,
                            difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
                            description: str = "",
                            optimal_value: Optional[float] = None,
                            optimal_parameters: Optional[Dict[str, float]] = None) -> BenchmarkProblem:
        """Create and register custom benchmark problem"""
        
        problem = BenchmarkProblem(
            name=name,
            param_space=param_space,
            objective_function=objective_function,
            optimal_value=optimal_value,
            optimal_parameters=optimal_parameters,
            category=category,
            difficulty=difficulty,
            description=description,
            properties={}
        )
        
        self.problems[name] = problem
        logger.info(f"Added custom problem: {name}")
        
        return problem
    
    def _initialize_mathematical_problems(self):
        """Initialize standard mathematical optimization problems"""
        
        # Sphere Function (Easy, Unimodal)
        def sphere_function(params):
            return sum(v**2 for v in params.values())
        
        self.problems["sphere_2d"] = BenchmarkProblem(
            name="sphere_2d",
            param_space={'x': (-5.0, 5.0), 'y': (-5.0, 5.0)},
            objective_function=sphere_function,
            optimal_value=0.0,
            optimal_parameters={'x': 0.0, 'y': 0.0},
            category=ProblemCategory.MATHEMATICAL,
            difficulty=DifficultyLevel.EASY,
            description="2D Sphere function - simple quadratic with global minimum at origin",
            properties={'unimodal': True, 'separable': True, 'convex': True}
        )
        
        # Rosenbrock Function (Medium, Valley-shaped)
        def rosenbrock_function(params):
            values = list(params.values())
            if len(values) < 2:
                return sum(v**2 for v in values)
            
            result = 0
            for i in range(len(values) - 1):
                result += 100 * (values[i+1] - values[i]**2)**2 + (1 - values[i])**2
            return result
        
        self.problems["rosenbrock_2d"] = BenchmarkProblem(
            name="rosenbrock_2d",
            param_space={'x': (-2.0, 2.0), 'y': (-1.0, 3.0)},
            objective_function=rosenbrock_function,
            optimal_value=0.0,
            optimal_parameters={'x': 1.0, 'y': 1.0},
            category=ProblemCategory.MATHEMATICAL,
            difficulty=DifficultyLevel.MEDIUM,
            description="2D Rosenbrock function - classic banana valley optimization problem",
            properties={'unimodal': True, 'separable': False, 'valley_shaped': True}
        )
        
        # Rastrigin Function (Hard, Multimodal)
        def rastrigin_function(params):
            values = list(params.values())
            A = 10
            n = len(values)
            return A * n + sum(x**2 - A * np.cos(2 * np.pi * x) for x in values)
        
        self.problems["rastrigin_2d"] = BenchmarkProblem(
            name="rastrigin_2d",
            param_space={'x': (-5.12, 5.12), 'y': (-5.12, 5.12)},
            objective_function=rastrigin_function,
            optimal_value=0.0,
            optimal_parameters={'x': 0.0, 'y': 0.0},
            category=ProblemCategory.MULTI_MODAL,
            difficulty=DifficultyLevel.HARD,
            description="2D Rastrigin function - highly multimodal with many local optima",
            properties={'multimodal': True, 'separable': True, 'local_optima': 'many'}
        )
        
        # Ackley Function (Hard, Multimodal)
        def ackley_function(params):
            values = list(params.values())
            n = len(values)
            sum_sq = sum(x**2 for x in values)
            sum_cos = sum(np.cos(2 * np.pi * x) for x in values)
            
            term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
            term2 = -np.exp(sum_cos / n)
            
            return term1 + term2 + 20 + np.e
        
        self.problems["ackley_2d"] = BenchmarkProblem(
            name="ackley_2d",
            param_space={'x': (-32.768, 32.768), 'y': (-32.768, 32.768)},
            objective_function=ackley_function,
            optimal_value=0.0,
            optimal_parameters={'x': 0.0, 'y': 0.0},
            category=ProblemCategory.MULTI_MODAL,
            difficulty=DifficultyLevel.HARD,
            description="2D Ackley function - multimodal with exponential terms",
            properties={'multimodal': True, 'separable': False, 'exponential': True}
        )
        
        # Schwefel Function (Extreme, Deceptive)
        def schwefel_function(params):
            values = list(params.values())
            n = len(values)
            return 418.9829 * n - sum(x * np.sin(np.sqrt(abs(x))) for x in values)
        
        self.problems["schwefel_2d"] = BenchmarkProblem(
            name="schwefel_2d",
            param_space={'x': (-500.0, 500.0), 'y': (-500.0, 500.0)},
            objective_function=schwefel_function,
            optimal_value=0.0,
            optimal_parameters={'x': 420.9687, 'y': 420.9687},
            category=ProblemCategory.MULTI_MODAL,
            difficulty=DifficultyLevel.EXTREME,
            description="2D Schwefel function - deceptive with global optimum far from center",
            properties={'multimodal': True, 'deceptive': True, 'large_domain': True}
        )
    
    def _initialize_strategy_problems(self):
        """Initialize strategy optimization problems"""
        
        # Portfolio Optimization Problem
        def portfolio_objective(params):
            """Simplified portfolio optimization with risk-return tradeoff"""
            weights = np.array(list(params.values()))
            
            # Ensure weights sum to 1 (constraint via penalty)
            weight_sum_penalty = 1000 * (abs(weights.sum() - 1.0))**2
            
            # Mock expected returns and covariance
            np.random.seed(42)  # For reproducibility
            n_assets = len(weights)
            expected_returns = np.random.normal(0.08, 0.02, n_assets)
            cov_matrix = np.random.rand(n_assets, n_assets)
            cov_matrix = 0.01 * (cov_matrix @ cov_matrix.T)  # Make positive definite
            
            # Portfolio return and risk
            portfolio_return = weights @ expected_returns
            portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
            
            # Risk-adjusted return (negative for minimization)
            risk_aversion = 0.5
            objective = -(portfolio_return - risk_aversion * portfolio_risk) + weight_sum_penalty
            
            return objective
        
        self.problems["portfolio_3_assets"] = BenchmarkProblem(
            name="portfolio_3_assets",
            param_space={'w1': (0.0, 1.0), 'w2': (0.0, 1.0), 'w3': (0.0, 1.0)},
            objective_function=portfolio_objective,
            optimal_value=None,  # Problem-dependent
            optimal_parameters=None,
            category=ProblemCategory.STRATEGY_OPTIMIZATION,
            difficulty=DifficultyLevel.MEDIUM,
            description="3-asset portfolio optimization with risk-return tradeoff",
            properties={'constraint': 'sum_to_one', 'financial': True}
        )
        
        # Strategy Parameter Optimization
        def strategy_params_objective(params):
            """Mock strategy parameter optimization"""
            lookback = params['lookback']
            threshold = params['threshold']
            risk_factor = params['risk_factor']
            
            # Simulate strategy performance based on parameters
            # Higher lookback = lower noise but slower adaptation
            # Higher threshold = fewer trades but higher precision
            # Higher risk_factor = higher return but higher volatility
            
            base_return = 0.1 * (1 / (1 + np.exp(-threshold * 10))) # Sigmoid for threshold
            lookback_penalty = 0.001 * lookback  # Penalty for high lookback
            risk_penalty = 0.05 * risk_factor**2  # Quadratic risk penalty
            
            # Add some noise to make it realistic
            np.random.seed(int(lookback + threshold * 100 + risk_factor * 1000))
            noise = np.random.normal(0, 0.01)
            
            # Return negative (for minimization) of risk-adjusted return
            return -(base_return - lookback_penalty - risk_penalty) + noise
        
        self.problems["strategy_parameters"] = BenchmarkProblem(
            name="strategy_parameters",
            param_space={
                'lookback': (5.0, 50.0),
                'threshold': (0.01, 0.1),
                'risk_factor': (0.5, 3.0)
            },
            objective_function=strategy_params_objective,
            optimal_value=None,
            optimal_parameters=None,
            category=ProblemCategory.STRATEGY_OPTIMIZATION,
            difficulty=DifficultyLevel.MEDIUM,
            description="Strategy parameter optimization with realistic tradeoffs",
            properties={'noisy': True, 'trading_strategy': True}
        )
    
    def _initialize_constrained_problems(self):
        """Initialize constrained optimization problems"""
        
        # Constrained Quadratic
        def constrained_quadratic(params):
            x, y = params['x'], params['y']
            
            # Objective: minimize (x-2)^2 + (y-1)^2
            objective = (x - 2)**2 + (y - 1)**2
            
            # Constraint: x + y <= 1 (penalty method)
            constraint_violation = max(0, x + y - 1)
            penalty = 1000 * constraint_violation**2
            
            return objective + penalty
        
        self.problems["constrained_quadratic"] = BenchmarkProblem(
            name="constrained_quadratic",
            param_space={'x': (-2.0, 2.0), 'y': (-2.0, 2.0)},
            objective_function=constrained_quadratic,
            optimal_value=1.25,  # At constraint boundary
            optimal_parameters={'x': 0.5, 'y': 0.5},
            category=ProblemCategory.CONSTRAINED,
            difficulty=DifficultyLevel.MEDIUM,
            description="Quadratic optimization with linear constraint",
            properties={'constraint_type': 'inequality', 'penalty_method': True}
        )
    
    def _initialize_multimodal_problems(self):
        """Initialize multimodal problems with many local optima"""
        
        # Egg Holder Function
        def egg_holder_function(params):
            x, y = params['x'], params['y']
            term1 = -(y + 47) * np.sin(np.sqrt(abs(x/2 + (y + 47))))
            term2 = -x * np.sin(np.sqrt(abs(x - (y + 47))))
            return term1 + term2
        
        self.problems["egg_holder"] = BenchmarkProblem(
            name="egg_holder",
            param_space={'x': (-512.0, 512.0), 'y': (-512.0, 512.0)},
            objective_function=egg_holder_function,
            optimal_value=-959.6407,
            optimal_parameters={'x': 512.0, 'y': 404.2319},
            category=ProblemCategory.MULTI_MODAL,
            difficulty=DifficultyLevel.EXTREME,
            description="Egg Holder function - highly multimodal with many local optima",
            properties={'multimodal': True, 'local_optima': 'very_many', 'rugged': True}
        )
        
        # Griewank Function
        def griewank_function(params):
            values = list(params.values())
            sum_sq = sum(x**2 for x in values) / 4000
            prod_cos = np.prod([np.cos(x / np.sqrt(i+1)) for i, x in enumerate(values)])
            return sum_sq - prod_cos + 1
        
        self.problems["griewank_2d"] = BenchmarkProblem(
            name="griewank_2d",
            param_space={'x': (-600.0, 600.0), 'y': (-600.0, 600.0)},
            objective_function=griewank_function,
            optimal_value=0.0,
            optimal_parameters={'x': 0.0, 'y': 0.0},
            category=ProblemCategory.MULTI_MODAL,
            difficulty=DifficultyLevel.HARD,
            description="Griewank function - multimodal with product of cosines",
            properties={'multimodal': True, 'separable': False, 'product_terms': True}
        )
    
    def _initialize_noisy_problems(self):
        """Initialize problems with noise to test robustness"""
        
        # Noisy Sphere
        def noisy_sphere(params):
            base_value = sum(v**2 for v in params.values())
            # Add Gaussian noise proportional to function value
            noise_level = 0.1
            noise = np.random.normal(0, noise_level * abs(base_value) + 0.01)
            return base_value + noise
        
        self.problems["noisy_sphere_2d"] = BenchmarkProblem(
            name="noisy_sphere_2d",
            param_space={'x': (-5.0, 5.0), 'y': (-5.0, 5.0)},
            objective_function=noisy_sphere,
            optimal_value=0.0,
            optimal_parameters={'x': 0.0, 'y': 0.0},
            category=ProblemCategory.NOISY,
            difficulty=DifficultyLevel.MEDIUM,
            description="Sphere function with Gaussian noise",
            properties={'noisy': True, 'noise_type': 'gaussian', 'robustness_test': True}
        )
        
        # Noisy Rosenbrock
        def noisy_rosenbrock(params):
            values = list(params.values())
            if len(values) < 2:
                return sum(v**2 for v in values)
            
            result = 0
            for i in range(len(values) - 1):
                result += 100 * (values[i+1] - values[i]**2)**2 + (1 - values[i])**2
            
            # Add noise with outliers (mixture model)
            if np.random.random() < 0.1:  # 10% chance of outlier
                noise = np.random.normal(0, result * 0.5)
            else:
                noise = np.random.normal(0, result * 0.05)
            
            return result + noise
        
        self.problems["noisy_rosenbrock_2d"] = BenchmarkProblem(
            name="noisy_rosenbrock_2d",
            param_space={'x': (-2.0, 2.0), 'y': (-1.0, 3.0)},
            objective_function=noisy_rosenbrock,
            optimal_value=0.0,
            optimal_parameters={'x': 1.0, 'y': 1.0},
            category=ProblemCategory.NOISY,
            difficulty=DifficultyLevel.HARD,
            description="Rosenbrock function with noise and outliers",
            properties={'noisy': True, 'outliers': True, 'mixture_noise': True}
        )
    
    def _initialize_high_dimensional_problems(self):
        """Initialize high-dimensional problems to test scalability"""
        
        # High-dimensional Sphere
        def sphere_10d(params):
            return sum(v**2 for v in params.values())
        
        param_space_10d = {f'x{i}': (-5.0, 5.0) for i in range(10)}
        optimal_params_10d = {f'x{i}': 0.0 for i in range(10)}
        
        self.problems["sphere_10d"] = BenchmarkProblem(
            name="sphere_10d",
            param_space=param_space_10d,
            objective_function=sphere_10d,
            optimal_value=0.0,
            optimal_parameters=optimal_params_10d,
            category=ProblemCategory.HIGH_DIMENSIONAL,
            difficulty=DifficultyLevel.MEDIUM,
            description="10-dimensional sphere function for scalability testing",
            properties={'high_dimensional': True, 'separable': True, 'scalability_test': True}
        )
        
        # High-dimensional Rosenbrock
        def rosenbrock_5d(params):
            values = list(params.values())
            result = 0
            for i in range(len(values) - 1):
                result += 100 * (values[i+1] - values[i]**2)**2 + (1 - values[i])**2
            return result
        
        param_space_5d = {f'x{i}': (-2.0, 2.0) for i in range(5)}
        optimal_params_5d = {f'x{i}': 1.0 for i in range(5)}
        
        self.problems["rosenbrock_5d"] = BenchmarkProblem(
            name="rosenbrock_5d",
            param_space=param_space_5d,
            objective_function=rosenbrock_5d,
            optimal_value=0.0,
            optimal_parameters=optimal_params_5d,
            category=ProblemCategory.HIGH_DIMENSIONAL,
            difficulty=DifficultyLevel.HARD,
            description="5-dimensional Rosenbrock function",
            properties={'high_dimensional': True, 'valley_shaped': True, 'non_separable': True}
        )
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the benchmark suite"""
        summary = {
            'total_problems': len(self.problems),
            'by_category': {},
            'by_difficulty': {},
            'dimensionality_stats': {},
            'problem_list': list(self.problems.keys())
        }
        
        # Count by category
        for category in ProblemCategory:
            count = len([p for p in self.problems.values() if p.category == category])
            summary['by_category'][category.value] = count
        
        # Count by difficulty
        for difficulty in DifficultyLevel:
            count = len([p for p in self.problems.values() if p.difficulty == difficulty])
            summary['by_difficulty'][difficulty.value] = count
        
        # Dimensionality statistics
        dimensions = [p.get_dimensionality() for p in self.problems.values()]
        summary['dimensionality_stats'] = {
            'min_dimensions': min(dimensions),
            'max_dimensions': max(dimensions),
            'avg_dimensions': np.mean(dimensions),
            'dimensions_distribution': {
                '2D': len([d for d in dimensions if d == 2]),
                '3D': len([d for d in dimensions if d == 3]),
                '5D': len([d for d in dimensions if d == 5]),
                '10D+': len([d for d in dimensions if d >= 10])
            }
        }
        
        return summary
    
    def validate_problem(self, problem_name: str) -> Dict[str, Any]:
        """Validate a benchmark problem"""
        if problem_name not in self.problems:
            return {'valid': False, 'error': f'Problem {problem_name} not found'}
        
        problem = self.problems[problem_name]
        validation_results = {'valid': True, 'checks': {}}
        
        try:
            # Test function evaluation
            test_params = {}
            for param, (low, high) in problem.param_space.items():
                test_params[param] = (low + high) / 2
            
            result = problem.evaluate(test_params)
            validation_results['checks']['function_evaluation'] = {
                'passed': isinstance(result, (int, float)) and not np.isnan(result),
                'test_value': result
            }
            
            # Test optimal solution if available
            if problem.optimal_parameters:
                optimal_result = problem.evaluate(problem.optimal_parameters)
                validation_results['checks']['optimal_evaluation'] = {
                    'passed': True,
                    'optimal_value': optimal_result,
                    'expected_value': problem.optimal_value
                }
            
            # Test parameter bounds
            validation_results['checks']['parameter_bounds'] = {
                'passed': all(low < high for low, high in problem.param_space.values()),
                'bounds_valid': True
            }
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['error'] = str(e)
        
        return validation_results
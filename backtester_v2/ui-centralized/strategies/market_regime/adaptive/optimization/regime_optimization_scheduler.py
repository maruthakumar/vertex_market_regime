"""
Regime Optimization Scheduler

This module implements intelligent scheduling and orchestration for regime
optimization tasks, coordinating all components of the adaptive system
for optimal performance and resource utilization.

Key Features:
- Intelligent task scheduling based on market conditions
- Resource-aware optimization orchestration
- Priority-based task management
- Adaptive scheduling algorithms
- Cross-component optimization coordination
- Performance-driven scheduling decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict, namedtuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
import threading
import queue
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
import json

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class TaskType(Enum):
    """Types of optimization tasks"""
    ASL_WEIGHT_UPDATE = "asl_weight_update"
    BOUNDARY_OPTIMIZATION = "boundary_optimization"
    TRANSITION_ANALYSIS = "transition_analysis"
    NOISE_FILTER_TUNING = "noise_filter_tuning"
    STABILITY_MONITORING = "stability_monitoring"
    PERFORMANCE_FEEDBACK = "performance_feedback"
    CONTINUOUS_LEARNING = "continuous_learning"
    SYSTEM_RECALIBRATION = "system_recalibration"
    DRIFT_DETECTION = "drift_detection"
    MODEL_VALIDATION = "model_validation"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class SchedulingMode(Enum):
    """Scheduling modes"""
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    ADAPTIVE = "adaptive"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class OptimizationTask:
    """Individual optimization task"""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    parameters: Dict[str, Any]
    dependencies: List[str]
    estimated_duration: float
    resource_requirements: Dict[str, float]
    deadline: Optional[datetime]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Priority comparison for heap queue"""
        return self.priority.value < other.priority.value


@dataclass
class SchedulingRule:
    """Rule for task scheduling"""
    condition: Callable[[Dict[str, Any]], bool]
    action: TaskType
    priority: TaskPriority
    parameters: Dict[str, Any]
    cooldown_period: timedelta
    last_triggered: Optional[datetime] = None


@dataclass
class ResourceUsage:
    """Resource usage tracking"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_usage: float
    disk_io: float
    timestamp: datetime


@dataclass
class SchedulerConfiguration:
    """Configuration for the scheduler"""
    # Scheduling behavior
    scheduling_mode: SchedulingMode = SchedulingMode.ADAPTIVE
    max_concurrent_tasks: int = 4
    task_timeout: float = 300.0  # seconds
    
    # Resource management
    max_cpu_usage: float = 0.8
    max_memory_usage: float = 0.7
    max_gpu_usage: float = 0.9
    resource_check_interval: float = 5.0  # seconds
    
    # Task management
    default_task_priority: TaskPriority = TaskPriority.MEDIUM
    priority_aging_factor: float = 0.1
    max_queue_size: int = 1000
    
    # Adaptive behavior
    performance_threshold: float = 0.7
    adaptation_sensitivity: float = 0.1
    market_condition_weight: float = 0.3
    
    # Scheduling intervals
    routine_optimization_interval: int = 3600  # seconds
    critical_monitoring_interval: int = 60     # seconds
    background_cleanup_interval: int = 7200    # seconds


class RegimeOptimizationScheduler:
    """
    Intelligent scheduler for regime optimization tasks
    """
    
    def __init__(self, config: Optional[SchedulerConfiguration] = None):
        """
        Initialize the optimization scheduler
        
        Args:
            config: Scheduler configuration
        """
        self.config = config or SchedulerConfiguration()
        
        # Task management
        self.task_queue = []  # Priority queue
        self.running_tasks: Dict[str, OptimizationTask] = {}
        self.completed_tasks: Dict[str, OptimizationTask] = {}
        self.failed_tasks: Dict[str, OptimizationTask] = {}
        
        # Scheduling rules
        self.scheduling_rules: List[SchedulingRule] = []
        self._initialize_default_rules()
        
        # Resource monitoring
        self.resource_usage_history = deque(maxlen=1000)
        self.current_resource_usage = ResourceUsage(0, 0, 0, 0, 0, datetime.now())
        
        # Execution infrastructure
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        self.scheduler_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.task_performance_history: Dict[TaskType, deque] = {
            task_type: deque(maxlen=100) for task_type in TaskType
        }
        
        # Market context
        self.current_market_conditions: Dict[str, Any] = {}
        self.system_performance_metrics: Dict[str, float] = {}
        
        # Adaptive scheduling state
        self.scheduling_effectiveness: Dict[TaskType, float] = {}
        self.adaptive_priorities: Dict[TaskType, float] = {}
        self.last_optimization: Dict[str, datetime] = {}
        
        # Statistics
        self.total_tasks_scheduled = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.average_task_duration: Dict[TaskType, float] = {}
        
        logger.info("RegimeOptimizationScheduler initialized")
    
    def _initialize_default_rules(self):
        """Initialize default scheduling rules"""
        
        # Critical performance degradation
        self.scheduling_rules.append(SchedulingRule(
            condition=lambda ctx: ctx.get('system_performance', 1.0) < 0.5,
            action=TaskType.SYSTEM_RECALIBRATION,
            priority=TaskPriority.CRITICAL,
            parameters={'full_recalibration': True},
            cooldown_period=timedelta(hours=1)
        ))
        
        # Concept drift detection
        self.scheduling_rules.append(SchedulingRule(
            condition=lambda ctx: ctx.get('drift_detected', False),
            action=TaskType.CONTINUOUS_LEARNING,
            priority=TaskPriority.HIGH,
            parameters={'aggressive_learning': True},
            cooldown_period=timedelta(minutes=30)
        ))
        
        # High volatility periods
        self.scheduling_rules.append(SchedulingRule(
            condition=lambda ctx: ctx.get('volatility', 0.0) > 0.4,
            action=TaskType.NOISE_FILTER_TUNING,
            priority=TaskPriority.HIGH,
            parameters={'high_volatility_mode': True},
            cooldown_period=timedelta(minutes=15)
        ))
        
        # Poor transition quality
        self.scheduling_rules.append(SchedulingRule(
            condition=lambda ctx: ctx.get('transition_quality', 1.0) < 0.6,
            action=TaskType.BOUNDARY_OPTIMIZATION,
            priority=TaskPriority.HIGH,
            parameters={'focus_on_transitions': True},
            cooldown_period=timedelta(minutes=45)
        ))
        
        # Routine optimization
        self.scheduling_rules.append(SchedulingRule(
            condition=lambda ctx: self._time_since_last_optimization('routine') > timedelta(hours=1),
            action=TaskType.ASL_WEIGHT_UPDATE,
            priority=TaskPriority.MEDIUM,
            parameters={'routine_update': True},
            cooldown_period=timedelta(hours=1)
        ))
        
        # Background stability monitoring
        self.scheduling_rules.append(SchedulingRule(
            condition=lambda ctx: self._time_since_last_optimization('stability') > timedelta(minutes=5),
            action=TaskType.STABILITY_MONITORING,
            priority=TaskPriority.LOW,
            parameters={'background_check': True},
            cooldown_period=timedelta(minutes=5)
        ))
    
    def start_scheduler(self):
        """Start the optimization scheduler"""
        
        if self.scheduler_running:
            logger.warning("Scheduler is already running")
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Optimization scheduler started")
    
    def stop_scheduler(self):
        """Stop the optimization scheduler"""
        
        self.scheduler_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        # Cancel pending tasks
        self._cancel_pending_tasks()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Optimization scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        
        logger.debug("Scheduler loop started")
        
        while self.scheduler_running:
            try:
                # Update resource usage
                self._update_resource_usage()
                
                # Check scheduling rules
                self._check_scheduling_rules()
                
                # Process task queue
                self._process_task_queue()
                
                # Clean up completed tasks
                self._cleanup_tasks()
                
                # Adaptive scheduling updates
                if self.config.scheduling_mode == SchedulingMode.ADAPTIVE:
                    self._update_adaptive_scheduling()
                
                # Sleep before next iteration
                time.sleep(self.config.resource_check_interval)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(1.0)
        
        logger.debug("Scheduler loop stopped")
    
    def schedule_task(self, task: OptimizationTask) -> bool:
        """
        Schedule an optimization task
        
        Args:
            task: Task to schedule
            
        Returns:
            True if task was scheduled successfully
        """
        if len(self.task_queue) >= self.config.max_queue_size:
            logger.warning("Task queue is full, cannot schedule new task")
            return False
        
        # Apply priority aging if task has been waiting
        if task.created_at:
            wait_time = datetime.now() - task.created_at
            if wait_time.total_seconds() > 300:  # 5 minutes
                aging_factor = min(wait_time.total_seconds() / 3600, 1.0)  # Max 1 hour aging
                new_priority_value = max(1, task.priority.value - aging_factor)
                
                # Convert back to enum
                for priority in TaskPriority:
                    if priority.value <= new_priority_value:
                        task.priority = priority
                        break
        
        # Add to priority queue
        heapq.heappush(self.task_queue, task)
        self.total_tasks_scheduled += 1
        
        logger.debug(f"Scheduled task: {task.task_type.value} (priority: {task.priority.value})")
        return True
    
    def _check_scheduling_rules(self):
        """Check scheduling rules and create tasks as needed"""
        
        # Prepare context for rule evaluation
        context = {
            **self.current_market_conditions,
            **self.system_performance_metrics,
            'current_time': datetime.now(),
            'running_tasks': len(self.running_tasks),
            'queue_size': len(self.task_queue)
        }
        
        for rule in self.scheduling_rules:
            try:
                # Check cooldown
                if (rule.last_triggered and 
                    datetime.now() - rule.last_triggered < rule.cooldown_period):
                    continue
                
                # Evaluate condition
                if rule.condition(context):
                    # Create and schedule task
                    task = OptimizationTask(
                        task_id=self._generate_task_id(),
                        task_type=rule.action,
                        priority=rule.priority,
                        parameters=rule.parameters.copy(),
                        dependencies=[],
                        estimated_duration=self._estimate_task_duration(rule.action),
                        resource_requirements=self._estimate_resource_requirements(rule.action),
                        deadline=None,
                        created_at=datetime.now()
                    )
                    
                    if self.schedule_task(task):
                        rule.last_triggered = datetime.now()
                        logger.debug(f"Rule triggered: {rule.action.value}")
                
            except Exception as e:
                logger.error(f"Error evaluating scheduling rule: {e}")
    
    def _process_task_queue(self):
        """Process tasks from the queue"""
        
        # Check if we can start new tasks
        if len(self.running_tasks) >= self.config.max_concurrent_tasks:
            return
        
        # Check resource availability
        if not self._has_available_resources():
            return
        
        # Get next task
        while self.task_queue and len(self.running_tasks) < self.config.max_concurrent_tasks:
            try:
                task = heapq.heappop(self.task_queue)
                
                # Check dependencies
                if not self._check_dependencies(task):
                    # Re-queue task
                    heapq.heappush(self.task_queue, task)
                    break
                
                # Check resource requirements
                if not self._check_resource_requirements(task):
                    # Re-queue task
                    heapq.heappush(self.task_queue, task)
                    break
                
                # Start task
                self._start_task(task)
                
            except Exception as e:
                logger.error(f"Error processing task queue: {e}")
                break
    
    def _start_task(self, task: OptimizationTask):
        """Start executing a task"""
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.scheduled_at = datetime.now()
        
        self.running_tasks[task.task_id] = task
        
        # Submit to executor
        future = self.executor.submit(self._execute_task, task)
        
        # Add callback for completion
        future.add_done_callback(lambda f: self._task_completed(task.task_id, f))
        
        logger.info(f"Started task: {task.task_type.value} (ID: {task.task_id})")
    
    def _execute_task(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute an optimization task"""
        
        start_time = time.time()
        
        try:
            # Route to appropriate handler
            result = self._route_task_execution(task)
            
            # Record performance
            execution_time = time.time() - start_time
            self._record_task_performance(task.task_type, execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_task_performance(task.task_type, execution_time, False)
            raise e
    
    def _route_task_execution(self, task: OptimizationTask) -> Dict[str, Any]:
        """Route task to appropriate execution handler"""
        
        handlers = {
            TaskType.ASL_WEIGHT_UPDATE: self._execute_asl_update,
            TaskType.BOUNDARY_OPTIMIZATION: self._execute_boundary_optimization,
            TaskType.TRANSITION_ANALYSIS: self._execute_transition_analysis,
            TaskType.NOISE_FILTER_TUNING: self._execute_noise_filter_tuning,
            TaskType.STABILITY_MONITORING: self._execute_stability_monitoring,
            TaskType.PERFORMANCE_FEEDBACK: self._execute_performance_feedback,
            TaskType.CONTINUOUS_LEARNING: self._execute_continuous_learning,
            TaskType.SYSTEM_RECALIBRATION: self._execute_system_recalibration,
            TaskType.DRIFT_DETECTION: self._execute_drift_detection,
            TaskType.MODEL_VALIDATION: self._execute_model_validation
        }
        
        handler = handlers.get(task.task_type)
        if not handler:
            raise ValueError(f"No handler for task type: {task.task_type}")
        
        return handler(task)
    
    def _execute_asl_update(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute ASL weight update"""
        
        logger.debug(f"Executing ASL update: {task.task_id}")
        
        # Simulate ASL weight update
        # In real implementation, this would call the actual ASL update methods
        time.sleep(np.random.uniform(1, 3))  # Simulate processing time
        
        result = {
            'task_type': 'asl_weight_update',
            'weights_updated': True,
            'improvement': np.random.uniform(0.01, 0.05),
            'new_weights': np.random.dirichlet(np.ones(5)).tolist()
        }
        
        return result
    
    def _execute_boundary_optimization(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute boundary optimization"""
        
        logger.debug(f"Executing boundary optimization: {task.task_id}")
        
        # Simulate boundary optimization
        time.sleep(np.random.uniform(5, 15))  # Simulate processing time
        
        result = {
            'task_type': 'boundary_optimization',
            'boundaries_optimized': True,
            'convergence_iterations': np.random.randint(10, 50),
            'improvement': np.random.uniform(0.02, 0.08)
        }
        
        return result
    
    def _execute_transition_analysis(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute transition analysis"""
        
        logger.debug(f"Executing transition analysis: {task.task_id}")
        
        # Simulate transition analysis
        time.sleep(np.random.uniform(2, 5))  # Simulate processing time
        
        result = {
            'task_type': 'transition_analysis',
            'patterns_analyzed': np.random.randint(10, 30),
            'new_patterns_found': np.random.randint(0, 5),
            'transition_quality_improvement': np.random.uniform(0.01, 0.03)
        }
        
        return result
    
    def _execute_noise_filter_tuning(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute noise filter tuning"""
        
        logger.debug(f"Executing noise filter tuning: {task.task_id}")
        
        # Simulate noise filter tuning
        time.sleep(np.random.uniform(1, 4))  # Simulate processing time
        
        result = {
            'task_type': 'noise_filter_tuning',
            'filters_tuned': ['microstructure', 'volume', 'volatility'],
            'noise_reduction': np.random.uniform(0.05, 0.15),
            'false_positive_reduction': np.random.uniform(0.02, 0.08)
        }
        
        return result
    
    def _execute_stability_monitoring(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute stability monitoring"""
        
        logger.debug(f"Executing stability monitoring: {task.task_id}")
        
        # Simulate stability monitoring
        time.sleep(np.random.uniform(0.5, 2))  # Simulate processing time
        
        result = {
            'task_type': 'stability_monitoring',
            'regimes_monitored': 12,
            'stability_scores': np.random.uniform(0.6, 0.9, 12).tolist(),
            'anomalies_detected': np.random.randint(0, 3),
            'overall_stability': np.random.uniform(0.7, 0.9)
        }
        
        return result
    
    def _execute_performance_feedback(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute performance feedback analysis"""
        
        logger.debug(f"Executing performance feedback: {task.task_id}")
        
        # Simulate performance feedback
        time.sleep(np.random.uniform(1, 3))  # Simulate processing time
        
        result = {
            'task_type': 'performance_feedback',
            'feedback_actions_generated': np.random.randint(2, 8),
            'system_performance_score': np.random.uniform(0.6, 0.9),
            'recommendations': ['tune_learning_rate', 'adjust_boundaries']
        }
        
        return result
    
    def _execute_continuous_learning(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute continuous learning update"""
        
        logger.debug(f"Executing continuous learning: {task.task_id}")
        
        # Simulate continuous learning
        time.sleep(np.random.uniform(3, 8))  # Simulate processing time
        
        result = {
            'task_type': 'continuous_learning',
            'models_updated': np.random.randint(2, 5),
            'learning_examples_processed': np.random.randint(100, 500),
            'accuracy_improvement': np.random.uniform(0.01, 0.05)
        }
        
        return result
    
    def _execute_system_recalibration(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute system recalibration"""
        
        logger.debug(f"Executing system recalibration: {task.task_id}")
        
        # Simulate system recalibration
        time.sleep(np.random.uniform(10, 30))  # Simulate processing time
        
        result = {
            'task_type': 'system_recalibration',
            'components_recalibrated': ['asl', 'boundaries', 'transitions'],
            'overall_improvement': np.random.uniform(0.05, 0.15),
            'recalibration_successful': True
        }
        
        return result
    
    def _execute_drift_detection(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute drift detection"""
        
        logger.debug(f"Executing drift detection: {task.task_id}")
        
        # Simulate drift detection
        time.sleep(np.random.uniform(1, 3))  # Simulate processing time
        
        result = {
            'task_type': 'drift_detection',
            'drift_detected': np.random.choice([True, False], p=[0.2, 0.8]),
            'drift_severity': np.random.uniform(0.0, 0.8),
            'affected_features': np.random.randint(0, 5)
        }
        
        return result
    
    def _execute_model_validation(self, task: OptimizationTask) -> Dict[str, Any]:
        """Execute model validation"""
        
        logger.debug(f"Executing model validation: {task.task_id}")
        
        # Simulate model validation
        time.sleep(np.random.uniform(2, 6))  # Simulate processing time
        
        result = {
            'task_type': 'model_validation',
            'models_validated': np.random.randint(3, 8),
            'validation_accuracy': np.random.uniform(0.7, 0.95),
            'models_requiring_update': np.random.randint(0, 3)
        }
        
        return result
    
    def _task_completed(self, task_id: str, future):
        """Handle task completion"""
        
        if task_id not in self.running_tasks:
            return
        
        task = self.running_tasks.pop(task_id)
        task.completed_at = datetime.now()
        
        try:
            result = future.result()
            task.status = TaskStatus.COMPLETED
            task.result = result
            self.completed_tasks[task_id] = task
            self.total_tasks_completed += 1
            
            logger.info(f"Task completed: {task.task_type.value} (ID: {task_id})")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.retry_count += 1
            
            # Retry if under limit
            if task.retry_count <= task.max_retries:
                task.status = TaskStatus.PENDING
                task.started_at = None
                heapq.heappush(self.task_queue, task)
                logger.warning(f"Task failed, retrying: {task.task_type.value} (attempt {task.retry_count})")
            else:
                self.failed_tasks[task_id] = task
                self.total_tasks_failed += 1
                logger.error(f"Task failed permanently: {task.task_type.value} - {e}")
    
    def _check_dependencies(self, task: OptimizationTask) -> bool:
        """Check if task dependencies are satisfied"""
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    def _check_resource_requirements(self, task: OptimizationTask) -> bool:
        """Check if resources are available for task"""
        
        required_cpu = task.resource_requirements.get('cpu', 0.1)
        required_memory = task.resource_requirements.get('memory', 0.1)
        
        available_cpu = self.config.max_cpu_usage - self.current_resource_usage.cpu_usage
        available_memory = self.config.max_memory_usage - self.current_resource_usage.memory_usage
        
        return available_cpu >= required_cpu and available_memory >= required_memory
    
    def _has_available_resources(self) -> bool:
        """Check if system has available resources"""
        
        return (self.current_resource_usage.cpu_usage < self.config.max_cpu_usage and
                self.current_resource_usage.memory_usage < self.config.max_memory_usage)
    
    def _update_resource_usage(self):
        """Update current resource usage"""
        
        # Simulate resource monitoring
        # In real implementation, this would query actual system resources
        base_cpu = 0.1 + len(self.running_tasks) * 0.15
        base_memory = 0.2 + len(self.running_tasks) * 0.1
        
        self.current_resource_usage = ResourceUsage(
            cpu_usage=min(0.95, base_cpu + np.random.uniform(-0.05, 0.1)),
            memory_usage=min(0.95, base_memory + np.random.uniform(-0.05, 0.1)),
            gpu_usage=np.random.uniform(0.0, 0.3),
            network_usage=np.random.uniform(0.0, 0.2),
            disk_io=np.random.uniform(0.0, 0.4),
            timestamp=datetime.now()
        )
        
        self.resource_usage_history.append(self.current_resource_usage)
    
    def _record_task_performance(self, task_type: TaskType, execution_time: float, success: bool):
        """Record task performance metrics"""
        
        performance_data = {
            'execution_time': execution_time,
            'success': success,
            'timestamp': datetime.now()
        }
        
        self.task_performance_history[task_type].append(performance_data)
        
        # Update average duration
        successful_executions = [
            p['execution_time'] for p in self.task_performance_history[task_type]
            if p['success']
        ]
        
        if successful_executions:
            self.average_task_duration[task_type] = np.mean(successful_executions)
    
    def _estimate_task_duration(self, task_type: TaskType) -> float:
        """Estimate task duration based on historical data"""
        
        if task_type in self.average_task_duration:
            return self.average_task_duration[task_type]
        
        # Default estimates
        default_durations = {
            TaskType.ASL_WEIGHT_UPDATE: 2.0,
            TaskType.BOUNDARY_OPTIMIZATION: 10.0,
            TaskType.TRANSITION_ANALYSIS: 3.0,
            TaskType.NOISE_FILTER_TUNING: 2.5,
            TaskType.STABILITY_MONITORING: 1.0,
            TaskType.PERFORMANCE_FEEDBACK: 2.0,
            TaskType.CONTINUOUS_LEARNING: 5.0,
            TaskType.SYSTEM_RECALIBRATION: 20.0,
            TaskType.DRIFT_DETECTION: 2.0,
            TaskType.MODEL_VALIDATION: 4.0
        }
        
        return default_durations.get(task_type, 5.0)
    
    def _estimate_resource_requirements(self, task_type: TaskType) -> Dict[str, float]:
        """Estimate resource requirements for task type"""
        
        # Resource requirement estimates
        requirements = {
            TaskType.ASL_WEIGHT_UPDATE: {'cpu': 0.1, 'memory': 0.05},
            TaskType.BOUNDARY_OPTIMIZATION: {'cpu': 0.3, 'memory': 0.15},
            TaskType.TRANSITION_ANALYSIS: {'cpu': 0.15, 'memory': 0.1},
            TaskType.NOISE_FILTER_TUNING: {'cpu': 0.1, 'memory': 0.05},
            TaskType.STABILITY_MONITORING: {'cpu': 0.05, 'memory': 0.03},
            TaskType.PERFORMANCE_FEEDBACK: {'cpu': 0.1, 'memory': 0.05},
            TaskType.CONTINUOUS_LEARNING: {'cpu': 0.25, 'memory': 0.2},
            TaskType.SYSTEM_RECALIBRATION: {'cpu': 0.4, 'memory': 0.3},
            TaskType.DRIFT_DETECTION: {'cpu': 0.1, 'memory': 0.08},
            TaskType.MODEL_VALIDATION: {'cpu': 0.2, 'memory': 0.15}
        }
        
        return requirements.get(task_type, {'cpu': 0.1, 'memory': 0.1})
    
    def _update_adaptive_scheduling(self):
        """Update adaptive scheduling parameters"""
        
        if self.config.scheduling_mode != SchedulingMode.ADAPTIVE:
            return
        
        # Calculate task type effectiveness
        for task_type, history in self.task_performance_history.items():
            if len(history) >= 5:
                recent_success_rate = np.mean([h['success'] for h in history[-10:]])
                recent_avg_time = np.mean([h['execution_time'] for h in history[-10:] if h['success']])
                
                # Effectiveness = success_rate / normalized_time
                normalized_time = recent_avg_time / self._estimate_task_duration(task_type)
                effectiveness = recent_success_rate / max(normalized_time, 0.1)
                
                self.scheduling_effectiveness[task_type] = effectiveness
                
                # Adjust adaptive priorities
                if effectiveness > 1.2:  # High effectiveness
                    self.adaptive_priorities[task_type] = min(1.0, self.adaptive_priorities.get(task_type, 0.5) + 0.1)
                elif effectiveness < 0.8:  # Low effectiveness
                    self.adaptive_priorities[task_type] = max(0.1, self.adaptive_priorities.get(task_type, 0.5) - 0.1)
    
    def _cleanup_tasks(self):
        """Clean up old completed and failed tasks"""
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean completed tasks
        old_completed = [
            task_id for task_id, task in self.completed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        
        for task_id in old_completed:
            del self.completed_tasks[task_id]
        
        # Clean failed tasks
        old_failed = [
            task_id for task_id, task in self.failed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        
        for task_id in old_failed:
            del self.failed_tasks[task_id]
    
    def _cancel_pending_tasks(self):
        """Cancel all pending tasks"""
        
        cancelled_count = 0
        
        while self.task_queue:
            task = heapq.heappop(self.task_queue)
            task.status = TaskStatus.CANCELLED
            cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} pending tasks")
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"task_{timestamp}_{np.random.randint(1000, 9999)}"
    
    def _time_since_last_optimization(self, optimization_type: str) -> timedelta:
        """Get time since last optimization of given type"""
        
        if optimization_type not in self.last_optimization:
            return timedelta(days=1)  # Force optimization if never done
        
        return datetime.now() - self.last_optimization[optimization_type]
    
    def update_market_conditions(self, conditions: Dict[str, Any]):
        """Update current market conditions for scheduling decisions"""
        
        self.current_market_conditions.update(conditions)
        
        # Mark time for routine optimizations
        if 'volatility' in conditions:
            self.last_optimization['market_update'] = datetime.now()
    
    def update_system_performance(self, metrics: Dict[str, float]):
        """Update system performance metrics"""
        
        self.system_performance_metrics.update(metrics)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        
        status = {
            'scheduler_running': self.scheduler_running,
            'queue_status': {
                'pending_tasks': len(self.task_queue),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks)
            },
            'statistics': {
                'total_scheduled': self.total_tasks_scheduled,
                'total_completed': self.total_tasks_completed,
                'total_failed': self.total_tasks_failed,
                'success_rate': self.total_tasks_completed / max(self.total_tasks_scheduled, 1)
            },
            'resource_usage': {
                'cpu': self.current_resource_usage.cpu_usage,
                'memory': self.current_resource_usage.memory_usage,
                'gpu': self.current_resource_usage.gpu_usage
            },
            'performance_metrics': {
                task_type.value: {
                    'avg_duration': self.average_task_duration.get(task_type, 0.0),
                    'effectiveness': self.scheduling_effectiveness.get(task_type, 0.0),
                    'recent_tasks': len(history)
                }
                for task_type, history in self.task_performance_history.items()
                if len(history) > 0
            }
        }
        
        return status
    
    def export_scheduler_data(self, filepath: str):
        """Export comprehensive scheduler data"""
        
        export_data = {
            'configuration': self.config.__dict__,
            'status': self.get_scheduler_status(),
            'current_market_conditions': self.current_market_conditions,
            'system_performance_metrics': self.system_performance_metrics,
            'recent_completed_tasks': [
                {
                    'task_id': task.task_id,
                    'task_type': task.task_type.value,
                    'priority': task.priority.value,
                    'duration': (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else 0,
                    'result_summary': str(task.result)[:200] if task.result else None
                }
                for task in list(self.completed_tasks.values())[-20:]
            ],
            'scheduling_rules': [
                {
                    'action': rule.action.value,
                    'priority': rule.priority.value,
                    'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for rule in self.scheduling_rules
            ],
            'adaptive_parameters': {
                'scheduling_effectiveness': {
                    k.value: v for k, v in self.scheduling_effectiveness.items()
                },
                'adaptive_priorities': {
                    k.value: v for k, v in self.adaptive_priorities.items()
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Scheduler data exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create scheduler
    config = SchedulerConfiguration(
        scheduling_mode=SchedulingMode.ADAPTIVE,
        max_concurrent_tasks=3,
        max_cpu_usage=0.7,
        max_memory_usage=0.6
    )
    
    scheduler = RegimeOptimizationScheduler(config)
    
    # Start scheduler
    scheduler.start_scheduler()
    
    # Simulate some market conditions and system performance updates
    try:
        for i in range(20):
            # Update market conditions
            market_conditions = {
                'volatility': np.random.uniform(0.1, 0.5),
                'trend': np.random.uniform(-0.02, 0.02),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'drift_detected': np.random.choice([True, False], p=[0.1, 0.9])
            }
            scheduler.update_market_conditions(market_conditions)
            
            # Update system performance
            system_performance = {
                'system_performance': np.random.uniform(0.6, 0.9),
                'transition_quality': np.random.uniform(0.5, 0.8),
                'overall_accuracy': np.random.uniform(0.6, 0.85)
            }
            scheduler.update_system_performance(system_performance)
            
            # Manually schedule a task
            if i % 5 == 0:
                manual_task = OptimizationTask(
                    task_id=f"manual_task_{i}",
                    task_type=TaskType.MODEL_VALIDATION,
                    priority=TaskPriority.HIGH,
                    parameters={'manual_trigger': True},
                    dependencies=[],
                    estimated_duration=3.0,
                    resource_requirements={'cpu': 0.2, 'memory': 0.1},
                    deadline=None,
                    created_at=datetime.now()
                )
                scheduler.schedule_task(manual_task)
            
            time.sleep(2)
        
        # Let it run for a bit
        time.sleep(10)
        
        # Get status
        status = scheduler.get_scheduler_status()
        print("Scheduler Status:")
        print(f"Running: {status['scheduler_running']}")
        print(f"Pending tasks: {status['queue_status']['pending_tasks']}")
        print(f"Running tasks: {status['queue_status']['running_tasks']}")
        print(f"Completed tasks: {status['queue_status']['completed_tasks']}")
        print(f"Success rate: {status['statistics']['success_rate']:.2%}")
        
    finally:
        # Stop scheduler
        scheduler.stop_scheduler()
        print("Scheduler stopped")
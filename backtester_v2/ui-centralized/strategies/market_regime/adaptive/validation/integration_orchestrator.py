"""
Integration Orchestrator

This module orchestrates the complete integration of all adaptive market regime
formation system components, managing data flow, component coordination, and
ensuring seamless operation across all phases.

Key Features:
- End-to-end pipeline orchestration
- Component lifecycle management
- Data flow coordination
- State synchronization across components
- Error handling and recovery
- Performance optimization
- Real-time monitoring and control
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
import threading
import queue
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class ComponentState(Enum):
    """Component operational states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class PipelineStage(Enum):
    """Pipeline processing stages"""
    DATA_INGESTION = "data_ingestion"
    FEATURE_ENGINEERING = "feature_engineering"
    REGIME_SCORING = "regime_scoring"
    TRANSITION_ANALYSIS = "transition_analysis"
    NOISE_FILTERING = "noise_filtering"
    REGIME_DECISION = "regime_decision"
    BOUNDARY_OPTIMIZATION = "boundary_optimization"
    PERFORMANCE_FEEDBACK = "performance_feedback"
    LEARNING_UPDATE = "learning_update"
    MONITORING = "monitoring"


class OrchestrationMode(Enum):
    """Orchestration operational modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    REAL_TIME = "real_time"
    BATCH = "batch"


@dataclass
class ComponentInfo:
    """Information about a system component"""
    component_id: str
    component_type: str
    module_reference: Any
    state: ComponentState
    dependencies: List[str]
    initialization_params: Dict[str, Any]
    last_update: datetime
    error_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineTask:
    """Task in the processing pipeline"""
    task_id: str
    stage: PipelineStage
    component_id: str
    input_data: Dict[str, Any]
    timestamp: datetime
    priority: int = 5
    timeout: float = 60.0
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Priority comparison for queue"""
        return self.priority < other.priority


@dataclass
class DataFlowPath:
    """Data flow path between components"""
    source_component: str
    target_component: str
    data_type: str
    transformation: Optional[Callable] = None
    validation: Optional[Callable] = None
    enabled: bool = True


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestrator"""
    mode: OrchestrationMode = OrchestrationMode.ADAPTIVE
    max_workers: int = 4
    task_timeout: float = 60.0
    batch_size: int = 100
    real_time_interval: float = 1.0
    error_threshold: int = 10
    performance_monitoring: bool = True
    state_persistence: bool = True
    state_file: str = "orchestrator_state.json"


@dataclass
class SystemState:
    """Complete system state snapshot"""
    timestamp: datetime
    component_states: Dict[str, ComponentState]
    active_tasks: List[str]
    completed_tasks: int
    failed_tasks: int
    current_regime: int
    performance_metrics: Dict[str, Any]
    error_log: List[Dict[str, Any]]


class IntegrationOrchestrator:
    """
    Master orchestrator for the adaptive market regime formation system
    """
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        """
        Initialize the integration orchestrator
        
        Args:
            config: Orchestration configuration
        """
        self.config = config or OrchestrationConfig()
        
        # Component registry
        self.components: Dict[str, ComponentInfo] = {}
        self.component_order: List[str] = []
        
        # Data flow management
        self.data_flow_paths: List[DataFlowPath] = []
        self.data_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Pipeline management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, PipelineTask] = {}
        self.completed_tasks = deque(maxlen=10000)
        
        # Execution infrastructure
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.futures: Dict[str, Future] = {}
        
        # Orchestration state
        self.orchestrator_running = False
        self.orchestrator_thread: Optional[threading.Thread] = None
        self.current_regime = 0
        self.last_regime_update = datetime.now()
        
        # Performance tracking
        self.stage_latencies: Dict[PipelineStage, deque] = {
            stage: deque(maxlen=100) for stage in PipelineStage
        }
        self.component_errors: Dict[str, int] = defaultdict(int)
        
        # State management
        self.system_state_history = deque(maxlen=1000)
        self.state_lock = threading.Lock()
        
        # Initialize data flow paths
        self._initialize_data_flow_paths()
        
        logger.info(f"IntegrationOrchestrator initialized with mode: {config.mode.value}")
    
    def _initialize_data_flow_paths(self):
        """Initialize default data flow paths between components"""
        
        # Core data flows
        self.add_data_flow_path(DataFlowPath(
            source_component="data_ingestion",
            target_component="feature_engineering",
            data_type="market_data"
        ))
        
        self.add_data_flow_path(DataFlowPath(
            source_component="feature_engineering",
            target_component="adaptive_scoring_layer",
            data_type="features"
        ))
        
        self.add_data_flow_path(DataFlowPath(
            source_component="adaptive_scoring_layer",
            target_component="intelligent_transition_manager",
            data_type="regime_scores"
        ))
        
        self.add_data_flow_path(DataFlowPath(
            source_component="intelligent_transition_manager",
            target_component="regime_stability_monitor",
            data_type="regime_decision"
        ))
        
        # Feedback loops
        self.add_data_flow_path(DataFlowPath(
            source_component="performance_feedback_system",
            target_component="continuous_learning_engine",
            data_type="performance_metrics"
        ))
        
        self.add_data_flow_path(DataFlowPath(
            source_component="continuous_learning_engine",
            target_component="adaptive_scoring_layer",
            data_type="model_updates"
        ))
        
        self.add_data_flow_path(DataFlowPath(
            source_component="regime_optimization_scheduler",
            target_component="dynamic_boundary_optimizer",
            data_type="optimization_tasks"
        ))
    
    def register_component(self, component_id: str, component_type: str,
                         module_reference: Any, dependencies: List[str] = None,
                         initialization_params: Dict[str, Any] = None):
        """
        Register a component with the orchestrator
        
        Args:
            component_id: Unique component identifier
            component_type: Type of component
            module_reference: Reference to the component module
            dependencies: List of component dependencies
            initialization_params: Parameters for initialization
        """
        component_info = ComponentInfo(
            component_id=component_id,
            component_type=component_type,
            module_reference=module_reference,
            state=ComponentState.UNINITIALIZED,
            dependencies=dependencies or [],
            initialization_params=initialization_params or {},
            last_update=datetime.now()
        )
        
        self.components[component_id] = component_info
        self._update_component_order()
        
        logger.info(f"Registered component: {component_id} ({component_type})")
    
    def add_data_flow_path(self, path: DataFlowPath):
        """Add a data flow path between components"""
        
        self.data_flow_paths.append(path)
        logger.debug(f"Added data flow: {path.source_component} -> {path.target_component}")
    
    def initialize_system(self) -> bool:
        """
        Initialize all registered components in dependency order
        
        Returns:
            True if all components initialized successfully
        """
        logger.info("Initializing system components...")
        
        success = True
        
        for component_id in self.component_order:
            if not self._initialize_component(component_id):
                success = False
                break
        
        if success:
            logger.info("✅ All components initialized successfully")
        else:
            logger.error("❌ Component initialization failed")
        
        return success
    
    def _initialize_component(self, component_id: str) -> bool:
        """Initialize a single component"""
        
        if component_id not in self.components:
            logger.error(f"Component not found: {component_id}")
            return False
        
        component = self.components[component_id]
        
        # Check dependencies
        for dep in component.dependencies:
            if dep not in self.components:
                logger.error(f"Dependency not found: {dep}")
                return False
            
            dep_state = self.components[dep].state
            if dep_state not in [ComponentState.READY, ComponentState.RUNNING]:
                logger.error(f"Dependency not ready: {dep} (state: {dep_state.value})")
                return False
        
        try:
            # Update state
            component.state = ComponentState.INITIALIZING
            
            # Component-specific initialization
            if hasattr(component.module_reference, 'initialize'):
                component.module_reference.initialize(**component.initialization_params)
            
            # Update state
            component.state = ComponentState.READY
            component.last_update = datetime.now()
            
            logger.info(f"Initialized component: {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize component {component_id}: {e}")
            component.state = ComponentState.ERROR
            component.error_count += 1
            return False
    
    def start_orchestration(self):
        """Start the orchestration system"""
        
        if self.orchestrator_running:
            logger.warning("Orchestrator is already running")
            return
        
        # Verify all components are ready
        for component_id, component in self.components.items():
            if component.state != ComponentState.READY:
                logger.error(f"Component not ready: {component_id} (state: {component.state.value})")
                return
        
        self.orchestrator_running = True
        
        # Start orchestration thread based on mode
        if self.config.mode == OrchestrationMode.REAL_TIME:
            self.orchestrator_thread = threading.Thread(
                target=self._real_time_orchestration_loop,
                daemon=True
            )
        else:
            self.orchestrator_thread = threading.Thread(
                target=self._batch_orchestration_loop,
                daemon=True
            )
        
        self.orchestrator_thread.start()
        
        # Update component states
        for component in self.components.values():
            component.state = ComponentState.RUNNING
        
        logger.info(f"Orchestration started in {self.config.mode.value} mode")
    
    def stop_orchestration(self):
        """Stop the orchestration system"""
        
        logger.info("Stopping orchestration...")
        
        self.orchestrator_running = False
        
        # Wait for orchestration thread
        if self.orchestrator_thread:
            self.orchestrator_thread.join(timeout=5.0)
        
        # Cancel active tasks
        self._cancel_active_tasks()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Update component states
        for component in self.components.values():
            component.state = ComponentState.SHUTDOWN
        
        # Save final state
        if self.config.state_persistence:
            self._save_system_state()
        
        logger.info("Orchestration stopped")
    
    def _real_time_orchestration_loop(self):
        """Real-time orchestration loop"""
        
        logger.debug("Starting real-time orchestration loop")
        
        while self.orchestrator_running:
            try:
                start_time = time.time()
                
                # Process pipeline stages
                self._process_pipeline_stages()
                
                # Process task queue
                self._process_task_queue()
                
                # Monitor component health
                self._monitor_component_health()
                
                # Update system state
                self._update_system_state()
                
                # Sleep to maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.real_time_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                time.sleep(1.0)
    
    def _batch_orchestration_loop(self):
        """Batch processing orchestration loop"""
        
        logger.debug("Starting batch orchestration loop")
        
        while self.orchestrator_running:
            try:
                # Collect batch of data
                batch_data = self._collect_batch_data()
                
                if batch_data:
                    # Process batch through pipeline
                    self._process_batch(batch_data)
                
                # Process task queue
                self._process_task_queue()
                
                # Monitor and update
                self._monitor_component_health()
                self._update_system_state()
                
                # Small delay between batches
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in batch orchestration: {e}")
                time.sleep(1.0)
    
    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data through the complete pipeline
        
        Args:
            market_data: Market data to process
            
        Returns:
            Processing results including regime decision
        """
        timestamp = datetime.now()
        
        # Create initial task
        task = PipelineTask(
            task_id=f"market_data_{timestamp.timestamp()}",
            stage=PipelineStage.DATA_INGESTION,
            component_id="data_ingestion",
            input_data=market_data,
            timestamp=timestamp,
            priority=1  # High priority
        )
        
        # Add to queue
        self.task_queue.put(task)
        
        # Wait for completion if in sequential mode
        if self.config.mode == OrchestrationMode.SEQUENTIAL:
            return self._wait_for_task_completion(task.task_id)
        
        return {'task_id': task.task_id, 'status': 'processing'}
    
    def _process_pipeline_stages(self):
        """Process data through pipeline stages"""
        
        # Check each data flow path
        for path in self.data_flow_paths:
            if not path.enabled:
                continue
            
            # Check if source has data
            source_buffer = self.data_buffers.get(path.source_component)
            if source_buffer and len(source_buffer) > 0:
                # Process available data
                data = source_buffer.popleft()
                
                # Apply transformation if defined
                if path.transformation:
                    data = path.transformation(data)
                
                # Validate if defined
                if path.validation and not path.validation(data):
                    logger.warning(f"Data validation failed for path: "
                                 f"{path.source_component} -> {path.target_component}")
                    continue
                
                # Create task for target component
                task = PipelineTask(
                    task_id=f"{path.target_component}_{time.time()}",
                    stage=self._get_stage_for_component(path.target_component),
                    component_id=path.target_component,
                    input_data=data,
                    timestamp=datetime.now()
                )
                
                self.task_queue.put(task)
    
    def _process_task_queue(self):
        """Process tasks from the queue"""
        
        # Process tasks up to worker limit
        while not self.task_queue.empty() and len(self.active_tasks) < self.config.max_workers:
            try:
                task = self.task_queue.get_nowait()
                self._execute_task(task)
            except queue.Empty:
                break
    
    def _execute_task(self, task: PipelineTask):
        """Execute a pipeline task"""
        
        # Check if component exists and is running
        if task.component_id not in self.components:
            logger.error(f"Component not found: {task.component_id}")
            return
        
        component = self.components[task.component_id]
        if component.state != ComponentState.RUNNING:
            logger.warning(f"Component not running: {task.component_id} (state: {component.state.value})")
            # Re-queue task
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                self.task_queue.put(task)
            return
        
        # Add to active tasks
        self.active_tasks[task.task_id] = task
        
        # Submit to executor
        future = self.executor.submit(self._execute_component_task, task, component)
        self.futures[task.task_id] = future
        
        # Add callback
        future.add_done_callback(lambda f: self._task_completed(task.task_id, f))
    
    def _execute_component_task(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Execute task on a specific component"""
        
        start_time = time.time()
        
        try:
            # Route to appropriate handler
            result = self._route_task_to_component(task, component)
            
            # Record latency
            latency = time.time() - start_time
            self.stage_latencies[task.stage].append(latency)
            
            # Update component metrics
            component.performance_metrics['last_execution_time'] = latency
            component.performance_metrics['total_executions'] = component.performance_metrics.get('total_executions', 0) + 1
            
            return result
            
        except Exception as e:
            # Record error
            component.error_count += 1
            self.component_errors[task.component_id] += 1
            
            # Check error threshold
            if component.error_count >= self.config.error_threshold:
                component.state = ComponentState.ERROR
                logger.error(f"Component {task.component_id} exceeded error threshold")
            
            raise e
    
    def _route_task_to_component(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Route task to the appropriate component handler"""
        
        module = component.module_reference
        
        # Standard processing method names by stage
        stage_methods = {
            PipelineStage.DATA_INGESTION: "ingest_data",
            PipelineStage.FEATURE_ENGINEERING: "engineer_features",
            PipelineStage.REGIME_SCORING: "calculate_regime_scores",
            PipelineStage.TRANSITION_ANALYSIS: "analyze_transitions",
            PipelineStage.NOISE_FILTERING: "filter_noise",
            PipelineStage.REGIME_DECISION: "decide_regime",
            PipelineStage.BOUNDARY_OPTIMIZATION: "optimize_boundaries",
            PipelineStage.PERFORMANCE_FEEDBACK: "generate_feedback",
            PipelineStage.LEARNING_UPDATE: "update_learning",
            PipelineStage.MONITORING: "monitor_system"
        }
        
        method_name = stage_methods.get(task.stage, "process")
        
        # Call component method
        if hasattr(module, method_name):
            method = getattr(module, method_name)
            result = method(task.input_data)
        else:
            # Default processing
            result = self._default_component_processing(task, component)
        
        return result
    
    def _default_component_processing(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Default processing for components without specific handlers"""
        
        # Simulate processing based on component type
        component_processors = {
            "adaptive_scoring_layer": self._process_asl,
            "transition_matrix_analyzer": self._process_transition_analyzer,
            "dynamic_boundary_optimizer": self._process_boundary_optimizer,
            "intelligent_transition_manager": self._process_transition_manager,
            "regime_stability_monitor": self._process_stability_monitor,
            "adaptive_noise_filter": self._process_noise_filter,
            "performance_feedback_system": self._process_performance_feedback,
            "continuous_learning_engine": self._process_learning_engine,
            "regime_optimization_scheduler": self._process_scheduler
        }
        
        processor = component_processors.get(component.component_type)
        if processor:
            return processor(task, component)
        
        # Generic processing
        return {'processed': True, 'component': component.component_id, 'data': task.input_data}
    
    def _process_asl(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Process task for Adaptive Scoring Layer"""
        
        asl = component.module_reference
        market_data = task.input_data
        
        # Calculate regime scores
        regime_scores = asl.calculate_regime_scores(market_data)
        
        # Store result
        result = {
            'regime_scores': regime_scores,
            'timestamp': task.timestamp,
            'component': 'adaptive_scoring_layer'
        }
        
        # Add to buffer for downstream components
        self.data_buffers['adaptive_scoring_layer'].append(result)
        
        return result
    
    def _process_transition_analyzer(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Process task for Transition Matrix Analyzer"""
        
        analyzer = component.module_reference
        input_data = task.input_data
        
        # Analyze transitions if we have regime sequence
        if 'regime_sequence' in input_data:
            results = analyzer.analyze_transitions(
                input_data['regime_sequence'],
                input_data.get('timestamps'),
                input_data.get('features')
            )
            
            return {
                'transition_analysis': results,
                'timestamp': task.timestamp,
                'component': 'transition_matrix_analyzer'
            }
        
        return {'processed': True}
    
    def _process_boundary_optimizer(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Process task for Dynamic Boundary Optimizer"""
        
        optimizer = component.module_reference
        input_data = task.input_data
        
        if 'performance_data' in input_data:
            result = optimizer.optimize_boundaries(
                input_data['performance_data'],
                input_data.get('market_conditions', {})
            )
            
            return {
                'optimization_result': result,
                'timestamp': task.timestamp,
                'component': 'dynamic_boundary_optimizer'
            }
        
        return {'processed': True}
    
    def _process_transition_manager(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Process task for Intelligent Transition Manager"""
        
        manager = component.module_reference
        input_data = task.input_data
        
        if 'regime_scores' in input_data:
            # Get proposed regime
            regime_scores = input_data['regime_scores']
            proposed_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
            
            # Evaluate transition
            decision = manager.evaluate_transition(
                proposed_regime,
                regime_scores,
                input_data.get('market_data', {})
            )
            
            result = {
                'transition_decision': decision,
                'proposed_regime': proposed_regime,
                'approved': decision.approved,
                'timestamp': task.timestamp,
                'component': 'intelligent_transition_manager'
            }
            
            # Update current regime if approved
            if decision.approved:
                self.current_regime = proposed_regime
                self.last_regime_update = datetime.now()
            
            # Add to buffer
            self.data_buffers['intelligent_transition_manager'].append(result)
            
            return result
        
        return {'processed': True}
    
    def _process_stability_monitor(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Process task for Regime Stability Monitor"""
        
        monitor = component.module_reference
        input_data = task.input_data
        
        if 'regime_decision' in input_data:
            decision = input_data['regime_decision']
            
            # Update stability monitoring
            monitor.update_regime_data(
                self.current_regime,
                input_data.get('regime_scores', {}),
                input_data.get('market_data', {}),
                input_data.get('prediction_accuracy')
            )
            
            # Get stability report
            report = monitor.get_stability_report()
            
            return {
                'stability_report': report,
                'timestamp': task.timestamp,
                'component': 'regime_stability_monitor'
            }
        
        return {'processed': True}
    
    def _process_noise_filter(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Process task for Adaptive Noise Filter"""
        
        noise_filter = component.module_reference
        input_data = task.input_data
        
        if 'regime_scores' in input_data:
            result = noise_filter.filter_regime_signal(
                input_data['regime_scores'],
                input_data.get('market_data', {}),
                self.current_regime
            )
            
            return {
                'noise_filter_result': result,
                'filtered_scores': result.filtered_signal,
                'has_noise': result.has_noise,
                'timestamp': task.timestamp,
                'component': 'adaptive_noise_filter'
            }
        
        return {'processed': True}
    
    def _process_performance_feedback(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Process task for Performance Feedback System"""
        
        feedback_system = component.module_reference
        input_data = task.input_data
        
        if 'performance_metrics' in input_data:
            # Record performance metrics
            for metric in input_data['performance_metrics']:
                feedback_system.record_performance_metric(metric)
            
            # Get feedback actions
            actions = feedback_system.get_pending_feedback_actions()
            
            result = {
                'feedback_actions': actions,
                'system_performance': feedback_system.get_system_performance_score(),
                'timestamp': task.timestamp,
                'component': 'performance_feedback_system'
            }
            
            # Add to buffer for learning engine
            self.data_buffers['performance_feedback_system'].append(result)
            
            return result
        
        return {'processed': True}
    
    def _process_learning_engine(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Process task for Continuous Learning Engine"""
        
        learning_engine = component.module_reference
        input_data = task.input_data
        
        if 'learning_example' in input_data:
            # Add learning example
            learning_engine.add_learning_example(input_data['learning_example'])
            
            # Get learning statistics
            stats = learning_engine.get_learning_statistics()
            
            return {
                'learning_stats': stats,
                'models_updated': True,
                'timestamp': task.timestamp,
                'component': 'continuous_learning_engine'
            }
        
        return {'processed': True}
    
    def _process_scheduler(self, task: PipelineTask, component: ComponentInfo) -> Dict[str, Any]:
        """Process task for Regime Optimization Scheduler"""
        
        scheduler = component.module_reference
        input_data = task.input_data
        
        if 'optimization_task' in input_data:
            # Schedule optimization task
            opt_task = input_data['optimization_task']
            scheduled = scheduler.schedule_task(opt_task)
            
            return {
                'task_scheduled': scheduled,
                'scheduler_status': scheduler.get_scheduler_status(),
                'timestamp': task.timestamp,
                'component': 'regime_optimization_scheduler'
            }
        
        return {'processed': True}
    
    def _task_completed(self, task_id: str, future: Future):
        """Handle task completion"""
        
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks.pop(task_id)
        
        try:
            result = future.result()
            
            # Add to completed tasks
            self.completed_tasks.append({
                'task': task,
                'result': result,
                'completed_at': datetime.now()
            })
            
            # Process next stage if needed
            self._process_next_stage(task, result)
            
        except Exception as e:
            logger.error(f"Task failed: {task_id} - {e}")
            
            # Retry if under limit
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                self.task_queue.put(task)
            else:
                # Task failed permanently
                logger.error(f"Task {task_id} failed after {task.retry_count} retries")
        
        finally:
            # Clean up
            if task_id in self.futures:
                del self.futures[task_id]
    
    def _process_next_stage(self, completed_task: PipelineTask, result: Dict[str, Any]):
        """Process the next stage in the pipeline based on completed task"""
        
        # Define pipeline flow
        next_stages = {
            PipelineStage.DATA_INGESTION: PipelineStage.FEATURE_ENGINEERING,
            PipelineStage.FEATURE_ENGINEERING: PipelineStage.REGIME_SCORING,
            PipelineStage.REGIME_SCORING: PipelineStage.NOISE_FILTERING,
            PipelineStage.NOISE_FILTERING: PipelineStage.TRANSITION_ANALYSIS,
            PipelineStage.TRANSITION_ANALYSIS: PipelineStage.REGIME_DECISION,
            PipelineStage.REGIME_DECISION: PipelineStage.PERFORMANCE_FEEDBACK,
            PipelineStage.PERFORMANCE_FEEDBACK: PipelineStage.LEARNING_UPDATE
        }
        
        next_stage = next_stages.get(completed_task.stage)
        if next_stage:
            # Create task for next stage
            next_component = self._get_component_for_stage(next_stage)
            if next_component:
                next_task = PipelineTask(
                    task_id=f"{next_stage.value}_{time.time()}",
                    stage=next_stage,
                    component_id=next_component,
                    input_data=result,
                    timestamp=datetime.now(),
                    priority=completed_task.priority
                )
                
                self.task_queue.put(next_task)
    
    def _monitor_component_health(self):
        """Monitor health of all components"""
        
        for component_id, component in self.components.items():
            # Check if component is responsive
            if hasattr(component.module_reference, 'health_check'):
                try:
                    health = component.module_reference.health_check()
                    if not health:
                        logger.warning(f"Component {component_id} health check failed")
                        component.error_count += 1
                except Exception as e:
                    logger.error(f"Health check error for {component_id}: {e}")
                    component.error_count += 1
            
            # Check error threshold
            if component.error_count >= self.config.error_threshold:
                if component.state != ComponentState.ERROR:
                    component.state = ComponentState.ERROR
                    logger.error(f"Component {component_id} marked as ERROR")
                    
                    # Attempt recovery
                    self._attempt_component_recovery(component_id)
    
    def _attempt_component_recovery(self, component_id: str):
        """Attempt to recover a failed component"""
        
        logger.info(f"Attempting recovery for component: {component_id}")
        
        component = self.components[component_id]
        
        # Reset error count
        component.error_count = 0
        
        # Re-initialize component
        if self._initialize_component(component_id):
            component.state = ComponentState.RUNNING
            logger.info(f"Component {component_id} recovered successfully")
        else:
            logger.error(f"Failed to recover component {component_id}")
    
    def _update_system_state(self):
        """Update and record system state"""
        
        with self.state_lock:
            # Create state snapshot
            state = SystemState(
                timestamp=datetime.now(),
                component_states={
                    comp_id: comp.state 
                    for comp_id, comp in self.components.items()
                },
                active_tasks=list(self.active_tasks.keys()),
                completed_tasks=len(self.completed_tasks),
                failed_tasks=sum(1 for comp in self.components.values() if comp.error_count > 0),
                current_regime=self.current_regime,
                performance_metrics=self._calculate_performance_metrics(),
                error_log=self._get_recent_errors()
            )
            
            self.system_state_history.append(state)
            
            # Persist state if enabled
            if self.config.state_persistence:
                self._save_system_state()
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics"""
        
        metrics = {}
        
        # Component metrics
        metrics['component_health'] = {
            comp_id: {
                'state': comp.state.value,
                'error_count': comp.error_count,
                'last_execution_time': comp.performance_metrics.get('last_execution_time', 0)
            }
            for comp_id, comp in self.components.items()
        }
        
        # Stage latencies
        metrics['stage_latencies'] = {
            stage.value: {
                'avg': np.mean(list(latencies)) if latencies else 0,
                'max': np.max(list(latencies)) if latencies else 0,
                'min': np.min(list(latencies)) if latencies else 0
            }
            for stage, latencies in self.stage_latencies.items()
            if latencies
        }
        
        # Task metrics
        metrics['task_metrics'] = {
            'active_tasks': len(self.active_tasks),
            'queued_tasks': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'task_success_rate': self._calculate_task_success_rate()
        }
        
        # System metrics
        metrics['system_metrics'] = {
            'uptime': (datetime.now() - self.system_state_history[0].timestamp).total_seconds() if self.system_state_history else 0,
            'current_regime': self.current_regime,
            'regime_stable_duration': (datetime.now() - self.last_regime_update).total_seconds()
        }
        
        return metrics
    
    def _calculate_task_success_rate(self) -> float:
        """Calculate task success rate"""
        
        if not self.completed_tasks:
            return 0.0
        
        successful = sum(
            1 for task_info in self.completed_tasks 
            if 'error' not in task_info.get('result', {})
        )
        
        return successful / len(self.completed_tasks)
    
    def _get_recent_errors(self) -> List[Dict[str, Any]]:
        """Get recent error log entries"""
        
        errors = []
        
        # Component errors
        for comp_id, error_count in self.component_errors.items():
            if error_count > 0:
                errors.append({
                    'component': comp_id,
                    'error_count': error_count,
                    'timestamp': datetime.now().isoformat()
                })
        
        return errors[-10:]  # Last 10 errors
    
    def _collect_batch_data(self) -> List[Dict[str, Any]]:
        """Collect data for batch processing"""
        
        batch = []
        
        # Collect from input buffer
        input_buffer = self.data_buffers.get('input', deque())
        
        while len(batch) < self.config.batch_size and input_buffer:
            batch.append(input_buffer.popleft())
        
        return batch
    
    def _process_batch(self, batch_data: List[Dict[str, Any]]):
        """Process a batch of data"""
        
        # Create batch task
        task = PipelineTask(
            task_id=f"batch_{time.time()}",
            stage=PipelineStage.DATA_INGESTION,
            component_id="batch_processor",
            input_data={'batch': batch_data},
            timestamp=datetime.now(),
            priority=5
        )
        
        self.task_queue.put(task)
    
    def _wait_for_task_completion(self, task_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Wait for a specific task to complete"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check completed tasks
            for task_info in self.completed_tasks:
                if task_info['task'].task_id == task_id:
                    return task_info['result']
            
            time.sleep(0.1)
        
        return {'error': 'Task timeout', 'task_id': task_id}
    
    def _cancel_active_tasks(self):
        """Cancel all active tasks"""
        
        for task_id, future in self.futures.items():
            if not future.done():
                future.cancel()
        
        self.active_tasks.clear()
        self.futures.clear()
    
    def _update_component_order(self):
        """Update component processing order based on dependencies"""
        
        # Topological sort
        visited = set()
        order = []
        
        def visit(comp_id):
            if comp_id in visited:
                return
            
            visited.add(comp_id)
            
            if comp_id in self.components:
                for dep in self.components[comp_id].dependencies:
                    visit(dep)
                
                order.append(comp_id)
        
        for comp_id in self.components:
            visit(comp_id)
        
        self.component_order = order
    
    def _get_stage_for_component(self, component_id: str) -> PipelineStage:
        """Get pipeline stage for a component"""
        
        stage_mapping = {
            'data_ingestion': PipelineStage.DATA_INGESTION,
            'feature_engineering': PipelineStage.FEATURE_ENGINEERING,
            'adaptive_scoring_layer': PipelineStage.REGIME_SCORING,
            'transition_matrix_analyzer': PipelineStage.TRANSITION_ANALYSIS,
            'adaptive_noise_filter': PipelineStage.NOISE_FILTERING,
            'intelligent_transition_manager': PipelineStage.REGIME_DECISION,
            'dynamic_boundary_optimizer': PipelineStage.BOUNDARY_OPTIMIZATION,
            'performance_feedback_system': PipelineStage.PERFORMANCE_FEEDBACK,
            'continuous_learning_engine': PipelineStage.LEARNING_UPDATE,
            'regime_stability_monitor': PipelineStage.MONITORING
        }
        
        return stage_mapping.get(component_id, PipelineStage.MONITORING)
    
    def _get_component_for_stage(self, stage: PipelineStage) -> Optional[str]:
        """Get component ID for a pipeline stage"""
        
        component_mapping = {
            PipelineStage.DATA_INGESTION: 'data_ingestion',
            PipelineStage.FEATURE_ENGINEERING: 'feature_engineering',
            PipelineStage.REGIME_SCORING: 'adaptive_scoring_layer',
            PipelineStage.TRANSITION_ANALYSIS: 'transition_matrix_analyzer',
            PipelineStage.NOISE_FILTERING: 'adaptive_noise_filter',
            PipelineStage.REGIME_DECISION: 'intelligent_transition_manager',
            PipelineStage.BOUNDARY_OPTIMIZATION: 'dynamic_boundary_optimizer',
            PipelineStage.PERFORMANCE_FEEDBACK: 'performance_feedback_system',
            PipelineStage.LEARNING_UPDATE: 'continuous_learning_engine',
            PipelineStage.MONITORING: 'regime_stability_monitor'
        }
        
        return component_mapping.get(stage)
    
    def _save_system_state(self):
        """Save system state to file"""
        
        try:
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'current_regime': self.current_regime,
                'component_states': {
                    comp_id: {
                        'state': comp.state.value,
                        'error_count': comp.error_count,
                        'last_update': comp.last_update.isoformat()
                    }
                    for comp_id, comp in self.components.items()
                },
                'performance_metrics': self._calculate_performance_metrics(),
                'completed_tasks_count': len(self.completed_tasks),
                'system_uptime': (datetime.now() - self.system_state_history[0].timestamp).total_seconds() if self.system_state_history else 0
            }
            
            with open(self.config.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
    
    def load_system_state(self, filepath: str):
        """Load system state from file"""
        
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            self.current_regime = state_data.get('current_regime', 0)
            
            # Update component states
            for comp_id, comp_state in state_data.get('component_states', {}).items():
                if comp_id in self.components:
                    self.components[comp_id].error_count = comp_state.get('error_count', 0)
            
            logger.info(f"System state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load system state: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        with self.state_lock:
            status = {
                'orchestrator_running': self.orchestrator_running,
                'orchestration_mode': self.config.mode.value,
                'current_regime': self.current_regime,
                'components': {
                    comp_id: {
                        'state': comp.state.value,
                        'error_count': comp.error_count,
                        'type': comp.component_type,
                        'performance': comp.performance_metrics
                    }
                    for comp_id, comp in self.components.items()
                },
                'tasks': {
                    'active': len(self.active_tasks),
                    'queued': self.task_queue.qsize(),
                    'completed': len(self.completed_tasks),
                    'success_rate': self._calculate_task_success_rate()
                },
                'performance': self._calculate_performance_metrics(),
                'data_flow': {
                    'active_paths': sum(1 for p in self.data_flow_paths if p.enabled),
                    'buffer_sizes': {
                        comp: len(buffer) 
                        for comp, buffer in self.data_buffers.items()
                    }
                }
            }
            
            return status
    
    def export_orchestration_report(self, filepath: str):
        """Export comprehensive orchestration report"""
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'configuration': {
                'mode': self.config.mode.value,
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size,
                'error_threshold': self.config.error_threshold
            },
            'component_registry': {
                comp_id: {
                    'type': comp.component_type,
                    'dependencies': comp.dependencies,
                    'initialization_params': comp.initialization_params
                }
                for comp_id, comp in self.components.items()
            },
            'data_flow_paths': [
                {
                    'source': path.source_component,
                    'target': path.target_component,
                    'data_type': path.data_type,
                    'enabled': path.enabled
                }
                for path in self.data_flow_paths
            ],
            'performance_summary': {
                'total_tasks_completed': len(self.completed_tasks),
                'average_task_latency': np.mean([
                    np.mean(list(latencies)) 
                    for latencies in self.stage_latencies.values() 
                    if latencies
                ]) if any(self.stage_latencies.values()) else 0,
                'component_error_rates': {
                    comp_id: errors / max(comp.performance_metrics.get('total_executions', 1), 1)
                    for comp_id, (comp, errors) in zip(
                        self.components.items(),
                        [(comp_id, self.component_errors.get(comp_id, 0)) for comp_id in self.components]
                    )
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Orchestration report exported to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create orchestrator
    config = OrchestrationConfig(
        mode=OrchestrationMode.REAL_TIME,
        max_workers=3,
        real_time_interval=0.5
    )
    
    orchestrator = IntegrationOrchestrator(config)
    
    # Mock components
    class MockComponent:
        def __init__(self, name):
            self.name = name
        
        def health_check(self):
            return True
        
        def calculate_regime_scores(self, data):
            return {i: np.random.random() for i in range(12)}
    
    # Register components
    orchestrator.register_component(
        component_id="adaptive_scoring_layer",
        component_type="adaptive_scoring_layer",
        module_reference=MockComponent("ASL")
    )
    
    orchestrator.register_component(
        component_id="intelligent_transition_manager",
        component_type="intelligent_transition_manager",
        module_reference=MockComponent("ITM"),
        dependencies=["adaptive_scoring_layer"]
    )
    
    # Initialize system
    if orchestrator.initialize_system():
        # Start orchestration
        orchestrator.start_orchestration()
        
        # Process some market data
        for i in range(5):
            market_data = {
                'timestamp': datetime.now(),
                'spot_price': 100 + np.random.randn(),
                'volume': np.random.randint(1000, 5000),
                'volatility': np.random.uniform(0.1, 0.3)
            }
            
            result = orchestrator.process_market_data(market_data)
            print(f"Processed market data {i}: {result}")
            
            time.sleep(1)
        
        # Get status
        status = orchestrator.get_system_status()
        print("\nSystem Status:")
        print(f"Running: {status['orchestrator_running']}")
        print(f"Current Regime: {status['current_regime']}")
        print(f"Active Tasks: {status['tasks']['active']}")
        print(f"Completed Tasks: {status['tasks']['completed']}")
        
        # Stop orchestration
        orchestrator.stop_orchestration()
    
    print("Orchestration example completed")
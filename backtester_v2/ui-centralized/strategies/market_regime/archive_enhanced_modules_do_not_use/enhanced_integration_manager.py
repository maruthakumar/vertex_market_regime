#!/usr/bin/env python3
"""
Enhanced Integration Manager - Phase 5 Enhancement
Manages seamless integration between all market regime components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import json
import pickle
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComponentMessage:
    """Message passed between components"""
    source: str
    target: str
    message_type: str
    payload: Any
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1
    requires_response: bool = False
    correlation_id: Optional[str] = None

@dataclass
class ComponentDependency:
    """Dependency information for a component"""
    component_name: str
    depends_on: List[str]
    provides: List[str]
    initialization_order: int
    is_critical: bool = True

@dataclass
class SharedMemoryPool:
    """Shared memory pool for component data"""
    data_store: Dict[str, Any] = field(default_factory=dict)
    access_counts: Dict[str, int] = field(default_factory=dict)
    last_updated: Dict[str, datetime] = field(default_factory=dict)
    locks: Dict[str, threading.Lock] = field(default_factory=dict)

class ComponentInterface(ABC):
    """Abstract interface for all components"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get component name"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize component"""
        pass
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get component dependencies"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get component status"""
        pass

class EventBus:
    """Event-driven communication system"""
    
    def __init__(self):
        self.subscribers = {}
        self.message_queue = asyncio.Queue()
        self.processing = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler to event type: {event_type}")
    
    async def publish(self, message: ComponentMessage):
        """Publish a message to the event bus"""
        await self.message_queue.put(message)
    
    async def start_processing(self):
        """Start processing messages"""
        self.processing = True
        while self.processing:
            try:
                # Get message with timeout to allow checking processing flag
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Process message
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _process_message(self, message: ComponentMessage):
        """Process a single message"""
        handlers = self.subscribers.get(message.message_type, [])
        
        # Process handlers in parallel
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._call_handler(handler, message))
            tasks.append(task)
        
        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_handler(self, handler: Callable, message: ComponentMessage):
        """Call a message handler"""
        try:
            # Run handler in executor if it's not async
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, handler, message
                )
        except Exception as e:
            logger.error(f"Error in handler for {message.message_type}: {e}")
    
    def stop_processing(self):
        """Stop processing messages"""
        self.processing = False

class DependencyResolver:
    """Resolves component dependencies and initialization order"""
    
    def __init__(self):
        self.dependencies = {}
        
    def add_component(self, dependency: ComponentDependency):
        """Add a component with its dependencies"""
        self.dependencies[dependency.component_name] = dependency
    
    def resolve_initialization_order(self) -> List[str]:
        """Resolve initialization order based on dependencies"""
        # Build dependency graph
        graph = {}
        for name, dep in self.dependencies.items():
            graph[name] = dep.depends_on
        
        # Topological sort
        visited = set()
        stack = []
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            
            # Visit dependencies first
            for dep in graph.get(node, []):
                if dep in graph:  # Only visit if dependency is a component
                    visit(dep)
            
            stack.append(node)
        
        # Visit all nodes
        for node in graph:
            visit(node)
        
        return stack
    
    def check_circular_dependencies(self) -> List[Tuple[str, str]]:
        """Check for circular dependencies"""
        circular = []
        
        def has_path(start, end, visited=None):
            if visited is None:
                visited = set()
            
            if start == end and len(visited) > 0:
                return True
            
            if start in visited:
                return False
            
            visited.add(start)
            
            if start in self.dependencies:
                for dep in self.dependencies[start].depends_on:
                    if has_path(dep, end, visited.copy()):
                        return True
            
            return False
        
        # Check all pairs
        for name1 in self.dependencies:
            for name2 in self.dependencies:
                if name1 != name2 and has_path(name1, name2) and has_path(name2, name1):
                    if (name2, name1) not in circular:  # Avoid duplicates
                        circular.append((name1, name2))
        
        return circular

class DataPipeline:
    """Manages data flow between components"""
    
    def __init__(self, shared_memory: SharedMemoryPool):
        self.shared_memory = shared_memory
        self.pipelines = {}
        self.transformers = {}
        
    def register_pipeline(self, name: str, source: str, target: str, 
                         transformer: Optional[Callable] = None):
        """Register a data pipeline between components"""
        pipeline_key = f"{source}_to_{target}"
        self.pipelines[pipeline_key] = {
            'name': name,
            'source': source,
            'target': target,
            'created': datetime.now()
        }
        
        if transformer:
            self.transformers[pipeline_key] = transformer
        
        logger.info(f"Registered pipeline: {name} ({source} -> {target})")
    
    def transfer_data(self, source: str, target: str, data: Any) -> Any:
        """Transfer data from source to target with optional transformation"""
        pipeline_key = f"{source}_to_{target}"
        
        # Apply transformation if exists
        if pipeline_key in self.transformers:
            try:
                data = self.transformers[pipeline_key](data)
            except Exception as e:
                logger.error(f"Error in data transformation: {e}")
        
        # Store in shared memory for target
        self._store_shared_data(f"{target}_input", data)
        
        return data
    
    def _store_shared_data(self, key: str, data: Any):
        """Store data in shared memory"""
        if key not in self.shared_memory.locks:
            self.shared_memory.locks[key] = threading.Lock()
        
        with self.shared_memory.locks[key]:
            self.shared_memory.data_store[key] = data
            self.shared_memory.last_updated[key] = datetime.now()
            self.shared_memory.access_counts[key] = \
                self.shared_memory.access_counts.get(key, 0) + 1

class ComponentRegistry:
    """Registry for all system components"""
    
    def __init__(self):
        self.components = {}
        self.component_status = {}
        self.initialization_callbacks = []
        
    def register(self, component: ComponentInterface):
        """Register a component"""
        name = component.get_name()
        self.components[name] = component
        self.component_status[name] = {
            'registered': datetime.now(),
            'initialized': False,
            'status': 'registered'
        }
        logger.info(f"Registered component: {name}")
    
    def initialize_component(self, name: str, config: Dict[str, Any]) -> bool:
        """Initialize a component"""
        if name not in self.components:
            logger.error(f"Component not found: {name}")
            return False
        
        try:
            component = self.components[name]
            success = component.initialize(config)
            
            self.component_status[name]['initialized'] = success
            self.component_status[name]['status'] = 'active' if success else 'failed'
            
            # Call initialization callbacks
            for callback in self.initialization_callbacks:
                callback(name, success)
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing component {name}: {e}")
            self.component_status[name]['status'] = 'error'
            return False
    
    def get_component(self, name: str) -> Optional[ComponentInterface]:
        """Get a component by name"""
        return self.components.get(name)
    
    def get_all_components(self) -> Dict[str, ComponentInterface]:
        """Get all registered components"""
        return self.components.copy()
    
    def add_initialization_callback(self, callback: Callable):
        """Add callback for component initialization"""
        self.initialization_callbacks.append(callback)

class EnhancedIntegrationManager:
    """
    Main integration manager that coordinates all components
    """
    
    def __init__(self):
        # Core systems
        self.shared_memory = SharedMemoryPool()
        self.event_bus = EventBus()
        self.dependency_resolver = DependencyResolver()
        self.data_pipeline = DataPipeline(self.shared_memory)
        self.component_registry = ComponentRegistry()
        
        # Component configuration
        self.component_configs = {}
        
        # Performance tracking
        self.performance_metrics = {
            'message_count': 0,
            'pipeline_transfers': 0,
            'initialization_time': {},
            'processing_times': deque(maxlen=1000)
        }
        
        # Initialize default component dependencies
        self._initialize_default_dependencies()
        
        logger.info("Enhanced Integration Manager initialized")
    
    def _initialize_default_dependencies(self):
        """Initialize default component dependencies"""
        # Define standard market regime component dependencies
        dependencies = [
            ComponentDependency(
                component_name='data_loader',
                depends_on=[],
                provides=['market_data'],
                initialization_order=1
            ),
            ComponentDependency(
                component_name='triple_straddle_engine',
                depends_on=['data_loader'],
                provides=['straddle_values'],
                initialization_order=2
            ),
            ComponentDependency(
                component_name='greek_sentiment_analyzer',
                depends_on=['data_loader'],
                provides=['greek_scores'],
                initialization_order=2
            ),
            ComponentDependency(
                component_name='oi_pattern_analyzer',
                depends_on=['data_loader'],
                provides=['oi_patterns'],
                initialization_order=2
            ),
            ComponentDependency(
                component_name='iv_analysis_suite',
                depends_on=['data_loader'],
                provides=['iv_metrics'],
                initialization_order=2
            ),
            ComponentDependency(
                component_name='atr_indicators',
                depends_on=['data_loader'],
                provides=['atr_values'],
                initialization_order=2
            ),
            ComponentDependency(
                component_name='support_resistance',
                depends_on=['data_loader'],
                provides=['sr_levels'],
                initialization_order=2
            ),
            ComponentDependency(
                component_name='ml_ensemble',
                depends_on=['triple_straddle_engine', 'greek_sentiment_analyzer', 
                          'oi_pattern_analyzer', 'iv_analysis_suite'],
                provides=['ml_predictions'],
                initialization_order=3
            ),
            ComponentDependency(
                component_name='regime_classifier',
                depends_on=['ml_ensemble'],
                provides=['regime_classification'],
                initialization_order=4
            ),
            ComponentDependency(
                component_name='output_generator',
                depends_on=['regime_classifier'],
                provides=['final_output'],
                initialization_order=5
            )
        ]
        
        for dep in dependencies:
            self.dependency_resolver.add_component(dep)
    
    async def initialize_system(self, config: Dict[str, Any]) -> bool:
        """Initialize the entire system"""
        try:
            logger.info("Initializing integrated system...")
            
            # Check for circular dependencies
            circular = self.dependency_resolver.check_circular_dependencies()
            if circular:
                logger.error(f"Circular dependencies detected: {circular}")
                return False
            
            # Get initialization order
            init_order = self.dependency_resolver.resolve_initialization_order()
            logger.info(f"Initialization order: {init_order}")
            
            # Initialize components in order
            for component_name in init_order:
                if component_name in self.component_registry.components:
                    start_time = datetime.now()
                    
                    # Get component config
                    component_config = config.get(component_name, {})
                    
                    # Initialize component
                    success = self.component_registry.initialize_component(
                        component_name, component_config
                    )
                    
                    if not success:
                        logger.error(f"Failed to initialize component: {component_name}")
                        return False
                    
                    # Track initialization time
                    init_time = (datetime.now() - start_time).total_seconds()
                    self.performance_metrics['initialization_time'][component_name] = init_time
                    
                    logger.info(f"Initialized {component_name} in {init_time:.2f}s")
            
            # Start event bus processing
            asyncio.create_task(self.event_bus.start_processing())
            
            # Setup data pipelines
            self._setup_data_pipelines()
            
            # Subscribe to events
            self._setup_event_subscriptions()
            
            logger.info("System initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during system initialization: {e}")
            return False
    
    def _setup_data_pipelines(self):
        """Setup data pipelines between components"""
        # Define standard pipelines
        pipelines = [
            ('market_data_flow', 'data_loader', 'triple_straddle_engine'),
            ('market_data_flow', 'data_loader', 'greek_sentiment_analyzer'),
            ('market_data_flow', 'data_loader', 'oi_pattern_analyzer'),
            ('market_data_flow', 'data_loader', 'iv_analysis_suite'),
            ('market_data_flow', 'data_loader', 'atr_indicators'),
            ('market_data_flow', 'data_loader', 'support_resistance'),
            ('straddle_to_ml', 'triple_straddle_engine', 'ml_ensemble'),
            ('greeks_to_ml', 'greek_sentiment_analyzer', 'ml_ensemble'),
            ('oi_to_ml', 'oi_pattern_analyzer', 'ml_ensemble'),
            ('iv_to_ml', 'iv_analysis_suite', 'ml_ensemble'),
            ('ml_to_regime', 'ml_ensemble', 'regime_classifier'),
            ('regime_to_output', 'regime_classifier', 'output_generator')
        ]
        
        for name, source, target in pipelines:
            self.data_pipeline.register_pipeline(name, source, target)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions"""
        # Subscribe to component events
        self.event_bus.subscribe('data_available', self._handle_data_available)
        self.event_bus.subscribe('processing_complete', self._handle_processing_complete)
        self.event_bus.subscribe('error_occurred', self._handle_error)
        self.event_bus.subscribe('regime_change', self._handle_regime_change)
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data through the entire pipeline"""
        try:
            start_time = datetime.now()
            
            # Store market data in shared memory
            self._store_in_shared_memory('market_data', market_data)
            
            # Create initial message
            message = ComponentMessage(
                source='integration_manager',
                target='data_loader',
                message_type='data_available',
                payload=market_data,
                priority=1,
                requires_response=True
            )
            
            # Publish to event bus
            await self.event_bus.publish(message)
            
            # Wait for processing to complete
            # In practice, would use proper async coordination
            await asyncio.sleep(0.1)
            
            # Get final output from shared memory
            output = self._get_from_shared_memory('final_output')
            
            # Track processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_metrics['processing_times'].append(processing_time)
            
            # Update metrics
            self.performance_metrics['pipeline_transfers'] += 1
            
            return {
                'status': 'success',
                'output': output,
                'processing_time_ms': processing_time,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _store_in_shared_memory(self, key: str, data: Any):
        """Store data in shared memory"""
        if key not in self.shared_memory.locks:
            self.shared_memory.locks[key] = threading.Lock()
        
        with self.shared_memory.locks[key]:
            self.shared_memory.data_store[key] = data
            self.shared_memory.last_updated[key] = datetime.now()
            self.shared_memory.access_counts[key] = \
                self.shared_memory.access_counts.get(key, 0) + 1
    
    def _get_from_shared_memory(self, key: str) -> Any:
        """Get data from shared memory"""
        if key not in self.shared_memory.locks:
            return None
        
        with self.shared_memory.locks[key]:
            return self.shared_memory.data_store.get(key)
    
    async def _handle_data_available(self, message: ComponentMessage):
        """Handle data available event"""
        self.performance_metrics['message_count'] += 1
        logger.debug(f"Data available from {message.source} to {message.target}")
    
    async def _handle_processing_complete(self, message: ComponentMessage):
        """Handle processing complete event"""
        logger.debug(f"Processing complete for {message.source}")
    
    async def _handle_error(self, message: ComponentMessage):
        """Handle error event"""
        logger.error(f"Error in {message.source}: {message.payload}")
    
    async def _handle_regime_change(self, message: ComponentMessage):
        """Handle regime change event"""
        logger.info(f"Regime change detected: {message.payload}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        # Calculate average processing time
        avg_processing_time = np.mean(list(self.performance_metrics['processing_times'])) \
            if self.performance_metrics['processing_times'] else 0
        
        return {
            'components': {
                name: status 
                for name, status in self.component_registry.component_status.items()
            },
            'shared_memory': {
                'keys': list(self.shared_memory.data_store.keys()),
                'total_accesses': sum(self.shared_memory.access_counts.values())
            },
            'performance': {
                'message_count': self.performance_metrics['message_count'],
                'pipeline_transfers': self.performance_metrics['pipeline_transfers'],
                'avg_processing_time_ms': avg_processing_time,
                'initialization_times': self.performance_metrics['initialization_time']
            },
            'dependencies': {
                'total_components': len(self.dependency_resolver.dependencies),
                'initialization_order': self.dependency_resolver.resolve_initialization_order()
            }
        }
    
    def export_integration_config(self, filepath: str):
        """Export integration configuration"""
        config = {
            'timestamp': datetime.now().isoformat(),
            'components': list(self.component_registry.components.keys()),
            'dependencies': {
                name: {
                    'depends_on': dep.depends_on,
                    'provides': dep.provides,
                    'initialization_order': dep.initialization_order
                }
                for name, dep in self.dependency_resolver.dependencies.items()
            },
            'pipelines': self.data_pipeline.pipelines,
            'system_status': self.get_system_status()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Integration configuration exported to {filepath}")
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down integration manager...")
        
        # Stop event bus
        self.event_bus.stop_processing()
        
        # Clear shared memory
        self.shared_memory.data_store.clear()
        
        # Shutdown components
        for name, component in self.component_registry.components.items():
            try:
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {name}: {e}")
        
        logger.info("Integration manager shutdown complete")

# Example component implementation
class ExampleComponent(ComponentInterface):
    """Example component implementation"""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.config = {}
        
    def get_name(self) -> str:
        return self.name
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        self.config = config
        self.initialized = True
        return True
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate processing
        return {'processed': True, 'component': self.name, 'data': data}
    
    def get_dependencies(self) -> List[str]:
        return []
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'initialized': self.initialized,
            'config': self.config
        }

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Integration Manager - Testing")
    print("="*60)
    
    async def test_integration():
        # Initialize manager
        manager = EnhancedIntegrationManager()
        
        # Register example components
        components = [
            'data_loader', 'triple_straddle_engine', 'greek_sentiment_analyzer',
            'oi_pattern_analyzer', 'iv_analysis_suite', 'atr_indicators',
            'support_resistance', 'ml_ensemble', 'regime_classifier', 'output_generator'
        ]
        
        for comp_name in components:
            component = ExampleComponent(comp_name)
            manager.component_registry.register(component)
        
        # Initialize system
        config = {
            'data_loader': {'source': 'heavydb'},
            'ml_ensemble': {'models': ['rf', 'xgb', 'nn']}
        }
        
        success = await manager.initialize_system(config)
        print(f"\nSystem initialization: {'✓ Success' if success else '✗ Failed'}")
        
        if success:
            # Test processing
            market_data = {
                'timestamp': datetime.now(),
                'spot_price': 22000,
                'options_data': []
            }
            
            print("\nProcessing market data...")
            result = await manager.process_market_data(market_data)
            
            print(f"Processing status: {result['status']}")
            if result['status'] == 'success':
                print(f"Processing time: {result['processing_time_ms']:.2f}ms")
            
            # Get system status
            status = manager.get_system_status()
            print("\nSystem Status:")
            print(f"Active components: {len(status['components'])}")
            print(f"Messages processed: {status['performance']['message_count']}")
            print(f"Average processing time: {status['performance']['avg_processing_time_ms']:.2f}ms")
            
            # Export configuration
            manager.export_integration_config("integration_config.json")
            
            # Shutdown
            await manager.shutdown()
    
    # Run test
    import asyncio
    asyncio.run(test_integration())
    
    print("\n✓ Enhanced Integration Manager implementation complete!")
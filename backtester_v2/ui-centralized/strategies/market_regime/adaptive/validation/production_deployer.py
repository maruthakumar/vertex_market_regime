"""
Production Deployer

This module handles the production deployment of the adaptive market regime
formation system, including environment setup, dependency management,
configuration deployment, and production monitoring setup.

Key Features:
- Automated deployment pipeline
- Environment validation and setup
- Configuration management
- Database connection verification
- Service health monitoring
- Rollback capabilities
- Performance baseline establishment
- Production monitoring integration
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
import hashlib
import requests
import psutil
import socket

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Deployment pipeline stages"""
    PRE_DEPLOYMENT = "pre_deployment"
    ENVIRONMENT_SETUP = "environment_setup"
    DEPENDENCY_CHECK = "dependency_check"
    DATABASE_SETUP = "database_setup"
    CONFIGURATION = "configuration"
    COMPONENT_DEPLOYMENT = "component_deployment"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_BASELINE = "performance_baseline"
    MONITORING_SETUP = "monitoring_setup"
    POST_DEPLOYMENT = "post_deployment"
    VERIFICATION = "verification"


class DeploymentStatus(Enum):
    """Deployment status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    VERIFIED = "verified"


class EnvironmentType(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """Production deployment configuration"""
    environment: EnvironmentType = EnvironmentType.PRODUCTION
    deployment_path: str = "/opt/adaptive_regime_system"
    config_path: str = "/etc/adaptive_regime"
    log_path: str = "/var/log/adaptive_regime"
    data_path: str = "/var/lib/adaptive_regime"
    
    # Database configuration
    heavydb_host: str = "localhost"
    heavydb_port: int = 6274
    heavydb_user: str = "admin"
    heavydb_password: str = "HyperInteractive"
    heavydb_database: str = "heavyai"
    
    # Service configuration
    service_name: str = "adaptive_regime_system"
    service_port: int = 8080
    worker_processes: int = 4
    max_memory_gb: float = 16.0
    
    # Monitoring configuration
    enable_monitoring: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    
    # Deployment options
    force_deployment: bool = False
    backup_existing: bool = True
    verify_deployment: bool = True
    rollback_on_failure: bool = True


@dataclass
class DeploymentStep:
    """Individual deployment step"""
    step_id: str
    stage: DeploymentStage
    description: str
    command: Optional[str] = None
    validation_function: Optional[callable] = None
    rollback_function: Optional[callable] = None
    critical: bool = True
    timeout: int = 300


@dataclass
class DeploymentResult:
    """Result of deployment operation"""
    deployment_id: str
    start_time: datetime
    end_time: datetime
    status: DeploymentStatus
    environment: EnvironmentType
    stages_completed: List[DeploymentStage]
    stages_failed: List[DeploymentStage]
    deployment_path: str
    configuration: Dict[str, Any]
    verification_results: Dict[str, Any]
    rollback_performed: bool = False
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ServiceHealth:
    """Service health status"""
    service_name: str
    status: str
    pid: Optional[int]
    memory_usage_mb: float
    cpu_percent: float
    uptime_seconds: float
    last_check: datetime
    endpoints_healthy: Dict[str, bool]


class ProductionDeployer:
    """
    Handles production deployment of the adaptive regime system
    """
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """
        Initialize production deployer
        
        Args:
            config: Deployment configuration
        """
        self.config = config or DeploymentConfig()
        
        # Deployment state
        self.deployment_id = self._generate_deployment_id()
        self.deployment_steps: List[DeploymentStep] = []
        self.completed_steps: List[DeploymentStep] = []
        self.failed_steps: List[DeploymentStep] = []
        
        # Backup information
        self.backup_path: Optional[str] = None
        self.original_config: Optional[Dict[str, Any]] = None
        
        # Service state
        self.service_pid: Optional[int] = None
        self.service_start_time: Optional[datetime] = None
        
        # Initialize deployment steps
        self._initialize_deployment_steps()
        
        logger.info(f"ProductionDeployer initialized for {config.environment.value}")
    
    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(f"{timestamp}_{self.config.environment.value}".encode()).hexdigest()[:8]
        return f"deploy_{timestamp}_{hash_suffix}"
    
    def _initialize_deployment_steps(self):
        """Initialize deployment pipeline steps"""
        
        steps = [
            # Pre-deployment
            DeploymentStep(
                step_id="pre_deploy_backup",
                stage=DeploymentStage.PRE_DEPLOYMENT,
                description="Backup existing deployment",
                validation_function=self._validate_backup,
                rollback_function=self._restore_backup,
                critical=False
            ),
            
            DeploymentStep(
                step_id="pre_deploy_validation",
                stage=DeploymentStage.PRE_DEPLOYMENT,
                description="Validate deployment environment",
                validation_function=self._validate_environment,
                critical=True
            ),
            
            # Environment setup
            DeploymentStep(
                step_id="create_directories",
                stage=DeploymentStage.ENVIRONMENT_SETUP,
                description="Create deployment directories",
                command=f"mkdir -p {self.config.deployment_path} {self.config.config_path} {self.config.log_path} {self.config.data_path}",
                validation_function=self._validate_directories,
                critical=True
            ),
            
            DeploymentStep(
                step_id="set_permissions",
                stage=DeploymentStage.ENVIRONMENT_SETUP,
                description="Set directory permissions",
                command=f"chmod 755 {self.config.deployment_path} {self.config.config_path} && chmod 775 {self.config.log_path} {self.config.data_path}",
                critical=True
            ),
            
            # Dependency check
            DeploymentStep(
                step_id="check_python",
                stage=DeploymentStage.DEPENDENCY_CHECK,
                description="Check Python version",
                validation_function=self._validate_python_version,
                critical=True
            ),
            
            DeploymentStep(
                step_id="check_dependencies",
                stage=DeploymentStage.DEPENDENCY_CHECK,
                description="Check Python dependencies",
                validation_function=self._validate_dependencies,
                critical=True
            ),
            
            # Database setup
            DeploymentStep(
                step_id="check_heavydb",
                stage=DeploymentStage.DATABASE_SETUP,
                description="Verify HeavyDB connection",
                validation_function=self._validate_heavydb_connection,
                critical=True
            ),
            
            DeploymentStep(
                step_id="validate_data",
                stage=DeploymentStage.DATABASE_SETUP,
                description="Validate market data availability",
                validation_function=self._validate_market_data,
                critical=True
            ),
            
            # Configuration deployment
            DeploymentStep(
                step_id="deploy_config",
                stage=DeploymentStage.CONFIGURATION,
                description="Deploy configuration files",
                validation_function=self._deploy_configuration,
                rollback_function=self._rollback_configuration,
                critical=True
            ),
            
            # Component deployment
            DeploymentStep(
                step_id="deploy_code",
                stage=DeploymentStage.COMPONENT_DEPLOYMENT,
                description="Deploy application code",
                validation_function=self._deploy_application_code,
                rollback_function=self._rollback_application_code,
                critical=True
            ),
            
            DeploymentStep(
                step_id="create_service",
                stage=DeploymentStage.COMPONENT_DEPLOYMENT,
                description="Create system service",
                validation_function=self._create_system_service,
                rollback_function=self._remove_system_service,
                critical=True
            ),
            
            # Integration test
            DeploymentStep(
                step_id="integration_test",
                stage=DeploymentStage.INTEGRATION_TEST,
                description="Run integration tests",
                validation_function=self._run_integration_tests,
                critical=True,
                timeout=600
            ),
            
            # Performance baseline
            DeploymentStep(
                step_id="performance_baseline",
                stage=DeploymentStage.PERFORMANCE_BASELINE,
                description="Establish performance baseline",
                validation_function=self._establish_performance_baseline,
                critical=False,
                timeout=900
            ),
            
            # Monitoring setup
            DeploymentStep(
                step_id="setup_monitoring",
                stage=DeploymentStage.MONITORING_SETUP,
                description="Setup production monitoring",
                validation_function=self._setup_monitoring,
                critical=self.config.enable_monitoring
            ),
            
            # Post-deployment
            DeploymentStep(
                step_id="start_service",
                stage=DeploymentStage.POST_DEPLOYMENT,
                description="Start production service",
                command=f"systemctl start {self.config.service_name}",
                validation_function=self._validate_service_running,
                rollback_function=self._stop_service,
                critical=True
            ),
            
            # Verification
            DeploymentStep(
                step_id="verify_deployment",
                stage=DeploymentStage.VERIFICATION,
                description="Verify deployment health",
                validation_function=self._verify_deployment,
                critical=True
            )
        ]
        
        self.deployment_steps = steps
    
    def deploy(self) -> DeploymentResult:
        """
        Execute full production deployment
        
        Returns:
            Deployment result with status and details
        """
        logger.info(f"Starting deployment {self.deployment_id} to {self.config.environment.value}")
        
        start_time = datetime.now()
        stages_completed = []
        stages_failed = []
        error_messages = []
        warnings = []
        
        try:
            # Execute deployment steps
            for step in self.deployment_steps:
                logger.info(f"Executing: {step.description}")
                
                try:
                    if not self._execute_step(step):
                        if step.critical:
                            stages_failed.append(step.stage)
                            raise Exception(f"Critical step failed: {step.step_id}")
                        else:
                            warnings.append(f"Non-critical step failed: {step.description}")
                    else:
                        self.completed_steps.append(step)
                        if step.stage not in stages_completed:
                            stages_completed.append(step.stage)
                            
                except Exception as e:
                    logger.error(f"Step failed: {step.step_id} - {e}")
                    error_messages.append(f"{step.step_id}: {str(e)}")
                    self.failed_steps.append(step)
                    
                    if self.config.rollback_on_failure:
                        self._perform_rollback()
                        
                        return DeploymentResult(
                            deployment_id=self.deployment_id,
                            start_time=start_time,
                            end_time=datetime.now(),
                            status=DeploymentStatus.ROLLED_BACK,
                            environment=self.config.environment,
                            stages_completed=stages_completed,
                            stages_failed=stages_failed,
                            deployment_path=self.config.deployment_path,
                            configuration=self._get_deployment_config(),
                            verification_results={},
                            rollback_performed=True,
                            error_messages=error_messages,
                            warnings=warnings
                        )
                    else:
                        raise
            
            # Deployment successful
            verification_results = self._get_verification_results()
            
            return DeploymentResult(
                deployment_id=self.deployment_id,
                start_time=start_time,
                end_time=datetime.now(),
                status=DeploymentStatus.VERIFIED if verification_results.get('healthy', False) else DeploymentStatus.COMPLETED,
                environment=self.config.environment,
                stages_completed=stages_completed,
                stages_failed=stages_failed,
                deployment_path=self.config.deployment_path,
                configuration=self._get_deployment_config(),
                verification_results=verification_results,
                rollback_performed=False,
                error_messages=error_messages,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            error_messages.append(str(e))
            
            return DeploymentResult(
                deployment_id=self.deployment_id,
                start_time=start_time,
                end_time=datetime.now(),
                status=DeploymentStatus.FAILED,
                environment=self.config.environment,
                stages_completed=stages_completed,
                stages_failed=stages_failed,
                deployment_path=self.config.deployment_path,
                configuration=self._get_deployment_config(),
                verification_results={},
                rollback_performed=False,
                error_messages=error_messages,
                warnings=warnings
            )
    
    def _execute_step(self, step: DeploymentStep) -> bool:
        """Execute a single deployment step"""
        
        # Execute command if specified
        if step.command:
            result = subprocess.run(
                step.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=step.timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Command failed: {result.stderr}")
                return False
        
        # Execute validation function if specified
        if step.validation_function:
            try:
                return step.validation_function()
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                return False
        
        return True
    
    def _validate_backup(self) -> bool:
        """Validate backup creation"""
        
        if not self.config.backup_existing:
            return True
        
        try:
            # Check if deployment exists
            if not os.path.exists(self.config.deployment_path):
                logger.info("No existing deployment to backup")
                return True
            
            # Create backup
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_path = f"/tmp/adaptive_regime_backup_{backup_timestamp}"
            
            shutil.copytree(self.config.deployment_path, self.backup_path)
            
            # Backup configuration
            if os.path.exists(self.config.config_path):
                config_backup = f"{self.backup_path}_config"
                shutil.copytree(self.config.config_path, config_backup)
            
            logger.info(f"Backup created at: {self.backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def _restore_backup(self) -> bool:
        """Restore from backup"""
        
        if not self.backup_path or not os.path.exists(self.backup_path):
            logger.warning("No backup available to restore")
            return False
        
        try:
            # Remove current deployment
            if os.path.exists(self.config.deployment_path):
                shutil.rmtree(self.config.deployment_path)
            
            # Restore from backup
            shutil.copytree(self.backup_path, self.config.deployment_path)
            
            # Restore configuration
            config_backup = f"{self.backup_path}_config"
            if os.path.exists(config_backup):
                if os.path.exists(self.config.config_path):
                    shutil.rmtree(self.config.config_path)
                shutil.copytree(config_backup, self.config.config_path)
            
            logger.info("Backup restored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def _validate_environment(self) -> bool:
        """Validate deployment environment"""
        
        checks = []
        
        # Check OS
        if sys.platform.startswith('linux'):
            checks.append(('os', True, "Linux OS detected"))
        else:
            checks.append(('os', False, f"Unsupported OS: {sys.platform}"))
        
        # Check disk space
        deployment_stat = os.statvfs('/')
        free_gb = (deployment_stat.f_frsize * deployment_stat.f_bavail) / (1024**3)
        
        if free_gb > 10:
            checks.append(('disk_space', True, f"Sufficient disk space: {free_gb:.1f}GB"))
        else:
            checks.append(('disk_space', False, f"Insufficient disk space: {free_gb:.1f}GB"))
        
        # Check memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb >= self.config.max_memory_gb * 0.5:
            checks.append(('memory', True, f"Sufficient memory: {available_gb:.1f}GB"))
        else:
            checks.append(('memory', False, f"Insufficient memory: {available_gb:.1f}GB"))
        
        # Check network connectivity
        try:
            socket.create_connection((self.config.heavydb_host, self.config.heavydb_port), timeout=5)
            checks.append(('network', True, "Network connectivity verified"))
        except:
            checks.append(('network', False, "Cannot connect to HeavyDB"))
        
        # Evaluate results
        for check_name, passed, message in checks:
            if passed:
                logger.info(f"✅ {message}")
            else:
                logger.error(f"❌ {message}")
        
        return all(passed for _, passed, _ in checks)
    
    def _validate_directories(self) -> bool:
        """Validate directory creation"""
        
        required_dirs = [
            self.config.deployment_path,
            self.config.config_path,
            self.config.log_path,
            self.config.data_path
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                logger.error(f"Directory not found: {directory}")
                return False
            
            if not os.access(directory, os.W_OK):
                logger.error(f"Directory not writable: {directory}")
                return False
        
        return True
    
    def _validate_python_version(self) -> bool:
        """Validate Python version"""
        
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            logger.info(f"Python version OK: {sys.version}")
            return True
        else:
            logger.error(f"Python {required_version} or higher required, found {current_version}")
            return False
    
    def _validate_dependencies(self) -> bool:
        """Validate Python dependencies"""
        
        required_packages = [
            'numpy',
            'pandas',
            'scikit-learn',
            'scipy',
            'joblib',
            'pyheavydb'  # or 'heavyai'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                # Try alternative names
                if package == 'pyheavydb':
                    try:
                        __import__('heavyai')
                    except ImportError:
                        missing_packages.append(package)
                else:
                    missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            return False
        
        return True
    
    def _validate_heavydb_connection(self) -> bool:
        """Validate HeavyDB connection"""
        
        try:
            # Try to import connector
            try:
                from pyheavydb import connect
            except ImportError:
                from heavyai import connect
            
            # Test connection
            conn = connect(
                host=self.config.heavydb_host,
                port=self.config.heavydb_port,
                user=self.config.heavydb_user,
                password=self.config.heavydb_password,
                dbname=self.config.heavydb_database
            )
            
            # Test query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            conn.close()
            
            logger.info("HeavyDB connection verified")
            return True
            
        except Exception as e:
            logger.error(f"HeavyDB connection failed: {e}")
            return False
    
    def _validate_market_data(self) -> bool:
        """Validate market data availability"""
        
        try:
            # Import connector
            try:
                from pyheavydb import connect
            except ImportError:
                from heavyai import connect
            
            import pandas as pd
            
            # Connect to HeavyDB
            conn = connect(
                host=self.config.heavydb_host,
                port=self.config.heavydb_port,
                user=self.config.heavydb_user,
                password=self.config.heavydb_password,
                dbname=self.config.heavydb_database
            )
            
            # Check data availability
            query = "SELECT COUNT(*) as count FROM nifty_option_chain"
            df = pd.read_sql(query, conn)
            row_count = df['count'].iloc[0]
            
            conn.close()
            
            if row_count > 1000000:  # Expect at least 1M rows
                logger.info(f"Market data verified: {row_count:,} rows")
                return True
            else:
                logger.error(f"Insufficient market data: {row_count} rows")
                return False
                
        except Exception as e:
            logger.error(f"Market data validation failed: {e}")
            return False
    
    def _deploy_configuration(self) -> bool:
        """Deploy configuration files"""
        
        try:
            # Create main configuration
            main_config = {
                'environment': self.config.environment.value,
                'deployment_id': self.deployment_id,
                'deployment_time': datetime.now().isoformat(),
                'service': {
                    'name': self.config.service_name,
                    'port': self.config.service_port,
                    'workers': self.config.worker_processes,
                    'max_memory_gb': self.config.max_memory_gb
                },
                'database': {
                    'heavydb': {
                        'host': self.config.heavydb_host,
                        'port': self.config.heavydb_port,
                        'user': self.config.heavydb_user,
                        'database': self.config.heavydb_database
                    }
                },
                'logging': {
                    'level': self.config.log_level,
                    'path': self.config.log_path
                },
                'monitoring': {
                    'enabled': self.config.enable_monitoring,
                    'metrics_port': self.config.metrics_port
                }
            }
            
            # Save configuration
            config_file = os.path.join(self.config.config_path, 'adaptive_regime.json')
            with open(config_file, 'w') as f:
                json.dump(main_config, f, indent=2)
            
            # Create logging configuration
            log_config = {
                'version': 1,
                'disable_existing_loggers': False,
                'formatters': {
                    'standard': {
                        'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                    }
                },
                'handlers': {
                    'file': {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'filename': os.path.join(self.config.log_path, 'adaptive_regime.log'),
                        'maxBytes': 104857600,  # 100MB
                        'backupCount': 10,
                        'formatter': 'standard'
                    }
                },
                'root': {
                    'level': self.config.log_level,
                    'handlers': ['file']
                }
            }
            
            log_config_file = os.path.join(self.config.config_path, 'logging.json')
            with open(log_config_file, 'w') as f:
                json.dump(log_config, f, indent=2)
            
            logger.info("Configuration deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration deployment failed: {e}")
            return False
    
    def _rollback_configuration(self) -> bool:
        """Rollback configuration changes"""
        
        try:
            # Remove deployed configuration
            config_files = [
                os.path.join(self.config.config_path, 'adaptive_regime.json'),
                os.path.join(self.config.config_path, 'logging.json')
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    os.remove(config_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
            return False
    
    def _deploy_application_code(self) -> bool:
        """Deploy application code"""
        
        try:
            # Get source path (current adaptive directory)
            source_path = Path(__file__).parent.parent  # Go up to adaptive directory
            
            # Create deployment structure
            deployment_structure = [
                'core',
                'analysis', 
                'intelligence',
                'optimization',
                'validation',
                'config',
                'utils'
            ]
            
            for directory in deployment_structure:
                dest_dir = os.path.join(self.config.deployment_path, directory)
                os.makedirs(dest_dir, exist_ok=True)
            
            # Copy Python files
            for root, dirs, files in os.walk(source_path):
                # Skip test directories
                if 'test' in root or '__pycache__' in root:
                    continue
                
                for file in files:
                    if file.endswith('.py'):
                        src_file = os.path.join(root, file)
                        
                        # Determine destination
                        rel_path = os.path.relpath(src_file, source_path)
                        dest_file = os.path.join(self.config.deployment_path, rel_path)
                        
                        # Create destination directory
                        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                        
                        # Copy file
                        shutil.copy2(src_file, dest_file)
            
            # Create __init__.py files
            for root, dirs, files in os.walk(self.config.deployment_path):
                init_file = os.path.join(root, '__init__.py')
                if not os.path.exists(init_file):
                    open(init_file, 'a').close()
            
            logger.info("Application code deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Code deployment failed: {e}")
            return False
    
    def _rollback_application_code(self) -> bool:
        """Rollback application code deployment"""
        
        try:
            # Remove deployed code
            if os.path.exists(self.config.deployment_path):
                shutil.rmtree(self.config.deployment_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Code rollback failed: {e}")
            return False
    
    def _create_system_service(self) -> bool:
        """Create systemd service"""
        
        try:
            # Create service file content
            service_content = f"""[Unit]
Description=Adaptive Market Regime Formation System
After=network.target

[Service]
Type=simple
User=adaptive_regime
Group=adaptive_regime
WorkingDirectory={self.config.deployment_path}
Environment="PYTHONPATH={self.config.deployment_path}"
ExecStart=/usr/bin/python3 -m adaptive_regime_service
Restart=always
RestartSec=10
StandardOutput=append:{self.config.log_path}/service.log
StandardError=append:{self.config.log_path}/service_error.log

# Resource limits
MemoryLimit={self.config.max_memory_gb}G
CPUQuota={self.config.worker_processes * 100}%

[Install]
WantedBy=multi-user.target
"""
            
            # Write service file
            service_file = f"/etc/systemd/system/{self.config.service_name}.service"
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(service_content)
                temp_file = f.name
            
            # Move to system directory (requires sudo)
            subprocess.run(f"sudo mv {temp_file} {service_file}", shell=True, check=True)
            subprocess.run(f"sudo chmod 644 {service_file}", shell=True, check=True)
            
            # Create service user if not exists
            try:
                subprocess.run("id adaptive_regime", shell=True, check=True)
            except:
                subprocess.run("sudo useradd -r -s /bin/false adaptive_regime", shell=True, check=True)
            
            # Set ownership
            subprocess.run(f"sudo chown -R adaptive_regime:adaptive_regime {self.config.deployment_path}", shell=True, check=True)
            subprocess.run(f"sudo chown -R adaptive_regime:adaptive_regime {self.config.log_path}", shell=True, check=True)
            subprocess.run(f"sudo chown -R adaptive_regime:adaptive_regime {self.config.data_path}", shell=True, check=True)
            
            # Reload systemd
            subprocess.run("sudo systemctl daemon-reload", shell=True, check=True)
            
            logger.info("System service created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Service creation failed: {e}")
            return False
    
    def _remove_system_service(self) -> bool:
        """Remove system service"""
        
        try:
            service_file = f"/etc/systemd/system/{self.config.service_name}.service"
            
            # Stop service if running
            subprocess.run(f"sudo systemctl stop {self.config.service_name}", shell=True)
            
            # Remove service file
            subprocess.run(f"sudo rm -f {service_file}", shell=True)
            
            # Reload systemd
            subprocess.run("sudo systemctl daemon-reload", shell=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Service removal failed: {e}")
            return False
    
    def _run_integration_tests(self) -> bool:
        """Run integration tests"""
        
        try:
            # Import test modules
            sys.path.insert(0, self.config.deployment_path)
            
            # Run basic integration test
            test_results = []
            
            # Test 1: Import modules
            try:
                from core.adaptive_scoring_layer import AdaptiveScoringLayer
                from analysis.transition_matrix_analyzer import TransitionMatrixAnalyzer
                from optimization.continuous_learning_engine import ContinuousLearningEngine
                test_results.append(('module_import', True, "Module imports successful"))
            except Exception as e:
                test_results.append(('module_import', False, f"Module import failed: {e}"))
            
            # Test 2: Component initialization
            try:
                asl = AdaptiveScoringLayer()
                analyzer = TransitionMatrixAnalyzer()
                test_results.append(('component_init', True, "Component initialization successful"))
            except Exception as e:
                test_results.append(('component_init', False, f"Component initialization failed: {e}"))
            
            # Test 3: Database connectivity
            try:
                from pyheavydb import connect
                conn = connect(
                    host=self.config.heavydb_host,
                    port=self.config.heavydb_port,
                    user=self.config.heavydb_user,
                    password=self.config.heavydb_password,
                    dbname=self.config.heavydb_database
                )
                conn.close()
                test_results.append(('database_connect', True, "Database connection successful"))
            except Exception as e:
                test_results.append(('database_connect', False, f"Database connection failed: {e}"))
            
            # Evaluate results
            for test_name, passed, message in test_results:
                if passed:
                    logger.info(f"✅ {message}")
                else:
                    logger.error(f"❌ {message}")
            
            return all(passed for _, passed, _ in test_results)
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            return False
    
    def _establish_performance_baseline(self) -> bool:
        """Establish performance baseline"""
        
        try:
            baseline_metrics = {}
            
            # Measure import time
            import time
            start_time = time.time()
            from core.adaptive_scoring_layer import AdaptiveScoringLayer
            import_time = time.time() - start_time
            baseline_metrics['module_import_time'] = import_time
            
            # Measure component initialization
            start_time = time.time()
            asl = AdaptiveScoringLayer()
            init_time = time.time() - start_time
            baseline_metrics['component_init_time'] = init_time
            
            # Measure scoring performance
            import numpy as np
            market_data = {
                'regime_count': 12,
                'volatility': 0.2,
                'trend': 0.01,
                'volume_ratio': 1.0
            }
            
            start_time = time.time()
            for _ in range(100):
                scores = asl.calculate_regime_scores(market_data)
            scoring_time = (time.time() - start_time) / 100
            baseline_metrics['regime_scoring_time_ms'] = scoring_time * 1000
            
            # Save baseline
            baseline_file = os.path.join(self.config.data_path, 'performance_baseline.json')
            with open(baseline_file, 'w') as f:
                json.dump({
                    'deployment_id': self.deployment_id,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': baseline_metrics
                }, f, indent=2)
            
            logger.info(f"Performance baseline established: {baseline_metrics}")
            return True
            
        except Exception as e:
            logger.error(f"Performance baseline failed: {e}")
            return False
    
    def _setup_monitoring(self) -> bool:
        """Setup production monitoring"""
        
        if not self.config.enable_monitoring:
            return True
        
        try:
            # Create monitoring configuration
            monitoring_config = {
                'metrics': {
                    'port': self.config.metrics_port,
                    'endpoint': '/metrics',
                    'collect_interval': 60
                },
                'alerts': {
                    'cpu_threshold': 80,
                    'memory_threshold': 90,
                    'error_rate_threshold': 0.05,
                    'latency_threshold_ms': 100
                },
                'dashboards': {
                    'system_health': True,
                    'regime_analysis': True,
                    'performance_metrics': True
                }
            }
            
            # Save monitoring configuration
            monitoring_file = os.path.join(self.config.config_path, 'monitoring.json')
            with open(monitoring_file, 'w') as f:
                json.dump(monitoring_config, f, indent=2)
            
            # Create monitoring scripts
            self._create_monitoring_scripts()
            
            logger.info("Monitoring setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return False
    
    def _create_monitoring_scripts(self):
        """Create monitoring helper scripts"""
        
        # Health check script
        health_check_script = f"""#!/bin/bash
curl -s http://localhost:{self.config.service_port}/health || exit 1
"""
        
        script_path = os.path.join(self.config.deployment_path, 'scripts')
        os.makedirs(script_path, exist_ok=True)
        
        health_check_file = os.path.join(script_path, 'health_check.sh')
        with open(health_check_file, 'w') as f:
            f.write(health_check_script)
        
        os.chmod(health_check_file, 0o755)
    
    def _validate_service_running(self) -> bool:
        """Validate service is running"""
        
        try:
            # Check systemctl status
            result = subprocess.run(
                f"systemctl is-active {self.config.service_name}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.stdout.strip() != "active":
                logger.error(f"Service not active: {result.stdout}")
                return False
            
            # Get service PID
            result = subprocess.run(
                f"systemctl show -p MainPID {self.config.service_name}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            pid_line = result.stdout.strip()
            if pid_line.startswith("MainPID="):
                self.service_pid = int(pid_line.split("=")[1])
                self.service_start_time = datetime.now()
            
            logger.info(f"Service running with PID: {self.service_pid}")
            return True
            
        except Exception as e:
            logger.error(f"Service validation failed: {e}")
            return False
    
    def _stop_service(self) -> bool:
        """Stop the service"""
        
        try:
            subprocess.run(f"sudo systemctl stop {self.config.service_name}", shell=True)
            return True
        except:
            return False
    
    def _verify_deployment(self) -> bool:
        """Verify deployment health"""
        
        try:
            # Wait for service to stabilize
            time.sleep(5)
            
            # Check service health
            health = self.check_service_health()
            
            if health.status != "healthy":
                logger.error(f"Service unhealthy: {health.status}")
                return False
            
            # Check endpoints
            unhealthy_endpoints = [
                endpoint for endpoint, healthy in health.endpoints_healthy.items()
                if not healthy
            ]
            
            if unhealthy_endpoints:
                logger.error(f"Unhealthy endpoints: {unhealthy_endpoints}")
                return False
            
            logger.info("Deployment verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Deployment verification failed: {e}")
            return False
    
    def check_service_health(self) -> ServiceHealth:
        """Check service health status"""
        
        try:
            # Get process info
            if self.service_pid and psutil.pid_exists(self.service_pid):
                process = psutil.Process(self.service_pid)
                memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
                cpu_percent = process.cpu_percent(interval=1.0)
                uptime = (datetime.now() - self.service_start_time).total_seconds()
                status = "healthy"
            else:
                memory_usage = 0
                cpu_percent = 0
                uptime = 0
                status = "not_running"
            
            # Check endpoints
            endpoints = {
                'health': f"http://localhost:{self.config.service_port}/health",
                'metrics': f"http://localhost:{self.config.metrics_port}/metrics" if self.config.enable_monitoring else None
            }
            
            endpoints_healthy = {}
            
            for endpoint_name, url in endpoints.items():
                if url:
                    try:
                        response = requests.get(url, timeout=5)
                        endpoints_healthy[endpoint_name] = response.status_code == 200
                    except:
                        endpoints_healthy[endpoint_name] = False
            
            # Determine overall health
            if status == "healthy" and all(endpoints_healthy.values()):
                status = "healthy"
            elif status == "healthy":
                status = "degraded"
            
            return ServiceHealth(
                service_name=self.config.service_name,
                status=status,
                pid=self.service_pid,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_percent,
                uptime_seconds=uptime,
                last_check=datetime.now(),
                endpoints_healthy=endpoints_healthy
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            
            return ServiceHealth(
                service_name=self.config.service_name,
                status="error",
                pid=None,
                memory_usage_mb=0,
                cpu_percent=0,
                uptime_seconds=0,
                last_check=datetime.now(),
                endpoints_healthy={}
            )
    
    def _perform_rollback(self):
        """Perform deployment rollback"""
        
        logger.warning("Performing deployment rollback...")
        
        # Execute rollback functions for completed steps in reverse order
        for step in reversed(self.completed_steps):
            if step.rollback_function:
                try:
                    step.rollback_function()
                    logger.info(f"Rolled back: {step.description}")
                except Exception as e:
                    logger.error(f"Rollback failed for {step.step_id}: {e}")
    
    def _get_deployment_config(self) -> Dict[str, Any]:
        """Get deployment configuration"""
        
        return {
            'deployment_id': self.deployment_id,
            'environment': self.config.environment.value,
            'deployment_path': self.config.deployment_path,
            'config_path': self.config.config_path,
            'service_name': self.config.service_name,
            'service_port': self.config.service_port,
            'monitoring_enabled': self.config.enable_monitoring
        }
    
    def _get_verification_results(self) -> Dict[str, Any]:
        """Get deployment verification results"""
        
        health = self.check_service_health()
        
        return {
            'healthy': health.status == "healthy",
            'service_status': health.status,
            'memory_usage_mb': health.memory_usage_mb,
            'cpu_percent': health.cpu_percent,
            'uptime_seconds': health.uptime_seconds,
            'endpoints_healthy': health.endpoints_healthy
        }
    
    def generate_deployment_report(self, result: DeploymentResult, filepath: str):
        """Generate deployment report"""
        
        report = {
            'deployment_summary': {
                'deployment_id': result.deployment_id,
                'environment': result.environment.value,
                'status': result.status.value,
                'duration': (result.end_time - result.start_time).total_seconds(),
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat()
            },
            'stages': {
                'completed': [stage.value for stage in result.stages_completed],
                'failed': [stage.value for stage in result.stages_failed]
            },
            'configuration': result.configuration,
            'verification': result.verification_results,
            'issues': {
                'errors': result.error_messages,
                'warnings': result.warnings
            },
            'rollback_performed': result.rollback_performed,
            'deployment_path': result.deployment_path
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Deployment report generated: {filepath}")


# Example usage
if __name__ == "__main__":
    # Create deployment configuration
    config = DeploymentConfig(
        environment=EnvironmentType.STAGING,  # Use staging for testing
        deployment_path="/tmp/adaptive_regime_test",
        config_path="/tmp/adaptive_regime_test/config",
        log_path="/tmp/adaptive_regime_test/logs",
        data_path="/tmp/adaptive_regime_test/data",
        enable_monitoring=False,  # Disable for testing
        force_deployment=True
    )
    
    # Create deployer
    deployer = ProductionDeployer(config)
    
    # Execute deployment
    result = deployer.deploy()
    
    # Generate report
    report_path = f"/tmp/deployment_report_{result.deployment_id}.json"
    deployer.generate_deployment_report(result, report_path)
    
    # Display results
    print(f"\nDeployment {result.deployment_id}")
    print(f"Status: {result.status.value}")
    print(f"Environment: {result.environment.value}")
    print(f"Duration: {(result.end_time - result.start_time).total_seconds():.1f} seconds")
    print(f"Stages completed: {len(result.stages_completed)}")
    print(f"Stages failed: {len(result.stages_failed)}")
    
    if result.error_messages:
        print("\nErrors:")
        for error in result.error_messages:
            print(f"- {error}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    
    print(f"\nReport saved to: {report_path}")
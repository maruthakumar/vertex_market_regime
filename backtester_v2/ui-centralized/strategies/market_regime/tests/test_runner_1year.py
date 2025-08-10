#!/usr/bin/env python3
"""
Market Regime 1-Year Analysis Test Runner
========================================
Enhanced test runner for comprehensive 1-year Market Regime analysis
with automated testing, monitoring, and result validation.

Features:
- Automated end-to-end testing for 1-year periods
- Parallel testing for multiple configurations
- Real-time progress monitoring
- Comprehensive result analysis
- Automatic retry mechanisms
- Performance benchmarking

Author: Market Regime Testing Team
Date: 2025-06-27
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import requests
from dataclasses import dataclass, asdict
import traceback

# Import Market Regime components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_market_regime_engine import EnhancedMarketRegimeEngine
from correlation_based_regime_formation_engine import CorrelationBasedRegimeFormationEngine
from sophisticated_regime_formation_engine import SophisticatedRegimeFormationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'market_regime_1year_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfiguration:
    """Test configuration for 1-year analysis"""
    name: str
    regime_mode: str  # '8_REGIME' or '18_REGIME'
    start_date: str
    end_date: str
    index_name: str = 'NIFTY'
    dte_adaptation: bool = True
    dynamic_weights: bool = True
    excel_config_path: Optional[str] = None
    parallel_workers: int = 4
    batch_size: int = 10000
    enable_caching: bool = True
    
@dataclass
class TestResult:
    """Test result container"""
    config_name: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    total_records: int
    regime_distribution: Dict[str, int]
    transition_count: int
    avg_confidence: float
    performance_metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]
    
class MarketRegime1YearTestRunner:
    """Enhanced test runner for 1-year Market Regime analysis"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/market_regime"
        self.results_dir = Path("market_regime_1year_results")
        self.results_dir.mkdir(exist_ok=True)
        self.cache_dir = Path("market_regime_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_tracker = {
            'processing_times': [],
            'memory_usage': [],
            'regime_transitions': [],
            'confidence_scores': []
        }
        
    async def run_1year_analysis(self, config: TestConfiguration) -> TestResult:
        """Run complete 1-year analysis for a configuration"""
        logger.info(f"Starting 1-year analysis for config: {config.name}")
        start_time = datetime.now()
        
        result = TestResult(
            config_name=config.name,
            start_time=start_time,
            end_time=start_time,
            total_duration=0,
            total_records=0,
            regime_distribution={},
            transition_count=0,
            avg_confidence=0,
            performance_metrics={},
            errors=[],
            warnings=[]
        )
        
        try:
            # Step 1: Upload configuration if provided
            if config.excel_config_path:
                await self._upload_configuration(config)
            
            # Step 2: Start backtest
            backtest_id = await self._start_backtest(config)
            logger.info(f"Backtest started with ID: {backtest_id}")
            
            # Step 3: Monitor progress
            await self._monitor_backtest(backtest_id, result)
            
            # Step 4: Retrieve and analyze results
            results_data = await self._retrieve_results(backtest_id, config)
            
            # Step 5: Analyze regime patterns
            analysis = self._analyze_regime_patterns(results_data)
            
            # Step 6: Calculate performance metrics
            metrics = self._calculate_performance_metrics(results_data)
            
            # Update result
            result.regime_distribution = analysis['regime_distribution']
            result.transition_count = analysis['transition_count']
            result.avg_confidence = analysis['avg_confidence']
            result.performance_metrics = metrics
            result.total_records = len(results_data)
            
        except Exception as e:
            logger.error(f"Error in 1-year analysis: {str(e)}")
            result.errors.append(str(e))
            traceback.print_exc()
            
        finally:
            result.end_time = datetime.now()
            result.total_duration = (result.end_time - result.start_time).total_seconds()
            
        # Save results
        self._save_results(result, config)
        
        return result
    
    async def _upload_configuration(self, config: TestConfiguration):
        """Upload Excel configuration file"""
        with open(config.excel_config_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.api_base}/config/upload",
                files=files
            )
            if response.status_code != 200:
                raise Exception(f"Failed to upload configuration: {response.text}")
    
    async def _start_backtest(self, config: TestConfiguration) -> str:
        """Start Market Regime backtest"""
        payload = {
            'regime_mode': config.regime_mode,
            'dte_adaptation': config.dte_adaptation,
            'dynamic_weights': config.dynamic_weights,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'index_name': config.index_name
        }
        
        response = requests.post(
            f"{self.api_base}/backtest/start",
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to start backtest: {response.text}")
            
        return response.json()['backtest_id']
    
    async def _monitor_backtest(self, backtest_id: str, result: TestResult):
        """Monitor backtest progress with real-time updates"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.get(
                    f"{self.api_base}/backtest/status/{backtest_id}"
                )
                
                if response.status_code != 200:
                    retry_count += 1
                    await asyncio.sleep(5)
                    continue
                
                status_data = response.json()
                status = status_data.get('status')
                progress = status_data.get('progress', 0)
                
                logger.info(f"Backtest progress: {progress}% - Status: {status}")
                
                if status == 'completed':
                    break
                elif status == 'failed':
                    raise Exception(f"Backtest failed: {status_data.get('error')}")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.warning(f"Error monitoring backtest: {str(e)}")
                retry_count += 1
                await asyncio.sleep(5)
    
    async def _retrieve_results(self, backtest_id: str, config: TestConfiguration) -> pd.DataFrame:
        """Retrieve and parse backtest results"""
        # Get CSV output for the date range
        params = {
            'start_time': config.start_date,
            'end_time': config.end_date,
            'include_all_columns': True
        }
        
        response = requests.get(
            f"{self.base_url}/api/v2/regime/output/csv",
            params=params
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve results: {response.text}")
        
        # Parse CSV data
        import io
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        return df
    
    def _analyze_regime_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze regime patterns from results"""
        analysis = {
            'regime_distribution': {},
            'transition_count': 0,
            'avg_confidence': 0,
            'regime_durations': {},
            'transition_matrix': {}
        }
        
        if 'regime_classification' in df.columns:
            # Calculate regime distribution
            regime_counts = df['regime_classification'].value_counts()
            analysis['regime_distribution'] = regime_counts.to_dict()
            
            # Count transitions
            regime_changes = df['regime_classification'].ne(df['regime_classification'].shift())
            analysis['transition_count'] = regime_changes.sum() - 1  # Subtract first row
            
            # Calculate average confidence
            if 'confidence_score' in df.columns:
                analysis['avg_confidence'] = df['confidence_score'].mean()
            
            # Calculate regime durations
            regime_groups = df.groupby((df['regime_classification'] != df['regime_classification'].shift()).cumsum())
            durations = regime_groups.size()
            
            for regime in df['regime_classification'].unique():
                regime_durations = durations[regime_groups['regime_classification'].first() == regime]
                analysis['regime_durations'][regime] = {
                    'avg_duration': regime_durations.mean(),
                    'max_duration': regime_durations.max(),
                    'min_duration': regime_durations.min()
                }
            
            # Build transition matrix
            for i in range(len(df) - 1):
                from_regime = df.iloc[i]['regime_classification']
                to_regime = df.iloc[i + 1]['regime_classification']
                
                if from_regime not in analysis['transition_matrix']:
                    analysis['transition_matrix'][from_regime] = {}
                
                if to_regime not in analysis['transition_matrix'][from_regime]:
                    analysis['transition_matrix'][from_regime][to_regime] = 0
                    
                analysis['transition_matrix'][from_regime][to_regime] += 1
        
        return analysis
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {
            'total_processing_time': 0,
            'avg_processing_time_per_record': 0,
            'peak_memory_usage': 0,
            'regime_stability_score': 0,
            'confidence_consistency': 0
        }
        
        # Calculate processing metrics
        if 'processing_time' in df.columns:
            metrics['total_processing_time'] = df['processing_time'].sum()
            metrics['avg_processing_time_per_record'] = df['processing_time'].mean()
        
        # Calculate regime stability (fewer transitions = more stable)
        if 'regime_classification' in df.columns:
            transitions = (df['regime_classification'] != df['regime_classification'].shift()).sum()
            metrics['regime_stability_score'] = 1 - (transitions / len(df))
        
        # Calculate confidence consistency
        if 'confidence_score' in df.columns:
            metrics['confidence_consistency'] = 1 - df['confidence_score'].std()
        
        return metrics
    
    def _save_results(self, result: TestResult, config: TestConfiguration):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"test_result_{config.name}_{timestamp}.json"
        
        result_data = asdict(result)
        result_data['configuration'] = asdict(config)
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {result_file}")
    
    async def run_parallel_tests(self, configurations: List[TestConfiguration]) -> List[TestResult]:
        """Run multiple test configurations in parallel"""
        logger.info(f"Running {len(configurations)} test configurations in parallel")
        
        tasks = []
        for config in configurations:
            task = asyncio.create_task(self.run_1year_analysis(config))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Test {configurations[i].name} failed: {str(result)}")
                # Create error result
                error_result = TestResult(
                    config_name=configurations[i].name,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    total_duration=0,
                    total_records=0,
                    regime_distribution={},
                    transition_count=0,
                    avg_confidence=0,
                    performance_metrics={},
                    errors=[str(result)],
                    warnings=[]
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    def generate_comparison_report(self, results: List[TestResult]):
        """Generate comparison report for multiple test results"""
        report_path = self.results_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Regime 1-Year Analysis Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .success { color: green; }
                .error { color: red; }
                .warning { color: orange; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Market Regime 1-Year Analysis Comparison Report</h1>
            <p>Generated: {timestamp}</p>
            
            <h2>Test Summary</h2>
            <table>
                <tr>
                    <th>Configuration</th>
                    <th>Duration (s)</th>
                    <th>Records</th>
                    <th>Transitions</th>
                    <th>Avg Confidence</th>
                    <th>Status</th>
                </tr>
        """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        for result in results:
            status = "Success" if not result.errors else "Failed"
            status_class = "success" if not result.errors else "error"
            
            html_content += f"""
                <tr>
                    <td>{result.config_name}</td>
                    <td>{result.total_duration:.2f}</td>
                    <td>{result.total_records}</td>
                    <td>{result.transition_count}</td>
                    <td>{result.avg_confidence:.2%}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Regime Distribution Comparison</h2>
            <div class="chart" id="regime-distribution-chart"></div>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr>
                    <th>Configuration</th>
                    <th>Processing Time</th>
                    <th>Stability Score</th>
                    <th>Confidence Consistency</th>
                </tr>
        """
        
        for result in results:
            metrics = result.performance_metrics
            html_content += f"""
                <tr>
                    <td>{result.config_name}</td>
                    <td>{metrics.get('avg_processing_time_per_record', 0):.3f}s</td>
                    <td>{metrics.get('regime_stability_score', 0):.2%}</td>
                    <td>{metrics.get('confidence_consistency', 0):.2%}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Detailed Results</h2>
        """
        
        for result in results:
            html_content += f"""
            <h3>{result.config_name}</h3>
            <h4>Regime Distribution:</h4>
            <ul>
            """
            
            for regime, count in result.regime_distribution.items():
                percentage = (count / result.total_records * 100) if result.total_records > 0 else 0
                html_content += f"<li>{regime}: {count} ({percentage:.1f}%)</li>"
            
            html_content += "</ul>"
            
            if result.errors:
                html_content += "<h4 class='error'>Errors:</h4><ul>"
                for error in result.errors:
                    html_content += f"<li>{error}</li>"
                html_content += "</ul>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Comparison report saved to: {report_path}")
        return report_path

# Command-line interface
async def main():
    """Main entry point for 1-year test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Regime 1-Year Analysis Test Runner')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--index', default='NIFTY', help='Index name (default: NIFTY)')
    parser.add_argument('--configs', nargs='+', help='Test configuration names')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--compare', action='store_true', help='Generate comparison report')
    
    args = parser.parse_args()
    
    # Create test configurations
    configurations = []
    
    # Default configurations if none specified
    if not args.configs:
        args.configs = ['8_regime_basic', '18_regime_advanced']
    
    for config_name in args.configs:
        if '8_regime' in config_name:
            regime_mode = '8_REGIME'
        else:
            regime_mode = '18_REGIME'
        
        config = TestConfiguration(
            name=config_name,
            regime_mode=regime_mode,
            start_date=args.start_date,
            end_date=args.end_date,
            index_name=args.index
        )
        configurations.append(config)
    
    # Run tests
    runner = MarketRegime1YearTestRunner()
    
    if args.parallel:
        results = await runner.run_parallel_tests(configurations)
    else:
        results = []
        for config in configurations:
            result = await runner.run_1year_analysis(config)
            results.append(result)
    
    # Generate comparison report
    if args.compare and len(results) > 1:
        runner.generate_comparison_report(results)
    
    # Print summary
    print("\n=== Test Summary ===")
    for result in results:
        print(f"\nConfiguration: {result.config_name}")
        print(f"Duration: {result.total_duration:.2f}s")
        print(f"Records: {result.total_records}")
        print(f"Transitions: {result.transition_count}")
        print(f"Avg Confidence: {result.avg_confidence:.2%}")
        print(f"Errors: {len(result.errors)}")

if __name__ == "__main__":
    asyncio.run(main())
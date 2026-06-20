"""
Advanced utilities and helper functions for the 6G spectrum allocation environment.

Includes:
- Experiment tracking and logging
- Visualization utilities
- Performance profiling
- Configuration builders
- Experiment reproducibility tools

Author: Research Team
Date: 2024
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import asdict

from config import Config
from metrics import EpisodeMetrics, StatisticsCollector

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Track and log experiment configurations, results, and metadata.
    
    Useful for:
    - Experiment reproducibility
    - Hyperparameter tracking
    - Results comparison
    - Automated report generation
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "./experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of experiment
            output_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            'experiment_name': experiment_name,
            'timestamp': self.timestamp,
            'config': None,
            'results': [],
        }
        
        logger.info(f"Initialized ExperimentTracker: {self.experiment_dir}")
    
    def save_config(self, config: Config):
        """
        Save configuration to JSON.
        
        Args:
            config: Config object
        """
        self.metadata['config'] = config.to_dict()
        
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.metadata['config'], f, indent=2)
        
        logger.info(f"Saved config to {config_path}")
    
    def record_result(self, result_name: str, metrics: EpisodeMetrics):
        """
        Record episode result.
        
        Args:
            result_name: Name of result
            metrics: EpisodeMetrics object
        """
        result = {
            'name': result_name,
            'timestamp': datetime.now().isoformat(),
            'throughput_mbps': metrics.total_throughput_mbps,
            'delay_ms': metrics.average_delay_ms,
            'fairness': metrics.jain_fairness_index,
            'utilization': metrics.resource_utilization,
            'reward': metrics.episode_reward,
            'packets_dropped': metrics.packets_dropped,
        }
        
        self.metadata['results'].append(result)
    
    def save_results(self):
        """Save all results to JSON."""
        results_path = self.experiment_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.metadata['results'], f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
    
    def generate_report(self) -> str:
        """
        Generate experiment report.
        
        Returns:
            Report as formatted string
        """
        if not self.metadata['results']:
            return "No results recorded yet."
        
        results = self.metadata['results']
        
        # Compute statistics
        throughputs = [r['throughput_mbps'] for r in results]
        delays = [r['delay_ms'] for r in results]
        fairness = [r['fairness'] for r in results]
        
        report = (
            f"\n{'='*80}\n"
            f"Experiment Report: {self.experiment_name}\n"
            f"{'='*80}\n"
            f"Timestamp: {self.timestamp}\n"
            f"Results: {len(results)} episodes\n"
            f"\n"
            f"Throughput (Mbps):\n"
            f"  Mean: {np.mean(throughputs):.2f}\n"
            f"  Std: {np.std(throughputs):.2f}\n"
            f"  Min: {np.min(throughputs):.2f}\n"
            f"  Max: {np.max(throughputs):.2f}\n"
            f"\n"
            f"Delay (ms):\n"
            f"  Mean: {np.mean(delays):.2f}\n"
            f"  Std: {np.std(delays):.2f}\n"
            f"  Min: {np.min(delays):.2f}\n"
            f"  Max: {np.max(delays):.2f}\n"
            f"\n"
            f"Fairness Index:\n"
            f"  Mean: {np.mean(fairness):.4f}\n"
            f"  Std: {np.std(fairness):.4f}\n"
            f"  Min: {np.min(fairness):.4f}\n"
            f"  Max: {np.max(fairness):.4f}\n"
            f"{'='*80}\n"
        )
        
        return report
    
    def finalize(self):
        """Finalize experiment and save all data."""
        self.save_results()
        report = self.generate_report()
        
        report_path = self.experiment_dir / "report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Experiment finalized: {self.experiment_dir}")
        print(report)


class ConfigurationBuilder:
    """
    Builder pattern for creating environment configurations.
    
    Simplifies creation of configurations for different scenarios.
    """
    
    @staticmethod
    def create_light_scenario() -> Config:
        """Create lightweight config for quick testing."""
        config = Config()
        config.env.num_users = 10
        config.env.num_resource_blocks = 25
        config.env.episode_length = 500
        return config
    
    @staticmethod
    def create_medium_scenario() -> Config:
        """Create medium-scale config for standard experiments."""
        config = Config()
        config.env.num_users = 20
        config.env.num_resource_blocks = 50
        config.env.episode_length = 1000
        return config
    
    @staticmethod
    def create_heavy_scenario() -> Config:
        """Create large-scale config for stress testing."""
        config = Config()
        config.env.num_users = 100
        config.env.num_resource_blocks = 200
        config.env.episode_length = 2000
        return config
    
    @staticmethod
    def create_fairness_focused() -> Config:
        """Create config emphasizing fairness over throughput."""
        config = Config()
        config.reward.throughput_weight = 0.2
        config.reward.fairness_weight = 0.5
        config.reward.delay_weight = 0.2
        config.reward.queue_penalty_weight = 0.1
        return config
    
    @staticmethod
    def create_throughput_focused() -> Config:
        """Create config emphasizing throughput."""
        config = Config()
        config.reward.throughput_weight = 0.6
        config.reward.fairness_weight = 0.1
        config.reward.delay_weight = 0.2
        config.reward.queue_penalty_weight = 0.1
        return config
    
    @staticmethod
    def create_delay_sensitive() -> Config:
        """Create config emphasizing low delay."""
        config = Config()
        config.reward.throughput_weight = 0.2
        config.reward.delay_weight = 0.5
        config.reward.fairness_weight = 0.2
        config.reward.queue_penalty_weight = 0.1
        return config


class PerformanceProfiler:
    """
    Profile environment and algorithm performance.
    
    Measures:
    - Inference time per step
    - Memory usage
    - Computational complexity
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.metrics = {}
    
    def profile_environment_step(self, env, n_steps: int = 1000) -> Dict[str, float]:
        """
        Profile environment step execution.
        
        Args:
            env: SpectrumAllocationEnv instance
            n_steps: Number of steps to profile
        
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        obs, _ = env.reset()
        times = []
        
        for _ in range(n_steps):
            action = env.action_space.sample()
            
            start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        times = np.array(times)
        
        return {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'total_steps': n_steps,
        }
    
    def profile_agent_inference(
        self,
        env,
        agent,
        n_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Profile agent inference time.
        
        Args:
            env: Environment instance
            agent: Agent with get_action() method
            n_steps: Number of steps to profile
        
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        obs, _ = env.reset()
        times = []
        
        for _ in range(n_steps):
            start = time.perf_counter()
            action = agent.get_action(obs)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            
            obs, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        times = np.array(times)
        
        return {
            'mean_time_us': np.mean(times) * 1e6,
            'std_time_us': np.std(times) * 1e6,
            'min_time_us': np.min(times) * 1e6,
            'max_time_us': np.max(times) * 1e6,
            'total_steps': n_steps,
        }


class SweepExperiment:
    """
    Perform systematic hyperparameter sweep experiments.
    
    Example:
    ```python
    sweep = SweepExperiment("num_users_sweep")
    sweep.add_parameter("num_users", [10, 20, 50, 100])
    
    for config in sweep.generate_configs():
        env = SpectrumAllocationEnv(config.env, ...)
        # Run experiment with this config
    ```
    """
    
    def __init__(self, name: str):
        """Initialize sweep experiment."""
        self.name = name
        self.parameters = {}
    
    def add_parameter(self, name: str, values: List[Any]):
        """
        Add parameter to sweep.
        
        Args:
            name: Parameter name (e.g., "num_users")
            values: List of values to sweep
        """
        self.parameters[name] = values
    
    def generate_configs(self):
        """
        Generate all configurations from parameter sweep.
        
        Yields:
            Config objects with swept parameters set
        """
        import itertools
        
        param_names = list(self.parameters.keys())
        param_values = [self.parameters[name] for name in param_names]
        
        for values in itertools.product(*param_values):
            config = Config()
            
            for param_name, value in zip(param_names, values):
                # Set parameter in config
                if hasattr(config.env, param_name):
                    setattr(config.env, param_name, value)
                elif hasattr(config.channel, param_name):
                    setattr(config.channel, param_name, value)
                elif hasattr(config.traffic, param_name):
                    setattr(config.traffic, param_name, value)
                elif hasattr(config.reward, param_name):
                    setattr(config.reward, param_name, value)
            
            yield config
    
    def count_configs(self) -> int:
        """Count total number of configurations in sweep."""
        count = 1
        for values in self.parameters.values():
            count *= len(values)
        return count


def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured at level {logging.getLevelName(log_level)}")


if __name__ == "__main__":
    # Example usage
    setup_logging(logging.INFO)
    
    # Test ConfigurationBuilder
    print("Available scenarios:")
    print("- Light:", ConfigurationBuilder.create_light_scenario().env.num_users, "users")
    print("- Medium:", ConfigurationBuilder.create_medium_scenario().env.num_users, "users")
    print("- Heavy:", ConfigurationBuilder.create_heavy_scenario().env.num_users, "users")
    
    # Test SweepExperiment
    sweep = SweepExperiment("test_sweep")
    sweep.add_parameter("num_users", [10, 20, 50])
    sweep.add_parameter("num_resource_blocks", [25, 50])
    
    print(f"\nSweep configurations: {sweep.count_configs()}")
    
    # Test ExperimentTracker
    tracker = ExperimentTracker("test_experiment")
    config = Config()
    tracker.save_config(config)
    
    # Record some dummy results
    from metrics import EpisodeMetrics
    metrics = EpisodeMetrics(
        total_throughput_mbps=50.0,
        average_delay_ms=75.0,
        jain_fairness_index=0.75,
        resource_utilization=0.85,
        episode_reward=100.0
    )
    tracker.record_result("test_result", metrics)
    tracker.finalize()

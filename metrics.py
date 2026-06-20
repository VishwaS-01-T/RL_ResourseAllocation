"""
Metrics and evaluation module for 6G spectrum allocation environment.

Implements performance metrics for evaluating spectrum allocation policies:
1. Throughput (total and per-user)
2. Delay (average and distribution)
3. Jain Fairness Index
4. Resource Utilization
5. Episode Reward

Author: Research Team
Date: 2024
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """
    Aggregated metrics for a complete episode.
    
    Attributes:
        total_throughput_mbps: Total throughput across all users (Mbps)
        per_user_throughput: Throughput per user (Mbps)
        average_delay_ms: Average packet delay (ms)
        per_user_delay: Delay per user (ms)
        jain_fairness_index: Fairness metric (0-1, higher is fairer)
        resource_utilization: Fraction of resource blocks used (0-1)
        episode_reward: Cumulative episode reward
        packets_dropped: Total packets dropped due to queue overflow
        queue_violations: Number of timesteps with QoS violations
    """
    
    total_throughput_mbps: float = 0.0
    per_user_throughput: List[float] = field(default_factory=list)
    average_delay_ms: float = 0.0
    per_user_delay: List[float] = field(default_factory=list)
    jain_fairness_index: float = 0.0
    resource_utilization: float = 0.0
    episode_reward: float = 0.0
    packets_dropped: int = 0
    queue_violations: int = 0
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        return (
            f"EpisodeMetrics:\n"
            f"  Total Throughput: {self.total_throughput_mbps:.2f} Mbps\n"
            f"  Avg Delay: {self.average_delay_ms:.2f} ms\n"
            f"  Jain Fairness: {self.jain_fairness_index:.4f}\n"
            f"  Resource Util: {self.resource_utilization:.2%}\n"
            f"  Episode Reward: {self.episode_reward:.2f}\n"
            f"  Packets Dropped: {self.packets_dropped}\n"
            f"  QoS Violations: {self.queue_violations}"
        )


class MetricsCalculator:
    """
    Calculates performance metrics for spectrum allocation.
    
    Mathematical Definitions:
    ========================
    
    1. Throughput:
       T_total = Σ_i T_i (sum across all users)
       T_i = bits_transmitted_i / episode_duration
    
    2. Average Delay:
       D_avg = (Σ_i D_i) / num_packets
       where D_i is delay of packet i
    
    3. Jain Fairness Index:
       J = (Σ_i T_i)² / (n · Σ_i T_i²)
       Range: [1/n, 1], where 1 is perfectly fair
    
    4. Resource Utilization:
       U = (Σ_t allocated_rbs_t) / (episode_length × num_rbs)
       Range: [0, 1]
    
    5. QoS Fairness:
       Measures how well users' minimum rate requirements are met
    """
    
    def __init__(self, config):
        """
        Initialize metrics calculator.
        
        Args:
            config: EnvironmentConfig object
        """
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset metrics tracking."""
        self.episode_throughputs = []  # Per-user throughput (Mbps)
        self.episode_delays = []  # Per-user delay (ms)
        self.episode_rewards = 0.0
        self.total_allocated_rbs = 0
        self.total_packets_dropped = 0
        self.qos_violations = 0
        self.timestep_count = 0
    
    @staticmethod
    def jain_fairness_index(throughputs: List[float]) -> float:
        """
        Calculate Jain Fairness Index.
        
        Formula: J = (Σ T_i)² / (n · Σ T_i²)
        
        Properties:
        - Range: [1/n, 1] where 1 is perfect fairness
        - Interpretation: 
          - 1.0 = all users have equal throughput (perfect fairness)
          - 0.5 = moderate fairness
          - 0.1 = poor fairness (high variation)
        
        Args:
            throughputs: List of throughput values
        
        Returns:
            Jain Fairness Index (0-1)
        """
        if len(throughputs) == 0:
            return 0.0
        
        throughputs = np.array(throughputs, dtype=np.float32)
        n = len(throughputs)
        
        numerator = np.sum(throughputs) ** 2
        denominator = n * np.sum(throughputs ** 2)
        
        if denominator == 0:
            return 0.0
        
        return float(numerator / denominator)
    
    @staticmethod
    def calculate_throughput(
        packets_transmitted: List[int],
        packet_size_bits: int,
        episode_duration_seconds: float
    ) -> Tuple[float, List[float]]:
        """
        Calculate total and per-user throughput.
        
        Args:
            packets_transmitted: Packets sent per user
            packet_size_bits: Size of each packet
            episode_duration_seconds: Episode length in seconds
        
        Returns:
            Tuple of (total_throughput_mbps, per_user_throughput_mbps)
        """
        per_user_throughput = [
            (pkt * packet_size_bits) / (episode_duration_seconds * 1e6)
            for pkt in packets_transmitted
        ]
        total_throughput = sum(per_user_throughput)
        
        return total_throughput, per_user_throughput
    
    @staticmethod
    def calculate_average_delay(
        per_user_delays: List[float],
        packets_transmitted: List[int]
    ) -> Tuple[float, List[float]]:
        """
        Calculate weighted average delay.
        
        Args:
            per_user_delays: Average delay per user (ms)
            packets_transmitted: Packets sent per user
        
        Returns:
            Tuple of (average_delay_ms, per_user_delay_ms)
        """
        total_packets = sum(packets_transmitted)
        
        if total_packets == 0:
            return 0.0, per_user_delays
        
        weighted_delay = sum(
            delay * pkt_count for delay, pkt_count in 
            zip(per_user_delays, packets_transmitted)
        ) / total_packets
        
        return weighted_delay, per_user_delays
    
    @staticmethod
    def calculate_resource_utilization(
        total_allocated_rbs: int,
        episode_length: int,
        num_rbs: int
    ) -> float:
        """
        Calculate resource block utilization.
        
        Utilization = (allocated RBs) / (max possible RBs in episode)
        
        Args:
            total_allocated_rbs: Total RBs allocated throughout episode
            episode_length: Number of timesteps
            num_rbs: Total available RBs per timestep
        
        Returns:
            Utilization ratio (0-1)
        """
        max_possible_rbs = episode_length * num_rbs
        if max_possible_rbs == 0:
            return 0.0
        return min(1.0, total_allocated_rbs / max_possible_rbs)
    
    def record_timestep(
        self,
        allocated_rbs: int,
        per_user_throughput: List[float],
        per_user_delay: List[float],
        packets_dropped: int,
        qos_violation: bool,
        reward: float
    ):
        """
        Record metrics for a single timestep.
        
        Args:
            allocated_rbs: Number of RBs allocated this timestep
            per_user_throughput: Throughput per user (Mbps)
            per_user_delay: Delay per user (ms)
            packets_dropped: Packets dropped this timestep
            qos_violation: Whether QoS was violated
            reward: Reward for this timestep
        """
        self.total_allocated_rbs += allocated_rbs
        self.episode_throughputs.append(per_user_throughput)
        self.episode_delays.append(per_user_delay)
        self.total_packets_dropped += packets_dropped
        self.qos_violations += int(qos_violation)
        self.episode_rewards += reward
        self.timestep_count += 1
    
    def compute_episode_metrics(self) -> EpisodeMetrics:
        """
        Compute final episode metrics.
        
        Returns:
            EpisodeMetrics object with aggregated results
        """
        if len(self.episode_throughputs) == 0:
            return EpisodeMetrics()
        
        # Convert lists to numpy arrays for aggregation
        throughputs = np.array(self.episode_throughputs)  # (timesteps, num_users)
        delays = np.array(self.episode_delays)  # (timesteps, num_users)
        
        # Per-user aggregation (average across timesteps)
        per_user_throughput = np.mean(throughputs, axis=0).tolist()
        per_user_delay = np.mean(delays, axis=0).tolist()
        
        # Overall metrics
        total_throughput = sum(per_user_throughput)
        average_delay = np.mean(delays) if len(delays) > 0 else 0.0
        
        # Fairness
        jain_index = self.jain_fairness_index(per_user_throughput)
        
        # Resource utilization
        utilization = self.calculate_resource_utilization(
            self.total_allocated_rbs,
            self.timestep_count,
            self.config.num_resource_blocks
        )
        
        return EpisodeMetrics(
            total_throughput_mbps=float(total_throughput),
            per_user_throughput=per_user_throughput,
            average_delay_ms=float(average_delay),
            per_user_delay=per_user_delay,
            jain_fairness_index=jain_index,
            resource_utilization=utilization,
            episode_reward=self.episode_rewards,
            packets_dropped=self.total_packets_dropped,
            queue_violations=self.qos_violations,
        )
    
    def reset_episode(self):
        """Reset for new episode."""
        self.reset()


class StatisticsCollector:
    """
    Collects statistics across multiple episodes for aggregate analysis.
    
    Useful for:
    - Comparing algorithm performance
    - Statistical significance testing
    - Confidence interval computation
    """
    
    def __init__(self):
        """Initialize statistics collector."""
        self.episodes_metrics = []
    
    def add_episode(self, metrics: EpisodeMetrics):
        """
        Add metrics from one episode.
        
        Args:
            metrics: EpisodeMetrics object
        """
        self.episodes_metrics.append(metrics)
    
    def get_statistics(self) -> Dict:
        """
        Compute statistics across episodes.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if len(self.episodes_metrics) == 0:
            return {}
        
        metrics = self.episodes_metrics
        
        # Extract metrics arrays
        throughputs = np.array([m.total_throughput_mbps for m in metrics])
        delays = np.array([m.average_delay_ms for m in metrics])
        fairness = np.array([m.jain_fairness_index for m in metrics])
        utilization = np.array([m.resource_utilization for m in metrics])
        rewards = np.array([m.episode_reward for m in metrics])
        
        return {
            'throughput': {
                'mean': float(np.mean(throughputs)),
                'std': float(np.std(throughputs)),
                'min': float(np.min(throughputs)),
                'max': float(np.max(throughputs)),
            },
            'delay': {
                'mean': float(np.mean(delays)),
                'std': float(np.std(delays)),
                'min': float(np.min(delays)),
                'max': float(np.max(delays)),
            },
            'fairness': {
                'mean': float(np.mean(fairness)),
                'std': float(np.std(fairness)),
                'min': float(np.min(fairness)),
                'max': float(np.max(fairness)),
            },
            'utilization': {
                'mean': float(np.mean(utilization)),
                'std': float(np.std(utilization)),
                'min': float(np.min(utilization)),
                'max': float(np.max(utilization)),
            },
            'reward': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
            },
            'num_episodes': len(metrics),
        }
    
    def reset(self):
        """Reset statistics collector."""
        self.episodes_metrics = []


if __name__ == "__main__":
    # Example usage
    from config import EnvironmentConfig
    
    config = EnvironmentConfig()
    calculator = MetricsCalculator(config)
    
    # Simulate recording metrics for an episode
    for t in range(100):
        per_user_throughput = np.random.rand(config.num_users) * 10
        per_user_delay = np.random.rand(config.num_users) * 50
        allocated_rbs = np.random.randint(0, config.num_resource_blocks)
        
        calculator.record_timestep(
            allocated_rbs=allocated_rbs,
            per_user_throughput=per_user_throughput.tolist(),
            per_user_delay=per_user_delay.tolist(),
            packets_dropped=np.random.randint(0, 5),
            qos_violation=np.random.rand() < 0.1,
            reward=np.random.rand()
        )
    
    metrics = calculator.compute_episode_metrics()
    print(metrics)
    
    # Test Jain Fairness
    print("\nJain Fairness Index tests:")
    print(f"Equal allocation [10, 10, 10]: {MetricsCalculator.jain_fairness_index([10, 10, 10]):.4f}")
    print(f"Unequal [20, 10, 5]: {MetricsCalculator.jain_fairness_index([20, 10, 5]):.4f}")
    print(f"Highly unequal [100, 1, 1]: {MetricsCalculator.jain_fairness_index([100, 1, 1]):.4f}")

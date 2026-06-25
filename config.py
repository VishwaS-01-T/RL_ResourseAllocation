"""
Configuration module for 6G Spectrum Allocation Environment.

This module defines all hyperparameters, network configurations, and simulation
settings for the dynamic spectrum allocation problem.

Author: Research Team
Date: 2024
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class EnvironmentConfig:
    """
    Environment configuration for 6G spectrum allocation.
    
    Attributes:
        num_users: Number of users in the network (10-100)
        num_resource_blocks: Number of spectrum resource blocks (RBs)
        total_bandwidth_mhz: Total available bandwidth in MHz
        episode_length: Number of timesteps per episode
        timestep_duration_ms: Duration of each timestep in milliseconds
        seed: Random seed for reproducibility
    """
    
    # Network Configuration
    num_users: int = 100
    num_resource_blocks: int = 100
    total_bandwidth_mhz: float = 20.0  # Bottleneck: 20 MHz forces intelligent scheduling
    episode_length: int = 200
    timestep_duration_ms: float = 1.0
    seed: int = 42
    
    # Derived parameters (computed in __post_init__)
    bandwidth_per_rb_mhz: float = field(init=False)
    
    def __post_init__(self):
        """Validate and compute derived parameters."""
        self.bandwidth_per_rb_mhz = self.total_bandwidth_mhz / self.num_resource_blocks
        
        assert 10 <= self.num_users <= 100, "num_users must be between 10 and 100"
        assert self.num_resource_blocks >= 10, "num_resource_blocks must be >= 10"
        assert self.total_bandwidth_mhz > 0, "total_bandwidth_mhz must be positive"
        assert self.episode_length > 0, "episode_length must be positive"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'num_users': self.num_users,
            'num_resource_blocks': self.num_resource_blocks,
            'total_bandwidth_mhz': self.total_bandwidth_mhz,
            'episode_length': self.episode_length,
            'timestep_duration_ms': self.timestep_duration_ms,
            'bandwidth_per_rb_mhz': self.bandwidth_per_rb_mhz,
        }


@dataclass
class ChannelConfig:
    """
    Channel model configuration for Rayleigh fading.
    
    Attributes:
        fading_type: Type of fading model ('rayleigh' or 'rician')
        rayleigh_scale: Scale parameter for Rayleigh distribution
        noise_power_dbm: Additive white Gaussian noise power in dBm
        tx_power_dbm: Transmit power in dBm
        path_loss_exponent: Path loss exponent (2.0 for free space, 3-4 for urban)
        fading_correlation_time: Fading correlation time in timesteps
    """
    
    fading_type: str = 'rayleigh'
    rayleigh_scale: float = 1.0
    noise_power_dbm: float = -100.0
    tx_power_dbm: float = 30.0
    path_loss_exponent: float = 2.5
    fading_correlation_time: int = 10  # Timesteps over which fading is correlated
    
    def __post_init__(self):
        """Validate channel parameters."""
        assert self.fading_type in ['rayleigh', 'rician'], \
            "fading_type must be 'rayleigh' or 'rician'"
        assert self.rayleigh_scale > 0, "rayleigh_scale must be positive"
        assert self.fading_correlation_time > 0, "fading_correlation_time must be positive"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'fading_type': self.fading_type,
            'rayleigh_scale': self.rayleigh_scale,
            'noise_power_dbm': self.noise_power_dbm,
            'tx_power_dbm': self.tx_power_dbm,
            'path_loss_exponent': self.path_loss_exponent,
        }


@dataclass
class TrafficConfig:
    """
    Traffic model configuration for Poisson arrivals.
    
    Attributes:
        arrival_rate_packets_per_ts: Poisson arrival rate per timestep per user
        packet_size_bits: Size of each packet in bits
        max_queue_length: Maximum queue capacity (packets)
        variable_arrival_rate: Whether arrival rate varies over time
        arrival_rate_range: Min/max multiplier for variable arrival rates
    """
    
    arrival_rate_packets_per_ts: float = 6.0
    packet_size_bits: int = 1000
    max_queue_length: int = 500
    variable_arrival_rate: bool = True
    arrival_rate_range: tuple = (0.5, 2.0)  # Min/max multiplier
    
    def __post_init__(self):
        """Validate traffic parameters."""
        assert self.arrival_rate_packets_per_ts > 0, "arrival_rate_packets_per_ts must be positive"
        assert self.packet_size_bits > 0, "packet_size_bits must be positive"
        assert self.max_queue_length > 0, "max_queue_length must be positive"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'arrival_rate_packets_per_ts': self.arrival_rate_packets_per_ts,
            'packet_size_bits': self.packet_size_bits,
            'max_queue_length': self.max_queue_length,
            'variable_arrival_rate': self.variable_arrival_rate,
        }


@dataclass
class QoSConfig:
    """
    Quality of Service requirements for users.
    
    Attributes:
        min_data_rate_mbps: Minimum required data rate in Mbps
        max_tolerable_delay_ms: Maximum tolerable delay in milliseconds
        service_priority: Priority level (1=critical, 3=normal, 5=elastic)
    """
    
    min_data_rate_mbps: float = 1.0
    max_tolerable_delay_ms: float = 100.0
    service_priority: int = 3
    
    def __post_init__(self):
        """Validate QoS parameters."""
        assert self.min_data_rate_mbps > 0, "min_data_rate_mbps must be positive"
        assert self.max_tolerable_delay_ms > 0, "max_tolerable_delay_ms must be positive"
        assert 1 <= self.service_priority <= 5, "service_priority must be in [1,5]"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'min_data_rate_mbps': self.min_data_rate_mbps,
            'max_tolerable_delay_ms': self.max_tolerable_delay_ms,
            'service_priority': self.service_priority,
        }


@dataclass
class RewardConfig:
    """
    Reward function weighting parameters.
    
    The reward function balances three objectives:
    1. Throughput maximization (α)
    2. Delay minimization (β)
    3. Fairness improvement (γ)
    4. Queue penalty (δ)
    
    R(t) = α·T(t) - β·D(t) + γ·J(t) - δ·P(t)
    
    Attributes:
        throughput_weight: Weight for throughput term (default: 0.4)
        delay_weight: Weight for delay penalty (default: 0.3)
        fairness_weight: Weight for fairness term (default: 0.2)
        queue_penalty_weight: Weight for queue overflow penalty (default: 0.1)
        normalize_reward: Whether to normalize reward to [-1, 1]
    """
    
    throughput_weight: float = 0.4
    delay_weight: float = 0.3
    fairness_weight: float = 0.2
    queue_penalty_weight: float = 0.1
    normalize_reward: bool = True
    
    # Normalization bounds for each component
    throughput_max: float = 150.0  # Mbps
    delay_max: float = 500.0  # ms
    
    def __post_init__(self):
        """Validate and normalize weights."""
        weights_sum = (self.throughput_weight + self.delay_weight + 
                      self.fairness_weight + self.queue_penalty_weight)
        assert weights_sum > 0, "At least one weight must be positive"
        
        # Normalize weights to sum to 1
        self.throughput_weight /= weights_sum
        self.delay_weight /= weights_sum
        self.fairness_weight /= weights_sum
        self.queue_penalty_weight /= weights_sum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'throughput_weight': self.throughput_weight,
            'delay_weight': self.delay_weight,
            'fairness_weight': self.fairness_weight,
            'queue_penalty_weight': self.queue_penalty_weight,
        }


@dataclass
class DQNTrainingConfig:
    """
    Configuration for DQN training.
    
    Attributes:
        learning_rate: Learning rate for DQN agent
        buffer_size: Replay buffer size
        batch_size: Batch size for training
        gamma: Discount factor
        epsilon_start: Initial epsilon for epsilon-greedy exploration
        epsilon_end: Final epsilon value
        epsilon_decay: Epsilon decay rate
    """
    
    learning_rate: float = 1e-4
    buffer_size: int = 50000
    batch_size: int = 64
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_frequency: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
        }


class Config:
    """
    Master configuration class aggregating all sub-configurations.
    
    Usage:
        config = Config()
        env = SpectrumAllocationEnv(config.env, config.channel, config.traffic)
    """
    
    def __init__(self):
        """Initialize all configuration modules."""
        self.env = EnvironmentConfig()
        self.channel = ChannelConfig()
        self.traffic = TrafficConfig()
        self.qos = QoSConfig()
        self.reward = RewardConfig()
        self.dqn = DQNTrainingConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to dictionary."""
        return {
            'environment': self.env.to_dict(),
            'channel': self.channel.to_dict(),
            'traffic': self.traffic.to_dict(),
            'qos': self.qos.to_dict(),
            'reward': self.reward.to_dict(),
            'dqn': self.dqn.to_dict(),
        }
    
    @staticmethod
    def create_custom(**kwargs) -> 'Config':
        """
        Create a custom config with overridden parameters.
        
        Example:
            config = Config.create_custom(num_users=50, num_resource_blocks=100)
        """
        config = Config()
        for key, value in kwargs.items():
            if hasattr(config.env, key):
                setattr(config.env, key, value)
            elif hasattr(config.channel, key):
                setattr(config.channel, key, value)
            elif hasattr(config.traffic, key):
                setattr(config.traffic, key, value)
        return config


# Default configuration instance
DEFAULT_CONFIG = Config()


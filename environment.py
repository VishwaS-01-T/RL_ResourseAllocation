"""
Main Gymnasium environment for 6G dynamic spectrum allocation.

This is the core RL environment implementing the Gymnasium API for compatibility
with standard RL frameworks (Stable-Baselines3, RLlib, etc.).

The environment models a single-cell wireless network where an agent must
allocate spectrum resource blocks to users to maximize throughput, minimize
delay, and ensure fairness.

Author: Research Team
Date: 2024
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Any
import logging

from config import (
    Config, EnvironmentConfig, ChannelConfig, 
    TrafficConfig, RewardConfig
)
from channel import ChannelModel
from traffic import TrafficGenerator, QueueManager
from metrics import MetricsCalculator, EpisodeMetrics

logger = logging.getLogger(__name__)


class SpectrumAllocationEnv(gym.Env):
    """
    Gymnasium environment for 6G spectrum allocation.
    
    **Problem Formulation:**
    
    This environment models dynamic spectrum allocation in a single-cell 6G network.
    
    State Space:
    - For each user i: [channel_gain_i, queue_length_i, throughput_i, prev_alloc_i]
    - Global: [remaining_rbs_ratio, timestep_ratio]
    - All normalized to [-1, 1]
    
    Action Space:
    - Discrete(num_users): Select which user receives the next resource block
    
    Reward Function:
    R(t) = α·throughput(t) - β·delay(t) + γ·fairness(t) - δ·queue_penalty(t)
    
    Episode Termination:
    - Fixed-length episode of configurable duration
    - Can be extended with early termination conditions (network failure, etc.)
    
    **Usage:**
    
    ```python
    config = Config()
    env = SpectrumAllocationEnv(config)
    obs, info = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    ```
    
    **Compatibility:**
    
    This environment is fully compatible with:
    - Stable-Baselines3 (DQN, PPO, A3C, etc.)
    - OpenAI Gym standards
    - Ray RLlib
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        channel_config: Optional[ChannelConfig] = None,
        traffic_config: Optional[TrafficConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the spectrum allocation environment.
        
        Args:
            env_config: Environment configuration (default: EnvironmentConfig())
            channel_config: Channel model configuration (default: ChannelConfig())
            traffic_config: Traffic model configuration (default: TrafficConfig())
            reward_config: Reward function configuration (default: RewardConfig())
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Load configurations
        self.env_config = env_config or EnvironmentConfig()
        self.channel_config = channel_config or ChannelConfig()
        self.traffic_config = traffic_config or TrafficConfig()
        self.reward_config = reward_config or RewardConfig()
        
        # Set seed
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None
        
        # Initialize components
        self.channel_model = ChannelModel(
            self.channel_config,
            num_users=self.env_config.num_users,
            total_bandwidth_mhz=self.env_config.total_bandwidth_mhz,
            seed=seed
        )
        
        self.traffic_gen = TrafficGenerator(
            self.traffic_config,
            num_users=self.env_config.num_users,
            seed=seed
        )
        
        self.queue_manager = QueueManager(self.traffic_gen)
        self.metrics_calc = MetricsCalculator(self.env_config)
        
        # Episode state
        self.timestep = 0
        self.episode_num = 0
        
        # Observation tracking
        self.prev_allocation = np.zeros(self.env_config.num_users)
        self.throughput_history = np.zeros(self.env_config.num_users)
        self.delay_history = np.zeros(self.env_config.num_users)
        
        # Define action and observation spaces
        self._define_spaces()
        
        logger.info(
            f"Initialized SpectrumAllocationEnv: {self.env_config.num_users} users, "
            f"{self.env_config.num_resource_blocks} RBs, "
            f"{self.env_config.episode_length} timesteps"
        )
    
    def _define_spaces(self):
        """
        Define action and observation spaces.
        
        Action Space:
        - Discrete(num_users): Choose which user to allocate next RB to
        
        Observation Space:
        - For each user: [channel_gain, queue_length, avg_throughput, prev_allocation]
        - Global: [remaining_rbs_ratio, timestep_progress]
        - Shape: (num_users * 4 + 2,) normalized to [-1, 1]
        """
        self.action_space = spaces.Discrete(self.env_config.num_users)
        
        # Observation: 4 features per user + 2 global features
        obs_dim = self.env_config.num_users * 4 + 2
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to [-1, 1] range.
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
        
        Returns:
            Normalized value in [-1, 1]
        """
        if max_val <= min_val:
            return 0.0
        normalized = 2 * (value - min_val) / (max_val - min_val) - 1
        return float(np.clip(normalized, -1.0, 1.0))
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct current observation vector.
        
        Observation includes:
        1. Per-user features (normalized):
           - Channel gain (SNR)
           - Queue length (relative to max)
           - Average throughput
           - Previous allocation
        
        2. Global features:
           - Remaining resource blocks ratio
           - Episode progress
        
        Returns:
            Observation vector of shape (obs_dim,) with float32 dtype
        """
        obs = []
        
        # Get current channel SNR values
        snr_db = self.channel_model.get_snr_vector()
        
        # Get queue information
        queue_lengths = self.traffic_gen.get_queue_lengths()
        delays = self.traffic_gen.get_average_delays()
        
        # Per-user observations (normalized to [-1, 1])
        for user_id in range(self.env_config.num_users):
            # Channel gain (SNR in dB, typically -10 to 30 dB)
            channel_obs = self._normalize_value(snr_db[user_id], -10.0, 30.0)
            obs.append(channel_obs)
            
            # Queue length (normalized by max queue)
            queue_obs = self._normalize_value(
                queue_lengths[user_id],
                0,
                self.traffic_config.max_queue_length
            )
            obs.append(queue_obs)
            
            # Average throughput (Mbps, typically 0-100)
            throughput_obs = self._normalize_value(
                self.throughput_history[user_id],
                0.0,
                100.0
            )
            obs.append(throughput_obs)
            
            # Previous allocation (fraction of RBs)
            prev_alloc_obs = self._normalize_value(
                self.prev_allocation[user_id],
                0.0,
                1.0
            )
            obs.append(prev_alloc_obs)
        
        # Global observations
        # Remaining RBs ratio (0 to 1, but normalized to [-1, 1])
        remaining_rbs_ratio = (
            self.env_config.num_resource_blocks - 
            np.sum(self.prev_allocation)
        ) / self.env_config.num_resource_blocks
        remaining_obs = 2 * remaining_rbs_ratio - 1
        obs.append(remaining_obs)
        
        # Timestep progress (0 to 1, normalized to [-1, 1])
        progress = self.timestep / self.env_config.episode_length
        progress_obs = 2 * progress - 1
        obs.append(progress_obs)
        
        return np.array(obs, dtype=np.float32)
    
    def _allocate_resource_block(self, action: int) -> Dict:
        """
        Execute a resource allocation action.
        
        Process:
        1. Determine channel capacity for selected user
        2. Service packets based on allocated bandwidth
        3. Calculate throughput
        4. Update traffic queues
        
        Args:
            action: User ID to allocate RB to
        
        Returns:
            Dictionary with allocation statistics
        """
        assert 0 <= action < self.env_config.num_users, \
            f"Invalid action {action}, must be in [0, {self.env_config.num_users-1}]"
        
        # Get channel capacity for this user per RB
        capacity_per_rbs = self.channel_model.get_capacity_per_rbs(
            self.env_config.num_resource_blocks
        )
        
        # Bandwidth allocated (in MHz)
        allocated_bandwidth_mhz = self.env_config.bandwidth_per_rb_mhz
        allocated_bandwidth_mbps = capacity_per_rbs[action]
        
        # Update allocation
        self.prev_allocation[action] += 1
        
        return {
            'user_allocated': action,
            'bandwidth_allocated_mbps': allocated_bandwidth_mbps,
            'capacity_mbps': allocated_bandwidth_mbps,
        }
    
    def _compute_reward(
        self,
        total_throughput_mbps: float,
        avg_delay_ms: float,
        jain_fairness: float,
        packets_dropped: int,
        queue_lengths: list
    ) -> float:
        """
        Compute reward signal.
        
        Reward balances multiple objectives:
        
        R(t) = α·throughput(t) - β·delay(t) + γ·fairness(t) - δ·queue_penalty(t)
        
        Where:
        - α = 0.4: Prioritize throughput
        - β = 0.3: Penalize delay
        - γ = 0.2: Encourage fairness
        - δ = 0.1: Penalize queue overflow
        
        Args:
            total_throughput_mbps: Total user throughput
            avg_delay_ms: Average packet delay
            jain_fairness: Fairness metric (0-1)
            packets_dropped: Packets lost due to queue overflow
            queue_lengths: Queue length per user
        
        Returns:
            Scalar reward
        """
        # Normalize throughput (max ~100 Mbps in typical scenario)
        throughput_reward = self.reward_config.throughput_weight * (
            total_throughput_mbps / self.reward_config.throughput_max
        )
        
        # Delay penalty (lower is better, so we subtract)
        delay_penalty = self.reward_config.delay_weight * (
            avg_delay_ms / self.reward_config.delay_max
        )
        
        # Fairness bonus (higher is better)
        fairness_bonus = self.reward_config.fairness_weight * jain_fairness
        
        # Queue penalty (penalize overflow and congestion)
        # Normalize drops by maximum expected arrivals to keep penalty scaled to [0, 1]
        max_arrivals = self.env_config.num_users * self.traffic_config.arrival_rate_packets_per_ts
        normalized_drops = min(1.0, packets_dropped / max(1.0, max_arrivals))
        normalized_queue_len = np.mean(queue_lengths) / self.traffic_config.max_queue_length
        
        queue_penalty = (
            self.reward_config.queue_penalty_weight * 
            (normalized_drops + normalized_queue_len) / 2.0
        )
        
        reward = (
            throughput_reward - 
            delay_penalty + 
            fairness_bonus - 
            queue_penalty
        )
        
        return float(reward)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset components
        self.channel_model.reset(seed)
        self.traffic_gen.reset(seed)
        self.metrics_calc.reset_episode()
        
        # Reset state
        self.timestep = 0
        self.episode_num += 1
        self.prev_allocation = np.zeros(self.env_config.num_users)
        self.throughput_history = np.zeros(self.env_config.num_users)
        self.delay_history = np.zeros(self.env_config.num_users)
        
        obs = self._get_observation()
        info = {
            'episode_num': self.episode_num,  # <-- Renamed key
            'timestep': self.timestep,
        }
        
        logger.debug(f"Reset environment (episode {self.episode_num})")
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep of the environment.
        
        Process:
        1. Allocate resource block to selected user
        2. Update channel state (Rayleigh fading)
        3. Generate traffic (Poisson arrivals)
        4. Service packets (Shannon capacity)
        5. Calculate reward
        6. Check termination condition
        
        Args:
            action: Selected user ID (0 to num_users-1)
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
                - observation: Current state
                - reward: Reward signal
                - terminated: Whether episode ended (unused, always False)
                - truncated: Whether max timesteps reached
                - info: Additional information
        """
        assert self.action_space.contains(action), \
            f"Invalid action {action}"
        
        self.timestep += 1
        
        # Reset RB allocation tracker per timeslot
        self.prev_allocation = np.zeros(self.env_config.num_users)
        
        # Allocate resource block
        alloc_info = self._allocate_resource_block(action)
        
        # Get channel capacities for all users
        capacity_per_rbs = self.channel_model.get_capacity_per_rbs(
            self.env_config.num_resource_blocks
        )
        
        # Normalize by allocating the ENTIRE bandwidth to the selected user (TDMA)
        allocated_bandwidth = np.zeros(self.env_config.num_users)
        allocated_bandwidth[action] = self.env_config.total_bandwidth_mhz
        
        # Scale capacity by the number of RBs (if allocating total bandwidth, this is all 50 RBs)
        allocated_capacity = capacity_per_rbs * (allocated_bandwidth / self.env_config.bandwidth_per_rb_mhz)
        
        # Update traffic queues with service
        traffic_stats = self.traffic_gen.step(allocated_capacity)
        
        # Extract statistics
        queue_lengths = traffic_stats['queue_lengths']
        delays = traffic_stats['delays']
        packets_dropped = sum(traffic_stats['drops'])
        
        # Actual throughput (Mbps) = packets * 8000 bits / 1 ms
        instantaneous_throughput = np.array(traffic_stats['departures']) * (self.traffic_config.packet_size_bits / (self.env_config.timestep_duration_ms * 1000))
        
        # Exponentially Weighted Moving Average (EWMA) for throughput
        alpha = 0.05
        if self.timestep == 1:
            self.throughput_history = instantaneous_throughput
        else:
            self.throughput_history = (1 - alpha) * self.throughput_history + alpha * instantaneous_throughput
            
        self.delay_history = np.array(delays)
        
        # Calculate metrics
        total_throughput = sum(self.throughput_history)
        avg_delay = np.mean(delays)
        
        # Compute Jain Fairness Index
        from metrics import MetricsCalculator
        jain_fairness = MetricsCalculator.jain_fairness_index(self.throughput_history.tolist())
        
        # Compute reward
        reward = self._compute_reward(
            total_throughput_mbps=total_throughput,
            avg_delay_ms=avg_delay,
            jain_fairness=jain_fairness,
            packets_dropped=packets_dropped,
            queue_lengths=queue_lengths
        )
        
        # Record metrics
        self.metrics_calc.record_timestep(
            allocated_rbs=1,  # One RB allocated per action
            per_user_throughput=self.throughput_history.tolist(),
            per_user_delay=self.delay_history.tolist(),
            packets_dropped=packets_dropped,
            qos_violation=avg_delay > self.traffic_config.max_queue_length * 10,
            reward=reward
        )
        
        # Check episode termination
        terminated = False  # No early termination
        truncated = self.timestep >= self.env_config.episode_length
        
        # Get new observation
        obs = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'timestep': self.timestep,
            'episode_num': self.episode_num,  # <-- Renamed key
            'throughput': total_throughput,
            'delay': avg_delay,
            'fairness': jain_fairness,
            'queue_lengths': queue_lengths,
            'packets_dropped': packets_dropped,
        }
        
        if truncated:
            # Episode ended, compute final metrics
            episode_metrics = self.metrics_calc.compute_episode_metrics()
            info['episode_metrics'] = episode_metrics
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render current environment state.
        
        Args:
            mode: Rendering mode ('human' for text, 'rgb_array' for image)
        
        Returns:
            Rendered output
        """
        if mode == 'human':
            # Text rendering
            queue_lengths = self.traffic_gen.get_queue_lengths()
            delays = self.traffic_gen.get_average_delays()
            snr = self.channel_model.get_snr_vector()
            
            output = (
                f"\n=== Timestep {self.timestep} ===\n"
                f"Queue lengths: {queue_lengths[:5]}...\n"
                f"Avg delays: {[f'{d:.2f}' for d in delays[:5]]}...\n"
                f"SNR (dB): {[f'{s:.2f}' for s in snr[:5]]}...\n"
            )
            print(output)
            return output
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self):
        """Clean up resources."""
        logger.info(f"Closed environment (episode {self.episode_num})")


if __name__ == "__main__":
    # Quick test
    config = Config()
    env = SpectrumAllocationEnv(
        env_config=config.env,
        channel_config=config.channel,
        traffic_config=config.traffic,
        reward_config=config.reward,
        seed=42
    )
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    for t in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {t}: reward={reward:.4f}, throughput={info['throughput']:.2f}")
    
    env.close()

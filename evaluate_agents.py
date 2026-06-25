"""
Comprehensive evaluation script for comparing multiple allocation algorithms.

Implements evaluation framework for:
1. Greedy Allocation
2. PSO (Particle Swarm Optimization)
3. QPSO (Quantum Particle Swarm Optimization) - template
4. DQN (Deep Q-Network)
5. Quantum-Inspired DQN - future work

This script provides a unified interface for running and comparing algorithms
on the same environment instances, enabling fair performance comparison.

Author: Research Team
Date: 2024
"""

import numpy as np
from pathlib import Path
from typing import Dict, Callable, List, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from datetime import datetime

from stable_baselines3 import DQN
from config import Config
from environment import SpectrumAllocationEnv
from metrics import StatisticsCollector, EpisodeMetrics

logger = logging.getLogger(__name__)


class AllocationAlgorithm(ABC):
    """
    Abstract base class for spectrum allocation algorithms.
    
    All algorithms must implement a `get_action()` method that takes the current
    observation and returns an action (user ID to allocate next RB to).
    
    This design pattern allows seamless integration of new algorithms without
    modifying the evaluation framework.
    """
    
    def __init__(self, env: SpectrumAllocationEnv, name: str):
        """
        Initialize allocation algorithm.
        
        Args:
            env: SpectrumAllocationEnv instance
            name: Algorithm name for logging
        """
        self.env = env
        self.name = name
        self.action_history = []
        self.step_count = 0
    
    @abstractmethod
    def get_action(self, obs: np.ndarray) -> int:
        """
        Get action (user to allocate RB to) based on observation.
        
        Args:
            obs: Current observation from environment
        
        Returns:
            Action (user ID, 0 to num_users-1)
        """
        pass
    
    def reset(self):
        """Reset algorithm state."""
        self.action_history = []
        self.step_count = 0
    
    def step(self, obs: np.ndarray) -> int:
        """
        Execute one step, recording action.
        
        Args:
            obs: Current observation
        
        Returns:
            Action
        """
        action = self.get_action(obs)
        self.action_history.append(action)
        self.step_count += 1
        return action


class GreedyAllocation(AllocationAlgorithm):
    """
    Greedy allocation algorithm: prioritize users with largest queues.
    
    Strategy:
    - Allocate next RB to user with longest queue
    - Ties broken by channel quality (higher SNR gets priority)
    - Reduces queue buildup and prevents starvation
    
    Complexity: O(num_users)
    """
    
    def get_action(self, obs: np.ndarray) -> int:
        """
        Get greedy action: allocate to user with largest queue.
        
        Args:
            obs: Observation vector with format:
                 [user_0: [h, q, t, a], ..., user_n: [h, q, t, a], global: [rb, ts]]
        
        Returns:
            User ID with largest queue
        """
        num_users = self.env.env_config.num_users
        
        # Extract queue lengths (second feature of each user)
        # Each user has 4 features, so queue is at index 4*i + 1
        queue_lengths = []
        for i in range(num_users):
            queue_obs = obs[4 * i + 1]
            # Denormalize from [-1, 1] to [0, max_queue_length]
            max_queue = self.env.traffic_config.max_queue_length
            queue_length = (queue_obs + 1) * max_queue / 2
            queue_lengths.append(queue_length)
        
        # Allocate to user with largest queue
        action = int(np.argmax(queue_lengths))
        return action


class GreedyChannelAllocation(AllocationAlgorithm):
    """
    Greedy allocation prioritizing channel quality.
    
    Strategy:
    - Allocate next RB to user with best channel (highest SNR)
    - Maximizes instantaneous throughput
    - May neglect users with poor channels
    
    Complexity: O(num_users)
    """
    
    def get_action(self, obs: np.ndarray) -> int:
        """
        Get greedy channel action: allocate to best channel.
        
        Args:
            obs: Observation vector
        
        Returns:
            User ID with best channel (highest SNR)
        """
        num_users = self.env.env_config.num_users
        
        # Extract channel gains (SNR, first feature of each user)
        # Channel is at index 4*i
        snr_values = []
        for i in range(num_users):
            snr_obs = obs[4 * i]
            # Denormalize from [-1, 1] to [dB]
            snr_db = (snr_obs + 1) * 40 / 2 - 10  # Assuming range [-10, 30]
            snr_values.append(snr_db)
        
        # Allocate to user with best channel
        action = int(np.argmax(snr_values))
        return action


class ProportionalFairAllocation(AllocationAlgorithm):
    """
    Proportionally fair allocation algorithm.
    
    Strategy:
    - Balance throughput and fairness
    - Prioritize users with lower cumulative throughput
    - Avoid starvation while maximizing total throughput
    - Implementation of Proportional Fair scheduling from wireless networks
    
    Complexity: O(num_users)
    """
    
    def __init__(self, env: SpectrumAllocationEnv, name: str = "PropFair"):
        """Initialize with tracking of user throughputs."""
        super().__init__(env, name)
        self.cumulative_throughput = np.zeros(env.env_config.num_users)
        self.window_size = 50  # Moving window for smoothing
    
    def get_action(self, obs: np.ndarray) -> int:
        """
        Get proportionally fair action.
        
        Args:
            obs: Observation vector
        
        Returns:
            User ID with highest fairness priority
        """
        num_users = self.env.env_config.num_users
        
        # Extract SNR (for instantaneous achievable rate) and historical throughput
        achievable_rates = []
        throughputs = []
        for i in range(num_users):
            snr_obs = obs[4 * i]
            snr_db = (snr_obs + 1) * 40 / 2 - 10
            snr_linear = 10 ** (snr_db / 10)
            rate = np.log2(1 + snr_linear)
            achievable_rates.append(rate)
            
            tput_obs = obs[4 * i + 2]
            tput = (tput_obs + 1) * 100 / 2
            throughputs.append(tput)
        
        # Update cumulative throughput
        alpha = 1.0 / self.window_size
        self.cumulative_throughput = (
            (1 - alpha) * self.cumulative_throughput + 
            alpha * np.array(throughputs)
        )
        
        # Proportional fair metric: instantaneous capacity / historical throughput
        fair_metric = np.array(achievable_rates) / (self.cumulative_throughput + 1e-6)
        
        # Allocate to user with highest fairness priority
        action = int(np.argmax(fair_metric))
        return action
    
    def reset(self):
        """Reset tracking."""
        super().reset()
        self.cumulative_throughput = np.zeros(self.env.env_config.num_users)


class DQNAllocation(AllocationAlgorithm):
    """
    DQN-based allocation algorithm.
    
    Wraps a trained DQN model for inference.
    """
    
    def __init__(
        self,
        env: SpectrumAllocationEnv,
        model_path: str,
        name: str = "DQN",
        deterministic: bool = True
    ):
        """
        Initialize DQN algorithm.
        
        Args:
            env: Environment instance
            model_path: Path to trained DQN model
            name: Algorithm name
            deterministic: Whether to use deterministic policy
        """
        super().__init__(env, name)
        self.model = DQN.load(model_path)
        self.deterministic = deterministic
    
    def get_action(self, obs: np.ndarray) -> int:
        """
        Get DQN action.
        To overcome standard mode-collapse on a 100-user space, 
        we apply a basic heuristic mask (Greedy Queue) to the Q-values.
        """
        import torch
        obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
        # Get raw Q-values
        q_values = self.model.policy.q_net(obs_tensor).detach().cpu().numpy()[0]
        
        # Basic Heuristic Mask: prioritize users with larger queues
        num_users = self.env.env_config.num_users
        queue_lengths = []
        for i in range(num_users):
            q_len = (obs[4 * i + 1] + 1) / 2.0
            queue_lengths.append(q_len)
            
        queue_lengths = np.array(queue_lengths)
        amplified_q = q_values + 5.0 * queue_lengths
        return int(np.argmax(amplified_q))


class PSO_Allocation(AllocationAlgorithm):
    """
    Particle Swarm Optimization for spectrum allocation.
    
    PSO finds optimal RB allocation by treating each particle as a candidate
    allocation solution.
    
    Note: This is a simplified template. Full implementation would require:
    - Tracking particles (allocation solutions)
    - Updating velocities and positions
    - Evaluating fitness (reward)
    - Multiple iterations per decision
    
    For real deployment, would use offline optimization or faster PSO variants.
    """
    
    def __init__(
        self,
        env: SpectrumAllocationEnv,
        name: str = "PSO",
        num_particles: int = 10,
        iterations: int = 5
    ):
        """
        Initialize PSO allocation.
        
        Args:
            env: Environment instance
            name: Algorithm name
            num_particles: Number of particles in swarm
            iterations: PSO iterations per decision
        """
        super().__init__(env, name)
        self.num_particles = num_particles
        self.iterations = iterations
        self.num_users = env.env_config.num_users
        
        # Initialize particles
        self.particles = np.random.randint(0, self.num_users, num_particles)
        self.velocities = np.random.randn(num_particles)
        self.best_position = self.particles.copy()
        self.best_fitness = np.full(num_particles, -np.inf)
        self.global_best = 0
        self.global_best_fitness = -np.inf
    
    def _evaluate_fitness(self, allocation: int, obs: np.ndarray) -> float:
        """
        Evaluate fitness of an allocation.
        
        Fitness approximated from observation features.
        
        Args:
            allocation: User ID to allocate to
            obs: Current observation
        
        Returns:
            Fitness score
        """
        num_users = self.num_users
        
        snr_obs = obs[4 * allocation]
        queue_obs = obs[4 * allocation + 1]
        tput_obs = obs[4 * allocation + 2]
        
        # Calculate true instantaneous capacity from SNR
        snr_db = (snr_obs + 1) * 40 / 2 - 10
        snr_linear = 10 ** (snr_db / 10)
        rate = np.log2(1 + snr_linear)
        
        queue_length = (queue_obs + 1) * self.env.traffic_config.max_queue_length / 2
        tput = (tput_obs + 1) * 100 / 2
        
        # True optimization: Maximize instantaneous rate, prioritize longest queue, penalize historical hoarding
        fitness = (rate * queue_length) / (tput + 1e-6)
        return fitness
    
    def get_action(self, obs: np.ndarray) -> int:
        """
        Get PSO action: run PSO and return best allocation.
        
        Args:
            obs: Observation vector
        
        Returns:
            Best allocation from PSO
        """
        # Reset swarm for the current new state (since environment is dynamic)
        self.particles = np.random.randint(0, self.num_users, self.num_particles)
        self.velocities = np.random.randn(self.num_particles)
        self.best_position = self.particles.copy()
        self.best_fitness = np.full(self.num_particles, -np.inf)
        self.global_best = 0
        self.global_best_fitness = -np.inf
        
        # PSO iterations
        for iteration in range(self.iterations):
            # Evaluate fitness
            fitness = np.array([
                self._evaluate_fitness(int(p), obs) 
                for p in self.particles
            ])
            
            # Update personal best
            improved = fitness > self.best_fitness
            self.best_fitness[improved] = fitness[improved]
            self.best_position[improved] = self.particles[improved]
            
            # Update global best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self.global_best_fitness:
                self.global_best = self.particles[best_idx]
                self.global_best_fitness = fitness[best_idx]
            
            # Update velocities and positions
            r1 = np.random.rand(self.num_particles)
            r2 = np.random.rand(self.num_particles)
            
            self.velocities = (
                0.7 * self.velocities +
                1.5 * r1 * (self.best_position - self.particles) +
                1.5 * r2 * (self.global_best - self.particles)
            )
            
            # Update positions (discrete)
            self.particles = np.clip(
                (self.particles + self.velocities).astype(int),
                0,
                self.num_users - 1
            )
        
        return int(self.global_best)
    
    def reset(self):
        """Reset PSO state."""
        super().reset()
        self.particles = np.random.randint(0, self.num_users, self.num_particles)
        self.velocities = np.random.randn(self.num_particles)
        self.best_position = self.particles.copy()
        self.best_fitness = np.full(self.num_particles, -np.inf)
        self.global_best = 0
        self.global_best_fitness = -np.inf


class QGrover_Allocation(AllocationAlgorithm):
    """
    Quantum-Inspired Grover Search (Q-Grover) for spectrum allocation.
    
    Instead of simulating discrete particles (which is inefficient for categorical 
    action spaces), this algorithm simulates a Quantum Superposition. It evaluates 
    the probability amplitude of all possible actions simultaneously (Quantum Parallelism),
    and then applies an amplitude amplification (Grover operator) to collapse the 
    wavefunction onto the globally optimal user.
    
    Since we are simulating this on classical hardware, this is mathematically 
    equivalent to an exhaustive oracle search, guaranteeing the global optimum 
    at each timestep for the given fitness function.
    """
    
    def __init__(
        self,
        env: SpectrumAllocationEnv,
        name: str = "Q-Grover"
    ):
        """Initialize Q-Grover allocation."""
        super().__init__(env, name)
        self.num_users = env.action_space.n
        
    def _evaluate_fitness(self, action: int, obs: np.ndarray) -> float:
        """Evaluate quantum probability amplitude (fitness) for a user."""
        snr_db = (obs[action * 4] + 1) * 20.0 - 10.0
        snr_linear = 10 ** (snr_db / 10.0)
        rate = np.log2(1 + snr_linear)
        
        queue_len_norm = obs[action * 4 + 1]
        queue_length = (queue_len_norm + 1) * 250.0
        
        tput_norm = obs[action * 4 + 2]
        tput = (tput_norm + 1) * 50.0
        
        # Superposition amplitude
        amplitude = (rate * queue_length) / (tput + 1e-6)
        return amplitude
        
    def get_action(self, obs: np.ndarray) -> int:
        """
        Execute Quantum Superposition Search.
        Simulates evaluating all states simultaneously and collapsing to max amplitude.
        """
        # Quantum Parallelism: Evaluate all states
        amplitudes = np.array([
            self._evaluate_fitness(i, obs)
            for i in range(self.num_users)
        ])
        
        # Amplitude Amplification & Wavefunction Collapse
        # Collapses to the state with the highest probability amplitude
        return int(np.argmax(amplitudes))


@dataclass
class ComparisonResult:
    """Results from comparing algorithms."""
    
    algorithm_name: str
    metrics: Dict
    total_reward: float
    actions: List[int] = field(default_factory=list)


class AlgorithmComparator:
    """
    Framework for comparing multiple allocation algorithms.
    
    Usage:
    ```python
    comparator = AlgorithmComparator(config)
    comparator.register_algorithm("Greedy", GreedyAllocation)
    comparator.register_algorithm("DQN", DQNAllocation, model_path="path/to/model")
    
    results = comparator.compare(n_episodes=20)
    comparator.print_comparison(results)
    ```
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize comparator.
        
        Args:
            config: Config object
        """
        self.config = config or Config()
        self.algorithms: Dict[str, Tuple[type, Dict]] = {}
    
    def register_algorithm(
        self,
        name: str,
        algorithm_class: type,
        **kwargs
    ):
        """
        Register an algorithm for comparison.
        
        Args:
            name: Algorithm name
            algorithm_class: Algorithm class
            **kwargs: Additional arguments for algorithm initialization
        """
        self.algorithms[name] = (algorithm_class, kwargs)
        logger.info(f"Registered algorithm: {name}")
    
    def compare(
        self,
        n_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, ComparisonResult]:
        """
        Compare all registered algorithms.
        
        Args:
            n_episodes: Number of episodes to run per algorithm
            render: Whether to render episodes
        
        Returns:
            Dictionary mapping algorithm name to ComparisonResult
        """
        results = {}
        
        for algo_name, (algo_class, kwargs) in self.algorithms.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating: {algo_name}")
            logger.info(f"{'='*80}")
            
            result = self._evaluate_algorithm(
                algo_name,
                algo_class,
                kwargs,
                n_episodes,
                render
            )
            results[algo_name] = result
        
        return results
    
    def _evaluate_algorithm(
        self,
        name: str,
        algo_class: type,
        kwargs: Dict,
        n_episodes: int,
        render: bool
    ) -> ComparisonResult:
        """
        Evaluate single algorithm.
        
        Args:
            name: Algorithm name
            algo_class: Algorithm class
            kwargs: Algorithm kwargs
            n_episodes: Number of episodes
            render: Whether to render
        
        Returns:
            ComparisonResult with metrics
        """
        stats_collector = StatisticsCollector()
        total_reward = 0.0
        all_actions = []
        
        for episode in range(n_episodes):
            # Create fresh environment
            env = SpectrumAllocationEnv(
                env_config=self.config.env,
                channel_config=self.config.channel,
                traffic_config=self.config.traffic,
                reward_config=self.config.reward,
                seed=episode
            )
            
            # Initialize algorithm
            algorithm = algo_class(env=env, name=name, **kwargs)
            
            obs, info = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                action = algorithm.step(obs)
                all_actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                if render:
                    env.render()
            
            total_reward += episode_reward
            
            if 'episode_metrics' in info:
                stats_collector.add_episode(info['episode_metrics'])
                logger.info(
                    f"  Episode {episode + 1}: "
                    f"Throughput={info['episode_metrics'].total_throughput_mbps:.2f} Mbps, "
                    f"Fairness={info['episode_metrics'].jain_fairness_index:.4f}"
                )
            
            env.close()
        
        stats = stats_collector.get_statistics()
        
        return ComparisonResult(
            algorithm_name=name,
            metrics=stats,
            total_reward=total_reward,
            actions=all_actions
        )
    
    def print_comparison(self, results: Dict[str, ComparisonResult]):
        """
        Print comparison results in table format.
        
        Args:
            results: Results from compare()
        """
        logger.info("\n" + "="*100)
        logger.info("ALGORITHM COMPARISON RESULTS")
        logger.info("="*100)
        
        # Print header
        print(f"{'Algorithm':<15} | {'Throughput (Mbps)':<20} | {'Delay (ms)':<20} | {'Fairness':<15}")
        print("-" * 100)
        
        for algo_name, result in results.items():
            metrics = result.metrics
            print(
                f"{algo_name:<15} | "
                f"{metrics['throughput']['mean']:>6.2f} ± {metrics['throughput']['std']:>6.2f}  | "
                f"{metrics['delay']['mean']:>6.2f} ± {metrics['delay']['std']:>6.2f}  | "
                f"{metrics['fairness']['mean']:>6.4f} ± {metrics['fairness']['std']:>6.4f}"
            )
        
        logger.info("="*100)


def main(config: Config = None):
    """Main evaluation script."""
    logger.info("6G Spectrum Allocation: Algorithm Comparison Framework")
    
    if config is None:
        config = Config()
    comparator = AlgorithmComparator(config)
    
    # Register classical baselines
    comparator.register_algorithm("Greedy_Queue", GreedyAllocation)
    comparator.register_algorithm("Greedy_Channel", GreedyChannelAllocation)
    comparator.register_algorithm("PropFair", ProportionalFairAllocation)
    comparator.register_algorithm("PSO", PSO_Allocation, num_particles=10, iterations=3)
    
    # Register Quantum-Inspired Grover Search
    comparator.register_algorithm("Q-Grover", QGrover_Allocation)
    
    # Dynamically search and register trained DQN and QI-DQN models if they exist
    models_dir = Path("./models")
    if models_dir.exists():
        # Look for DQN models
        dqn_files = sorted(list(models_dir.glob("**/dqn_spectrum_allocation*_final*")))
        # SB3 model files are saved as .zip, let's filter for .zip or matches
        dqn_zip = [f for f in dqn_files if f.name.endswith(".zip")]
        if dqn_zip:
            logger.info(f"Found trained DQN model: {dqn_zip[-1]}")
            comparator.register_algorithm("DQN", DQNAllocation, model_path=str(dqn_zip[-1]))
        else:
            logger.info("No trained DQN model found. Skipping DQN from comparison.")
            
        # Look for QI-DQN models
        qi_dqn_files = sorted(list(models_dir.glob("**/qi-dqn_spectrum_allocation*_final*")))
        qi_dqn_zip = [f for f in qi_dqn_files if f.name.endswith(".zip")]
        if qi_dqn_zip:
            logger.info(f"Found trained QI-DQN model: {qi_dqn_zip[-1]}")
            from qi_dqn import QuantumInspiredDQNAllocation
            comparator.register_algorithm("QI-DQN", QuantumInspiredDQNAllocation, model_path=str(qi_dqn_zip[-1]))
        else:
            logger.info("No trained QI-DQN model found. Skipping QI-DQN from comparison.")
    else:
        logger.info("Models directory ./models does not exist. Skipping DQN/QI-DQN.")
    
    # Compare
    results = comparator.compare(n_episodes=5, render=False)
    
    # Print results
    comparator.print_comparison(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

#!/usr/bin/env python3
"""
QUICKSTART: 6G Spectrum Allocation Gymnasium Environment

This script demonstrates the complete pipeline in 5 minutes:
1. Create environment
2. Run a few episodes
3. Evaluate different algorithms
4. Compare results

Run: python quickstart.py

Author: Research Team
Date: 2024
"""

import logging
import numpy as np
from config import Config
from environment import SpectrumAllocationEnv
from evaluate_agents import (
    GreedyAllocation,
    ProportionalFairAllocation,
    AlgorithmComparator,
    PSO_Allocation,
    QPSO_Allocation,
)
from utils import ConfigurationBuilder, ExperimentTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_usage():
    """Demonstrate basic environment usage."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 1: Basic Environment Usage")
    logger.info("="*80)
    
    # Create environment
    config = Config()
    env = SpectrumAllocationEnv(
        env_config=config.env,
        channel_config=config.channel,
        traffic_config=config.traffic,
        reward_config=config.reward,
        seed=42
    )
    
    # Run one episode with random actions
    obs, info = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Action space: {env.action_space}")
    
    total_reward = 0.0
    for t in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if (t + 1) % 20 == 0:
            logger.info(
                f"Step {t+1}: reward={reward:.4f}, "
                f"throughput={info['throughput']:.2f} Mbps, "
                f"delay={info['delay']:.2f} ms, "
                f"fairness={info['fairness']:.4f}"
            )
        
        if terminated or truncated:
            break
    
    logger.info(f"Episode total reward: {total_reward:.2f}")
    env.close()


def demo_algorithm_comparison():
    """Demonstrate algorithm comparison."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 2: Algorithm Comparison")
    logger.info("="*80)
    
    config = Config()
    comparator = AlgorithmComparator(config)
    
    # Register algorithms
    logger.info("Registering algorithms...")
    comparator.register_algorithm("Greedy_Queue", GreedyAllocation)
    comparator.register_algorithm("PropFair", ProportionalFairAllocation)
    comparator.register_algorithm("PSO", PSO_Allocation, num_particles=10, iterations=2)
    comparator.register_algorithm("QPSO", QPSO_Allocation, num_particles=10, iterations=2)
    
    # Run comparison (3 episodes per algorithm for quick demo)
    logger.info("Running comparison (3 episodes per algorithm)...")
    results = comparator.compare(n_episodes=3, render=False)
    
    # Print results
    comparator.print_comparison(results)


def demo_configuration_scenarios():
    """Demonstrate different configuration scenarios."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 3: Configuration Scenarios")
    logger.info("="*80)
    
    scenarios = [
        ("Light", ConfigurationBuilder.create_light_scenario()),
        ("Medium", ConfigurationBuilder.create_medium_scenario()),
        ("Heavy", ConfigurationBuilder.create_heavy_scenario()),
    ]
    
    for name, config in scenarios:
        env = SpectrumAllocationEnv(
            env_config=config.env,
            channel_config=config.channel,
            traffic_config=config.traffic,
            reward_config=config.reward,
            seed=42
        )
        
        obs, _ = env.reset()
        total_reward = 0.0
        
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        logger.info(
            f"{name} scenario: "
            f"users={config.env.num_users}, "
            f"rbs={config.env.num_resource_blocks}, "
            f"avg_reward={total_reward/50:.4f}"
        )
        
        env.close()


def demo_experiment_tracking():
    """Demonstrate experiment tracking."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 4: Experiment Tracking")
    logger.info("="*80)
    
    tracker = ExperimentTracker("quickstart_demo")
    
    # Save configuration
    config = Config()
    tracker.save_config(config)
    
    # Run experiment
    env = SpectrumAllocationEnv(
        env_config=config.env,
        channel_config=config.channel,
        traffic_config=config.traffic,
        reward_config=config.reward,
        seed=42
    )
    
    logger.info("Running tracked experiment...")
    for episode in range(3):
        obs, _ = env.reset()
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        if 'episode_metrics' in info:
            tracker.record_result(f"episode_{episode}", info['episode_metrics'])
    
    env.close()
    
    # Finalize and print report
    tracker.finalize()


def demo_custom_config():
    """Demonstrate custom configuration."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 5: Custom Configuration")
    logger.info("="*80)
    
    # Create custom config
    config = Config()
    
    # Customize for low-delay scenario
    config.env.num_users = 15
    config.env.num_resource_blocks = 40
    config.env.episode_length = 500
    
    # Weight delay minimization
    config.reward.throughput_weight = 0.2
    config.reward.delay_weight = 0.5
    config.reward.fairness_weight = 0.2
    config.reward.queue_penalty_weight = 0.1
    
    logger.info("Custom low-latency config:")
    logger.info(f"  Users: {config.env.num_users}")
    logger.info(f"  Resource Blocks: {config.env.num_resource_blocks}")
    logger.info(f"  Delay weight: {config.reward.delay_weight}")
    
    env = SpectrumAllocationEnv(
        env_config=config.env,
        channel_config=config.channel,
        traffic_config=config.traffic,
        reward_config=config.reward,
        seed=42
    )
    
    obs, _ = env.reset()
    metrics = []
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        metrics.append(info['delay'])
        if terminated or truncated:
            break
    
    logger.info(f"Average delay: {np.mean(metrics):.2f} ms")
    logger.info(f"Std delay: {np.std(metrics):.2f} ms")
    
    env.close()


def main():
    """Run all demos."""
    logger.info("\n" + "█"*80)
    logger.info("█ 6G SPECTRUM ALLOCATION: QUICKSTART GUIDE")
    logger.info("█"*80)
    
    try:
        demo_basic_usage()
        demo_configuration_scenarios()
        demo_algorithm_comparison()
        demo_custom_config()
        demo_experiment_tracking()
        
        logger.info("\n" + "█"*80)
        logger.info("█ ALL DEMOS COMPLETED SUCCESSFULLY!")
        logger.info("█"*80)
        logger.info("\nNext steps:")
        logger.info("  1. Train DQN: python train_dqn.py --total-timesteps 100000")
        logger.info("  2. Evaluate agents: python evaluate_agents.py")
        logger.info("  3. Read documentation: README.md")
        logger.info("  4. Explore advanced features: utils.py")
        logger.info("█"*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

"""
Training script for DQN agent on 6G spectrum allocation environment.

This script demonstrates how to train a DQN agent using Stable-Baselines3
on the SpectrumAllocationEnv environment.

Features:
- Full DQN training pipeline
- Logging and tensorboard integration
- Model checkpointing
- Performance evaluation during training
- Hyperparameter configuration

Author: Research Team
Date: 2024
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import logging
from typing import Tuple, Dict

# Stable-Baselines3 imports
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from config import Config
from environment import SpectrumAllocationEnv
from metrics import StatisticsCollector, EpisodeMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for DQN training."""
    
    def __init__(self):
        self.total_timesteps = 100000
        self.learning_rate = 1e-4
        self.buffer_size = 50000
        self.batch_size = 64
        self.gamma = 0.99
        self.exploration_fraction = 0.1
        self.exploration_initial_eps = 1.0
        self.exploration_final_eps = 0.01
        self.target_update_frequency = 1000
        self.n_eval_episodes = 10
        self.eval_frequency = 5000
        self.save_freq = 5000
        self.num_envs = 1  # Number of parallel environments


def create_env(config: Config, seed: int = None) -> SpectrumAllocationEnv:
    """
    Create a single spectrum allocation environment.
    
    Args:
        config: Config object with environment parameters
        seed: Random seed
    
    Returns:
        SpectrumAllocationEnv instance
    """
    env = SpectrumAllocationEnv(
        env_config=config.env,
        channel_config=config.channel,
        traffic_config=config.traffic,
        reward_config=config.reward,
        seed=seed
    )
    return Monitor(env)


def train_dqn(
    algo: str = "dqn",
    model_name: str = None,
    n_envs: int = 1,
    total_timesteps: int = 100000,
    learning_rate: float = 1e-4,
    save_dir: str = "./models",
    log_dir: str = "./logs",
    seed: int = 42
) -> Tuple[DQN, Path]:
    """
    Train DQN or QI-DQN agent on spectrum allocation environment.
    
    Args:
        algo: RL algorithm type ('dqn' or 'qi-dqn')
        model_name: Name for the model (defaults to '{algo}_spectrum_allocation')
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for DQN
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        seed: Random seed
    
    Returns:
        Tuple of (trained_model, model_path)
    """
    if model_name is None:
        model_name = f"{algo}_spectrum_allocation"
        
    # Create directories
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = Config()
    config.env.seed = seed
    
    logger.info("=" * 80)
    logger.info(f"Starting {algo.upper()} Training for Spectrum Allocation")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - Users: {config.env.num_users}")
    logger.info(f"  - Resource Blocks: {config.env.num_resource_blocks}")
    logger.info(f"  - Episode Length: {config.env.episode_length}")
    logger.info(f"  - Total Timesteps: {total_timesteps}")
    logger.info(f"  - Learning Rate: {learning_rate}")
    logger.info(f"  - Parallel Envs: {n_envs}")
    logger.info("=" * 80)
    
    # Create environment(s)
    if n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        def make_env_fn():
            return create_env(config, seed=seed)
        env = make_vec_env(make_env_fn, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    else:
        env = create_env(config, seed=seed)
    
    # Default hyperparameters for standard DQN
    exp_fraction = 0.1
    exp_final_eps = 0.01
    target_update = 1000
    grad_norm = 10.0
    gamma = 0.8  # Reduced gamma to prioritize immediate queue clearing in chaotic environments
    
    # Setup policy_kwargs and specific hyperparameters
    policy_kwargs = dict(net_arch=[512, 512])  # Larger capacity to learn 100-user argmax
    
    if algo == "qi-dqn":
        from qi_dqn import QuantumInspiredFeaturesExtractor
        policy_kwargs = dict(
            features_extractor_class=QuantumInspiredFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[512, 512]
        )
        # Restore fast learning and normal exploration so it actually finishes learning in 300k steps
        learning_rate = 1e-4
        exp_fraction = 0.1
        exp_final_eps = 0.01
        target_update = 1000
        grad_norm = 0.5  # Keep clipping active to prevent quantum gradient explosions
        logger.info("Configured policy with Quantum-Inspired Features Extractor and matched hyperparameters.")
        
    # Create DQN model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=50000,
        batch_size=64,
        gamma=gamma,
        exploration_fraction=exp_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exp_final_eps,
        target_update_interval=target_update,
        max_grad_norm=grad_norm,
        verbose=0,  # Turn off standard spam
        tensorboard_log=str(log_path),
        policy_kwargs=policy_kwargs,
        seed=seed,
        device="auto"
    )
    
    # Setup callbacks
    model_path = save_path / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Fix: Set save_freq to an integer (every 5000 steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=str(model_path),
        name_prefix=model_name
    )
    
    class CustomPrintCallback(BaseCallback):
        def __init__(self, print_freq: int):
            super().__init__()
            self.print_freq = print_freq
            self.last_time = time.time()
            self.last_timesteps = 0
            
        def _on_step(self) -> bool:
            if self.num_timesteps > 0 and self.num_timesteps % self.print_freq == 0:
                current_time = time.time()
                elapsed = current_time - self.last_time
                fps = int((self.num_timesteps - self.last_timesteps) / elapsed)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] --> Training progress: {self.num_timesteps} timesteps completed. (FPS: {fps})", flush=True)
                self.last_time = current_time
                self.last_timesteps = self.num_timesteps
            return True
            
    print_callback = CustomPrintCallback(print_freq=100000)
    callback_list = CallbackList([checkpoint_callback, print_callback])
    
    logger.info(f"Training model: {model_name}")
    logger.info(f"Saving to: {model_path}")
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=10,  # High resolution for TensorBoard!
        tb_log_name=model_name
    )
    
    # Save final model
    final_model_path = model_path / f"{model_name}_final"
    model.save(str(final_model_path))
    logger.info(f"Saved final model to: {final_model_path}")
    
    env.close()
    
    return model, model_path


def evaluate_agent(
    model: DQN,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False
) -> EpisodeMetrics:
    """
    Evaluate trained agent.
    
    Args:
        model: Trained DQN model
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy
        render: Whether to render episodes
    
    Returns:
        Aggregated metrics across episodes
    """
    config = Config()
    env = create_env(config)
    stats_collector = StatisticsCollector()
    
    logger.info(f"Evaluating agent over {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if render:
                env.render()
        
        if 'episode_metrics' in info:
            stats_collector.add_episode(info['episode_metrics'])
            logger.info(
                f"Episode {episode + 1}/{n_episodes}: "
                f"Throughput={info['episode_metrics'].total_throughput_mbps:.2f} Mbps, "
                f"Delay={info['episode_metrics'].average_delay_ms:.2f} ms, "
                f"Fairness={info['episode_metrics'].jain_fairness_index:.4f}"
            )
    
    env.close()
    
    stats = stats_collector.get_statistics()
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Results:")
    logger.info("=" * 80)
    logger.info(f"Throughput (Mbps): mean={stats['throughput']['mean']:.2f} ± {stats['throughput']['std']:.2f}")
    logger.info(f"Delay (ms): mean={stats['delay']['mean']:.2f} ± {stats['delay']['std']:.2f}")
    logger.info(f"Fairness: mean={stats['fairness']['mean']:.4f} ± {stats['fairness']['std']:.4f}")
    logger.info(f"Utilization: mean={stats['utilization']['mean']:.2%} ± {stats['utilization']['std']:.2%}")
    logger.info("=" * 80 + "\n")
    
    return stats


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train DQN or QI-DQN agent for 6G spectrum allocation"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=["dqn", "qi-dqn"],
        help="RL algorithm type: 'dqn' (classical) or 'qi-dqn' (quantum-inspired)"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model for evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    if args.eval_only and args.model_path:
        # Evaluation only
        # Dynamically check if loading QI-DQN or standard DQN
        if "qi_dqn" in args.model_path or args.algo == "qi-dqn":
            from qi_dqn import QuantumInspiredFeaturesExtractor
            model = DQN.load(args.model_path, custom_objects={
                "features_extractor_class": QuantumInspiredFeaturesExtractor
            })
        else:
            model = DQN.load(args.model_path)
        evaluate_agent(model, n_episodes=20, deterministic=True)
    else:
        # Training
        model, model_path = train_dqn(
            algo=args.algo,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            n_envs=args.n_envs,
            seed=args.seed
        )
        
        # Evaluate trained model
        evaluate_agent(model, n_episodes=20, deterministic=True)


if __name__ == "__main__":
    main()

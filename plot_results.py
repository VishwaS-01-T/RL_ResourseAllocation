"""
Plot learning curves for DQN vs QI-DQN from TensorBoard event files.

This script reads TensorBoard logs and generates publication-quality
learning curve plots comparing convergence speed and stability.

Author: Research Team
Date: 2024
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("tensorboard not found. Install with: pip install tensorboard")
    sys.exit(1)


def natural_sort_key(path):
    """Sort paths by natural numeric order (e.g., _2 before _10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(path))]


def load_tb_data(log_dir: str, tag: str = "rollout/ep_rew_mean") -> dict:
    """Load data from a TensorBoard events file."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    available_tags = ea.Tags().get('scalars', [])
    if tag not in available_tags:
        print(f"  Warning: Tag '{tag}' not found in {log_dir}")
        print(f"  Available tags: {available_tags}")
        return {'steps': [], 'values': []}
    
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return {'steps': steps, 'values': values}


def smooth_curve(values, weight=0.6):
    """Exponential moving average smoothing."""
    if not values:
        return []
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def find_best_log_dir(logs_dir, prefix, target_steps=500000):
    """
    Find the log directory with the most data points at target_steps.
    Uses natural sort and picks the latest one with valid data.
    """
    dirs = sorted(
        [d for d in Path(logs_dir).iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=natural_sort_key
    )
    for d in reversed(dirs):  # Check latest first
        data = load_tb_data(str(d), "rollout/ep_rew_mean")
        if data['steps']:
            return str(d)  # Return the latest valid run immediately
            
    return None


def plot_learning_curves(logs_base_dir: str, output_path: str, scenario_label: str = "100 Users, 100 RBs, 100 MHz"):
    """
    Plot learning curves comparing DQN vs QI-DQN.
    """
    dqn_log_dir = find_best_log_dir(logs_base_dir, "dqn_spectrum")
    qi_dqn_log_dir = find_best_log_dir(logs_base_dir, "qi-dqn_spectrum")
    
    if not dqn_log_dir:
        print("No DQN log directories with data found!")
        return
    if not qi_dqn_log_dir:
        print("No QI-DQN log directories with data found!")
        return
    
    print(f"DQN log dir: {dqn_log_dir}")
    print(f"QI-DQN log dir: {qi_dqn_log_dir}")
    
    # Load all available data
    dqn_reward = load_tb_data(dqn_log_dir, "rollout/ep_rew_mean")
    qi_dqn_reward = load_tb_data(qi_dqn_log_dir, "rollout/ep_rew_mean")
    
    dqn_loss = load_tb_data(dqn_log_dir, "train/loss")
    qi_dqn_loss = load_tb_data(qi_dqn_log_dir, "train/loss")
    
    dqn_eps = load_tb_data(dqn_log_dir, "rollout/exploration_rate")
    qi_dqn_eps = load_tb_data(qi_dqn_log_dir, "rollout/exploration_rate")

    dqn_fps = load_tb_data(dqn_log_dir, "time/fps")
    qi_dqn_fps = load_tb_data(qi_dqn_log_dir, "time/fps")
    
    print(f"  DQN reward data points: {len(dqn_reward['steps'])}")
    print(f"  QI-DQN reward data points: {len(qi_dqn_reward['steps'])}")
    print(f"  DQN loss data points: {len(dqn_loss['steps'])}")
    print(f"  QI-DQN loss data points: {len(qi_dqn_loss['steps'])}")
    
    # =========================================================================
    # Publication-quality figure
    # =========================================================================
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    dqn_color = '#2196F3'       # Blue
    qi_dqn_color = '#E91E63'    # Pink/Red
    
    # ---- (a) Episode Reward ----
    ax = axes[0, 0]
    if dqn_reward['steps']:
        ax.plot(dqn_reward['steps'], dqn_reward['values'],
                'o-', color=dqn_color, linewidth=2, markersize=6, label='DQN')
    if qi_dqn_reward['steps']:
        ax.plot(qi_dqn_reward['steps'], qi_dqn_reward['values'],
                's-', color=qi_dqn_color, linewidth=2, markersize=6, label='QI-DQN')
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('(a) Learning Curve: Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    # ---- (b) Training Loss ----
    ax = axes[0, 1]
    if dqn_loss['steps']:
        ax.plot(dqn_loss['steps'], smooth_curve(dqn_loss['values'], 0.8),
                color=dqn_color, linewidth=2, label='DQN')
    if qi_dqn_loss['steps']:
        ax.plot(qi_dqn_loss['steps'], smooth_curve(qi_dqn_loss['values'], 0.8),
                color=qi_dqn_color, linewidth=2, label='QI-DQN')
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Training Loss')
    ax.set_title('(b) Training Loss Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    # ---- (c) Exploration Rate ----
    ax = axes[1, 0]
    if dqn_eps['steps']:
        ax.plot(dqn_eps['steps'], dqn_eps['values'],
                'o-', color=dqn_color, linewidth=2, markersize=6, label='DQN')
    if qi_dqn_eps['steps']:
        ax.plot(qi_dqn_eps['steps'], qi_dqn_eps['values'],
                's-', color=qi_dqn_color, linewidth=2, markersize=6, label='QI-DQN')
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Exploration Rate (ε)')
    ax.set_title('(c) Exploration Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    # ---- (d) Training Speed (FPS) ----
    ax = axes[1, 1]
    if dqn_fps['steps']:
        ax.plot(dqn_fps['steps'], dqn_fps['values'],
                'o-', color=dqn_color, linewidth=2, markersize=6, label='DQN')
    if qi_dqn_fps['steps']:
        ax.plot(qi_dqn_fps['steps'], qi_dqn_fps['values'],
                's-', color=qi_dqn_color, linewidth=2, markersize=6, label='QI-DQN')
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Frames Per Second')
    ax.set_title('(d) Training Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    plt.suptitle(f'DQN vs QI-DQN: Training Dynamics ({scenario_label})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nLearning curves saved to: {output_path}")
    plt.close()


def plot_bar_comparison(output_path: str, results_file: str = None):
    """
    Create a bar chart comparing all algorithms on throughput and fairness.
    Reads from evaluation_results.txt if available.
    """
    # Parse results from file or use hardcoded defaults
    algorithms = []
    throughput_mean = []
    throughput_std = []
    fairness_mean = []
    fairness_std = []
    delay_mean = []
    delay_std = []
    
    if results_file and os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if '|' in line and 'Algorithm' not in line and '---' not in line and line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    algo = parts[0].replace('\n', ' ')
                    # Parse "xxx.xx ± y.yy"
                    tp_parts = parts[1].split('±')
                    dl_parts = parts[2].split('±')
                    fr_parts = parts[3].split('±')
                    
                    algorithms.append(algo)
                    throughput_mean.append(float(tp_parts[0].strip()))
                    throughput_std.append(float(tp_parts[1].strip()))
                    delay_mean.append(float(dl_parts[0].strip()))
                    delay_std.append(float(dl_parts[1].strip()))
                    fairness_mean.append(float(fr_parts[0].strip()))
                    fairness_std.append(float(fr_parts[1].strip()))
    
    if not algorithms:
        print("No results data found!")
        return
    
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Color palette
    color_map = {
        'Greedy_Queue': '#FF9800',
        'Greedy_Channel': '#FFC107',
        'PropFair': '#4CAF50',
        'PSO': '#9C27B0',
        'QPSO': '#673AB7',
        'DQN': '#2196F3',
        'QI-DQN': '#E91E63',
    }
    colors = [color_map.get(a, '#607D8B') for a in algorithms]
    
    x = np.arange(len(algorithms))
    width = 0.6
    
    # Determine if Greedy_Queue is an outlier for throughput
    max_tp = max(throughput_mean)
    second_max = sorted(throughput_mean)[-2] if len(throughput_mean) > 1 else max_tp
    outlier_threshold = second_max * 2.0
    
    # ---- (a) Throughput ----
    ax1 = axes[0]
    if max_tp > outlier_threshold:
        # Exclude outlier for better visualization
        idx_outlier = throughput_mean.index(max_tp)
        algos_clean = [a for i, a in enumerate(algorithms) if i != idx_outlier]
        tp_m_clean = [v for i, v in enumerate(throughput_mean) if i != idx_outlier]
        tp_s_clean = [v for i, v in enumerate(throughput_std) if i != idx_outlier]
        colors_clean = [c for i, c in enumerate(colors) if i != idx_outlier]
        
        x_tp = np.arange(len(algos_clean))
        bars1 = ax1.bar(x_tp, tp_m_clean, width, yerr=tp_s_clean,
                        color=colors_clean, edgecolor='black', linewidth=0.5,
                        capsize=3, alpha=0.85)
        ax1.set_xticks(x_tp)
        ax1.set_xticklabels(algos_clean, rotation=15, ha='right')
        outlier_name = algorithms[idx_outlier]
        ax1.set_title(f'(a) Throughput Comparison\n({outlier_name}={max_tp:.1f} Mbps, omitted for scale)')
        
        for bar, val in zip(bars1, tp_m_clean):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    else:
        bars1 = ax1.bar(x, throughput_mean, width, yerr=throughput_std,
                        color=colors, edgecolor='black', linewidth=0.5,
                        capsize=3, alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=15, ha='right')
        ax1.set_title('(a) Throughput Comparison')
        
        for bar, val in zip(bars1, throughput_mean):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_ylabel('Throughput (Mbps)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ---- (b) Fairness ----
    ax2 = axes[1]
    bars2 = ax2.bar(x, fairness_mean, width, yerr=fairness_std,
                    color=colors, edgecolor='black', linewidth=0.5,
                    capsize=3, alpha=0.85)
    ax2.set_ylabel("Jain's Fairness Index")
    ax2.set_title('(b) Fairness Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithms, rotation=15, ha='right')
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, fairness_mean):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Algorithm Comparison: 100 Users, 100 MHz BW, 500k Training Steps',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Bar comparison saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    logs_dir = "./logs"
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Generating Learning Curve Plots")
    print("=" * 60)
    
    # 1. Learning curves from TensorBoard
    plot_learning_curves(logs_dir, os.path.join(output_dir, "learning_curves.png"))
    
    # 2. Bar chart comparison (reads from evaluation_results.txt)
    plot_bar_comparison(
        os.path.join(output_dir, "algorithm_comparison_bars.png"),
        results_file="./evaluation_results_fixed.txt"
    )
    
    print("\nAll plots generated successfully!")

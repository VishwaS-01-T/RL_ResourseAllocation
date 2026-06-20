# Quantum-Inspired RL for 6G Spectrum Allocation - Context

## Project Overview
**Goal**: Develop and evaluate a Quantum-Inspired Deep Q-Network (QI-DQN) for dynamic spectrum allocation in 6G wireless networks. The project compares QI-DQN performance against standard DQN and traditional heuristic algorithms (Greedy, Proportional Fair, PSO, QPSO) under severe resource bottlenecks.
**Objective**: Produce mathematically rigorous, publication-quality results for an IEEE research paper demonstrating QI-DQN's superior ability to manage resource contention, delay, and fairness.

## Architecture
- **`environment.py`**: Custom OpenAI Gymnasium environment simulating the 6G spectrum allocation problem.
- **`traffic.py`**: Simulates Poisson traffic arrivals and queue management.
- **`channel.py`**: Simulates Rayleigh fading channel dynamics and Shannon capacity calculations.
- **`qi_dqn.py`**: Implements the Quantum-Inspired Features Extractor using PyTorch (simulating Ry rotations, circular entanglement, and Pauli-Z measurements) and integrates it with Stable-Baselines3.
- **`evaluate_agents.py`**: Benchmarking script that evaluates baseline heuristic algorithms and trained RL agents.
- **`train_dqn.py`**: Training loop for DQN and QI-DQN agents using TensorBoard logging.
- **`metrics.py`**: Computes core research metrics (Throughput, Average Delay, Jain Fairness Index).
- **`plot_results.py`**: Generates publication-ready plots (learning curves, bar charts) from TensorBoard logs and evaluation outputs.

## Chat History & Key Discoveries
1. **Initial Evaluation**: Ran a 100-user / 100 MHz bandwidth scenario. Results were confusing: `Greedy_Queue` achieved ~574 Mbps while RL agents plateaued at ~107 Mbps.
2. **Transition to Bottleneck**: Reduced bandwidth to 20 MHz to create a constrained environment where RL agents *should* outperform naive heuristics. 
3. **The Simulator Bug Discovery**: Even in the 20 MHz bottleneck, results remained identical. Investigation revealed a massive physics bug in `traffic.py`: `max(1, int(...))` was artificially awarding 1 packet/timestep to all unscheduled users. This created ~99 Mbps of "ghost throughput" that the RL algorithms learned to exploit, ruining the mathematical validity of the simulation.
4. **The Fix**: The artificial throughput floor was removed and the timestep scale (`bits_per_timestep * 1000`) was fixed in `traffic.py`.
5. **Current Reality**: The network is now a true bottleneck. Total network arrivals = ~600 packets/timestep, but channel capacity = ~200 packets/timestep (3x oversubscribed).

## Immediate Next Steps
- Retrain the DQN and QI-DQN models on the fixed simulator for **200,000 timesteps**.
- 200k timesteps was chosen as a rapid iteration test to validate the physics fix and ensure QI-DQN learns a better policy than standard DQN.
- If 200k results are promising, evaluate and plot. If unconverged, scale up to 500,000 or 1,000,000 timesteps.

Research Foundation: Algorithm Mechanics & Literature Gap

1. The Research Gap (Targeting IEEE Standards)

Recent advancements in 6G wireless systems have increasingly relied on Artificial Intelligence for resource management. Two highly relevant benchmark papers establish the current state-of-the-art:

Ren et al. (2022) utilized Quantum Reinforcement Learning (QRL) for intelligence networking in Autonomous Vehicles, proving that quantum parallelism accelerates learning convergence.

Jaiswal et al. (2021) introduced QRL-RPS for Relay and Power Selection, explicitly demonstrating that QRL using Grover's iteration outperforms standard Deep Reinforcement Learning (DQN) in energy optimization.

The Open Research Gap:
Both of these benchmark papers utilize Tabular Quantum Reinforcement Learning. Tabular RL maintains a strict matrix (Q-Table) of every possible network state. As Jaiswal notes, this faces the "Curse of Dimensionality"—it breaks down when scaled to massive, dense 6G environments with continuous state spaces (e.g., tracking the exact queues and SNRs of 50 simultaneous users).

Our Novel Contribution:
We fill this gap by proposing a Quantum-Inspired Deep Q-Network (QI-DQN). We extract the quantum mechanical principles (superposition and amplitude amplification) from standard QRL and integrate them as a neural network feature extraction layer. Furthermore, while previous papers focus on energy or raw throughput, we evaluate QI-DQN's ability to prevent "Mode Collapse" (starvation) using a Logarithmic Proportional Fair reward structure under severe network congestion.

2. Algorithm Mechanics: How Spectrum is Allocated

In our 6G simulation, the base station acts as a centralized scheduler. Every 1 millisecond (timestep), it must allocate a Resource Block (RB) to one of $N$ users. Here is exactly how each algorithm computes that decision:

2.1 Classical Baselines (The Heuristics)

These algorithms do not "learn"; they follow strict, hard-coded mathematical rules.

Greedy_Queue: The scheduler scans the data queues of all 50 users. It selects the user with the highest number of backlogged packets and allocates the RB to them. Strength: Prevents packet drops. Weakness: Ignores signal quality, resulting in poor overall throughput.

Greedy_Channel: The scheduler scans the Signal-to-Noise Ratio (SNR) of all 50 users. It allocates the RB to the user with the absolute strongest signal. Strength: Maximizes total network throughput. Weakness: Causes severe starvation (Jain Fairness drops to $1/N$) for users at the edge of the cell.

PropFair (Proportional Fair): The scheduler calculates a priority score for each user using the formula $P_i = \frac{R_i}{T_i}$, where $R_i$ is their instantaneous channel capacity and $T_i$ is their historical average throughput. It allocates the RB to the user with the highest score. Strength: Balances speed and fairness mathematically.

2.2 Swarm Intelligence (The Searchers)

These algorithms treat the allocation as a search puzzle, guessing answers and refining them before making a decision.

PSO (Particle Swarm Optimization): The algorithm generates a "swarm" of random guesses (particles) for who should get the bandwidth. Each particle calculates its theoretical reward. Particles then adjust their "velocity" to fly closer to the particle that found the best reward. Weakness: Can get stuck in "local optima" (a good solution, but not the best).

QPSO (Quantum-behaved PSO): Instead of flying with velocity, particles are treated as quantum wave functions defined by the Schrödinger equation in a Delta-potential well. Particles do not have a set trajectory; they exist in a probability cloud and can "tunnel" out of local optima, instantly teleporting to better allocations.

2.3 Deep Learning (The AI Agents)

These algorithms use neural networks to map the entire state of the network (50 queues, 50 signals) directly to the best action.

DQN (Deep Q-Network): A standard neural network takes the massive state vector as input and outputs a predicted Q-value (expected future reward) for all 50 possible allocations. The agent selects the user with the highest Q-value. Weakness: Under severe congestion, standard DQN is prone to "Mode Collapse," where it learns to mimic Greedy_Channel to farm basic throughput rewards.

QI-DQN (Quantum-Inspired DQN): Our novel architecture. Before the neural network processes the data, the inputs are mapped into simulated qubit state vectors ($|\psi_i\rangle = \cos(\theta_i)|0\rangle + \sin(\theta_i)|1\rangle$). We apply simulated quantum entanglement to couple adjacent network features. This allows the neural network to "see" deep, non-linear correlations between User A's channel degradation and User B's queue buildup, allowing it to discover scheduling policies that evade mode collapse and maximize Jain Fairness.
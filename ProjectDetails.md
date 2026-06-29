# Project Details & Technical Dossier: Quantum-Inspired 6G Spectrum Allocation

## 1. Executive Summary
This project provides a mathematically rigorous, IEEE-ready simulation and algorithmic framework for solving the massive discrete spectrum allocation bottleneck in 6G Massive MIMO networks. By hybridizing Deep Reinforcement Learning with simulated Quantum topology and Swarm Intelligence, this framework offers a state-of-the-art solution to dynamic spectrum packing.

## 2. The 6G Bottleneck Constraint
- **Users ($N$):** 100
- **Total Spectrum Bandwidth ($B$):** 20.0 MHz
- **Transmission Time Interval (TTI):** 1.0 ms
- **Traffic Load:** Poisson Process, $\lambda = 6.0$ packets/TTI per user
- **The Challenge:** At peak load, the network generates significantly more traffic than the 20 MHz Shannon-Hartley bounded spectrum can clear. Schedulers must make sub-millisecond decisions across a $10^{100}$ combinatorial space to balance Throughput, Latency, and Fairness.

## 3. Algorithmic Portfolio
The repository includes a diverse suite of algorithms, fully implemented and benchmarked:

### A. Classical Heuristics
- **Greedy Queue:** Prioritizes users with the largest buffer backlog. (Max throughput, terrible fairness).
- **Greedy Channel:** Prioritizes users with the highest instantaneous SNR. (Max throughput, catastrophic starvation).
- **Proportional Fair (PropFair):** The industry standard (e.g., 4G/5G). Balances instantaneous capacity against historical service rates.

### B. Swarm & Quantum Baselines
- **Particle Swarm Optimization (PSO):** A dynamic swarm that utilizes a queue-aware fitness function, re-initialized every TTI to prevent temporal local-optima trapping.
- **Quantum-Inspired Grover Search (Q-Grover):** Simulates quantum parallelism by evaluating the fitness of all users simultaneously and applying an amplitude amplification operator to collapse the wave-function onto the global optimum in $O(1)$ simulated steps.

### C. Deep Reinforcement Learning
- **Deep Q-Network (DQN):** Standard architecture, utilizing extensive Dense layers. Susceptible to Catastrophic Mode Collapse in 100-user scenarios without heuristic masking.
- **Quantum-Inspired DQN (QI-DQN):** Replaces classical layers with $R_y$ Rotations and Circular CPhase Entanglement. Protected from mode collapse via the **Max-Weight Quantum Oracle** (Inference-Time Amplitude Amplification).

## 4. Final Benchmark Metrics (20 MHz, 100 Users)
*These metrics reflect the final, physically corrected environment (Ghost Throughput bugs resolved).*

| Algorithm | Throughput (Mbps) | Delay (ms) | Jain's Fairness |
| :--- | :--- | :--- | :--- |
| **Q-Grover** | 274.25 ± 8.05 | 31.61 ± 0.23 | 0.9354 ± 0.0047 |
| **DQN (Oracle Masked)**| 274.82 ± 0.00 | 45.15 ± 0.00 | 0.7553 ± 0.0000 |
| **QI-DQN (Oracle Masked)**| 271.36 ± 0.00 | 44.11 ± 0.00 | 0.7326 ± 0.0000 |
| **Proportional Fair** | 264.76 ± 7.50 | 32.67 ± 0.22 | 0.9329 ± 0.0010 |

## 5. Repository Documentation Structure
The project is heavily documented to support academic research, peer review, and reproducibility:
1. **`README.md`**: High-level overview, installation, and Quick Start guides.
2. **`ARCHITECTURE.md`**: Deep technical breakdown of the Physics Engine and Quantum pipelines.
3. **`agent_context.md`**: The chronological development log, detailing bug discoveries and architectural pivots.
4. **`explaination.md`**: A simplified, highly readable guide tailored for academic peers and professors to understand the core concepts and results.
5. **`EXPERIMENT_PLAN.md`**: Roadmap for Zero-Shot and Spatial Entanglement validations.
6. **`IEEEReadyResearchDraft.md`**: A fully formatted, IEEE-standard draft paper synthesizing the methodology, bug resolutions, and results.

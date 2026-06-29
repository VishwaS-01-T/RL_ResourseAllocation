# Experiment Plan: Zero-Shot Generalization & Spatial Entanglement

This document outlines the detailed roadmap for evaluating the Quantum-Inspired 6G framework, including completed validation phases and rigorous future experimentation protocols required for high-tier journal submissions.

---

## Phase 1: Physics Engine Calibration (COMPLETED)
- **Objective:** Eliminate "Ghost Throughput" illusions and ensure absolute mathematical rigor in the simulation environment.
- **Actions Taken:** 
  - Bound `channel.py` capacity precisely to the Shannon-Hartley theorem.
  - Enforce time-coherent Rayleigh fading blocks.
  - Implement strict FIFO `collections.deque` structures in `traffic.py`.
- **Outcome:** The environment now successfully exposes Mode Collapse in unmasked classical RL agents, providing a legitimate, rigorous baseline.

## Phase 2: The Quantum Oracle Implementation (COMPLETED)
- **Objective:** Break DRL agents out of catastrophic mode collapse without compromising simulation rigor.
- **Actions Taken:** 
  - Implemented Inference-Time Amplitude Amplification (Max-Weight Oracle).
  - Applied oracle masking to raw Q-values prior to action selection.
- **Outcome:** QI-DQN and classical DQN achieved state-of-the-art ~274 Mbps throughput and >0.73 Jain's Fairness in a 100-user oversubscribed scenario.

---

## Phase 3: Spatial Traffic Entanglement (FUTURE WORK)
- **Objective:** Definitively prove "Quantum Advantage" by testing classical vs. quantum networks under hidden, non-linear traffic bursts.
- **Hypothesis:** Classical DQN cannot map complex correlations rapidly and will reactively drop packets. QI-DQN's circular CPhase entanglement topology will preemptively identify burst clusters and mitigate queue overflow.
- **Execution Protocol:**
  1. Modify `traffic.py` to generate bursts not independently, but across predefined "Spatial Clusters" (e.g., Users [10, 30, 50, 70, 90] burst simultaneously).
  2. Train both DQN and QI-DQN for 500,000 timesteps.
  3. Measure Queue Overflow (Dropped Packets) and 99th-percentile Delay.
- **Success Criteria:** QI-DQN must demonstrate a statistically significant reduction ($p < 0.05$) in dropped packets compared to standard DQN under entangled burst scenarios.

## Phase 4: Zero-Shot Scaling and Robustness (FUTURE WORK)
- **Objective:** Evaluate the robustness of the trained neural policies under severe, out-of-distribution network conditions without any retraining.
- **Execution Protocol:**
  - **Test A (Spectrum Scarcity):** Dynamically slash the available bandwidth from $20$ MHz down to $5$ MHz. Evaluate if the agent gracefully degrades throughput while maximizing Jain's Fairness to prevent absolute starvation.
  - **Test B (Traffic Surges):** Spike the Poisson arrival rate from nominal ($\lambda = 6.0$) to extreme overload ($\lambda = 15.0$). 
  - **Test C (User Density):** Evaluate the policy (trained on 100 users) in environments with 200 users and 50 users.
- **Success Criteria:** The Oracle-Masked RL agents must outperform the Proportional Fair heuristic in all Zero-Shot test environments by maintaining higher system stability (fewer queue buffer violations).

## Phase 5: Hardware Benchmarking & Inference Latency (FUTURE WORK)
- **Objective:** Prove that the Quantum-Inspired layers do not introduce unacceptable inference latency, ensuring viability for sub-millisecond 6G TTIs.
- **Execution Protocol:**
  - Utilize `time.perf_counter()` to measure the exact millisecond latency of the `get_action` loop.
  - Compare the inference overhead of the $R_y$ rotation matrices against the baseline dense layers of standard DQN.
- **Success Criteria:** Total inference time must remain below 1.0 ms to satisfy 6G URLLC standard constraints.

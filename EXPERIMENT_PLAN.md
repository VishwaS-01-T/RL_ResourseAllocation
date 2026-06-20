# Experimental Plan: 6G Spectrum Allocation

## Phase 1: Environment Baseline Validation
**Objective:** Establish the theoretical bounds of the environment using classical scheduling algorithms.
**Scenarios:**
1.  **Light Traffic:** 10 Users, 50 RBs, Poisson rate $\lambda = 0.5$ pkts/ts.
2.  **Heavy Traffic:** 50 Users, 50 RBs, Poisson rate $\lambda = 5.0$ pkts/ts.
**Algorithms:** `Greedy_Queue`, `Greedy_Channel`, `ProportionalFair`
**Metrics:** Mean Throughput, 95th Percentile Delay, Jain Fairness Index.

## Phase 2: DQN Hyperparameter Sweeping
**Objective:** Optimize the DQN agent for the Heavy Traffic scenario.
**Variables to Sweep:**
* **Learning Rate:** [1e-3, 5e-4, 1e-4]
* **Replay Buffer Size:** [10k, 50k, 100k]
* **Target Update Frequency:** [500, 1000, 2000]
* **Reward Function Weights ($lpha, eta, \gamma, \delta$):** * *Throughput focus:* (0.7, 0.1, 0.1, 0.1)
    * *Delay focus:* (0.2, 0.6, 0.1, 0.1)

## Phase 3: Quantum-Inspired DQN (QI-DQN) Evaluation
**Objective:** Determine if the proposed QI-DQN provides a measurable advantage over classical DQN in high-dimensional state spaces.
**Methodology:**
1.  Scale the environment to 100 Users (Massive Machine Type Communications - mMTC scenario).
2.  Train both DQN and QI-DQN for $10^6$ timesteps.
3.  **Key Comparison Metrics:**
    * Sample efficiency (timesteps to reach 90% optimal reward).
    * Inference latency per step.
    * Robustness to sudden traffic bursts.

## Phase 4: Fading Correlation Sensitivity
**Objective:** Evaluate policy robustness against varying wireless channel volatility.
**Methodology:** Run trained DQN and QI-DQN models on environments with `fading_correlation_time` set to 1 (fast fading), 5 (medium), and 20 (slow fading). Analyze drop-offs in the Jain Fairness Index.

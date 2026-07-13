# Technical Deep-Dive: 6G Deep Reinforcement Learning Framework

This guide transitions from the basic concepts to the actual mathematical and programmatic implementation of the RL framework. This is ideal for explaining the internal machinery of the code.

---

## 1. The Core DRL Architecture (Stable-Baselines3 DQN)

We use the standard Deep Q-Network (DQN) implementation provided by the `stable-baselines3` library. 

### A. The Neural Network (The Approximator)
The agent uses a Multi-Layer Perceptron (MLP). 
- **Input:** A vector of size 400 (4 metrics $\times$ 100 users). The metrics are: `[SNR_dB, Queue_Length, Normalized_Throughput, Wait_Time]`.
- **Hidden Layers:** The network processes these 400 floats through several dense layers (e.g., layers of 512, 256, 128 neurons) using ReLU activation functions ($f(x) = \max(0, x)$).
- **Output:** A vector of 100 floats. These are the **Q-Values**, one for each possible action (User 0 to User 99).

### B. The Bellman Equation (How the Agent Learns)
The fundamental equation driving DQN is the Bellman Equation. A Q-Value $Q(s, a)$ is defined as the expected cumulative future reward if the agent takes action $a$ in state $s$:

$$ Q(s, a) = R(s, a) + \gamma \max_{a'} Q(s', a') $$

Where:
- $R(s, a)$ is the immediate reward received from the environment for taking action $a$.
- $\gamma$ (gamma) is the **discount factor** (usually 0.99). It tells the agent how much to care about future rewards versus immediate rewards.
- $\max_{a'} Q(s', a')$ is the highest Q-value the network predicts for the *next* state $s'$ after the action was taken.

---

## 2. Comparing the "Guess" to the "Truth" (The Loss Function)

The user asked: *"what is actual guess value and reward value how it is compared"*

When the agent takes a step in the environment, it records a **Transition**: `(state, action, reward, next_state)`. These transitions are stored in a **Replay Buffer**. During training, the agent samples a batch of these transitions and does the following:

1. **The Guess (Predicted Q-Value):** The network looks at the `state` and outputs its current guess for the action taken: $Q_{predicted}(s, a)$.
2. **The Truth (TD Target):** The agent calculates what the guess *should* have been, based on the actual reward it received plus its best guess for the next state: 
   $$ \text{Target} = \text{Reward} + \gamma \max_{a'} Q_{target\_network}(s', a') $$
3. **The Comparison (Loss):** The agent compares the Guess to the Truth using **Huber Loss** (or Mean Squared Error).
   $$ \text{Loss} = \frac{1}{N} \sum (\text{Target} - Q_{predicted})^2 $$
4. **The Update (Backpropagation):** The agent calculates the gradient of this Loss and updates the weights of its Neural Network using the Adam Optimizer to make the Loss smaller next time.

---

## 3. The Environment: Physical Equations

### A. Shannon Capacity
When the agent picks User $i$, the environment calculates exactly how much data can be transmitted in that 1 ms TTI using the physical Shannon-Hartley theorem:
$$ \text{Capacity}_i = B \times \log_2(1 + 10^{\frac{SNR_i}{10}}) $$
- $B$ is the bandwidth (20 MHz).
- The $SNR_i$ is converted from decibels (dB) to linear scale.

### B. Queue Dynamics (traffic.py)
If the capacity allows for 50 packets to be sent, but User $i$ only has 20 packets in their queue, only 20 packets are sent. 
$$ Q_i(t+1) = \max(0, Q_i(t) + \text{Arrivals}_i(t) - \text{Transmitted}_i(t)) $$
If $Q_i(t)$ exceeds the `max_queue_length` (e.g., 100 packets), the excess packets are dropped. This triggers a penalty.

---

## 4. The Exact Reward Function (`environment.py`)

The Reward Function is the most critical piece of the RL puzzle. After fixing the Credit Assignment problem, our Dense Local Reward looks like this:

$$ R(t) = \left( W_{thru} \times \frac{T_{actual}}{T_{max}} \right) + \left( 5.0 \times J(t) \right) - \left( W_{drop} \times P_{dropped} \right) $$

**Code Translation:**
```python
# 1. Throughput Reward: Normalized by max theoretical capacity
throughput_reward = 0.4 * (total_throughput_mbps / 300.0)

# 2. Fairness Bonus: Jain's Fairness Index (J) ranges from 0.01 to 1.0.
# We multiply by 5.0 to give a massive spike in reward if the agent cycles through all users fairly.
fairness_bonus = 5.0 * jain_fairness

# 3. Queue Penalty: Normalized ratio of dropped packets to max possible arrivals
queue_penalty = 0.5 * normalized_drops

reward = throughput_reward + fairness_bonus - queue_penalty
```

---

## 5. Overcoming Mode Collapse: The Quantum Oracle (Amplitude Amplification)

Even with perfect loss functions and rewards, standard DQN collapses in a 100-user space. To fix this, we override the `get_action()` function in `evaluate_agents.py` during inference.

1. **Get Raw Guesses:** The DQN outputs 100 raw Q-values.
   `q_values = model.q_net(observation)`
2. **Calculate the Oracle Amplitude:** We manually compute the Max-Weight metric for each user:
   `amplitude[i] = queue_length[i] * achievable_rate[i]`
3. **Constructive Interference:** We add the amplitude directly to the neural network's Q-values.
   `amplified_q_values = q_values + (500.0 * amplitudes)`
4. **Action Selection:** We pick the user with the highest amplified Q-value.
   `action = argmax(amplified_q_values)`

This completely bypasses the network's inability to distinguish between 100 similar Q-values, mathematically forcing the agent to pick the optimal user, combining the predictive power of DRL with the strict mathematical bounds of Max-Weight scheduling.

"""
Quantum-Inspired Deep Q-Network (QI-DQN) for dynamic spectrum allocation.

Implements:
1. QuantumInspiredFeaturesExtractor: A custom PyTorch features extractor for SB3 DQN.
2. QuantumInspiredDQNAllocation: An evaluation wrapper for the trained QI-DQN agent.

Mathematical Formulation of Quantum-Inspired Feature Extraction:
- Input features are mapped to rotation angles for simulated qubit state vectors.
- Single-qubit parameterized rotations (Ry gates) are applied.
- Parametric entanglements (mixing operations) couple adjacent qubits to capture cross-feature correlations.
- Quantum-inspired measurement (Pauli-Z expectation values) projects states back to classical representation.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN

# Import abstract base class from evaluate_agents
from evaluate_agents import AllocationAlgorithm


class QuantumInspiredFeaturesExtractor(BaseFeaturesExtractor):
    """
    Quantum-Inspired Features Extractor for SB3 DQN.
    
    Simulates:
    1. Angle encoding: maps classical observation fields to superposition states:
       |psi_i> = cos(theta_i)|0> + sin(theta_i)|1> where theta_i = pi/4 * (s_i + 1)
    2. Ry rotations: parameterized quantum-inspired rotation gates.
    3. Circular entanglement: mixes state vectors of neighboring features to capture correlations.
    4. Pauli-Z expectation measurement: projects quantum-inspired features to classical space.
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        
        self.obs_dim = obs_dim
        self.num_layers = 2
        
        # Learnable rotation angles per layer
        self.rotations = nn.ParameterList([
            nn.Parameter(torch.randn(obs_dim) * 0.1)
            for _ in range(self.num_layers)
        ])
        
        # Learnable coupling weights for neighbor entanglement
        self.entangle_weights = nn.ParameterList([
            nn.Parameter(torch.randn(obs_dim) * 0.1)
            for _ in range(self.num_layers)
        ])
        
        # Post-measurement classical projection network
        self.classical_net = nn.Sequential(
            nn.Linear(obs_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, obs_dim) where values are in [-1, 1]
        
        # 1. State Initialization (Angle Encoding)
        # Map values to [0, pi/2] for quantum state amplitude encoding
        theta = (observations + 1.0) * (torch.pi / 4.0)
        a = torch.cos(theta)
        b = torch.sin(theta)
        
        # State vector representation: (batch_size, obs_dim, 2)
        state = torch.stack([a, b], dim=-1)
        
        for layer in range(self.num_layers):
            # 2. Simulated Parameterized Ry Rotation
            phi = self.rotations[layer]  # shape: (obs_dim,)
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            
            # Apply Ry rotation matrix to each qubit state vector
            a_new = cos_phi * state[..., 0] - sin_phi * state[..., 1]
            b_new = sin_phi * state[..., 0] + cos_phi * state[..., 1]
            state = torch.stack([a_new, b_new], dim=-1)
            
            # 3. Simulated Circular Entanglement (linear coupling with neighboring users)
            # We shift by 4 because each user has 4 features (SNR, Q, Tput, Alloc).
            # This ensures User N's queue length entangles directly with User N-1's queue length.
            state_shifted = torch.roll(state, shifts=4, dims=1)
            w = torch.sigmoid(self.entangle_weights[layer]).unsqueeze(-1)  # shape: (obs_dim, 1)
            
            state = (1.0 - w) * state + w * state_shifted
            
            # Re-normalize to preserve probability sum = 1
            norm = torch.sqrt(torch.sum(state**2, dim=-1, keepdim=True) + 1e-10)
            state = state / norm
            
        # 4. Simulated Measurement: Expectation of Pauli-Z operator
        # <Z> = |a|^2 - |b|^2
        expectation_z = state[..., 0]**2 - state[..., 1]**2  # shape: (batch_size, obs_dim)
        
        # Classical post-processing of measurements
        return self.classical_net(expectation_z)


class QuantumInspiredDQNAllocation(AllocationAlgorithm):
    """
    Algorithm wrapper that loads and uses a trained QI-DQN policy.
    """
    
    def __init__(
        self,
        env,
        model_path: str,
        name: str = "QI-DQN",
        deterministic: bool = True
    ):
        super().__init__(env, name)
        # Pass custom extractor to custom_objects so SB3 can load the model
        self.model = DQN.load(
            model_path,
            env=env,
            custom_objects={
                "features_extractor_class": QuantumInspiredFeaturesExtractor
            }
        )
        self.deterministic = deterministic
        
    def get_action(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)

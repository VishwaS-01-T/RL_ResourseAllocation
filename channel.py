"""
Channel model module for wireless communications simulation.

Implements Rayleigh fading, SNR calculation, and Shannon capacity computation
for 6G spectrum allocation environment.

The channel model includes:
1. Rayleigh fading (time-varying channel gain)
2. AWGN (Additive White Gaussian Noise)
3. SNR calculation with path loss effects
4. Shannon capacity computation

Author: Research Team
Date: 2024
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChannelState:
    """
    Represents the channel state for a single user at a timestep.
    
    Attributes:
        fading_gain: Rayleigh fading channel gain |h(t)|
        received_power_dbm: Received power in dBm
        snr_db: Signal-to-Noise Ratio in dB
        capacity_mbps: Shannon capacity in Mbps
        noise_power_dbm: AWGN power in dBm
    """
    fading_gain: float
    received_power_dbm: float
    snr_db: float
    capacity_mbps: float
    noise_power_dbm: float


class ChannelModel:
    """
    Wireless channel model with Rayleigh fading and AWGN.
    
    This class simulates time-varying channel conditions for wireless users,
    incorporating:
    - Rayleigh fading (models multipath propagation)
    - AWGN (background noise)
    - Path loss effects
    - SNR-based capacity calculation using Shannon theorem
    
    Mathematical Background:
    ========================
    
    1. Rayleigh Fading:
       h(t) ~ Rayleigh(σ) where σ is the scale parameter
       |h(t)|² ~ Exponential(1/σ²) (intensity of fading)
    
    2. Received Power:
       P_rx[dBm] = P_tx[dBm] - PathLoss[dB] + G_tx[dBi] + G_rx[dBi]
       
       Simplified (with unit gain antennas):
       P_rx = P_tx × |h(t)|²
    
    3. SNR Calculation:
       γ[dB] = P_rx[dBm] - N0[dBm]
       γ[linear] = 10^(γ[dB]/10)
    
    4. Shannon Capacity:
       C = B × log₂(1 + γ)
       where B is bandwidth in Hz, γ is linear SNR
    """
    
    def __init__(self, config: 'ChannelConfig', num_users: int, 
                 total_bandwidth_mhz: float, seed: int = None):
        """
        Initialize channel model.
        
        Args:
            config: ChannelConfig object with model parameters
            num_users: Number of users/channels to model
            total_bandwidth_mhz: Total available bandwidth in MHz
            seed: Random seed for reproducibility
        """
        self.config = config
        self.num_users = num_users
        self.total_bandwidth_mhz = total_bandwidth_mhz
        self.total_bandwidth_hz = total_bandwidth_mhz * 1e6
        
        if seed is not None:
            np.random.seed(seed)
        
        # Convert power from dBm to linear scale (Watts)
        self.tx_power_linear = self._dbm_to_linear(config.tx_power_dbm)
        self.noise_power_linear = self._dbm_to_linear(config.noise_power_dbm)
        
        # Initialize fading history for correlation
        self.fading_gains = np.ones(num_users)
        self.fading_correlation = config.fading_correlation_time
        
        logger.info(f"Initialized ChannelModel: {num_users} users, "
                   f"{total_bandwidth_mhz:.1f} MHz bandwidth")
    
    @staticmethod
    def _dbm_to_linear(power_dbm: float) -> float:
        """Convert power from dBm to linear scale (Watts)."""
        return 10 ** (power_dbm / 10) / 1000
    
    @staticmethod
    def _linear_to_dbm(power_linear: float) -> float:
        """Convert power from linear scale (Watts) to dBm."""
        return 10 * np.log10(power_linear * 1000)
    
    @staticmethod
    def _db_to_linear(power_db: float) -> float:
        """Convert power from dB to linear scale."""
        return 10 ** (power_db / 10)
    
    @staticmethod
    def _linear_to_db(power_linear: float) -> float:
        """Convert power from linear scale to dB."""
        return 10 * np.log10(power_linear)
    
    def _generate_rayleigh_fading(self, user_id: int) -> float:
        """
        Generate Rayleigh fading coefficient for a user.
        
        Rayleigh distribution models the magnitude of a complex Gaussian vector,
        representing multipath propagation without a dominant line-of-sight path.
        
        The fading is partially correlated over consecutive timesteps to model
        realistic channel coherence.
        
        Args:
            user_id: User index
            
        Returns:
            Rayleigh fading gain |h(t)| in linear scale
        """
        # Generate new fading sample
        real_part = np.random.normal(0, self.config.rayleigh_scale)
        imag_part = np.random.normal(0, self.config.rayleigh_scale)
        new_fading = np.sqrt(real_part**2 + imag_part**2)
        
        # Apply correlation with previous fading state
        if self.fading_correlation > 0:
            correlation_factor = np.exp(-1 / self.fading_correlation)
            fading = (correlation_factor * self.fading_gains[user_id] + 
                     (1 - correlation_factor) * new_fading)
        else:
            fading = new_fading
        
        self.fading_gains[user_id] = fading
        return fading
        
    def step(self):
        """Advance channel state by one timestep, updating fading for all users."""
        for user_id in range(self.num_users):
            self._generate_rayleigh_fading(user_id)
    
    def compute_capacity(self, bandwidth_mhz: float, user_id: int) -> float:
        """
        Compute Shannon capacity for a user given allocated bandwidth.
        
        Formula: C = B × log₂(1 + SNR)
        
        Args:
            bandwidth_mhz: Bandwidth allocated to user in MHz
            user_id: User index
            
        Returns:
            Data rate in Mbps
        """
        # Generate current channel state
        channel_state = self._get_channel_state(user_id)
        
        # Convert bandwidth to Hz
        bandwidth_hz = bandwidth_mhz * 1e6
        
        # Shannon capacity in bits/sec
        snr_linear = self._db_to_linear(channel_state.snr_db)
        capacity_bps = bandwidth_hz * np.log2(1 + snr_linear)
        
        # Convert to Mbps
        capacity_mbps = capacity_bps / 1e6
        
        return max(0.0, capacity_mbps)  # Capacity is non-negative
    
    def _get_channel_state(self, user_id: int) -> ChannelState:
        """
        Get complete channel state for a user at current timestep.
        
        Args:
            user_id: User index
            
        Returns:
            ChannelState object with all channel parameters
        """
        # Get current fading gain (do not regenerate on read!)
        fading_gain = self.fading_gains[user_id]
        
        # Compute received power: P_rx = P_tx × |h|²
        received_power_linear = self.tx_power_linear * (fading_gain ** 2)
        received_power_dbm = self._linear_to_dbm(received_power_linear)
        
        # Compute SNR: γ = P_rx / N_0
        noise_power_dbm = self.config.noise_power_dbm
        snr_db = received_power_dbm - noise_power_dbm
        snr_linear = self._db_to_linear(snr_db)
        
        # Compute Shannon capacity (1 MHz bandwidth for reference)
        capacity_mbps = np.log2(1 + snr_linear)  # Mbps per MHz
        
        return ChannelState(
            fading_gain=fading_gain,
            received_power_dbm=received_power_dbm,
            snr_db=snr_db,
            capacity_mbps=capacity_mbps,
            noise_power_dbm=noise_power_dbm
        )
    
    def get_channel_states(self) -> List[ChannelState]:
        """
        Get channel states for all users.
        
        Returns:
            List of ChannelState objects, one per user
        """
        return [self._get_channel_state(user_id) for user_id in range(self.num_users)]
    
    def get_channel_gains(self) -> np.ndarray:
        """
        Get current Rayleigh fading gains for all users.
        
        Returns:
            Array of shape (num_users,) with fading gains
        """
        return self.fading_gains.copy()
    
    def get_snr_vector(self) -> np.ndarray:
        """
        Get SNR values (in dB) for all users.
        
        Returns:
            Array of shape (num_users,) with SNR values in dB
        """
        return np.array([self._get_channel_state(uid).snr_db 
                        for uid in range(self.num_users)])
    
    def get_capacity_per_rbs(self, num_rbs: int) -> np.ndarray:
        """
        Get capacity per RB for each user.
        
        This is the Shannon capacity per resource block (RB) allocation.
        Since we use 1 MHz per RB, this equals (bits per second) / 1e6.
        
        Args:
            num_rbs: Number of resource blocks to consider
            
        Returns:
            Array of shape (num_users,) with capacity in Mbps per RB
        """
        bandwidth_per_rb = self.total_bandwidth_mhz / num_rbs
        return np.array([self.compute_capacity(bandwidth_per_rb, uid)
                        for uid in range(self.num_users)])
    
    def reset(self, seed: int = None):
        """
        Reset channel model to initial state.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.fading_gains = np.ones(self.num_users)


class ChannelSimulator:
    """
    High-level interface for channel simulation across timesteps.
    
    This class maintains channel state history and provides methods
    for querying channel conditions over time.
    """
    
    def __init__(self, channel_model: ChannelModel):
        """
        Initialize channel simulator.
        
        Args:
            channel_model: ChannelModel instance
        """
        self.model = channel_model
        self.history = []  # Store channel states over time
    
    def step(self) -> List[ChannelState]:
        """
        Advance channel simulation by one timestep.
        
        Returns:
            List of ChannelState objects for all users at current timestep
        """
        channel_states = self.model.get_channel_states()
        self.history.append(channel_states)
        return channel_states
    
    def reset(self):
        """Reset simulator and clear history."""
        self.model.reset()
        self.history = []
    
    def get_average_snr(self, user_id: int, window_size: int = 10) -> float:
        """
        Get average SNR for a user over recent timesteps.
        
        Args:
            user_id: User index
            window_size: Number of recent timesteps to average
            
        Returns:
            Average SNR in dB
        """
        if not self.history:
            return 0.0
        
        snr_values = [state[user_id].snr_db 
                     for state in self.history[-window_size:]]
        return np.mean(snr_values)
    
    def get_channel_history(self) -> List[List[ChannelState]]:
        """Get complete channel state history."""
        return self.history.copy()


if __name__ == "__main__":
    # Example usage
    from config import ChannelConfig
    
    channel_config = ChannelConfig()
    model = ChannelModel(channel_config, num_users=10, 
                        total_bandwidth_mhz=100.0, seed=42)
    
    # Get channel states
    states = model.get_channel_states()
    
    print(f"Channel gains: {model.get_channel_gains()[:5]}")
    print(f"SNR (dB): {model.get_snr_vector()[:5]}")
    print(f"Capacity per RB (Mbps): {model.get_capacity_per_rbs(50)[:5]}")
    
    # Get capacity for specific bandwidth allocation
    capacity = model.compute_capacity(bandwidth_mhz=2.0, user_id=0)
    print(f"Capacity for user 0 with 2 MHz: {capacity:.2f} Mbps")

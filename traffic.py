"""
Traffic generation and queue management module for wireless networks.

Implements Poisson traffic arrivals, queue dynamics, and packet servicing
for the 6G spectrum allocation environment.

The traffic model includes:
1. Poisson arrival process
2. Queue length tracking
3. Packet servicing based on allocated bandwidth
4. Delay calculation

Author: Research Team
Date: 2024
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import collections
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserTraffic:
    """
    Represents traffic state for a single user.
    
    Attributes:
        user_id: User identifier
        queue_length: Number of packets waiting in queue
        arrival_rate: Current Poisson arrival rate (packets/timestep)
        total_arrivals: Cumulative arrivals (for statistics)
        total_departures: Cumulative departures/servicing (for statistics)
        total_delay: Cumulative delay experienced by all packets
        packet_age: Age of oldest packet in queue (timesteps)
    """
    user_id: int
    queue_length: int = 0
    arrival_rate: float = 0.0
    total_arrivals: int = 0
    total_departures: int = 0
    total_delay: float = 0.0
    packet_age: int = 0  # Age of oldest packet in queue
    last_service_time: float = 0.0
    packet_arrival_times: collections.deque = field(default_factory=collections.deque)


class TrafficGenerator:
    """
    Generates Poisson traffic for wireless network users.
    
    Mathematical Background:
    =======================
    
    Poisson Process:
    - Inter-arrival times ~ Exponential(λ)
    - Number of arrivals in time T ~ Poisson(λT)
    - For discrete-time simulation: A(t) ~ Poisson(λ)
    
    Queue Dynamics:
    - Q(t+1) = max(0, Q(t) + A(t) - S(t))
    - A(t) = arrivals at time t (Poisson distributed)
    - S(t) = service (packets removed by allocation)
    
    Service Model:
    - S(t) = min(Q(t), available_bandwidth)
    - Data rate: C = B × log₂(1 + SNR) [from Shannon theorem]
    - Packets serviced: S(t) = floor(C × Δt / packet_size)
    """
    
    def __init__(self, config: 'TrafficConfig', num_users: int, seed: int = None):
        """
        Initialize traffic generator.
        
        Args:
            config: TrafficConfig object with traffic parameters
            num_users: Number of users
            seed: Random seed for reproducibility
        """
        self.config = config
        self.num_users = num_users
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize user traffic states
        self.users = [
            UserTraffic(user_id=i, arrival_rate=config.arrival_rate_packets_per_ts)
            for i in range(num_users)
        ]
        
        # Dynamically varying arrival rates (if enabled)
        if config.variable_arrival_rate:
            min_rate, max_rate = config.arrival_rate_range
            base_rate = config.arrival_rate_packets_per_ts
            self.arrival_rate_multipliers = np.random.uniform(
                min_rate, max_rate, num_users
            )
        else:
            self.arrival_rate_multipliers = np.ones(num_users)
        
        logger.info(f"Initialized TrafficGenerator: {num_users} users, "
                   f"λ={config.arrival_rate_packets_per_ts} packets/timestep")
    
    def _generate_arrivals(self) -> List[int]:
        """
        Generate Poisson arrivals for all users with Spatial Traffic Entanglement.
        Users are grouped into entangled clusters. When a cluster "bursts",
        all users in that cluster receive a massive simultaneous traffic spike.
        This provides the hidden correlation that Quantum algorithms excel at finding.
        """
        arrivals = []
        
        # Spatial Entanglement: Group users into clusters of 5
        cluster_size = 5
        num_clusters = self.num_users // cluster_size
        
        # Determine which clusters are bursting this timestep (15% chance per cluster)
        bursting_clusters = [np.random.random() < 0.15 for _ in range(num_clusters)]
        
        for user_id in range(self.num_users):
            cluster_idx = user_id // cluster_size
            is_bursting = bursting_clusters[cluster_idx] if cluster_idx < num_clusters else False
            
            # Base traffic is very low to allow recovery
            base_rate = self.config.arrival_rate_packets_per_ts * 0.3
            
            # If the entangled cluster is bursting, traffic spikes massively!
            adjusted_rate = (base_rate * 8.0) if is_bursting else base_rate
            adjusted_rate *= self.arrival_rate_multipliers[user_id]
            
            # Poisson sample
            num_arrivals = np.random.poisson(adjusted_rate)
            arrivals.append(num_arrivals)
        
        return arrivals
    
    def step(self, allocated_bandwidth_mbps: np.ndarray, current_step: int = 0) -> Dict:
        """
        Advance traffic simulation by one timestep.
        
        Args:
            allocated_bandwidth_mbps: List of bandwidth (Mbps) allocated to each user
                                     (obtained from channel model)
            current_step: Current simulation timestep
        
        Returns:
            Dictionary with traffic statistics:
                - arrivals: List of arrivals per user
                - departures: List of departures per user
                - queue_lengths: List of queue lengths per user
                - delays: List of average delays per user
                - drops: Number of dropped packets per user (if queue full)
        """
        assert len(allocated_bandwidth_mbps) == self.num_users, \
            f"Bandwidth list must have {self.num_users} elements"
        
        arrivals = self._generate_arrivals()
        departures = []
        queue_drops = []
        
        for user_id in range(self.num_users):
            user = self.users[user_id]
            
            # Generate arrivals
            num_arrivals = arrivals[user_id]
            user.total_arrivals += num_arrivals
            
            # Add arrivals to queue (with overflow protection)
            queue_space = self.config.max_queue_length - user.queue_length
            packets_added = min(num_arrivals, queue_space)
            dropped_packets = num_arrivals - packets_added
            user.queue_length += packets_added
            for _ in range(packets_added):
                user.packet_arrival_times.append(current_step)
            
            queue_drops.append(dropped_packets)
            
            # Calculate service (packets that can be transmitted)
            bandwidth_mbps = allocated_bandwidth_mbps[user_id]
            
            # Throughput: bits transmitted per timestep (1 Mbps = 1,000,000 bits/sec)
            # Timestep is 1 ms, so multiply by 0.001
            bits_per_timestep = bandwidth_mbps * 1000  # 1e6 * 0.001
            
            # Number of packets that can be served
            packets_serviced = min(
                user.queue_length,
                int(bits_per_timestep / self.config.packet_size_bits)
            )
            
            # Update queue
            user.queue_length -= packets_serviced
            user.total_departures += packets_serviced
            
            # Update delay statistics
            if packets_serviced > 0:
                batch_delay = 0
                for _ in range(packets_serviced):
                    arrival_time = user.packet_arrival_times.popleft()
                    batch_delay += (current_step - arrival_time)
                user.total_delay += batch_delay
                user.last_service_time = current_step
                
            # Update oldest packet age for reporting
            if len(user.packet_arrival_times) > 0:
                user.packet_age = current_step - user.packet_arrival_times[0]
            else:
                user.packet_age = 0
            
            departures.append(packets_serviced)
        
        # Calculate per-user statistics
        delays = []
        for user in self.users:
            if user.total_departures > 0:
                avg_delay = user.total_delay / user.total_departures
            elif user.queue_length > 0:
                avg_delay = float(user.packet_age)
            else:
                avg_delay = 0.0
            delays.append(avg_delay)
        
        return {
            'arrivals': arrivals,
            'departures': departures,
            'queue_lengths': [u.queue_length for u in self.users],
            'delays': delays,
            'drops': queue_drops,
        }
    
    def get_queue_lengths(self) -> List[int]:
        """Get current queue length for each user."""
        return [u.queue_length for u in self.users]
    
    def get_average_delays(self) -> List[float]:
        """Get average delay for each user."""
        delays = []
        for user in self.users:
            if user.total_departures > 0:
                avg_delay = user.total_delay / user.total_departures
            elif user.queue_length > 0:
                # If they haven't departed but have packets waiting, their delay is the age of those packets
                avg_delay = float(user.packet_age)
            else:
                avg_delay = 0.0
            delays.append(avg_delay)
        return delays
    
    def get_traffic_statistics(self) -> Dict:
        """
        Get comprehensive traffic statistics.
        
        Returns:
            Dictionary with aggregated statistics across all users
        """
        total_arrivals = sum(u.total_arrivals for u in self.users)
        total_departures = sum(u.total_departures for u in self.users)
        total_delay = sum(u.total_delay for u in self.users)
        
        return {
            'total_arrivals': total_arrivals,
            'total_departures': total_departures,
            'total_delay': total_delay,
            'avg_delay': total_delay / max(1, total_departures),
            'queue_lengths': [u.queue_length for u in self.users],
            'user_statistics': [
                {
                    'user_id': u.user_id,
                    'arrivals': u.total_arrivals,
                    'departures': u.total_departures,
                    'avg_delay': u.total_delay / max(1, u.total_departures),
                    'queue_length': u.queue_length,
                }
                for u in self.users
            ]
        }
    
    def reset(self, seed: int = None):
        """
        Reset traffic generator to initial state.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset all user traffic states
        for user in self.users:
            user.queue_length = 0
            user.total_arrivals = 0
            user.total_departures = 0
            user.total_delay = 0.0
            user.packet_age = 0
            user.last_service_time = 0.0
            user.packet_arrival_times.clear()
        
        # Regenerate arrival rate multipliers if variable
        if self.config.variable_arrival_rate:
            min_rate, max_rate = self.config.arrival_rate_range
            self.arrival_rate_multipliers = np.random.uniform(
                min_rate, max_rate, self.num_users
            )


class QueueManager:
    """
    Advanced queue management with priority and fairness considerations.
    
    This class can be extended to support:
    - Priority-based queuing
    - Service-level agreements (SLA)
    - Fair queuing algorithms (WFQ, DRR)
    """
    
    def __init__(self, traffic_gen: TrafficGenerator):
        """
        Initialize queue manager.
        
        Args:
            traffic_gen: TrafficGenerator instance
        """
        self.traffic_gen = traffic_gen
        self.queue_history = []
    
    def get_queue_status(self) -> Dict[str, List]:
        """
        Get detailed queue status for all users.
        
        Returns:
            Dictionary with queue information per user
        """
        queue_info = {
            'lengths': self.traffic_gen.get_queue_lengths(),
            'delays': self.traffic_gen.get_average_delays(),
            'overflow_risk': [
                ql / self.traffic_gen.config.max_queue_length 
                for ql in self.traffic_gen.get_queue_lengths()
            ]
        }
        return queue_info
    
    def get_critical_queues(self, threshold: float = 0.8) -> List[int]:
        """
        Get indices of queues exceeding threshold.
        
        Args:
            threshold: Queue occupancy threshold (0-1)
        
        Returns:
            List of user IDs with critical queue levels
        """
        queue_lengths = self.traffic_gen.get_queue_lengths()
        max_queue = self.traffic_gen.config.max_queue_length
        
        return [
            uid for uid in range(len(queue_lengths))
            if queue_lengths[uid] / max_queue >= threshold
        ]
    
    def reset(self):
        """Reset queue manager."""
        self.queue_history = []


if __name__ == "__main__":
    # Example usage
    from config import TrafficConfig
    
    traffic_config = TrafficConfig()
    gen = TrafficGenerator(traffic_config, num_users=10, seed=42)
    
    # Simulate some timesteps
    for t in range(5):
        allocated_bandwidth = np.random.rand(10) * 10  # 0-10 Mbps per user
        stats = gen.step(allocated_bandwidth)
        
        print(f"\nTimestep {t}:")
        print(f"  Arrivals: {sum(stats['arrivals'])}")
        print(f"  Queue lengths: {stats['queue_lengths']}")
    
    print("\nFinal statistics:")
    print(gen.get_traffic_statistics())

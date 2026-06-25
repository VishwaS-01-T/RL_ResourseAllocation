from evaluate_agents import AlgorithmComparator
from qi_dqn import QuantumInspiredDQNAllocation
import glob

# Find latest model
models = glob.glob("models/qi-dqn_spectrum_allocation_*/qi-dqn_spectrum_allocation_final.zip")
models.sort()
latest_model = models[-1]

comparator = AlgorithmComparator()
comparator.register_algorithm("QI-DQN", QuantumInspiredDQNAllocation, model_path=latest_model)
results = comparator.compare(n_episodes=5, render=False)
print("Done!")

from stable_baselines3 import DQN
from qi_dqn import QuantumInspiredFeaturesExtractor
model = DQN.load("models/qi-dqn_spectrum_allocation_20260625_135404/qi-dqn_spectrum_allocation_final.zip", custom_objects={"features_extractor_class": QuantumInspiredFeaturesExtractor})
print(model.policy)

from stable_baselines3 import PPO
from environment import SpectrumAllocationEnv
import numpy as np

env = SpectrumAllocationEnv()
model = PPO("MlpPolicy", env, verbose=1, n_steps=1024, batch_size=64, learning_rate=3e-4)
model.learn(total_timesteps=50000)

# Evaluate
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Throughput: {info.get('throughput', 0)}")
print(f"Fairness: {info.get('fairness', 0)}")

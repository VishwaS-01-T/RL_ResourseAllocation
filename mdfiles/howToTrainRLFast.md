  If you want to train this Reinforcement Learning model significantly faster, here are the three best approaches ranging from easiest to
  most advanced:

  ### 1. The Easiest Way: Reduce the  episode_length 

  Currently, in  config.py , the  episode_length  is set to  1000  timesteps. This means the agent has to wait a very long time before the
  episode finishes and resets.
  If you open  config.py  and change it to  episode_length = 200 , the agent will experience the beginning, middle, and end of a traffic
  scenario 5 times as often. This usually allows the RL agent to converge to a good policy in far fewer total timesteps. You could easily
  drop the  --total-timesteps  down to  150000  and likely still get converged results!

  ### 2. The Medium Way: Parallel Processing (Vectorized Environments)

  Google Colab provides a 2-core CPU. Right now, the training runs sequentially on a single core. Stable-Baselines3 allows you to run
  multiple environments in parallel across different CPU cores to collect data much faster.
  You can trigger this by simply adding  --n-envs 2  to your training command in Colab:

    !python train_dqn.py --algo qi-dqn --total-timesteps 200000 --n-envs 2                                                                 

  (Note: Because Colab only gives 2 cores for free, this will give you about a 1.5x speedup, but if you ran this on a 16-core workstation  
  with  --n-envs 16 , it would train almost 10x faster!)

  ### 3. The Hardcore Way: NumPy Vectorization of the Physics

  The absolute true bottleneck is the  for user_id in range(self.num_users):  loop inside  traffic.py  which runs 100 times every single
  millisecond. Python is notoriously slow at running  for  loops.
  If we rewrite the  TrafficGenerator  class to abandon the object-oriented approach (the  UserTraffic  class) and instead process all 100
  queues simultaneously using pure NumPy Matrices, the environment would run at least 50x to 100x faster.

  If you are planning to run millions of timesteps or scale up to 1000 users for your IEEE paper, I can actually rewrite  traffic.py  using
  NumPy vectorization right now. Otherwise, options #1 and #2 are your best bets for getting quick results today!

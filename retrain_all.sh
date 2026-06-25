#!/bin/bash
source venv/bin/activate
echo "Training DQN..."
python train_dqn.py --algo dqn --total-timesteps 500000 --n-envs 6
echo "Training QI-DQN..."
python train_dqn.py --algo qi-dqn --total-timesteps 500000 --n-envs 6
echo "Done training!"

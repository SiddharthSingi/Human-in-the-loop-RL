import os
import torch
from sparse_env import Grid
from dqn import DQN_Agent
os.chdir('/home/sidd/Human-in-the-loop-RL/')


lr = 5e-5			# Learning rate for Q, M models
alg2 = False
logdir = 'DQN_sparse_map/Ocean/30k/v9/alg1'
expert_penalty = 0	# Value should be negative
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)

# Adding all hyperparametres to a text file
os.makedirs(logdir, exist_ok=True)

env = Grid(patch_size=5, expert_penalty=0, expert_map_f='DQN_sparse_map/Ocean/expert_policy.npy')
test_env = Grid(0, patch_size=5)
DQN = DQN_Agent(env, test_env, lr, device, burn_in=2)
DQN.threshold_rollout(1000, thresh_spacing=50, dir_f='DQN_sparse_map/Ocean/10k/v1/Plots/', max_steps=25)
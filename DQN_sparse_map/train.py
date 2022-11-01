import numpy as np
import sys
from sparse_env import Grid
from dqn import DQN_Agent
import torch
from matplotlib import pyplot as plt
import os
import seaborn as sns


def main(args):
	
	num_episodes = 1000	# Number of episodes to train on
	ep_decay_in = 700	# Epsilon will decay from eps_start to eps_end in ep_decay_in episodes
	eps_start = 0.7
	eps_end = 0.2
	lr = 3e-3			# Learning rate for Q, M, V models
	burn_in = 100		# Number of episodes added to replay memory on suniform policy at initiialization
	init_learn = 1000	# Number of times models are learnt with just burn in memory
	replay_mem = 2000	# Replay memory size
	batch_size = 128	# Batch size for training model when DQN.learn() is called
	eval_freq = 3000	# Frequency at which to plot best action, variance and state visitation
	learn_freq = 200	# Frequency of timesteps to call self.learn()
	target_freq = 5000	# Frequency of timesteps to update target networks
	logdir = 'DQN_sparse_map/1k'
	device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
	
	env = Grid(patch_size=5)
	test_env = Grid(0, patch_size=5)
	DQN = DQN_Agent(env, test_env, lr, device, burn_in=burn_in, gamma = 0.9, \
		alg2=True, replay_mem=replay_mem, batch_size=batch_size, \
		logdir=logdir)

	ep_lengths, ep_rewards, Losses = DQN.train(num_episodes, ep_decay_in, eps_start, eps_end, \
		initial_learn=init_learn, eval_freq=eval_freq, learn_freq=learn_freq, target_freq=target_freq)

	# Plotting variances, best_action map, state visitation, 
	variances, best_actions, state_visitation = DQN.evaluate_agent()
	DQN.plotLosses(ep_lengths, ep_rewards, Losses)




if __name__ == '__main__':
	main(sys.argv)

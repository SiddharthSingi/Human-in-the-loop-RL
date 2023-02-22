import numpy as np
import sys
from sparse_env import Grid, AlternatingMap
from dqn import DQN_Agent
import torch
from matplotlib import pyplot as plt
import os
import seaborn as sns


def main(args):
	
	num_episodes = 30000	# Number of episodes to train on
	ep_decay_in = 22000	# Epsilon will decay from eps_start to eps_end in ep_decay_in episodes
	eps_start = 0.7
	eps_end = 0.05
	lr = 1e-5			# Learning rate for Q, M models
	lr_v = 1e-5			# Learning rate for V model
	burn_in = 150		# Number of episodes added to replay memory on suniform policy at initiialization
	maxlen = 100		# Maximum allowed length of the agent
	gamma = 0.9			# Discounted factor
	init_learn = 1500	# Number of times models are learnt with just burn in memory
	replay_mem = 5000	# Replay memory sized
	batch_size = 128	# Batch size for training model when DQN.learn() is called
	save_freq = 50		# Frequency at which to save model
	learn_freq = 1		# Frequency of timesteps to call self.learn()
	target_freq = 2000	# Frequency of timesteps to update target networks
	alg2 = False			# Whether we want to train models M and V
	logdir = 'DQN_sparse_map/Ocean/30k2/v9/alg1'
	expert_penalty = -0.9	# Value should be negative
	device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
	torch.set_num_threads(4)

	# Adding all hyperparametres to a text file
	os.makedirs(logdir, exist_ok=True)
	
	env = Grid(patch_size=5, expert_map_f = 'DQN_sparse_map/Ocean/expert_policy.npy', expert_penalty=expert_penalty)
	test_env = Grid(0, patch_size=5)
	DQN = DQN_Agent(env, test_env, lr, device, burn_in=burn_in, gamma=gamma, \
		alg2=alg2, replay_mem=replay_mem, batch_size=batch_size, \
		logdir=logdir, lr_v=lr_v)

	f = open(f'{logdir}/details.txt', 'w')
	f.write(f'\n\
	num_episodes = {num_episodes}	# Number of episodes to train on\n\
	ep_decay_in = {ep_decay_in}	# Epsilon will decay from eps_start to eps_end in ep_decay_in episodes\n\
	eps_start = {eps_start}\n\
	eps_end = {eps_end}\n\
	gamma = {gamma}\n\
	alg2 = {alg2}\n\
	lr = {lr}			# Learning rate for Q, M, V models\n\
	lr_v = {lr_v}		# Learning rate for V model\n\
	burn_in = {burn_in}		# Number of episodes added to replay memory on suniform policy at initiialization\n\
	init_learn = {init_learn}	# Number of times models are learnt with just burn in memory\n\
	replay_mem = {replay_mem}	# Replay memory size\n\
	batch_size = {batch_size}	# Batch size for training model when DQN.learn() is called\n\
	learn_freq = {learn_freq}	# Frequency of timesteps to call self.learn()\n\
	target_freq = {target_freq}	# Frequency of timesteps to update target networks\n\
	logdir = {logdir}\n\
	Alg1 Expert Penalty = {expert_penalty}\n\n\
	Neural network architecture = {DQN.Q.model}\n')
	f.close()

	ep_lengths, ep_rewards, Losses = DQN.train(num_episodes, ep_decay_in, eps_start, eps_end, \
		initial_learn=init_learn, learn_freq=learn_freq, target_freq=target_freq,\
			maxlen=maxlen, save_freq=save_freq)

	# Plotting variances, best_action map, state visitation, 
	variances, best_actions, state_visitation = DQN.evaluate_agent(num_episodes)
	DQN.plotLosses(ep_lengths, ep_rewards, Losses)

	# Training M and V models at the end
	# DQN.offline_training(5000, 5000, smoothing_number=250, target_freq=500)

if __name__ == '__main__':
	main(sys.argv)

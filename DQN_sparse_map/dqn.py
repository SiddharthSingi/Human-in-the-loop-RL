import os
import seaborn as sns
import random
import math
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from neural_model import NeuralNet
from replay import Replay_Memory

class QNet():
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, device, lr, logdir):
		# Define  network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		## Logs
		self.logdir = logdir
		self.lr = lr
		self.nA = 4
		self.model = NeuralNet()
		self.model_op = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		self.device = device
		self.model.to(self.device)

	def save_model_weights(self):
		# Helper function to save your model / weights.
		path = os.path.join(self.logdir, 'Qmodel.pth')
		torch.save(self.model.state_dict(), path)
		return path

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
		self.model.load_state_dict(torch.load(model_file))
	
	def load_model_weights(self, weight_file):
		# Optional Helper function to load model weights.
		return self.model.load_weights(weight_file)

	def choose_action(self, state, epsilon):
		'''
		Predicts the q values of a state and chooses an action based on epsilon value
		params:
			state in the format: [np.array([patch]), np.array(posn)], taken from Grid	
		Outputs: action to take
		'''
		patch = torch.from_numpy(np.expand_dims(state[0], axis=0)).to(self.device)
		posn = torch.from_numpy(np.expand_dims(state[1], axis=0)).to(self.device)
		x = [patch.float(), posn.float()]

		self.model.eval()	# Will notify batchnorm layers to work in eval mode
		with torch.no_grad():
			qvalues = self.model(x).view(-1)

		if random.random() < epsilon:
			action = np.random.randint(self.nA)
		else:
			action = torch.argmax(qvalues).item()
		return action

class MNet():
    # The network should take in state of the world as an input,
	# and output log M values of the actions available to the agent as the output.

	def __init__(self, device, lr, logdir):
		# Define  network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		## Logs
		self.logdir = logdir
		self.lr = lr
		self.model = NeuralNet()
		self.model_op = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		self.model.to(device)

	def save_model_weights(self):
		# Helper function to save your model / weights.
		path = os.path.join(self.logdir, 'Mmodel.pth')
		torch.save(self.model.state_dict(), path)
		return path

class VNet():
    # The network should take in state of the world as an input,
	# and output log variance of the actions available to the agent as the output.

	def __init__(self, device, lr, logdir):
		# Define  network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		## Logs
		self.logdir = logdir
		self.lr = lr
		self.model = NeuralNet()
		self.model_op = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		self.model.to(device)

	def save_model_weights(self):
		# Helper function to save your model / weights.
		path = os.path.join(self.logdir, 'Vmodel.pth')
		torch.save(self.model.state_dict(), path)
		return path

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
		self.model.load_state_dict(torch.load(model_file))

	def load_model_weights(self, weight_file):
		# Optional Helper function to load model weights.
		return self.model.load_weights(weight_file)

class DQN_Agent():
	'''
	DQN framework with Q net, M net, V net, training and testing protocols 
		params:
			alg2 (bool): Trains alg2 if True, else Vanilla Q learning
	'''

	def __init__(self, env, test_env, lr, device, burn_in=50, gamma=0.9, batch_size=64, alg2=True,\
					replay_mem=2000, logdir = 'DQN_sparse_map/'):
		self.env = env
		self.test_env = test_env
		self.nA = 4
		self.alg2 = alg2 
		self.device = device
		self.logdir = logdir
		model_f = os.path.join(self.logdir, 'models')
		plots_f = os.path.join(self.logdir, 'Plots')
		os.makedirs(model_f, exist_ok=True)
		os.makedirs(plots_f, exist_ok=True)

		self.Q = QNet(self.device, lr, model_f)
		self.target_Q = QNet(self.device, lr, model_f)

		self.criterion = nn.MSELoss()
		self.memory = Replay_Memory(self.device, batch_size=batch_size, memory_size=replay_mem)
		self.burn_in_memory(burn_in) # Inits memory

		if alg2:
			self.M = MNet(self.device, lr, model_f)
			self.target_M = MNet(self.device, lr, model_f)
			self.V = VNet(self.device, lr, model_f)

		self.training_episodes = 50
		self.trained_episodes = 0
		self.gamma = gamma
		self.batch_size = batch_size
		self.total_steps = 0
		self.T = 0
		self.expert_call = 0
		print('DQN Initialized')

	def learn(self):
		'''
		Samples experiences from replay buffer. 
		Train the Q, M, V models based on if self.alg2 is True
		'''

		# Setting all models to train mode
		self.Q.model.train()
		self.M.model.train()
		self.V.model.train()

		# Sample minibatch as tensors
		states, actions, rewards, next_states, dones = self.memory.sample_batch()

		batch_inds = torch.arange(self.batch_size, dtype=torch.int8)
		# Q(s,a)
		q_predicted = self.Q.model(states)
		q_predicted = q_predicted[batch_inds.long(), actions.long()]

		# r + gamma*max(Q(s',a'))
		with torch.no_grad():
			# Predicted values of next state from target model
			q_values_target = self.target_Q.model(next_states)
			max_q_targets, _ = torch.max(q_values_target, axis=1)

			# Final td target values to measure loss from
			q_targets = rewards + (1-dones.float())*self.gamma*max_q_targets

		# Q Loss and step
		QLoss = self.criterion(q_predicted, q_targets)
		self.Q.model_op.zero_grad()
		QLoss.backward()				
		self.Q.model_op.step()

		if self.alg2:
			# Algo2
			# M(s, a)
			# V(s, a)
			m_predicted = self.M.model(states)
			m_predicted = m_predicted[batch_inds.long(), actions.long()]
			v_predicted = self.V.model(states)
			v_predicted = v_predicted[batch_inds.long(), actions.long()]

			with torch.no_grad():
				# Predicted values of next state from target model
				m_values_target = self.target_M.model(next_states)
				max_m_targets, _ = torch.max(m_values_target, axis=1)

			# r**2 + 2*gamma*r*max(Q(s', a')) + (gamma**2)*max(M(s', a'))
			m_targets = rewards**2 + (1-dones.float())*2*self.gamma*rewards*max_q_targets\
						+ (1-dones.float())*(self.gamma**2)*max_m_targets

			# detach is used to avoid gradients flowing into M and Q when computing
			# loss for V model
			v_targets = m_predicted.detach() - q_predicted.detach()**2

			# M Loss and training
			MLoss = self.criterion(m_predicted, m_targets)
			self.M.model_op.zero_grad()
			MLoss.backward()
			self.M.model_op.step()

			# V Loss and training
			VLoss = self.criterion(v_predicted, v_targets)
			self.V.model_op.zero_grad()
			VLoss.backward()
			self.V.model_op.step()

		return QLoss.item(), MLoss.item(), VLoss.item()

	def train(self, num_episodes, decay_ep=False, eps_start=0.5, eps_end = 0.05, learn_freq=1000,\
		target_freq=10000, save_freq=200, eval_freq=10000, initial_learn = 1000, maxlen=200):
		'''
		Train DQN by decaying exploration scheme with target networks and replay buffer

		params:
			learn_freq: frequency of timesteps to call self.learn()
			eval_freq: frequency of timesteps to plot best_actions, variances and state visitation
			target_freq: frequency of timesteps to update target network
			save_freq: frequency of episodes to save Q, M, V models
			initial_learn: number of batches to call self.learn() on burn in memory
		'''
		
		ep_rewards = []
		ep_lengths = []
		Losses = {'Q': [], 'M': [], 'V': [], 'timestep': []}
		t = 0
		if not decay_ep: decay_ep=num_episodes
		lmda = math.log(eps_start/eps_end)/decay_ep

		# Updating the models based on burned in memory
		for _ in range(initial_learn):
			_, _, _ = self.learn()
		self.evaluate_agent()
		print('Initial Training done')

		print('Starting to play')
		for ep in range(num_episodes):
			self.epsilon = max(eps_start*math.exp(-lmda*ep), eps_end)
			print(f'Episode: {ep} has epsilon: {self.epsilon}')

			total_reward = 0	# Will be appended to ep_rewards
			ep_l = 0 			# Will be appended to ep_lengths

			state = self.env.reset() # state will consist of patch, and posn
			self.env.visited[int(state[1][0]), int(state[1][1])] += 1
			done = False
			print(f'Episode {ep} starting')

			while not done:
				
				# Choose action based on epsilon value
				action = self.Q.choose_action(state, self.epsilon)
				# print(action)
				next_state, reward, done = self.env.step(action)
				self.memory.cache(state, action, reward, next_state, done)
				state = next_state

				total_reward += reward
				ep_l += 1
				self.env.visited[int(state[1][0]), int(state[1][1])] += 1
				if ep_l==maxlen:
					done = True

				if t % learn_freq == 0:
					# This will sample experiences from replay buffer and train the models.
					qloss, mloss, vloss = self.learn()
					Losses['Q'].append(qloss)
					Losses['M'].append(mloss)
					Losses['V'].append(vloss)
					Losses['timestep'].append(t)
					# print(f'Learning. Qloss: {qloss}, Mloss: {mloss}, VLoss: {vloss}, timestep: {t}')

				if t % eval_freq == 0:
					# print(Losses)
					_, _, _ = self.evaluate_agent()
					self.plotLosses(ep_lengths, ep_rewards, Losses)
					print('Plots Updated in folder!')

				# Update Q and M target networks
				if t % target_freq == 0:
					self.target_Q.model.load_state_dict(self.Q.model.state_dict())
					if self.alg2:
						self.target_M.model.load_state_dict(self.M.model.state_dict())

				# Save models
				if ep % save_freq == 0:
					self.Q.save_model_weights()
					if self.alg2:
						self.M.save_model_weights()
						self.V.save_model_weights()
				t += 1
			print(f'Episode {ep} over')
			ep_rewards.append(total_reward)
			ep_lengths.append(ep_l)
		return ep_lengths, ep_rewards, Losses

	def evaluate_agent(self):
		'''
		Will plot the variance, best actions and state visitations using Q and V models
		'''

		# Setting all models to eval mode
		self.Q.model.eval()
		self.V.model.eval()

		plt.close('all')	# Close previous plt figures to avoid memory error
		rows, cols = np.indices((15,13))
		rows = rows.reshape(-1)
		cols = cols.reshape(-1)
		posns = np.stack((rows, cols), axis=1)

		# Will give a list of 210 elements, each having the patch and posn of all cells in the grid
		all_states = list(map(lambda x: self.env.yx_to_obs(x), posns))

		# Will convert all patches and posns into one tensor each to be passed to the models
		all_patches, all_posns = map(lambda x: torch.from_numpy(np.stack(x)).to(self.device), \
									zip(*all_states))

		# [Tensor_of_shape(210, 4, 3, 3), Tensor_of_shape(210, 2)]
		all_state_tensors = [all_patches.float(), all_posns.float()]

		# Q values and best actions
		qvalues = self.Q.model(all_state_tensors)
		best_actions = torch.argmax(qvalues, axis=1).reshape(-1)

		# Plotting Best actions
		best_actions_map = np.zeros((15,13))
		best_actions_map[rows, cols] = best_actions.cpu().numpy()

		# These states either have obstacles, traps or goals and their best action values
		# do not make any sense. Will be set to -1 to avoid confusion
		obstacle_x, obstacle_y = np.where(self.env.grid!=0)
		best_actions_map[obstacle_x, obstacle_y] = -1

		# Convert to arrows instead of numbers
		num_to_arrow = {0: u'\u2191', 1: u'\u2193', 2: u'\u2190', 3: u'\u2192', -1:'-1'}
		f = np.vectorize(lambda x: num_to_arrow[x])
		arrows = f(best_actions_map)

		fig, ax = plt.subplots(figsize=(8,8))
		ax.set_title('Best action in each state')
		np.save(os.path.join(self.logdir, 'Plots', 'Best_actions.npy'), best_actions_map)
		ax = sns.heatmap(best_actions_map, annot=arrows, fmt="s", cmap='cool', linewidths=.5)
		plt.savefig(os.path.join(self.logdir, 'Plots', 'Best_actions.jpg'))
		# plt.show()

		# Plotting Variances of every state
		variances = self.V.model(all_state_tensors)
		variances = variances[torch.arange(195), best_actions].reshape(15,13)

		# These states either have obstacles, traps or goals and their variance values
		# do not make any sense. Need to be zeroed
		variances[obstacle_x, obstacle_y] = 0

		fig, ax = plt.subplots(figsize=(20,20))
		ax.set_title('Variances values of every state')
		ax = sns.heatmap(variances.cpu().data.numpy().astype(int), annot=True, \
			fmt=".1g", cmap='cool', linewidths=1)
		np.save(os.path.join(self.logdir, 'Plots', 'Variances.npy'), variances.cpu().data.numpy().astype(int))
		plt.savefig(os.path.join(self.logdir, 'Plots', 'Variances.jpg'))
		# plt.show()

		# Plotting state visitation
		fig, ax = plt.subplots(figsize=(20,20))
		ax = sns.heatmap(self.env.visited, annot=False, fmt=".1g", cmap='cool', linewidths=1)
		ax.set_title('State visitation')
		np.save(os.path.join(self.logdir, 'Plots', 'State Visitation.npy'), self.env.visited)
		plt.savefig(os.path.join(self.logdir, 'Plots', 'State Visitation.jpg'))
		# plt.show()

		return variances.cpu().data.numpy(), best_actions_map, self.env.visited

	def burn_in_memory(self, burnin_ep=50):

		# Initialize replay memory with a burn_in number of episodes.
		for ep in range(burnin_ep):
			state = self.env.reset()
			done=False
			while not done:
				action = np.random.randint(self.nA)
				next_state, reward, done = self.env.step(action)

				self.memory.cache(state, action, reward, next_state, done)
				state = next_state

	def plotLosses(self, ep_lenghts, ep_rewards, Losses):
		
		plt.close('all')	# Close previous plt figures to avoid memory error
		# Visualizations
		# Plotting Rewards
		smoothing_number = 100
		x = np.arange(0, len(ep_rewards), smoothing_number)
		x = np.append(x, len(ep_rewards)-1)
		y = [np.average(ep_rewards[x[i]:x[i+1]]) for i in range(len(x)-1)]
		x = x[1:]

		fig, ax0 = plt.subplots(figsize=(8,6))
		ax0.set_title('Rewards per episode')
		ax0.plot(x, y)
		plt.savefig(os.path.join(self.logdir, 'Plots', 'rewards.jpg'), \
			bbox_inches ="tight",\
			dpi=250)
		# plt.show()


		# Plotting Episode Lengths
		x = np.arange(0, len(ep_lenghts), smoothing_number)
		x = np.append(x, len(ep_lenghts)-1)
		y = [np.average(ep_lenghts[x[i]:x[i+1]]) for i in range(len(x)-1)]
		x = x[1:]

		fig, ax0 = plt.subplots(figsize=(8,6))
		ax0.set_title('Episode Lengths')
		ax0.plot(x, y)
		plt.savefig(os.path.join(self.logdir, 'Plots', 'lengths.jpg'), \
			bbox_inches ="tight",\
			dpi=250)
		# plt.show()


		# Plotting Losses
		fig, ax = plt.subplots(figsize=(8,8))
		sns.lineplot(x=Losses['timestep'], y=Losses['Q'])
		ax.set_title('Q Losses')
		plt.savefig(os.path.join(self.logdir, 'Plots', 'QLosses.jpg'), \
			bbox_inches ="tight",\
			dpi=250)
		# plt.show()

		fig, ax = plt.subplots(figsize=(8,8))
		sns.lineplot(x=Losses['timestep'], y=Losses['M'])
		ax.set_title('M Losses')
		plt.savefig(os.path.join(self.logdir, 'Plots', 'MLosses.jpg'), \
			bbox_inches ="tight",\
			dpi=250)
		# plt.show()

		fig, ax = plt.subplots(figsize=(8,8))
		sns.lineplot(x=Losses['timestep'], y=Losses['V'])
		ax.set_title('V Losses')
		plt.savefig(os.path.join(self.logdir, 'Plots', 'VLosses.jpg'), \
			bbox_inches ="tight",\
			dpi=250)
		# plt.show()
	
		return

import os
import seaborn as sns
import random
import math
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from neural_model import NeuralNet
import pickle
from replay import Replay_Memory
from sparse_env import Grid

class QNet():
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, device, lr, logdir, alg2):
		# Define  network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		## Logs
		self.logdir = logdir
		self.lr = lr
		self.nA = 4 if alg2 else 5
		self.model = NeuralNet(self.nA)
		self.model_op = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		self.device = device
		self.model.to(self.device)

	def save_model_weights(self, ep_number, loss, dir_path=None):
		# Helper function to save your model / weights.
		path = os.path.join(self.logdir, f'Qmodel_{ep_number}.pth')
		if dir_path:
			path = os.path.join(dir_path, f'Qmodel_{ep_number}.pth')
		torch.save({
			'ep_number': ep_number,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.model_op.state_dict(),
			'loss': loss
		}, path)
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

	def __init__(self, device, lr, logdir, alg2):
		# Define  network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		## Logs
		self.logdir = logdir
		self.lr = lr
		self.nA = 4 if alg2 else 5
		self.model = NeuralNet(self.nA)
		self.model_op = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		self.model.to(device)

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
		self.model.load_state_dict(torch.load(model_file))

	def save_model_weights(self, ep_number, loss):
		# Helper function to save your model / weights.
		path = os.path.join(self.logdir, f'Mmodel_{ep_number}.pth')
		torch.save({
			'ep_number': ep_number,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.model_op.state_dict(),
			'loss': loss
		}, path)
		return path

class VNet():
	# The network should take in state of the world as an input,
	# and output log variance of the actions available to the agent as the output.

	def __init__(self, device, lr, logdir, alg2):
		# Define  network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		## Logs
		self.logdir = logdir
		self.lr = lr
		self.nA = 4 if alg2 else 5
		self.model = NeuralNet(self.nA)
		self.model_op = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		self.model.to(device)

	def save_model_weights(self, ep_number, loss):
		# Helper function to save your model / weights.
		path = os.path.join(self.logdir, f'Vmodel_{ep_number}.pth')
		torch.save({
			'ep_number': ep_number,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.model_op.state_dict(),
			'loss': loss
		}, path)
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

	def __init__(self, env, test_env, lr, device, burn_in=50, gamma=0.9, batch_size=64, alg2=True,
					replay_mem=2000, logdir='DQN_sparse_map-extra/', lr_v=1e-4):
		self.env = env
		self.test_env = test_env
		self.nA = 4
		self.alg2 = alg2 
		self.device = device
		self.logdir = logdir
		model_f = os.path.join(self.logdir, 'models')
		plots_f = os.path.join(self.logdir, 'Plots')
		qvalues_f = os.path.join(self.logdir, 'Qvalues')
		os.makedirs(qvalues_f, exist_ok=True)
		os.makedirs(model_f, exist_ok=True)
		os.makedirs(plots_f, exist_ok=True)

		self.Q = QNet(self.device, lr, model_f, alg2)
		self.target_Q = QNet(self.device, lr, model_f, alg2)

		self.criterion = nn.MSELoss()
		self.memory = Replay_Memory(self.device, batch_size=batch_size, memory_size=replay_mem)
		self.burn_in_memory(burn_in) # Inits memory

		if alg2:
			self.M = MNet(self.device, lr, model_f, alg2)
			self.target_M = MNet(self.device, lr, model_f, alg2)
			self.V = VNet(self.device, lr_v, model_f, alg2)

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
		if self.alg2:
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

				# Predicted values of current state from target model M and Q
				# for variance model true values
				m_cur_target = self.target_M.model(states)
				m_cur_target = m_cur_target[batch_inds.long(), actions.long()]

				q_cur_target = self.target_Q.model(states)
				q_cur_target = q_cur_target[batch_inds.long(), actions.long()]

			# r**2 + 2*gamma*r*max(Q(s', a')) + (gamma**2)*max(M(s', a'))
			m_targets = rewards**2 + (1-dones.float())*2*self.gamma*rewards*max_q_targets\
						+ (1-dones.float())*(self.gamma**2)*max_m_targets

			# m_cur_target and q_cur_target were built inside torch.no_grad to avoid gradients flowing
			# loss for V model
			v_targets = m_cur_target - q_cur_target**2

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
		
		else:
			return QLoss.item(), None, None

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
		Losses = {'Q': [], 'M': [], 'V': [], 'episode': []}
		t = 0
		if not decay_ep: decay_ep=num_episodes
		lmda = math.log(eps_start/eps_end)/decay_ep

		# Updating the models based on burned in memory
		for _ in range(initial_learn):
			_, _, _ = self.learn()
		self.evaluate_agent(0)
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
				action = self.target_Q.choose_action(state, self.epsilon)
				# print(action)
				next_state, reward, done = self.env.step(action)
				self.memory.cache(state, action, reward, next_state, done)
				state = next_state
				t += 1
				total_reward += reward
				ep_l += 1
				self.env.visited[int(state[1][0]), int(state[1][1])] += 1
				if ep_l==maxlen:
					done = True

				if t % learn_freq == 0:
					# This will sample experiences from replay buffer and train the models.
					qloss, mloss, vloss = self.learn()
					Losses['Q'].append(qloss)
					Losses['episode'].append(t)
					if self.alg2:
						Losses['M'].append(mloss)
						Losses['V'].append(vloss)
					# print(f'Learning. Qloss: {qloss}, Mloss: {mloss}, VLoss: {vloss}, timestep: {t}')

				# Update Q and M target networks
				if t % target_freq == 0:
					self.target_Q.model.load_state_dict(self.Q.model.state_dict())
					if self.alg2:
						self.target_M.model.load_state_dict(self.M.model.state_dict())

			# Save models
			if ep % save_freq == 0:
				_, _, _ = self.evaluate_agent(ep)
				self.plotLosses(ep_lengths, ep_rewards, Losses)
				qloss, mloss, vloss = self.learn()
				self.target_Q.save_model_weights(ep, qloss)
				print('Plots and arrays updated in folder!')

				Losses['Q'].append(qloss)
				Losses['episode'].append(ep)

				if self.alg2:
					Losses['M'].append(mloss)
					Losses['V'].append(vloss)
					self.target_M.save_model_weights(ep, mloss)
					self.V.save_model_weights(ep, vloss)
				
			print(f'Episode {ep} over')
			ep_rewards.append(total_reward)
			ep_lengths.append(ep_l)

		with open(os.path.join(self.logdir, 'Plots', 'Losses.pkl'), 'wb') as f:
			pickle.dump(Losses, f)

		return ep_lengths, ep_rewards, Losses

	def evaluate_agent(self, ep_num):
		'''
		Will plot the variance, best actions and state visitations using Q and V models
		'''

		m, n = self.env.grid.shape

		# Setting all models to eval mode
		self.Q.model.eval()

		plt.close('all')	# Close previous plt figures to avoid memory error
		rows, cols = np.indices((m, n))
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

		# These states either have obstacles, traps or goals and their best action values
		# do not make any sense. Will be set to -1 to avoid confusion
		obstacle_x, obstacle_y = np.where(self.env.grid!=0)

		# Q values and best actions
		qvalues = self.target_Q.model(all_state_tensors)
		# Saving Q values
		qvalues_arr = qvalues.reshape(m, n, -1)
		qvalues_arr[obstacle_x, obstacle_y, :] = -100
		np.save(os.path.join(self.logdir, 'Qvalues', f'Qvalues_{ep_num}.npy'), qvalues_arr.cpu().data.numpy().astype(float))

		best_actions = torch.argmax(qvalues, axis=1).reshape(-1)

		# Plotting Best actions
		best_actions_map = np.zeros((m, n))
		best_actions_map[rows, cols] = best_actions.cpu().numpy()
		best_actions_map[obstacle_x, obstacle_y] = -1

		# Convert to arrows instead of numbers
		num_to_arrow = {0: u'\u2191', 1: u'\u2193', 2: u'\u2190', 3: u'\u2192', -1:'-1', -3: 'G', 4: 'E'}
		f = np.vectorize(lambda x: num_to_arrow[x])
		arrows = f(best_actions_map)

		fig, ax = plt.subplots(figsize=(8,8))
		ax.set_title('Best action in each state')
		np.save(os.path.join(self.logdir, 'Plots', 'Best_actions.npy'), best_actions_map)
		ax = sns.heatmap(best_actions_map, annot=arrows, fmt="s", cmap='cool', linewidths=.5)
		plt.savefig(os.path.join(self.logdir, 'Plots', 'Best_actions.jpg'))
		# plt.show()

		# Plotting state visitation
		fig, ax = plt.subplots(figsize=(20,20))
		ax = sns.heatmap(self.env.visited, annot=False, fmt=".1g", cmap='cool', linewidths=1)
		ax.set_title('State visitation')
		np.save(os.path.join(self.logdir, 'Plots', 'State Visitation.npy'), self.env.visited)
		plt.savefig(os.path.join(self.logdir, 'Plots', 'State Visitation.jpg'))
		# plt.show()

		if self.alg2:
			self.M.model.eval()
			self.V.model.eval()

			# Plotting Moment of returns of every state
			moments = self.M.model(all_state_tensors)
			moments = moments[torch.arange(m*n), best_actions].reshape(m, n)

			# These states either have obstacles, traps or goals and their variance values
			# do not make any sense. Need to be zeroed
			moments[obstacle_x, obstacle_y] = 0

			fig, ax = plt.subplots(figsize=(20,20))
			ax.set_title('Predicted moment of return of every state')
			ax = sns.heatmap(moments.cpu().data.numpy().astype(int), annot=True, \
				fmt=".0f", cmap='cool', linewidths=1)
			np.save(os.path.join(self.logdir, 'Plots', 'Moments.npy'), moments.cpu().data.numpy().astype(int))
			plt.savefig(os.path.join(self.logdir, 'Plots', 'Moments.jpg'))
			# plt.show()


			# Plotting Variances of every state
			variances = self.V.model(all_state_tensors)
			variances = variances[torch.arange(m*n), best_actions].reshape(m, n)

			# These states either have obstacles, traps or goals and their variance values
			# do not make any sense. Need to be zeroed
			variances[obstacle_x, obstacle_y] = 0

			fig, ax = plt.subplots(figsize=(20,20))
			ax.set_title('Variances values of every state')
			ax = sns.heatmap(variances.cpu().data.numpy().astype(int), annot=True, \
				fmt=".0f", cmap='cool', linewidths=1)
			np.save(os.path.join(self.logdir, 'Plots', 'Variances.npy'), variances.cpu().data.numpy().astype(int))
			plt.savefig(os.path.join(self.logdir, 'Plots', 'Variances.jpg'))
			# plt.show()

			return variances.cpu().data.numpy(), best_actions_map, self.env.visited
		
		else:
			return None, best_actions_map, self.env.visited

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
		'''
		Plots
		Rewards, Episode length, Q, M, V Losses
			vs episode number
		'''
		
		plt.close('all')	# Close previous plt figures to avoid memory error
		# Visualizations
		# Plotting Rewards
		smoothing_number = 100
		x = np.arange(0, len(ep_rewards), smoothing_number)
		x = np.append(x, len(ep_rewards))
		y = [np.average(ep_rewards[x[i]:x[i+1]]) for i in range(len(x)-1)]
		x = x[1:]

		fig, ax0 = plt.subplots(figsize=(15,6))
		ax0.set_title('Returns per episode')
		ax0.plot(x, y)
		plt.savefig(os.path.join(self.logdir, 'Plots', 'returns.jpg'), \
			bbox_inches ="tight",\
			dpi=250)
		# plt.show()


		# Plotting Episode Lengths
		x = np.arange(0, len(ep_lenghts), smoothing_number)
		x = np.append(x, len(ep_lenghts))
		y = [np.average(ep_lenghts[x[i]:x[i+1]]) for i in range(len(x)-1)]
		x = x[1:]

		fig, ax0 = plt.subplots(figsize=(15,6))
		ax0.set_title('Episode Lengths')
		ax0.plot(x, y)
		plt.savefig(os.path.join(self.logdir, 'Plots', 'lengths.jpg'), \
			bbox_inches ="tight",\
			dpi=250)
		# plt.show()


		# Plotting Losses
		smoothing_number = 50

		fig, ax = plt.subplots(figsize=(15,6))
		plotlist = Losses['Q']
		x = np.arange(0, len(plotlist), smoothing_number)
		x = np.append(x, len(plotlist))
		y = [np.average(plotlist[x[i]:x[i+1]]) for i in range(len(x)-1)]
		x = x[1:]
		# x = Losses['episode']
		# y = plotlist
		sns.color_palette('muted')
		sns.lineplot(x=x, y=y)
		ax.set_title('Q Losses')
		plt.savefig(os.path.join(self.logdir, 'Plots', 'QLosses.jpg'), \
			bbox_inches ="tight",\
			dpi=250)
		# plt.show()

		if self.alg2:
			fig, ax = plt.subplots(figsize=(15,6))
			plotlist = Losses['M']
			x = np.arange(0, len(plotlist), smoothing_number)
			x = np.append(x, len(plotlist))
			y = [np.average(plotlist[x[i]:x[i+1]]) for i in range(len(x)-1)]
			x = x[1:]
			# x = Losses['episode']
			# y = plotlist
			sns.color_palette('muted')
			sns.lineplot(x=x, y=y)
			ax.set_title('M Losses')
			plt.savefig(os.path.join(self.logdir, 'Plots', 'MLosses.jpg'), \
				bbox_inches ="tight",\
				dpi=250)
			# plt.show()

			fig, ax = plt.subplots(figsize=(15,6))
			plotlist = Losses['V']
			x = np.arange(0, len(plotlist), smoothing_number)
			x = np.append(x, len(plotlist))
			y = [np.average(plotlist[x[i]:x[i+1]]) for i in range(len(x)-1)]
			x = x[1:]
			# x = Losses['episode']
			# y = plotlist
			sns.color_palette('muted')
			sns.lineplot(x=x, y=y)
			ax.set_title('V Losses')
			plt.savefig(os.path.join(self.logdir, 'Plots', 'VLosses.jpg'), \
				bbox_inches ="tight",\
				dpi=250)
			# plt.show()
	
		return

	def offline_training(self, m_learn, vlearn, qlearn, smoothing_number, target_freq=1000, save_freq=1000):
		'''
		Trains and plots the M and V models offline at the end
		mlearn: number of times to call learn to train M model
		vlearn: number of times to call learn to train V model
		
		If alg2 is False then only Q model is trained
		'''
		
		if self.alg2:
			# Setting all models to train mode
			self.Q.model.train()
			self.M.model.train()
			self.V.model.train()

			M_losses = []
			V_losses = []

			m, n = self.env.grid.shape
			
			# Training M model
			for i in range(m_learn):
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


				# M Loss and training
				MLoss = self.criterion(m_predicted, m_targets)
				self.M.model_op.zero_grad()
				MLoss.backward()
				self.M.model_op.step()
				M_losses.append(MLoss.item())

				if i%target_freq==0:
					self.target_M.model.load_state_dict(self.M.model.state_dict())
			
			# Training V model
			for _ in range(vlearn):

				# Sample minibatch as tensors
				states, actions, rewards, next_states, dones = self.memory.sample_batch()
				batch_inds = torch.arange(self.batch_size, dtype=torch.int8)

				# Algo2
				# M(s, a)
				# V(s, a)
				v_predicted = self.V.model(states)
				v_predicted = v_predicted[batch_inds.long(), actions.long()]

				with torch.no_grad():
					# Predicted values of current state from target model M and Q
					# for variance model true values
					m_cur_target = self.target_M.model(states)
					m_cur_target = m_cur_target[batch_inds.long(), actions.long()]

					q_cur_target = self.target_Q.model(states)
					q_cur_target = q_cur_target[batch_inds.long(), actions.long()]

				# m_cur_target and q_cur_target were built inside torch.no_grad to avoid gradients flowing
				# loss for V model
				v_targets = m_cur_target - q_cur_target**2

				# V Loss and training
				VLoss = self.criterion(v_predicted, v_targets)
				self.V.model_op.zero_grad()
				VLoss.backward()
				self.V.model_op.step()
				V_losses.append(VLoss.item())

			# Save the M and V models
			self.M.save_model_weights()
			self.V.save_model_weights()	

			'''
			Will plot the variance, best actions and state visitations using Q and V models
			'''
			os.makedirs(os.path.join(self.logdir, 'Plots', 'offline_trained'), exist_ok=True)
			# Setting all models to eval mode
			self.Q.model.eval()
			self.V.model.eval()

			m, n = self.env.grid.shape

			plt.close('all')	# Close previous plt figures to avoid memory error
			rows, cols = np.indices((m, n))
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
			best_actions_map = np.zeros((m, n))
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
			# np.save(os.path.join(self.logdir, 'Plots', 'Best_actions.npy'), best_actions_map)
			ax = sns.heatmap(best_actions_map, annot=arrows, fmt="s", cmap='cool', linewidths=.5)
			plt.savefig(os.path.join(self.logdir, 'Plots', 'offline_trained', 'Best_actions.jpg'))
			# plt.show()

			# Plotting Variances of every state
			variances = self.V.model(all_state_tensors)
			variances = variances[torch.arange(m*n), best_actions].reshape(m, n)

			# These states either have obstacles, traps or goals and their variance values
			# do not make any sense. Need to be zeroed
			variances[obstacle_x, obstacle_y] = 0

			fig, ax = plt.subplots(figsize=(10,10))
			ax.set_title('Variances values of every state')
			ax = sns.heatmap(variances.cpu().data.numpy().astype(int), annot=True, \
				fmt=".0f", cmap='cool', linewidths=1)
			# np.save(os.path.join(self.logdir, 'Plots', 'Variances.npy'), variances.cpu().data.numpy().astype(int))
			plt.savefig(os.path.join(self.logdir, 'Plots', 'offline_trained', 'Variances.jpg'))
			# plt.show()

			# Plotting state visitation
			fig, ax = plt.subplots(figsize=(20,20))
			ax = sns.heatmap(self.env.visited, annot=False, fmt=".1g", cmap='cool', linewidths=1)
			ax.set_title('State visitation')
			# np.save(os.path.join(self.logdir, 'Plots', 'State Visitation.npy'), self.env.visited)
			plt.savefig(os.path.join(self.logdir, 'Plots', 'offline_trained', 'State Visitation.jpg'))
			# plt.show()

			'''
			Plotting M and V Losses while offline training
			Saving Q, M, V models for evaluation later
			'''
			plt.close('all')	
			
			# Close previous plt figures to avoid memory error
			# Visualizations

			# Plotting M Losses
			x = np.arange(0, len(M_losses), smoothing_number)
			x = np.append(x, len(M_losses))
			y = [np.average(M_losses[x[i]:x[i+1]]) for i in range(len(x)-1)]
			x = x[1:]

			fig, ax0 = plt.subplots(figsize=(15,6))
			ax0.set_title('M Losses - Offline training')
			ax0.plot(x, y)
			plt.savefig(os.path.join(self.logdir, 'Plots', 'offline_trained', 'MLosses-offline.jpg'), \
				bbox_inches ="tight",\
				dpi=250)


			# Plotting V Losses
			x = np.arange(0, len(V_losses), smoothing_number)
			x = np.append(x, len(V_losses))
			y = [np.average(V_losses[x[i]:x[i+1]]) for i in range(len(x)-1)]
			x = x[1:]

			fig, ax0 = plt.subplots(figsize=(15,6))
			ax0.set_title('V Losses - Offline training')
			ax0.plot(x, y)
			plt.savefig(os.path.join(self.logdir, 'Plots', 'offline_trained', 'VLosses-offline.jpg'), \
				bbox_inches ="tight",\
				dpi=250)

			# Saving, Q, M, V models
			path = os.path.join(self.logdir, 'Plots', 'offline_trained')
			torch.save(self.Q.model, os.path.join(path, 'Qmodel.pt'))
			torch.save(self.M.model, os.path.join(path, 'Mmodel.pt'))
			torch.save(self.V.model, os.path.join(path, 'Vmodel.pt'))

		else:
			self.Q.model.train()
			self.target_Q.model.train()
			Q_losses = []
			os.makedirs(os.path.join(self.logdir, 'offline_trained'), exist_ok=True)
			os.makedirs(os.path.join(self.logdir, 'offline_trained', 'Qvalues'), exist_ok=True)
			os.makedirs(os.path.join(self.logdir, 'offline_trained', 'Plots'), exist_ok=True)
			os.makedirs(os.path.join(self.logdir, 'offline_trained', 'models'), exist_ok=True)

			for i in range(qlearn):
				# Sample minibatch as tensors
				states, actions, rewards, next_states, dones = self.memory.sample_batch()

				batch_inds = torch.arange(self.batch_size, dtype=torch.int8)
				# Q(s,a)
				q_predicted = self.Q.model(states)
				q_predicted = q_predicted[batch_inds.long(), actions.long()]

				# r + gamma*max(Q(s',a'))
				with torch.no_grad():
					# Predicted values of next state from target model
					q_next = self.target_Q.model(next_states)
					q_next = q_next.max(dim=1)[0]
					# Target Q values
					q_targets = rewards + (1-dones.float())*self.gamma*q_next

				# Compute loss
				loss = self.criterion(q_predicted, q_targets)
				Q_losses.append(loss.item())

				# Backpropagation
				self.Q.model_op.zero_grad()
				loss.backward()
				self.Q.model_op.step()

				# Update target network
				if i % target_freq == 0:
					self.target_Q.model.load_state_dict(self.Q.model.state_dict())
				
				if i % save_freq == 0:

					self.Q.model.eval()
					self.target_Q.model.eval()
					# Save Q values
					m, n = self.env.grid.shape
					plt.close('all')	# Close previous plt figures to avoid memory error
					rows, cols = np.indices((m, n))
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
					qvalues = self.target_Q.model(all_state_tensors)

					best_actions = torch.argmax(qvalues, axis=1).reshape(-1)

					# Plotting Best actions
					best_actions_map = np.zeros((m, n))
					best_actions_map[rows, cols] = best_actions.cpu().numpy()

					# These states either have obstacles, traps or goals and their best action values
					# do not make any sense. Will be set to -1 to avoid confusion
					obstacle_x, obstacle_y = np.where(self.env.grid!=0)
					best_actions_map[obstacle_x, obstacle_y] = -1

					# Saving Q values
					qvalues_arr = qvalues.reshape(m, n, -1)
					qvalues_arr[obstacle_x, obstacle_y, :] = -100
					np.save(os.path.join(self.logdir, 'offline_trained', 'Qvalues', f'Qvalues_{i}.npy'), qvalues_arr.cpu().data.numpy().astype(float))

					# Convert to arrows instead of numbers
					num_to_arrow = {0: u'\u2191', 1: u'\u2193', 2: u'\u2190', 3: u'\u2192', -1:'-1', 4: 'E'}
					f = np.vectorize(lambda x: num_to_arrow[x])
					arrows = f(best_actions_map)

					fig, ax = plt.subplots(figsize=(8,8))
					ax.set_title('Best action in each state')
					# np.save(os.path.join(self.logdir, 'Plots', 'Best_actions.npy'), best_actions_map)
					ax = sns.heatmap(best_actions_map, annot=arrows, fmt="s", cmap='cool', linewidths=.5)
					plt.savefig(os.path.join(self.logdir, 'offline_trained', 'Plots', 'Best_actions.jpg'))
					# plt.show()

					# Saving models
					self.target_Q.save_model_weights(i//save_freq, loss, \
				      os.path.join(self.logdir, 'offline_trained', 'models'))

					self.Q.model.train()
					self.target_Q.model.train()

			# Plotting losses
			plt.close('all')
			x = np.arange(0, len(Q_losses), smoothing_number)
			x = np.append(x, len(Q_losses))
			y = [np.average(Q_losses[x[i]:x[i+1]]) for i in range(len(x)-1)]
			x = x[1:]

			fig, ax0 = plt.subplots(figsize=(15,6))
			ax0.set_title('Q Losses - Offline training')
			ax0.plot(x, y)
			plt.savefig(os.path.join(self.logdir, 'offline_trained', 'Plots', 'QLosses-offline.jpg'), \
				bbox_inches ="tight",\
				dpi=250)

	def check_model_values(self, modelsdir, posn, alg2):
		'''
		Takes in models directory and prints the values of Q, M, V models at a specific position
		'''

		env = Grid(patch_size=5)
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

		Q = QNet(device, 1e-4, '15k/v5/', alg2)	# logdir does not matter here
		Q.load_model(os.path.join(modelsdir, 'Qmodel.pth'))
		Q.model.eval()
		actions = 5

		if alg2:
			M = MNet(device, 1e-4, '15k/v5/')
			M.load_model(os.path.join(modelsdir, 'Mmodel.pth'))
			V = VNet(device, 1e-4, '15k/v5/')
			V.load_model(os.path.join(modelsdir, 'Vmodel.pth'))
			M.model.eval()
			V.model.eval()
			actions = 4

		m, n = self.env.grid.shape

		rows, cols = np.indices((m, n))
		rows = rows.reshape(-1)
		cols = cols.reshape(-1)
		posns = np.stack((rows, cols), axis=1)

		# Will give a list of 210 elements, each having the patch and posn of all cells in the grid
		all_states = list(map(lambda x: env.yx_to_obs(x), posns))

		# Will convert all patches and posns into one tensor each to be passed to the models
		all_patches, all_posns = map(lambda x: torch.from_numpy(np.stack(x)).to(device), \
									zip(*all_states))

		# [Tensor_of_shape(210, 4, 3, 3), Tensor_of_shape(210, 2)]
		all_state_tensors = [all_patches.float(), all_posns.float()]

		# Q values and best actions
		qvalues = Q.model(all_state_tensors)
		best_actions = torch.argmax(qvalues, axis=1).reshape(-1)

		# Plotting Best actions
		best_actions_map = np.zeros((m, n))
		best_actions_map[rows, cols] = best_actions.cpu().numpy()

		# These states either have obstacles, traps or goals and their best action values
		# do not make any sense. Will be set to -1 to avoid confusion
		obstacle_x, obstacle_y = np.where(env.grid!=0)
		best_actions_map[obstacle_x, obstacle_y] = -1

		qvalues = qvalues.reshape(m, n, actions)

		if alg2:
			# Plotting Variances of every state
			variances = V.model(all_state_tensors)
			mvalues = M.model(all_state_tensors)

			mvalues = mvalues.reshape(m, n, actions)
			variances = variances.reshape(m, n, actions)

		print(f'Q Values: {qvalues[posn[0], posn[1]]}')
		if alg2:
			print(f'M Values: {mvalues[posn[0], posn[1]]}')
			print(f'V Values: {variances[posn[0], posn[1]]}')

	def policy_rollout(self, num_episodes, max_steps, qtable_f=False):
		"""
		Running Monte Carlo for average return we get from the action policy
		num_episodes: to run for Monte Carlo
		smoothing: smoothing number for reward plotting
		qtable_path: path for best_action.npy to use if you do not want to train again
		"""
		if type(qtable_f) == str:
			q_arr = np.load(qtable_f)
		else:
			q_arr = qtable_f
		best_actions = np.argmax(q_arr, axis=2)
		rewards_list = []
		ep_expert_calls = []
		max_steps_reached = []
		for ep in range(num_episodes):
			done = False
			patch, posn = self.env.reset()
			tot_reward = 0
			expert_called = 0
			step = 0
			posns_reached = [posn.tolist()]
			while not done and step<max_steps:
				action = best_actions[posn[0], posn[1]].astype(int)
				if action==4:
					expert_called+=1
				obs, reward, done = self.env.step(action)
				tot_reward += reward
				posn = obs[1]
				if posn.tolist() in posns_reached:
					done = True
				else:
					posns_reached.append(posn.tolist())
				# print(step)
				step+=1
			max_steps_reached.append(not done)
			rewards_list.append(tot_reward)
			ep_expert_calls.append(expert_called)

		avg_expert_calls = sum(ep_expert_calls)/len(ep_expert_calls)
		avg_return = sum(rewards_list)/len(rewards_list)
		percent_stuck = sum(max_steps_reached)/len(max_steps_reached)
		# print('Average number of expert calls made: ', avg_expert_calls)
		# print('Average return using best action plot: ', avg_return)
		# print('Percentage of episodes stuck in infinity loop: ', percent_stuck)

		return avg_expert_calls, avg_return, percent_stuck

	def zhanpeng_plot(self, num_episodes, max_steps, max_expert_calls, qtable_f=False, ):
		"""
		Running Monte Carlo for average return we get from the action policy
		num_episodes: to run for Monte Carlo
		smoothing: smoothing number for reward plotting
		qtable_path: path for best_action.npy to use if you do not want to train again
		"""
		if type(qtable_f) == str:
			q_arr = np.load(qtable_f)
		else:
			q_arr = qtable_f
		best_actions = np.argmax(q_arr, axis=2)
		rewards_list = []
		ep_expert_calls = []
		max_steps_reached = []
		for ep in range(num_episodes):
			done = False
			patch, posn = self.env.reset()
			tot_reward = 0
			expert_called = 0
			step = 0
			posns_reached = [posn.tolist()]
			while not done and step<max_steps:
				# If expert called once, then choose best action from remaininng actions
				if expert_called==max_expert_calls:
					action = np.argmax(q_arr[posn[0], posn[1], :4]).astype(int)
				else:
					action = best_actions[posn[0], posn[1]].astype(int)
				if action==4:
					expert_called+=1
				obs, reward, done = self.env.step(action)
				tot_reward += reward
				posn = obs[1]
				if posn.tolist() in posns_reached:
					done = True
				else:
					posns_reached.append(posn.tolist())
				# print(step)
				step+=1
			max_steps_reached.append(not done)
			rewards_list.append(tot_reward)
			ep_expert_calls.append(expert_called)

		avg_expert_calls = sum(ep_expert_calls)/len(ep_expert_calls)
		avg_return = sum(rewards_list)/len(rewards_list)
		percent_stuck = sum(max_steps_reached)/len(max_steps_reached)
		# print('Average number of expert calls made: ', avg_expert_calls)
		# print('Average return using best action plot: ', avg_return)
		# print('Percentage of episodes stuck in infinity loop: ', percent_stuck)

		return avg_expert_calls, avg_return, percent_stuck

	def get_action_table(self, variances, threshold, best_actions):
		"""
		returns an action table where action=4 if variance is above the threshold
		qtable: shape=(10,10,4,3) Learned from ALG2
		threshold: threshold for the variance value
		"""
		m, n = self.env.grid.shape
		rows, cols = np.indices((m, n))
		rows = rows.reshape(-1)
		cols = cols.reshape(-1)
		# best_actions = np.argmax(qtable[:, :, :, 0], axis=2).reshape(-1)
		# variances = qtable[:, :, :, 2][rows, cols, best_actions]
		action_table = np.where(variances<threshold, best_actions, 4).reshape(m, n)

		# # Visualize a policy
		# if threshold==3800:
		#     fig, ax = plt.subplots()
		#     ax = sns.heatmap(action_table, annot=True, fmt=".1f", cmap='cool', linewidths=.5)
		#     plt.show()

		return action_table

	def threshold_rollout(self, num_episodes, thresh_spacing, dir_f, max_steps=100):
		"""
		Will create a policy where the expert is called if the variance is greater than the threshold
		This policy is rolled on the environment for num_episodes episodes and the average of the return is
		plotted for each threshold.
		num_episodes: Average return is calculated for each policy for num_episodes
		qtable_f: filename for the qtable that must have been trained using the ALG2 class, shape (10,10,4,3)

		We use this to compute the 
		"""
		# For each state and action we have a q-value, 2nd reward moment and the variance using Bellman eqns
		variances = np.load(os.path.join(dir_f, 'Variances.npy'))
		best_actions = np.load(os.path.join(dir_f, 'Best_actions.npy'))
		max_thresh = np.max(variances)
		thresholds = []
		returns = []
		expert_calls = []
		max_steps_reached = []

		for thresh in range(0, max_thresh, thresh_spacing):
			action_table = self.get_action_table(variances, thresh, best_actions)
			cur_thresh_rewards = []
			cur_thresh_expert_calls = []
			cur_thresh_max_steps_reached = []
			for ep in range(num_episodes):
				done = False
				patch, posn = self.env.reset()
				tot_reward = 0
				expert_called = 0
				step = 0
				posns_reached = [posn.tolist()]
				while not done and step<max_steps:
					action = action_table[posn[0], posn[1]].astype(int)
					if action==4:
						expert_called+=1
					obs, reward, done = self.env.step(action)
					tot_reward += reward
					posn = obs[1]
					if posn.tolist() in posns_reached:
						done = True
					else:
						posns_reached.append(posn.tolist())
					# print(step)
					step+=1
				cur_thresh_rewards.append(tot_reward)
				cur_thresh_expert_calls.append(expert_called)
				cur_thresh_max_steps_reached.append(not done)

			avg_expert_calls = sum(cur_thresh_expert_calls)/len(cur_thresh_expert_calls)
			avg_return = sum(cur_thresh_rewards)/len(cur_thresh_rewards)
			percent_stuck = sum(cur_thresh_max_steps_reached)/len(cur_thresh_max_steps_reached)

			thresholds.append(thresh)
			returns.append(avg_return)
			expert_calls.append(avg_expert_calls)
			max_steps_reached.append(percent_stuck)
		
		print('Thresholds: ', thresholds)
		print('Average number of expert calls made: ', expert_calls)
		print('Average return using best action plot: ', returns)
		print('Percentage of episodes stuck in infinity loop: ', max_steps_reached)

		# # Saving the thresholdeds and their returns
		# thresh_rewards = np.array(thresh_rewards)
		# thresholds = np.array(thresholds)
		# os.makedirs(os.path.join(self.fname, 'thresh_rollout'))
		# np.savez(os.path.join(self.fname, 'thresh_rollout', 'thresholdrollout.npz'), thresh_rewards, thresholds)


		# # Policy of the best threshold
		# high_threshold = thresholds[np.argmax(thresh_rewards)]
		# best_action_table = self.get_action_table(qtable, high_threshold)
		# fig, ax = plt.subplots()
		# ax = sns.heatmap(best_action_table, annot=True, fmt=".1f", cmap='cool', linewidths=.5)
		# plt.savefig(os.path.join(self.fname, 'thresh_rollout', 'Best threshold policy.jpg'), \
		# 	bbox_inches ="tight",\
		# 	dpi=250)

		# # Visualizing the average return for every threshold
		# fig, ax = plt.subplots()
		# ax = sns.barplot(x=thresholds, y=thresh_rewards)
		# ax.set_xticks(np.arange(0, len(thresholds)+1, 10))
		# ax.set_xlabel('Thresholds')
		# ax.set_ylabel('Average Return (1000 episodes)')
		# ax.text(1, 0.2, f'Best threshold: {high_threshold}\n Highest average return: {np.max(thresh_rewards)}', \
		# 	horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
		# # ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
		# plt.savefig(os.path.join(self.fname, 'thresh_rollout', 'Thresholded action returns.jpg'), \
		# 	bbox_inches ="tight",\
		# 	dpi=250)

		return
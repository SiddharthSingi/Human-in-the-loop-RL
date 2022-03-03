from environment import Grid, ExpertGrid
import numpy as np
from matplotlib import pyplot as plt

class VanillaAgent():
    def __init__(self, table_size, grid) -> None:
        
        self.qtable = np.zeros(shape=table_size)    # Q-Learning table
        self.epsilon = 0.1  # Exploration epsilon
        self.grid = grid
        self.action_space = 4
        self.lr = 0.05
        self.gamma = 0.9
        self.rewards_list = []
        self.smoothing_num = 1000
        
    # Returns action to choose based on state and self.epsilon
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.qtable[state[0], state[1], :])
        return action

    # Updates the q table based on one state/action pair
    def learn(self, prev_state, action, reward, new_state):
        # Current q value
        qsa = self.qtable[prev_state[0], prev_state[1], action,]

        # Max q value of next state
        qmax = np.max(self.qtable[new_state[0], new_state[1], :])

        # Updating the table
        self.qtable[prev_state[0], prev_state[1], action] = \
            qsa*(1-self.lr) + self.lr*(reward + qmax)

    def play(self, num_episodes):
        
        for ep in range(num_episodes):
            done = False
            prev_obs = self.grid.reset()
            # print('episode: ', ep)
            tot_reward = 0
            while not done:
                action = self.choose_action(prev_obs)
                # print(self.grid.done)
                # print('prev obs: ', prev_obs)
                obs, reward, done = self.grid.step(action)
                tot_reward += reward
                self.learn(prev_obs, action, reward, obs)
                prev_obs = obs

            self.rewards_list.append(tot_reward)
        
        # Plot rewards
        x = np.arange(0, len(self.rewards_list), self.smoothing_num)
        y = [np.average(self.rewards_list[i:i+self.smoothing_num]) for i in x[:-1]]
        y = np.array(self.rewards_list)[x]

        fig, ax0 = plt.subplots(figsize=(8,6))
        ax0.set_title('Rewards per episode')
        ax0.plot(x, y)
        plt.show()

        # Best action in each state
        fig, ax1 = plt.subplots(figsize=(8,6))
        Z = np.argmax(agent.qtable, axis=2)
        ax1.matshow(Z, cmap='seismic')
        ax1.set_title('Best action in each state')
        for (i, j), z in np.ndenumerate(Z):
            ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.show()


class ALG1(VanillaAgent):
    def __init__(self, table_size, grid) -> None:
        super().__init__(table_size, grid)
        self.action_space = 5


class ALG2(VanillaAgent):
    def __init__(self, table_size, grid) -> None:
        super().__init__(table_size, grid)
        self.q_lr = 0.1
        self.m_lr = 0.1
        self.v_lr = 0.1

    def learn(self, prev_state, action, reward, new_state):

        # Current q value
        qsa = self.qtable[prev_state[0], prev_state[1], action, 0]

        # Max q value of next state
        qmax = np.max(self.qtable[new_state[0], new_state[1], :, 0])

        # Second order moment of rewards
        rew_moment = self.qtable[prev_state[0], prev_state[1], action, 1]

        # Max moment of next state
        moment_max = np.max(self.qtable[new_state[0], new_state[1], :, 1])

        # Variance of the rewards
        rew_var = self.qtable[prev_state[0], prev_state[1], action, 2]

        # Updating the q value
        self.qtable[prev_state[0], prev_state[1], action, 0] = \
            qsa*(1-self.q_lr) + self.lr*(reward + qmax)

        qsa = self.qtable[prev_state[0], prev_state[1], action, 0]

        # Updating the M value
        self.qtable[prev_state[0], prev_state[1], action, 1] = \
            rew_moment*(1-self.m_lr) + self.m_lr*(reward**2 +2*self.gamma*reward*qmax) \
                + self.m_lr*(self.gamma**2)*moment_max

        rew_moment = self.qtable[prev_state[0], prev_state[1], action, 1]

        # Updating the Variance, we will use the previously updated values to get a better estimate
        self.qtable[prev_state[0], prev_state[1], action, 2] = \
            rew_var*(1-self.v_lr) + self.v_lr*(rew_moment - qsa**2)

        return 

    # Returns action to choose based on state and self.epsilon
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            # We will still choose the action based on the highest q-value amonsgst all actions
            action = np.argmax(self.qtable[state[0], state[1], :, 0])
        return action

    
    def play(self, num_episodes):
        
        for ep in range(num_episodes):
            done = False
            prev_obs = self.grid.reset()
            # print('episode: ', ep)
            tot_reward = 0
            while not done:
                action = self.choose_action(prev_obs)
                # print(self.grid.done)
                # print('prev obs: ', prev_obs)
                obs, reward, done = self.grid.step(action)
                tot_reward += reward
                self.learn(prev_obs, action, reward, obs)
                prev_obs = obs

            self.rewards_list.append(tot_reward)
        
        # Visualizations
        # Plotting Rewards
        x = np.arange(0, len(self.rewards_list), self.smoothing_num)
        y = [np.average(self.rewards_list[i:i+self.smoothing_num]) for i in x[:-1]]
        y = np.array(self.rewards_list)[x]

        fig, ax0 = plt.subplots(figsize=(8,6))
        ax0.set_title('Rewards per episode')
        ax0.plot(x, y)
        plt.show()
        
        # Plotting MaxQ value
        fig, ax2 = plt.subplots(figsize=(8,15))
        Z = np.round(np.max(agent.qtable[:, :, :, 0], axis=2), 1)
        ax2.matshow(Z, cmap='seismic')
        ax2.set_title('MaxQ value')
        for (i, j), z in np.ndenumerate(Z):
            ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.show()
        

        # Plotting Action for each state with MaxQ value
        fig, ax2 = plt.subplots(figsize=(8,15))
        Z = np.argmax(agent.qtable[:, :, :, 0], axis=2)
        ax2.matshow(Z, cmap='seismic')
        ax2.set_title('Action with maxQ value')
        for (i, j), z in np.ndenumerate(Z):
            ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.show()


        # # Plotting Action with least Variance
        # fig, ax2 = plt.subplots(figsize=(8,15))
        # Z = np.round(np.argmin(agent.qtable[:, :, :, 2], axis=2), 1)
        # ax2.matshow(Z, cmap='seismic')
        # ax2.set_title('Action with least variance')
        # for (i, j), z in np.ndenumerate(Z):
        #     ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        # plt.show()


        # Plotting Variance of best action
        rows, cols = np.indices((10,10))
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        best_actions = np.argmax(agent.qtable[:, :, :, 0], axis=2).reshape(-1)

        fig, ax3 = plt.subplots(figsize=(8,15))
        np.set_printoptions(suppress=True)
        Z = np.round(agent.qtable[:, :, :, 2][rows, cols, best_actions], 1).reshape(10,10)
        print(Z)
        ax3.matshow(Z, cmap='seismic')
        ax3.set_title('Variance of best action')
        for (i, j), z in np.ndenumerate(Z):
            ax3.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.show()


class MonteCarlo(VanillaAgent):
    def __init__(self, table_size, grid) -> None:
        super().__init__(table_size, grid)


    def learn(self, epdata):
        epdata['prev_state'].reverse()
        epdata['reward'].reverse()
        epdata['action'].reverse()

        ret_to_go = 0
        for state, action, reward in zip(epdata['prev_state'], epdata['action'], epdata['reward']):

            mean = self.qtable[state[0], state[1], action, 0]
            count = self.qtable[state[0], state[1], action, 1]
            prev_var = self.qtable[state[0], state[1], action, 2]
            ret_to_go += reward

            new_var = (count/(count+1))*(prev_var + ((ret_to_go - mean)**2)/(count+1))

            self.qtable[state[0], state[1], action, 0] = (mean*count + ret_to_go) / (count+1)
            self.qtable[state[0], state[1], action, 1] += 1
            self.qtable[state[0], state[1], action, 2] = new_var

        return


    def play(self, num_episodes):
        
        for ep in range(num_episodes):
            done = False
            prev_obs = self.grid.reset()
            ep_data = {'prev_state': [], 'reward': [], 'action': []}
            # print('episode: ', ep)
            tot_reward = 0
            while not done:
                action = self.choose_action(prev_obs)
                # print(self.grid.done)
                # print('prev obs: ', prev_obs)
                obs, reward, done = self.grid.step(action)
                tot_reward += reward
                ep_data['prev_state'].append(prev_obs)
                ep_data['reward'].append(reward)
                ep_data['action'].append(action)
                # self.learn(prev_obs, action, reward, obs)
                prev_obs = obs
            self.learn(ep_data)
            self.rewards_list.append(tot_reward)
        
        # Plot rewards
        x = np.arange(0, len(self.rewards_list), self.smoothing_num)
        y = [np.average(self.rewards_list[i:i+self.smoothing_num]) for i in x[:-1]]
        y = np.array(self.rewards_list)[x]
        fig, ax = plt.subplots(figsize=(8,15))
        ax.plot(x, y)
        plt.show()
        
        # Plotting MaxQ value
        fig, ax2 = plt.subplots(figsize=(8,15))
        Z = np.round(np.max(agent.qtable[:, :, :, 0], axis=2), 1)
        ax2.matshow(Z, cmap='seismic')
        ax2.set_title('MaxQ value')
        for (i, j), z in np.ndenumerate(Z):
            ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.show()
        

        # Plotting Action for each state with MaxQ value
        fig, ax2 = plt.subplots(figsize=(8,15))
        Z = np.argmax(agent.qtable[:, :, :, 0], axis=2)
        ax2.matshow(Z, cmap='seismic')
        ax2.set_title('Action with maxQ value')
        for (i, j), z in np.ndenumerate(Z):
            ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.show()

        # Plotting Variance of best action
        rows, cols = np.indices((10,10))
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        best_actions = np.argmax(agent.qtable[:, :, :, 0], axis=2).reshape(-1)

        fig, ax3 = plt.subplots(figsize=(8,15))
        np.set_printoptions(suppress=True)
        Z = np.round(agent.qtable[:, :, :, 2][rows, cols, best_actions], 1).reshape(10,10)
        print(Z)
        ax3.matshow(Z, cmap='seismic')
        ax3.set_title('Variance of best action')
        for (i, j), z in np.ndenumerate(Z):
            ax3.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.show()


    # Returns action to choose based on state and self.epsilon
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            # We will still choose the action based on the highest q-value amonsgst all actions
            action = np.argmax(self.qtable[state[0], state[1], :, 0])
        return action


# grid = Grid()
# agent = VanillaAgent([10,10,4], grid)
# agent.play(100000)
# print(repr(np.argmax(agent.qtable, axis=2)))


grid = ExpertGrid()
# Our agent has 5 possible actions hence the q table size is 10,10,5
agent = ALG1([10,10,5], grid)
agent.play(1000000)
print(repr(np.argmax(agent.qtable, axis=2)))


# grid = Grid()
# # The agent can take 4 actions and for each action it finds the q-value, 2nd reward moment
# # and the variance of the returs using Bellmann Equations
# agent = ALG2([10,10,4,3], grid)
# agent.play(100000)
# np.set_printoptions(suppress=True)
# print(repr(np.round(np.average(agent.qtable[:, :, :, 2], axis=2), 1)))


# grid = Grid()
# # The agent can take 4 actions and stores the mean of the return, count
# # and variance of return for each (s,a) pair
# agent = MonteCarlo([10,10,4,3], grid)
# agent.play(100000)
# # np.set_printoptions(suppress=True)
# print(repr(np.round(np.average(agent.qtable[:, :, :, 2], axis=2), 1)))





from environment import Grid, ExpertGrid
import numpy as np
from matplotlib import pyplot as plt
import os
import math
# Uncomment the algorithm you'd like to run at the end

class VanillaAgent():
    def __init__(self, table_size, grid, fname) -> None:
        
        self.qtable = np.zeros(shape=table_size)    # Q-Learning table
        self.rollout_tbl = np.zeros(shape=np.append(np.array(table_size), 3))
        self.grid = grid
        self.action_space = 4
        self.gamma = 0.9
        self.rewards_list = []
        self.smoothing_num = 1000
        self.fname = fname
        self.lr = 0.1   # Q value learning rate
        self.epsilon = 0.15  # Exploration epsilon
        # Debugging
        self.avoid_states = [[1,2], [1,3], [2,4], [3,4], [4,2], [4,3], [2,1], [3,1], [4,4], [4,5],\
            [4,6], [5,7], [6,7], [7,6], [7,5], [7,4], [6,3], [5,3]]
        
    def choose_action(self, state):
        """
        Returns action to choose based on state and self.epsilon
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_space)
        else:
            action = np.argmax(self.qtable[state[0], state[1], :])
        return action
    
    def learn(self, prev_state, action, reward, new_state):
        """
        Updates the q table based on one state/action pair
        """
        # Current q value
        qsa = self.qtable[prev_state[0], prev_state[1], action]

        # Max q value of next state
        qmax = np.max(self.qtable[new_state[0], new_state[1], :])

        # Updating the table
        self.qtable[prev_state[0], prev_state[1], action] = \
            qsa*(1-self.lr) + self.lr*(reward + self.gamma*qmax)

    def play(self, num_episodes, decay_ep, eps_start=0.15, eps_end=0.01):
        """
        The agent will train with an exploitation epsilon and update self.qtable
        self.epsilon will exponentially decay from eps_start to eps_end in decay_ep

        """
        if not decay_ep: decay_ep=num_episodes
        lmda = math.log(eps_start/eps_end)/decay_ep
        for ep in range(num_episodes):
            done = False
            prev_obs = self.grid.reset()
            # print('episode: ', ep)
            tot_reward = 0
            self.epsilon = eps_start*math.exp(-lmda*ep)
            while not done:
                action = self.choose_action(prev_obs)
                # print(self.grid.done)
                # print('prev obs: ', prev_obs)
                obs, reward, done = self.grid.step(action)

                tot_reward += reward
                self.learn(prev_obs, action, reward, obs)
                prev_obs = obs

            self.rewards_list.append(tot_reward)
        
        # Save the qtable
        np.save(os.path.join(self.fname, 'qtable.npy'), self.qtable)

        # Plot rewards
        x = np.arange(0, len(self.rewards_list), self.smoothing_num)
        x = np.append(x, len(self.rewards_list)-1)
        y = [np.average(self.rewards_list[i:i+1]) for i in range(len(x))]
        fig, ax0 = plt.subplots(figsize=(8,6))
        ax0.set_title('Rewards per episode')
        ax0.plot(x, y)
        plt.savefig(os.path.join(self.fname, 'rewards.jpg'), \
            bbox_inches ="tight",\
            dpi=250)

        # Best action in each state
        fig, ax1 = plt.subplots(figsize=(8,6))
        Z = np.argmax(agent.qtable, axis=2)
        ax1.matshow(Z, cmap='cool')
        ax1.set_title('Best action in each state')
        for (i, j), z in np.ndenumerate(Z):
            ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.savefig(os.path.join(self.fname, 'best_action.jpg'), \
            bbox_inches ="tight",\
            dpi=250)

    def policy_rollout(self, num_episodes, smoothing=1000, qtable_path=False):
        """
        Running Monte Carlo for mean, state visitation and variance of rewards
        num_episodes: to run for Monte Carlo
        smoothing: smoothing number for reward plotting
        qtable_path: path for qtable.npy to use if you do not want to train again
        """
        
        if qtable_path:
            self.qtable = np.load(qtable_path)
        self.rewards_list.clear()
        fell_in_trap = 0
        ep_data = {'prev_state': [], 'reward': [], 'action': []}
        for ep in range(num_episodes):
            done = False
            prev_obs = self.grid.reset()
            ep_data.clear()
            ep_data['prev_state'] = []
            ep_data['reward'] = []
            ep_data['action'] = []
            # print('episode: ', ep)
            tot_reward = 0
            while not done:
                action = np.argmax(self.qtable[prev_obs[0], prev_obs[1], :])
                obs, reward, done = self.grid.step(action)
                if reward < -100:
                    fell_in_trap+=1
                    # in_avoid = prev_obs.tolist() in self.avoid_states
                    # print(f'prev_state:{prev_obs}, new:{obs}, action:{action}, reward:{reward}, done:{done}, {in_avoid}')
                tot_reward += reward
                ep_data['prev_state'].append(prev_obs)
                ep_data['reward'].append(reward)
                ep_data['action'].append(action)
                prev_obs = obs
            self.rollout_update(ep_data)
            self.rewards_list.append(tot_reward)

        # Plot rewards
        x = np.arange(0, len(self.rewards_list), smoothing)
        x = np.append(x, len(self.rewards_list)-1)
        y = [np.average(self.rewards_list[i:i+1]) for i in range(len(x))]
        fig, ax0 = plt.subplots(figsize=(8,6))
        ax0.set_title('Rewards per episode')
        ax0.plot(x, y)
        os.makedirs(os.path.join(self.fname, 'MC'))
        ax0.text(1, 0.5, f'Fell in trap {fell_in_trap*100/num_episodes:.3f} % times', horizontalalignment='right',
            verticalalignment='bottom', transform=ax0.transAxes)
        plt.savefig(os.path.join(self.fname, 'MC', 'rewards.jpg'), \
            bbox_inches ="tight",\
            dpi=250)

        # Mean of return from each state taking the best action
        fig, ax1 = plt.subplots(figsize=(8,6))
        Z = np.max(self.rollout_tbl[:, :, :, 0], axis=2)
        ax1.matshow(Z, cmap='cool')
        ax1.set_title('Mean of returns for each state')
        for (i, j), z in np.ndenumerate(Z):
            ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.savefig(os.path.join(self.fname, 'MC', 'Mean_returns.jpg'), \
            bbox_inches ="tight",\
            dpi=250)


        # Plotting Variance of best action
        rows, cols = np.indices((10,10))
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        best_actions = np.argmax(self.qtable[:, :, :], axis=2).reshape(-1)

        # Count of state visitation for best action
        fig, ax3 = plt.subplots(figsize=(8,15))
        np.set_printoptions(suppress=True)
        Z = self.rollout_tbl[:, :, :, 1][rows, cols, best_actions].reshape(10,10)
        ax3.matshow(Z, cmap='cool')
        ax3.set_title('State visitation count')
        for (i, j), z in np.ndenumerate(Z):
            ax3.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.savefig(os.path.join(self.fname, 'MC', 'counts.jpg'), \
            bbox_inches ="tight",\
            dpi=250)

        # Plotting Variance of best action
        fig, ax3 = plt.subplots(figsize=(8,15))
        np.set_printoptions(suppress=True)
        Z = np.round(np.sqrt(self.rollout_tbl[:, :, :, 2][rows, cols, best_actions]), 1).reshape(10,10)
        ax3.matshow(Z, cmap='cool')
        ax3.set_title('Standard deviation of return from each state')
        for (i, j), z in np.ndenumerate(Z):
            ax3.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.savefig(os.path.join(self.fname, 'MC', 'std_dev_returns.jpg'), \
            bbox_inches ="tight",\
            dpi=250)

    def rollout_update(self, epdata):
        """
        Takes the entire episode data and updates the rollout table,
        which stores the mean, state visitation and variance
        """

        epdata['prev_state'].reverse()
        epdata['reward'].reverse()
        epdata['action'].reverse()

        ret_to_go = 0

        for state, action, reward in zip(epdata['prev_state'], epdata['action'], epdata['reward']):

            mean = self.rollout_tbl[state[0], state[1], action, 0]
            count = self.rollout_tbl[state[0], state[1], action, 1]
            prev_var = self.rollout_tbl[state[0], state[1], action, 2]
            ret_to_go += reward

            new_var = (count/(count+1))*(prev_var + ((ret_to_go - mean)**2)/(count+1))

            self.rollout_tbl[state[0], state[1], action, 0] = (mean*count + ret_to_go) / (count+1)
            self.rollout_tbl[state[0], state[1], action, 1] += 1
            self.rollout_tbl[state[0], state[1], action, 2] = new_var


class ALG1(VanillaAgent):
    def __init__(self, table_size, grid, fname) -> None:
        super().__init__(table_size, grid, fname)
        self.action_space = 5


class ALG2(VanillaAgent):
    def __init__(self, table_size, grid, fname) -> None:
        super().__init__(table_size, grid, fname)
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
            qsa*(1-self.q_lr) + self.q_lr*(reward + self.gamma*qmax)

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

    
    def play(self, num_episodes, decay_ep=False, eps_start=0.15, eps_end=0.01, retrain=False):
        
        if not decay_ep: decay_ep=num_episodes
        if retrain: self.qtable = np.load(retrain)
        lmda = math.log(eps_start/eps_end)/decay_ep
        for ep in range(num_episodes):
            done = False
            prev_obs = self.grid.reset()
            self.epsilon = eps_start*math.exp(-lmda*ep)
            tot_reward = 0
            while not done:
                action = self.choose_action(prev_obs)
                # print('prev obs: ', prev_obs)
                obs, reward, done = self.grid.step(action)
                if ep==8000:
                    print(f'prev_state, action, next_state, reward: {prev_obs}, {action}, {obs}, {reward}, {done}')
                # if obs[0]==3 and obs[1]==3:
                #     print(f'Reached trap, reward, tot_r, done: {reward}, {tot_reward}, {done}')
                
                tot_reward += reward
                self.learn(prev_obs, action, reward, obs)
                prev_obs = obs

            self.rewards_list.append(tot_reward)
        

        # Save the qtable
        np.save(os.path.join(self.fname, 'qtable.npy'), self.qtable)

        # Visualizations
        # Plotting Rewards
        x = np.arange(0, len(self.rewards_list), self.smoothing_num)
        x = np.append(x, len(self.rewards_list)-1)
        y = [np.average(self.rewards_list[i:i+1]) for i in range(len(x))]
        fig, ax0 = plt.subplots(figsize=(8,6))
        ax0.set_title('Rewards per episode')
        ax0.plot(x, y)
        plt.savefig(os.path.join(self.fname, 'rewards.jpg'), \
            bbox_inches ="tight",\
            dpi=250)
        
        # Plotting MaxQ value
        fig, ax2 = plt.subplots(figsize=(8,15))
        Z = np.round(np.max(agent.qtable[:, :, :, 0], axis=2), 1)
        ax2.matshow(Z, cmap='cool')
        ax2.set_title('MaxQ value')
        for (i, j), z in np.ndenumerate(Z):
            ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.savefig(os.path.join(self.fname, 'MaxQ.jpg'), \
            bbox_inches ="tight",\
            dpi=250)
        

        # Plotting Action for each state with MaxQ value - Best action
        fig, ax2 = plt.subplots(figsize=(8,15))
        Z = np.argmax(agent.qtable[:, :, :, 0], axis=2)
        ax2.matshow(Z, cmap='cool')
        ax2.set_title('Best Action in each state')
        for (i, j), z in np.ndenumerate(Z):
            ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.savefig(os.path.join(self.fname, 'best_action.jpg'), \
            bbox_inches ="tight",
            dpi=250)


        # # Plotting Action with least Variance
        # fig, ax2 = plt.subplots(figsize=(8,15))
        # Z = np.round(np.argmin(agent.qtable[:, :, :, 2], axis=2), 1)
        # ax2.matshow(Z, cmap='cool')
        # ax2.set_title('Action with least variance')
        # for (i, j), z in np.ndenumerate(Z):
        #     ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        # plt.show()


        # Plotting Standard Deviation of return for best action
        rows, cols = np.indices((10,10))
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        best_actions = np.argmax(agent.qtable[:, :, :, 0], axis=2).reshape(-1)

        fig, ax3 = plt.subplots(figsize=(8,15))
        np.set_printoptions(suppress=True)
        Z = np.round(np.sqrt(agent.qtable[:, :, :, 2][rows, cols, best_actions]), 1).reshape(10,10)
        print(Z)
        ax3.matshow(Z, cmap='cool')
        ax3.set_title('Standard deviation of return for the best action')
        for (i, j), z in np.ndenumerate(Z):
            ax3.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.savefig(os.path.join(self.fname, 'std_dev_returns.jpg'), \
            bbox_inches ="tight",\
            dpi=250)


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
        Z = np.round(np.max(self.qtable[:, :, :, 0], axis=2), 1)
        ax2.matshow(Z, cmap='cool')
        ax2.set_title('MaxQ value')
        for (i, j), z in np.ndenumerate(Z):
            ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.show()
        

        # Plotting Action for each state with MaxQ value
        fig, ax2 = plt.subplots(figsize=(8,15))
        Z = np.argmax(self.qtable[:, :, :, 0], axis=2)
        ax2.matshow(Z, cmap='cool')
        ax2.set_title('Action with maxQ value')
        for (i, j), z in np.ndenumerate(Z):
            ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        plt.show()

        # Plotting Variance of best action
        rows, cols = np.indices((10,10))
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        best_actions = np.argmax(self.qtable[:, :, :, 0], axis=2).reshape(-1)

        fig, ax3 = plt.subplots(figsize=(8,15))
        np.set_printoptions(suppress=True)
        Z = np.round(self.qtable[:, :, :, 2][rows, cols, best_actions], 1).reshape(10,10)
        print(Z)
        ax3.matshow(Z, cmap='cool')
        ax3.set_title('Variance of return for the best action of each state')
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


# Vanilla
# grid = Grid()
# dpath = 'Outputs/Vanilla/Expertrun2'
# os.makedirs(dpath)
# agent = VanillaAgent([10,10,4], grid, dpath)
# # agent.play(10000000, 5e6, 0.20, 0.01)
# agent.policy_rollout(100000, 1000, 'Outputs/Vanilla/expertaction.npy')
# print(repr(np.argmax(agent.qtable, axis=2)))


## ALG1
grid = ExpertGrid()
dpath = 'Outputs/ALG1/100k.2'
os.makedirs(dpath)
# Our agent has 5 possible actions hence the q table size is 10,10,5
agent = ALG1([10,10,5], grid, dpath)
agent.play(100000, False, 0.25, 0.01)
agent.policy_rollout(100000, 1000)
print(repr(np.argmax(agent.qtable, axis=2)))


# # ALG2
# grid = Grid()
# dpath = 'Outputs/ALG2/10M.2/test' # Output folder name
# os.makedirs(dpath)
# # The agent can take 4 actions and for each action it finds the q-value, 2nd reward moment
# # and the variance of the returs using Bellmann Equations
# agent = ALG2([10,10,4,3], grid, dpath)
# agent.play(1000000, eps_start=0.25, eps_end=0.01, retrain=False)


# grid = Grid()
# # The agent can take 4 actions and stores the mean of the return, count
# # and variance of return for each (s,a) pair
# agent = MonteCarlo([10,10,4,3], grid)
# agent.play(1000000)
# # np.set_printoptions(suppress=True)
# print(repr(np.round(np.average(agent.qtable[:, :, :, 2], axis=2), 1)))





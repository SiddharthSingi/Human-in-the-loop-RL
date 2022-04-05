from threading import currentThread
from environment import Grid, ExpertGrid
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import seaborn as sns
# Uncomment the algorithm you'd like to run at the end

class VanillaAgent():
    def __init__(self, table_size, grid, dpath) -> None:
        
        self.qtable = np.zeros(shape=table_size)    # Q-Learning table
        self.rollout_tbl = np.zeros(shape=np.append(np.array(table_size), 3))
        self.grid = grid
        self.action_space = 4
        self.gamma = 0.9
        self.rewards_list = []
        self.smoothing_num = 1000
        self.fname = dpath
        self.lr = 0.1   # Q value learning rate
        self.epsilon = 0.15  # Exploration epsilon
        self.exploration_count = np.zeros((10, 10, 4))

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
                self.exploration_count[prev_obs[0], prev_obs[1], action] += 1
                # print(self.grid.done)
                # print('prev obs: ', prev_obs)
                obs, reward, done = self.grid.step(action)

                tot_reward += reward
                self.learn(prev_obs, action, reward, obs)
                prev_obs = obs

            self.rewards_list.append(tot_reward)
        
        # Save the qtable and exploration count
        np.save(os.path.join(self.fname, 'qtable.npy'), self.qtable)
        np.save(os.path.join(self.fname, 'exploration'), self.exploration_count)

        # Plot rewards
        x = np.arange(0, len(self.rewards_list), self.smoothing_num)
        x = np.append(x, len(self.rewards_list)-1)
        y = [np.average(self.rewards_list[x[i]:x[i+1]]) for i in range(len(x)-1)]
        x = x[:-1]
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
        print(f'Total Average Rewards: {sum(self.rewards_list)/len(self.rewards_list)}')
        x = np.arange(0, len(self.rewards_list), smoothing)
        x = np.append(x, len(self.rewards_list)-1)
        y = [np.average(self.rewards_list[x[i]:x[i+1]]) for i in range(len(x)-1)]
        x = x[:-1]
        
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
        self.exploration_count = np.zeros((10, 10, 5))



class ALG2(VanillaAgent):
    def __init__(self, table_size, grid, fname) -> None:
        super().__init__(table_size, grid, fname)
        self.q_lr = 0.1
        self.m_lr = 0.1
        self.v_lr = 0.1
        self.exploration_count = np.zeros((10, 10, 4))

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
                self.exploration_count[prev_obs[0], prev_obs[1], action] += 1
                # print('prev obs: ', prev_obs)
                obs, reward, done = self.grid.step(action)
                # if ep==8000:
                #     print(f'prev_state, action, next_state, reward: {prev_obs}, {action}, {obs}, {reward}, {done}')
                # if obs[0]==3 and obs[1]==3:
                #     print(f'Reached trap, reward, tot_r, done: {reward}, {tot_reward}, {done}')
                
                tot_reward += reward
                self.learn(prev_obs, action, reward, obs)
                prev_obs = obs

            self.rewards_list.append(tot_reward)
        

        # Save the qtable, exploration_count table
        np.save(os.path.join(self.fname, 'qtable.npy'), self.qtable)
        np.save(os.path.join(self.fname, 'exploration.npy'), self.exploration_count)

        # Visualizations
        # Plotting Rewards
        x = np.arange(0, len(self.rewards_list), self.smoothing_num)
        x = np.append(x, len(self.rewards_list)-1)
        y = [np.average(self.rewards_list[x[i]:x[i+1]]) for i in range(len(x)-1)]
        x = x[:-1]

        print('X: ', x, '\n')
        print('Y: ', y)
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

    def get_action_table(self, qtable, threshold):
        """
        returns an action table where action=4 if variance is above the threshold
        qtable: shape=(10,10,4,3) Learned from ALG2
        threshold: threshold for the variance value
        """
        rows, cols = np.indices((10,10))
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        best_actions = np.argmax(qtable[:, :, :, 0], axis=2).reshape(-1)
        variances = qtable[:, :, :, 2][rows, cols, best_actions]
        # best_actions = best_actions.reshape(10,10)
        action_table = np.where(variances<threshold, best_actions, 4).reshape(10,10)

        if threshold==5700:
            fig, ax = plt.subplots()
            ax = sns.heatmap(action_table, annot=True, fmt=".1f", cmap='cool', linewidths=.5)
            plt.show()

        return action_table
    
    def threshold_rollout(self, num_episodes, max_thresh, thresh_spacing, qtable_f):
        grid = ExpertGrid()
        qtable = np.load(qtable_f)
        thresh_rewards = []
        thresholds = []
        for thresh in range(0, max_thresh, thresh_spacing):
            action_table = self.get_action_table(qtable, thresh)
            cur_thresh_rewards = []
            for ep in range(num_episodes):
                done = False
                prev_obs = grid.reset()
                tot_reward = 0
                while not done:
                    action = action_table[prev_obs[0], prev_obs[1]]
                    obs, reward, done = grid.step(action)
                    tot_reward+=reward
                    prev_obs = obs
                cur_thresh_rewards.append(tot_reward)

            thresh_avg_reward = sum(cur_thresh_rewards) / len(cur_thresh_rewards)
            thresh_rewards.append(thresh_avg_reward)
            thresholds.append(thresh)

        thresh_rewards = np.array(thresh_rewards)
        thresholds = np.array(thresholds)
        np.savez(os.path.join(self.fname, 'thresholdrollout.npz'), thresh_rewards, thresholds)

        fig, ax = plt.subplots()
        ax = sns.barplot(x=thresholds, y=thresh_rewards)
        ax.set_xticks(np.arange(0, len(thresholds)+1, 10))
        ax.set_xlabel('Thresholds')
        ax.set_ylabel('Average Return (1000 episodes)')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        plt.show()

        return

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
        x = np.append(x, len(self.rewards_list)-1)
        y = [np.average(self.rewards_list[x[i]:x[i+1]]) for i in range(len(x)-1)]
        x = x[:-1]
        
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


# # Vanilla
# grid = Grid()
# dpath = 'Outputs/Vanilla/10M.5/test'
# num_episodes = 10000000
# ep_decay = 5e6
# smoothing_num = 1000
# eps_start = 0.3
# eps_end = 0.01

# if not os.path.exists(dpath):
#     os.makedirs(dpath)
# else:
#     raise Exception('This dpath already exists, be more careful dummy!')
# agent = VanillaAgent([10,10,4], grid, dpath)
# # agent.play(num_episodes, ep_decay, eps_start, eps_end)
# agent.policy_rollout(1000, smoothing=1, qtable_path='Outputs/Vanilla/10M.5/qtable.npy')
# print(repr(np.argmax(agent.qtable, axis=2)))

# f = open(os.path.join(dpath, 'description.txt'), 'w')
# f.write(f'Vanilla Training\n \
# episodes: {num_episodes}\n \
# smoothing num: {smoothing_num}\n \
# eps_start: {eps_start} \
# eps_end: {eps_end} \
# starting_states: states 1 step away from trap.')
# f.close()




# ALG1
# grid = ExpertGrid()
# dpath = 'Outputs/ALG1/1M.6/test'
# os.makedirs(dpath)

# # Our agent has 5 possible actions hence the q table size is 10,10,5
# agent = ALG1([10,10,5], grid, dpath)
# # agent.play(800000, 400000, 0.6, 0.07)
# agent.policy_rollout(1000, 1, qtable_path='Outputs/ALG1/1M.6/qtable.npy')
# print(repr(np.argmax(agent.qtable, axis=2)))




# # # ALG2
# num_episodes = 10000000
# ep_decay = 5e6
# smoothing_num = 1000
# eps_start = 0.4
# eps_end = 0.01

# grid = Grid()
# dpath = 'Outputs/ALG2/10M.5' # Output folder name
# if not os.path.exists(dpath):
#     os.makedirs(dpath)
# else:
#     raise Exception('This dpath already exists, be more careful dummy!')
# # The agent can take 4 actions and for each action it finds the q-value, 2nd reward moment
# # and the variance of the returs using Bellmann Equations
# agent = ALG2([10,10,4,3], grid, dpath)
# agent.play(num_episodes, ep_decay, eps_start, eps_end, retrain=False)

# f = open(os.path.join(dpath, 'description.txt'), 'w')
# f.write(f'ALG2 Training\n \
# episodes: {num_episodes}\n \
# smoothing num: {smoothing_num}\n \
# ep_decay: {ep_decay}\n \
# eps_start: {eps_start}\n \
# eps_end: {eps_end}\n\
# starting_states: states 1 step away from trap, plus few top left states.')
# f.close()

# Create the rewards plot with thresholding
grid = ExpertGrid()
agent = ALG2([10,10,4,3], grid, 'Outputs/ALG2/10M.4')
agent.threshold_rollout(1000, 8000, 100, 'Outputs/ALG2/10M.4/qtable.npy')


# grid = Grid()
# # The agent can take 4 actions and stores the mean of the return, count
# # and variance of return for each (s,a) pair
# agent = MonteCarlo([10,10,4,3], grid)
# agent.play(1000000)
# # np.set_printoptions(suppress=True)
# print(repr(np.round(np.average(agent.qtable[:, :, :, 2], axis=2), 1)))





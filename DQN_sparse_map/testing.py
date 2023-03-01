import os
import torch
from sparse_env import Grid
import numpy as np
from dqn import DQN_Agent
import pickle
os.chdir('/home/sidd/Human-in-the-loop-RL/')


lr = 5e-5			# Learning rate for Q, M models
alg2 = False
logdir = 'DQN_sparse_map/Ocean/30k/v9/alg1'
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)

# Adding all hyperparametres to a text file
os.makedirs(logdir, exist_ok=True)

env = Grid(patch_size=5, expert_penalty=0, expert_map_f='DQN_sparse_map/Ocean/expert_policy.npy')
test_env = Grid(patch_size=5, expert_penalty=0, expert_map_f='DQN_sparse_map/Ocean/expert_policy.npy')
DQN = DQN_Agent(env, test_env, lr, device, burn_in=2)
best_returns = {'versions': [], 'returns': [], 'expert_calls': [], 'episode_num': []}


for ver in range(0, 23):

    expert_calls_list = []
    avg_returns = []
    episode_num = []
    for ep in range(1000, 10000, 50):
        expert_calls, avg_return, _ = \
        DQN.policy_rollout(2000, max_steps=30, qtable_f=f'DQN_sparse_map/Ocean/10k5/algo1/v{ver}/Qvalues/Qvalues_{ep}.npy')
        avg_returns.append(avg_return)
        expert_calls_list.append(expert_calls)
        episode_num.append(ep)

    # print('Expert Calls: ', expert_calls_list)
    # print('Avg Returns: ', avg_returns)
    # print('Episode Num: ', episode_num)

    highest_return_ind = np.argmax(avg_returns)
    exp_highest_return = expert_calls_list[highest_return_ind]
    episode_high_return = episode_num[highest_return_ind]
    highest_return = avg_returns[highest_return_ind]
    print(f'For version {ver}, highest return is {highest_return} at episode {episode_high_return} with {exp_highest_return} expert calls')

    best_returns['versions'].append(ver)
    best_returns['returns'].append(highest_return)
    best_returns['expert_calls'].append(exp_highest_return)
    best_returns['episode_num'].append(episode_high_return)

# Save the dictionary
with open(f'DQN_sparse_map/Ocean/10k5/algo1/best_returns_algo1.pkl', 'wb') as f:
    pickle.dump(best_returns, f)


# import os
# import torch
# from sparse_env import Grid
# from dqn import DQN_Agent
# os.chdir('/home/sidd/Human-in-the-loop-RL/')


# lr = 5e-5			# Learning rate for Q, M models
# alg2 = False
# logdir = 'DQN_sparse_map/Ocean/30k/v9/alg1'
# expert_penalty = -0.9	# Value should be negative
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# torch.set_num_threads(4)

# # Adding all hyperparametres to a text file
# os.makedirs(logdir, exist_ok=True)

# env = Grid(patch_size=5, expert_penalty=0, expert_map_f='DQN_sparse_map/Ocean/expert_policy.npy')
# test_env = Grid(patch_size=5, expert_penalty=0, expert_map_f='DQN_sparse_map/Ocean/expert_policy.npy')
# DQN = DQN_Agent(env, test_env, lr, device, burn_in=2)
# expert_calls_list = []
# avg_returns = []

# expert_calls, avg_return, _ = \
#     DQN.zhanpeng_plot(1000, max_steps=25, qtable_f=f'DQN_sparse_map/Ocean/10k2/v2/Qvalues/Qvalues_550.npy')

# print('Expert Calls: ', expert_calls)
# print('Average Return: ', avg_return)
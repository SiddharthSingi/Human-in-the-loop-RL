import os
import numpy as np
from environment import Grid
import math
import seaborn as sns
import matplotlib.pyplot as plt

# expertaction = a = np.array([[1, 3, 3, 3, 3, 3, 3, 3, 3, 1],
#             [1, 2, 2, 3, 3, 1, 0, 0, 0, 1],
#             [1, 2, 0, 0, 3, 1, 1, 1, 0, 1],
#             [1, 2, 0, 0, 3, 3, 3, 3, 1, 1],
#             [1, 1, 1, 1, 3, 0, 3, 3, 1, 1],
#             [1, 2, 2, 2, 0, 0, 0, 3, 1, 1],
#             [1, 2, 0, 1, 0, 0, 0, 3, 1, 1],
#             [1, 2, 0, 3, 1, 1, 1, 3, 1, 1],
#             [1, 0, 0, 0, 3, 1, 1, 1, 1, 1],
#             [3, 3, 3, 3, 3, 3, 3, 3, 3, 0]])

# vanilla6 = np.load('Outputs/Vanilla/10M.6/qtable.npy')
# vanilla7 = np.load('Outputs/Vanilla/10M.7/qtable.npy')

# print(vanilla6[7,6, :])
# print(vanilla7[7,6, :])


# alg2 = np.load('Outputs/ALG2/10M.2/qtable.npy')

# np.set_printoptions(suppress=True)
# fig, ax = plt.subplots(1,2, figsize=(32,8))
# plt.tight_layout()
# sns.heatmap(vanilla[:, :, 0], annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[0])
# sns.heatmap(vanilla[:, :, 1], annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[1])
# sns.heatmap(vanilla[:, :, 2], annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[0])
# sns.heatmap(vanilla[:, :, 3], annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[1])

# sns.heatmap(alg2[:, :, 0, 0], annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[0])
# sns.heatmap(alg2[:, :, 1, 0], annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[1])
# sns.heatmap(alg2[:, :, 2, 0], annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[0])
# sns.heatmap(alg2[:, :, 3, 0], annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[1])

# plt.show()

# print(alg2[6, 7, :, 0])
# print(vanilla[6, 7, :])

# num_episodes = 100000
# smoothing_num = 1000
# eps_start = 0.2
# eps_end = 0.01
# dpath = 'Outputs/Vanilla/'


# f = open(os.path.join(dpath, 'description.txt'), 'w')
# f.write(f'Vanilla Training\n  \
# episodes: {num_episodes}\n \
# smoothing num: {smoothing_num}\n \
# eps_start: {eps_start} \
# eps_end: {eps_end} \
# starting_states: states 1 step away from trap.')
# f.close()

# exploration = np.load('Outputs/ALG2/10M.4/exploration.npy')
# np.set_printoptions(suppress=True)
# print(np.round(np.average(exploration[:,:,:], axis=2)/1000, 0))
# print(np.round(exploration[:,:,0]/1000, 0))
# print(np.round(exploration[:,:,1]/1000, 0))
# print(np.round(exploration[:,:,2]/1000, 0))
# print(np.round(exploration[:,:,3]/1000, 0))

# print(exploration[4,4,:])


# rows, cols = np.indices((5,5))
# rows = rows.reshape(-1, 1)
# cols = cols.reshape(-1, 1)


# print(rows)
# print(cols)


# dtype = [('row', np.int32), ('cols', np.int32)]

# comb = np.concatenate((rows, cols), axis=1)

# np.random.shuffle(comb)
# print(comb.shape)


# sorted_inds = np.lexsort((comb[:, 1], comb[:, 0]))
# new = comb[sorted_inds].reshape(5,5,-1)

# print(new)
# import matplotlib.pyplot as plt
# import seaborn as sns

# y = np.array([44.4, 63.0, 64.0, 67.8, 74.5, 77.4])

# x = np.array([2.5, 4.1, 4.5, 5.1, 7.6, 9.6])

# expert_penalties = [-20, -15, -25, -10, -5, -3]

# fig, ax = plt.subplots()
# plt.plot(x, y, 'bo-')
# ax.set_ylabel('Average Return')
# ax.set_xlabel('Average number of expert calls (Expert Penalty)')
# ax.set_title('Average return vs expert calls for New Grid')
# for i, label in enumerate(expert_penalties):
#     plt.annotate(label, (x[i]+0.2, y[i]-0.6))
# # ax.set_xticklabels(['0.5 (-25)','1.9 (-18)','4.7 (-10)','7.1 (-5)', '8.8 (-3)'])
# plt.show()

# a = np.array([[3, 3, 3, 3, 1, 2, 2, 2, 2, 2],
#             [3, 3, 3, 3, 1, 2, 2, 0, 3, 1],
#             [3, 3, 3, 3, 1, 2, 0, 0, 0, 1],
#             [3, 3, 3, 3, 1, 2, 2, 2, 2, 2],
#             [1, 2, 2, 3, 1, 2, 3, 3, 3, 1],
#             [1, 2, 0, 0, 1, 0, 0, 3, 1, 1],
#             [1, 2, 0, 0, 1, 0, 0, 3, 1, 1],
#             [1, 2, 0, 0, 1, 0, 0, 3, 1, 1],
#             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#             [3, 3, 3, 3, 1, 2, 2, 2, 2, 2]])

# fig, ax = plt.subplots()
# ax = sns.heatmap(a, annot=True, fmt=".1f", cmap='cool', linewidths=.5)
# plt.show()


# fig, ax = plt.subplots()
# ax.set_ylabel("Average return across 1000 episodes")
# ax.set_xlabel("Average number of expert calls made per episode")
# plt.plot(x, y, marker='o')
# plt.show()


# a = np.array([1,1,2,3,2,2])
# b = np.zeros_like(a)
# b[np.where(a==1)] = 1

# print(b)

# alg2 = np.load('ALG2_NewGrid.npy')
# alg1 = np.load('ALG1_NewGrid.npy')

# returns = alg2[:,1]
# calls = alg2[:,0]
# fig, ax = plt.subplots(figsize=(10,10))
# ax.set_ylabel("Average return across 1000 episodes")
# ax.set_xlabel("Average number of expert calls made per episode")
# ax.set_title('Average Return vs Expert calls for Old Grid')

# plt.plot(calls, returns, 'bo-', label='ALG2', ms=2, linewidth=1)
# ax.text(0.95, 0.2, f'Expert Penalty=0', \
#     horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

# # Superimposing data from ALG1
# x = alg1[0]
# y = alg1[1]
# expert_penalties = alg1[2]

# plt.plot(x, y, 'rv-', label='ALG1', ms=2, linewidth=1)

# # Uncomment if you would like to penalty labels on the markers
# for i, label in enumerate(expert_penalties):
#             plt.annotate(int(label), (x[i]+0.2, y[i]))

# plt.legend(loc='upper left')
# plt.show()


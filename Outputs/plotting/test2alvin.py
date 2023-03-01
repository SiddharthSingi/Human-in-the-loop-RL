import numpy as np
import matplotlib.pyplot as plt

alg2 = np.load('Outputs/plotting/ALG2_NewGrid.npy')
alg1 = np.load('Outputs/plotting/ALG1_NewGrid.npy')

# ALG2 data is in the form of [expert calls, average return]
calls = alg2[:,0]
returns = alg2[:,1]
fig, ax = plt.subplots(figsize=(10,10))
ax.set_ylabel("Average return across 1000 episodes")
ax.set_xlabel("Average number of expert calls made per episode")
ax.set_title('Average Return vs Expert calls for Old Grid')

plt.plot(calls, returns, 'bo-', label='ALG2', ms=2, linewidth=1)
ax.text(0.95, 0.2, f'Expert Penalty=0', \
    horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

# Superimposing data from ALG1
# ALG1 data is in the form of [expert calls, average return, expert penalties]
x = alg1[0]
y = alg1[1]
expert_penalties = alg1[2]

plt.plot(x, y, 'rv-', label='ALG1', ms=2, linewidth=1)

# Uncomment if you would like to penalty labels on the markers
for i, label in enumerate(expert_penalties):
            plt.annotate(int(label), (x[i]+0.2, y[i]))

plt.legend(loc='upper left')
plt.show()
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

class Grid:
    def __init__(self, expert_penalty=-5, patch_size = 5, expert_map_f='DQN_sparse_map/expert_map.npy') -> None:

        self.grid = np.zeros((15,13))
        # Obstacle
        self.grid[0:9, 2:10] = -1
        self.grid[3:6, 2:8] = 0
        self.grid[12:215, 2:10] = -1
        # Trap
        # self.grid[35:40, 10:30] = -2
        # Goal
        self.grid[0:15, 12] = 1

        # Array of all possible start positions for the agent
        self.possible_starts = np.where(self.grid==0)
        self.possible_starts = np.vstack((self.possible_starts[0], self.possible_starts[1])).T

        # self.possible_starts = [[i, j] for i in range(15) for j in [1,2,11,12]]

        # Used in the step function
        self.action_add = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )

        # Patch size that the agent is able to see. This must be an odd number
        self.patch_size = patch_size

        # Grid epsilon
        self.g_epsilon = 0.00

        self.expertaction = a = np.load(expert_map_f)
        self.expert_penalty = expert_penalty
        
        # Creates a padding around grid of self.patch_size with values -5 around
        self.padded_grid = np.pad(self.grid, self.patch_size, \
            'constant', constant_values=-5)

        self.visited = np.zeros_like(self.grid)

    def viz_grid(self, flag):
        '''
        Visualize the plot of the map
        params:
            flag: either 'obstacles', or 'expert'
        '''

        if flag=='obstacles': map=self.grid
        if flag=='expert': map=self.expertaction

        fig, ax = plt.subplots()
        ax = sns.heatmap(map, annot=True, fmt=".2g", cmap='cool', linewidths=.5)
        plt.show()

    def disp_actions(grid):

        num_to_arrow = {0: u'\u2191', 1: u'\u2193', 2: u'\u2190', 3: u'\u2192', -1:'-1'}
        f = np.vectorize(lambda x: num_to_arrow[x])
        arrows = f(grid)

        fig, ax = plt.subplots()
        ax = sns.heatmap(grid, annot=arrows, fmt="s", cmap='cool', linewidths=.5)
        plt.show()

    def reset(self, test=None):
        '''
        test: Optional param to start the episode at a particular cell position
        
        '''
        if test is not None:
            obs = self.yx_to_obs(test)
        else:
            yx_start = self.possible_starts[np.random.randint(len(self.possible_starts))]
            obs = self.yx_to_obs(yx_start)
        self.current_state = obs
        self.done = False
        return obs

    def yx_to_obs(self, yx):
        '''
        Our observable space for the agent are the n^2 blocks around it (including the current x, y)
        These n^2 blocks will be represented as 1 feature map, where obstacles and boundary wall
        will be represented as -1, empty cell as 0 and goal cells as 1

        params: 
            yx: [index of cell in the grid]
            n: odd number for the n^2 patch visible to the agent

        ouputs:
        List of
            observed patch: Numpy array of one hot encoded patch of environment around the yx position, 
                with 4 feature maps for empty space, obstacles, traps, goals
            yx_pos: Numpy array of y and x position of the agent
        '''
        n = self.patch_size
        assert n%2==1, 'n must be an odd number'
        y = yx[0]
        x = yx[1]

        s = n//2
        patch = np.zeros((n,n))

        # We are adding self.patch_size because the grid is padded with pad width
        # self.patch_size
        patch = self.padded_grid[y-s+n:y+s+1+n, x-s+n:x+s+1+n]

        '''
        Use this if you want to use feature maps for obstacles, traps, goals etc
        # Creating our feature maps
        empties = np.zeros_like(patch)
        empties[patch==0] = 1

        obstacles = np.zeros_like(patch)
        obstacles[patch==-1] = 1

        traps = np.zeros_like(patch)
        traps[patch==-2] = 1

        goals = np.zeros_like(patch)
        goals[patch==1] = 1

        obs_maps = np.stack((empties, obstacles, traps, goals), axis=0).astype('f')
        '''
        
        return [np.expand_dims(patch, axis=0), np.array((y, x)).astype('f')]

    def step(self, a: int) -> None:
        """
        action can be UP, DOWN, LEFT, RIGHT.
        with action values 0,1,2,3
        """

        # Check to see if the episode has already ended
        assert self.done==False, "Episode has ended,\
             you should not call step again"

        reward = -2.0 # For taking a step

        if a<4:
            if np.random.random() < self.g_epsilon:
                a = np.random.randint(4)
        else:
            reward -= self.expert_penalty
            a = self.expertaction[self.current_state[1][0], self.current_state[1][1]]
        
        # Moving to the new state
        new_state = (self.current_state[1] + self.action_add[a]).astype('int')

        # If you have reached a boundary
        if any(new_state >= self.grid.shape) or any(new_state < 0):
            # Boundary reached
            new_state = np.copy(self.current_state[1])
            done = False


        # Check value of cell         
        else:
            cell = self.grid[new_state[0], new_state[1]].astype('int')
            # Reached a trap
            if cell == -2: 
                reward += -100  # -100 for trap
                done = True
                
            # Reached an obstacle
            if cell == -1:  
                reward += -3    # -3 for obstacle
                done = False
                new_state = np.copy(self.current_state[1]).astype('int')
                
            # Reached another empty cell
            if cell == 0:
                done = False
                
            # If goal state is reached
            if cell == 1:
                reward += 200    # 100 for goal
                done = True

        new_obs = self.yx_to_obs(new_state.astype('int'))
        self.current_state = new_obs
        self.done = done

        return new_obs, reward, done


# This code will also be run when importing this file
gridtest = Grid(patch_size=5)
gridtest.viz_grid('obstacles')
obs = gridtest.reset([1,8])
print(obs)
# new_obs, reward, done = gridtest.step(0)

# np.set_printoptions(threshold=sys.maxsize)
# print('cur: ', obs[0], obs[1])
# print('new: ', new_obs[0], new_obs[1])
# print(done)
# from collections import defaultdict
# counter = defaultdict(int)
# for _ in range(100):
#     state = gridtest.reset()
#     counter[str(int(state[1][0])) + str(int(state[1][1]))] += 1

# print(counter)

# print(obs)
# totalrew = 0
# totalrew += reward
# totalrew += reward

# print(totalrew)




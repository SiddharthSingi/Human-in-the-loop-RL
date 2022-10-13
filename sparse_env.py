import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

class Grid:
    def __init__(self, expert_penalty=-5) -> None:

        self.grid = np.zeros((40,40))
        # Obstacle
        self.grid[10:30, 10:30] = -1
        self.grid[20, 10:25] = 0
        # Trap
        self.grid[35:40, 10:30] = -2
        # Goal
        self.grid[20, 39] = 1

        # Array of all possible start positions for the agent
        self.possible_starts = np.where(self.grid==0)
        self.possible_starts = np.vstack((self.possible_starts[0], self.possible_starts[1])).T

        # Used in the step function
        self.action_add = np.array(
            [[-1, 0], [1, 0], [0, -1], [0, 1]]
        )

        # Patch size that the agent is able to see. This must be an odd number
        self.patch_size = 3

        # Grid epsilon
        self.g_epsilon = 0.00

        self.expertaction = a = np.array([[1, 3, 3, 3, 3, 3, 3, 3, 3, 1],
            [1, 2, 2, 3, 3, 1, 0, 0, 0, 1],
            [1, 2, 0, 0, 3, 1, 1, 1, 0, 1],
            [1, 2, 0, 0, 3, 3, 3, 3, 1, 1],
            [1, 1, 1, 1, 3, 0, 3, 3, 1, 1],
            [1, 2, 2, 2, 0, 0, 0, 3, 1, 1],
            [1, 2, 0, 1, 0, 0, 0, 3, 1, 1],
            [1, 2, 0, 3, 1, 1, 1, 3, 1, 1],
            [1, 0, 0, 0, 3, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 0]])

        self.expert_penalty = expert_penalty

    def viz_grid(self):
        fig, ax = plt.subplots()
        ax = sns.heatmap(self.grid, annot=True, fmt=".2g", cmap='cool', linewidths=.5)
        plt.show()

    def reset(self, test=None):
        '''
        test: Optional param to start the episode at a particular cell position
        '''
        if test is not None:
            obs = self.yx_to_obs(test, self.patch_size)
        else:
            yx_start = self.possible_starts[np.random.randint(len(self.possible_starts))]
            obs = self.yx_to_obs(yx_start, self.patch_size)
        self.current_state = obs
        self.done = False
        return obs

    def yx_to_obs(self, yx, n):
        '''
        Our observable space for the agent are the n^2 blocks around it (including the current x, y)
        These 9 blocks will be represented as 4 feature maps, each map being a one hot encoded map of
        empty space, obstacles, traps and goal.
        Apart from this we will also be giving the x and y values as input

        params: 
            yx: [index of cell in the grid]
            n: odd number for the n^2 patch visible to the agent
        '''

        assert n%2==1, 'n must be an odd number'
        y = yx[0]
        x = yx[1]

        s = n//2
        patch = self.grid[y-s:y+s+1, x-s:x+s+1]

        # Creating our feature maps
        empties = np.zeros_like(patch)
        empties[patch==0] = 1

        obstacles = np.zeros_like(patch)
        obstacles[patch==-1] = 1

        traps = np.zeros_like(patch)
        traps[patch==-2] = 1

        goals = np.zeros_like(patch)
        goals[patch==1] = 1

        obs_maps = np.stack((empties, obstacles, traps, goals), axis=0)
        return (obs_maps, np.array((y, x)))


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
        
        new_state = self.current_state[1] + self.action_add[a]


        # If you have reached a boundary
        if any(new_state >= self.grid.shape) or any(new_state < 0):
            # Boundary reached
            new_state = np.copy(self.current_state[1])
            done = False


        # Check value of cell         
        else:
            cell = self.grid[new_state[0], new_state[1]]
            # Reached a trap
            if cell == -2: 
                reward += -100  # -100 for trap
                done = True
                
            # Reached an obstacle
            if cell == -1:  
                reward += -3    # -3 for obstacle
                done = False
                new_state = np.copy(self.current_state[1])
                
            # Reached another empty cell
            if cell == 0:
                done = False
                
            # If goal state is reached
            if cell == 1:
                reward += 100    # 100 for goal
                done = True

        new_obs = self.yx_to_obs(new_state, self.patch_size)
        self.current_state = new_obs
        self.done = done

        return new_obs, reward, done



grid = Grid()
grid.viz_grid()
a = grid.reset([9,10])
new_obs, reward, done = grid.step(1)
print(new_obs[0])
print(new_obs[1])



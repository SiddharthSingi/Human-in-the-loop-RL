import numpy as np

class Grid:
    def __init__(self) -> None:
        # This will save all the obstacles and traps
        #-1 is obstacle, -2 is trap
        self.grid = np.array([
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,-1,-1,-1,0],
            [0,0,-2,-2,0,0,0,0,-1,0],
            [0,0,-2,-2,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,-2,-2,-2,0,0,0],
            [0,0,-1,0,-2,-2,-2,0,0,0],
            [0,0,-1,0,0,0,0,0,0,0],
            [0,-1,-1,-1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1]
            ])

        # array of possible start positions for the agent
        self.possible_starts = np.where(self.grid==0)
        self.possible_starts = np.vstack((self.possible_starts[0], self.possible_starts[1])).T
        # We are avoiding some states to start off in that arr adjacent to traps
        avoid_states = [[1,2], [1,3], [2,4], [3,4], [4,2], [4,3], [2,1], [3,1], [4,4], [4,5],\
            [4,6], [5,7], [6,7], [7,6], [7,5], [7,4], [6,3], [5,3]]
        self.possible_starts = np.array([x for x in self.possible_starts if x.tolist() not in avoid_states])
        
        #Debugging - delete later
        # These states are 2 steps from the trap
        start_states = [[0,2], [0,3], [1,4],[2,5], [3,5], [3,6], [4,7], [5,8], [6,8],\
            [7,7], [8,6], [8,5], [8,4], [5,2], [4,1], [3,0], [2,0]]
        self.possible_starts = np.array(start_states)

        # grid_epsilon is the probability that the action executed will be
        # different than the one passed to the environment
        self.g_epsilon = 0.3

        self.action_space = 4
        self.done = False
        return

    def reset(self, test=None):
        obs = self.possible_starts[np.random.randint(len(self.possible_starts))]
        if test is not None:
            obs = np.array(test)
        self.current_state = obs
        self.done = False
        return obs


    def step(self, a):
        """
        action can be UP, DOWN, LEFT, RIGHT.
        with action values 0,1,2,3
        """

        # Check to see if the episode has already ended
        assert self.done==False, "Episode has ended,\
             you should not call step again"

        # There is a self.g_epsilon prob that the action will be 
        # chosen randomly
        if np.random.random() < self.g_epsilon:
            action = np.random.randint(4)
            # print(f'Randomly action {action} chosen')
        else:
            action = a
        new_state = np.copy(self.current_state)

        #UP
        if action==0:
            new_state[0] = self.current_state[0] - 1

        #DOWN
        if action==1:
            new_state[0] = self.current_state[0] + 1

        #LEFT
        if action==2:
            new_state[1] = self.current_state[1] - 1

        #RIGHT
        if action==3:
            new_state[1] = self.current_state[1] + 1   

        # Reward Engineering
        reward = 0

        # If you have reached a boundary
        if any(new_state >= self.grid.shape) or any(new_state < 0):
            # Boundary reached
            reward += -2
            new_state = np.copy(self.current_state)
            done = False
            

        # Check value of cell         
        else:
            cell = self.grid[new_state[0], new_state[1]]
            # Reached a trap
            if cell == -2: 
                reward += -102  # -100 for trap -2 for the step
                done = True
                

            # Reached an obstacle
            if cell == -1:  
                reward += -5    # -3 for obstacle -2 for step
                done = False
                new_state = np.copy(self.current_state)
                

            # Reached another empty cell
            if cell == 0:
                reward += -2    # -2 for the step
                done = False
                

            # If goal state is reached
            if cell == 1:
                reward += 98    # 100 for goal -2 for step
                done = True

        self.current_state = np.copy(new_state)
        self.done = done
        return new_state, reward, done


class ExpertGrid(Grid):
    def __init__(self) -> None:
        super().__init__()
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

    def step(self, a):
        """
        action can be UP, DOWN, LEFT, RIGHT, CallExpert
        with action values 0,1,2,3,4
        """

        # Check to see if the episode has already ended
        assert self.done==False, "Episode has ended,\
             you should not call step again"

        # There is a self.g_epsilon prob that the action will be 
        # chosen randomly, this does not include the expert action
        if np.random.random() < self.g_epsilon:
            action = np.random.randint(4)
            # print(f'Randomly action {action} chosen')
        else:
            action = a
        new_state = np.copy(self.current_state)

        # Reward Engineering
        reward = 0

        # CallExpert
        if a==4:
            action = self.expertaction[self.current_state[0], self.current_state[1]]
            reward += -5
            # print('Expert called!')

        #UP
        if action==0:
            new_state[0] = self.current_state[0] - 1

        #DOWN
        if action==1:
            new_state[0] = self.current_state[0] + 1

        #LEFT
        if action==2:
            new_state[1] = self.current_state[1] - 1

        #RIGHT
        if action==3:
            new_state[1] = self.current_state[1] + 1   

        # If you have reached a boundary
        if any(new_state >= self.grid.shape) or any(new_state < 0):
            # Boundary reached
            reward += -2
            new_state = np.copy(self.current_state)
            done = False
            

        # Check value of cell         
        else:
            cell = self.grid[new_state[0], new_state[1]]
            # Reached a trap
            if cell == -2: 
                reward += -102  # -100 for trap -2 for the step
                done = True
                
            # Reached an obstacle
            if cell == -1:  
                reward += -5    # -3 for obstacle -2 for step
                done = False
                new_state = np.copy(self.current_state)
                
            # Reached another empty cell
            if cell == 0:
                reward += -2    # -2 for the step
                done = False


            # If goal state is reached
            if cell == 1:
                reward += 98    # 100 for goal -2 for step
                done = True

        self.current_state = np.copy(new_state)
        self.done = done
        return new_state, reward, done

# grid = Grid()
# obs = grid.reset(test=[0,0])
# done = False
# while not done:
#     a = np.random.randint(4)
#     state, reward, done = grid.step(a)
#     print('action: ', a, 'state: ', state, 'reward: ', reward, 'done: ', done)
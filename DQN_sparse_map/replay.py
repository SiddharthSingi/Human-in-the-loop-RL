import random
from collections import deque
import torch

class Replay_Memory():
    def __init__(self, device, batch_size=64, memory_size=2500):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.replay_memory = deque(maxlen=memory_size)
        self.device = device
        self.batch_size = batch_size


    def sample_batch(self):
        '''
        Samples batches from replay memory to be used as training input for the models.
        Outputs:

        '''
        batch = random.sample(self.replay_memory, self.batch_size)

        # Stacking all replay memory elements efficiently
        patches_cur, posns_cur, actions, rewards, patches_next, \
            posns_next, dones = map(torch.stack, zip(*batch))

        # Returning state and next_state in the format that is acceptable to the model
        states = [patches_cur.float(), posns_cur.float()]
        next_states = [patches_next.float(), posns_next.float()]

        return (states, actions.squeeze(), rewards.squeeze(), next_states, dones.squeeze())


    def cache(self, state, action, reward, next_state, done):
        '''
        Inputs:
            state ([patch, posn]),
            actions (int),
            rewards (int),
            next_state ([patch, posn]),
            dones (bool)
        '''

        patches_cur = torch.from_numpy(state[0]).to(self.device)
        posns_cur = torch.from_numpy(state[1]).to(self.device)
        actions = torch.Tensor([action]).to(self.device)
        rewards = torch.Tensor([reward]).to(self.device)
        patches_next = torch.from_numpy(next_state[0]).to(self.device)
        posns_next = torch.from_numpy(next_state[1]).to(self.device)
        dones = torch.Tensor([done]).to(self.device)

        self.replay_memory.append((patches_cur, posns_cur, actions, rewards, \
                                    patches_next, posns_next, dones))

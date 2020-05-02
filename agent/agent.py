import abc

import numpy as np

class Agent(abc.ABC):

    @abc.abstractmethod
    def act(self, state, eps=0.):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def load(self, filename):
        pass

    @abc.abstractmethod
    def save(self, filename):
        pass


class RandomAgent(Agent):

    valid_action_types = ['discrete', 'continuous']

    def __init__(self, action_size, action_type='discrete', num_agents=1):

        if action_type not in self.valid_action_types:
            raise TypeError(f"Action type {action_type} not implemented")

        self.action_size = action_size
        self.action_type = action_type
        self.num_agents = num_agents

    def act(self, state):

        if self.action_type == 'discrete':
            return np.random.randint(self.action_size, size=self.num_agents)
        elif self.action_type == 'continuous':
            return np.random.randn(self.num_agents, self.action_size)

    def step(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

import abc

import numpy as np

class Agent(abc.ABC):

    @abc.abstractmethod
    def new_episode(self):
        pass

    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def load(self, directory):
        pass

    @abc.abstractmethod
    def save(self, directory):
        pass


class RandomAgent(Agent):
    """Agent which acts randomly and does not learn.

    This class can be used for simulating a naive agent in environments,
    creating agent baselines and in pytest test cases.

    """
    valid_action_types = ['discrete', 'continuous']

    def __init__(self, space_size, action_size, action_type='discrete', num_agents=1):
        """Initializes an instance of RandomAgent.

        Args:
            space_size (int): the space size per agent
            action_size (int): the max integer output(s), when using discrete action type,
                and the number of outputs per agent, when using continous action space
            action_type (str): either 'discrete' or 'continuous'
            num_agents (int): number of agents

        """
        if action_type not in self.valid_action_types:
            raise TypeError(f"Action type {action_type} not implemented")

        self.space_size = space_size
        self.action_size = action_size
        self.action_type = action_type
        self.num_agents = num_agents

        self.expected_state_shape = (self.num_agents, self.space_size)

    def new_episode(self):
        return

    def act(self, state):
        """Returns a random action for agent(s).

        Args:
            state (np.ndarray): the input is disregarded, only shape is checked

        """
        if np.shape(state) != self.expected_state_shape:
            raise TypeError("input shape does not match expectation")

        if self.action_type == 'discrete':
            if self.num_agents == 1:
                return np.random.randint(self.action_size)
            else:
                return np.random.randint(self.action_size, size=self.num_agents)
        elif self.action_type == 'continuous':
            return np.random.randn(self.num_agents, self.action_size)

    def step(self, state=None, action=None, reward=None, next_state=None, done=None):
        return

    def save(self, directory):
        return

    def load(self, directory):
        return

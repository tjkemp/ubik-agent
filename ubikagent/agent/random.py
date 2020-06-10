from .agent import Agent

import numpy as np


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
        self.randomness = True

        self.expected_state_shape = (self.num_agents, self.space_size)

    def new_episode(self):
        # RandomAgent has no metrics to output
        return

    def act(self, state):
        """Returns a random action for agent(s).

        Args:
            state (np.ndarray): the input is disregarded, only shape is checked

        Raises:
            TypeError: if state shape does not match expectation

        Returns:
            np.ndarray: a random move, an integer (for discrete agent) or array
                of floats for continuous agent

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

class RandomGymAgent(RandomAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: num_agents > 1 not implemented

    def act(self, state):
        """Returns a random action for agent(s).

        Args:
            state (np.ndarray): the input is disregarded, only shape is checked

        Raises:
            TypeError: if state shape does not match expectation

        Returns:
            np.ndarray: a random move, an integer (for discrete agent) or array
                of floats for continuous agent

        """
        if state not in self.space_size:
            raise TypeError("state given to act() is not in the expected state space")

        return self.action_size.sample()
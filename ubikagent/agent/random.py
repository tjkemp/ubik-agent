import numpy as np

from ubikagent.agent.abc import Agent


class RandomAgent(Agent):

    def __init__(self, state_size, action_size, seed):
        """Initializes an instance of RandomAgent.

        Only supports single agent environments.

        Args:
            state_size (gym.spaces.space):  the observation space size
            action_size (gym.spaces.space): the action space size
            seed (int): seed number for randomness

        """
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """Returns a random action from the agent.

        Args:
            state (any): the input is disregarded, only its shape is checked that
                it matches `self.state_size`

        Raises:
            TypeError: if state shape does not match expectation

        Returns:
            np.ndarray: the random action

        """
        if state not in self.state_size:
            raise TypeError("state given to act() is not in the expected state space")

        return self.action_size.sample()

    def step(self, state=None, action=None, reward=None, next_state=None, done=None):
        return

    def new_episode(self):
        return

    def save(self, directory):
        raise NotImplementedError("RandomAgent has no state to save")

    def load(self, directory):
        raise NotImplementedError("RandomAgent has no state to save or load")

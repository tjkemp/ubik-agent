import os
from collections import defaultdict
import pickle

import numpy as np

from ubikagent.agent.abc import Agent

class SarsaAgent(Agent):

    def __init__(
            self,
            action_size,
            alpha=0.05,
            epsilon=1.0,
            epsilon_decay=0.9,
            epsilon_min=0.1,
            gamma=1.0,
            algorithm='expected_sarsa'):
        """Initialize a SarsaMax or Expected Sarsa agent.

        Args:
            action_size (int): number of actions agent can take
            alpha (float): learning rate
            epsilon (float): controls amount o fexploration [0, 1]
            epsilon_decay (float): controls how fast epsilon decays
            epsilon_min (float): minimum epsilon
            gamma (float): controls how much future reward is valued [0, 1]
            algorithm (str): either 'expected_sarsa' (default), or 'sarsamax'

        Raises:
            NotImplementedError: if improper algorithm name is provided

        """
        self.action_size = action_size
        self.Q = defaultdict(lambda: np.zeros(self.action_size, dtype=np.float32))

        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        if algorithm == 'expected_sarsa':
            self.algorithm = algorithm
        elif algorithm == 'sarsamax':
            self.algorithm = algorithm
        else:
            raise NotImplementedError(
                f"Algorithm {algorithm} is not implemented")

        self.num_episodes = 0

    def new_episode(self):
        """Function is called when a new episode starts."""
        return {'epsilon': self.epsilon}

    def act(self, state):
        """Selects an action given the state.

        Args:
            state (integer): the current state of the environment

        Returns:
            integer: action compatible with the task's action space

        """
        policy_s = self._epsilon_greedy_probabilities(state)
        action_t = np.random.choice(self.action_size, p=policy_s)
        return action_t

    def _epsilon_greedy_probabilities(self, state):
        """Calculates epsilon greedy probabilities for actions given the state.

        The action with the highest expected value gets the largest
        probabilty and the rest of the actions get each an equally small
        probability, depending on the size of `self.epsilon`.

        Args:
            state (integer): state for which to calculate action probabilities

        Returns:
            list of floats: probability of each action at
            the given state according to the current policy.

        """
        q_state = self.Q[state]
        probs = np.ones_like(q_state) * (self.epsilon / self.action_size)
        best_action = np.argmax(q_state)
        probs[best_action] = (1 - self.epsilon) + (self.epsilon / self.action_size)
        return probs

    def step(self, state, action, reward, next_state, done):
        """Update the agent's knowledge, using the most recently sampled tuple.

        Args:
            state: the previous state of the environment
            action: the agent's previous choice of action
            reward: last reward received
            next_state: the current state of the environment
            done: whether the episode is complete (True or False)

        """
        q_value = self._updated_reward(
            state,
            action,
            reward,
            next_state)

        self.Q[state][action] = q_value

        if done:
            self.num_episodes += 1
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def _updated_reward(
            self,
            state_t,
            action_t,
            reward_next,
            state_next):
        """Calculates the update to value function using either Sarsamax
        or Expected Sarsa algorithm."""

        q_current = self.Q[state_t][action_t]

        if self.algorithm == 'sarsamax':
            q_value = (1 - self.alpha) * q_current + \
                self.alpha * (reward_next + self.gamma * np.max(self.Q[state_next]))

        elif self.algorithm == 'expected_sarsa':
            policy_state_t = self._epsilon_greedy_probabilities(state_next)
            reward_expected = np.dot(policy_state_t, self.Q[state_next])
            q_value = (1 - self.alpha) * q_current + \
                self.alpha * (reward_next + self.gamma * reward_expected)
        return q_value

    def load(self, directory, filename='model.json'):
        """Load a learned model from a file."""

        load_path = os.path.join(directory, filename)
        with open(load_path, 'rb') as input_file:
            model = pickle.load(input_file)

        self.Q = defaultdict(lambda: np.zeros(self.action_size, dtype=np.float32))
        self.Q.update(model)

    def save(self, directory, filename='model.json'):
        """Save a learned model into a file."""

        save_path = os.path.join(directory, filename)
        with open(save_path, 'wb') as output_file:
            pickle.dump(dict(self.Q), output_file)

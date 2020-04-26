import numpy as np
from collections import defaultdict


class SarsaAgent:

    def __init__(
            self,
            num_states,
            alpha=0.01,
            epsilon=0.1,
            gamma=1.0,
            algorithm='expected_sarsa'):
        """Initialize agent."""

        self.num_states = num_states
        self.Q = defaultdict(lambda: np.zeros(self.num_states))

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        if algorithm == 'expected_sarsa':
            self.algorithm = algorithm
        elif algorithm == 'sarsamax':
            self.algorithm = algorithm
        else:
            raise NotImplementedError(
                f"Algorithm {algorithm} is not implemented")

        self.num_episodes = 0

    def _epsilon_greedy_probabilities(self, q_state):
        """Calculates epsilon greedy probabilities for the next state
        given its value function.

        Args:
            q_state (list of floats): Values for the state
                the Q table. Each location and its value represents
                an action and expected reward for the action.

        Returns:
            list of floats: probability of each action at
            the given state according to the current policy.

        """
        probs = np.ones_like(q_state) * \
            self.epsilon / self.num_states
        best_action = np.argmax(q_state)
        probs[best_action] = (1 - self.epsilon) + \
            (self.epsilon / self.num_states)

        return probs

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
            q = (1 - self.alpha) * \
                q_current + \
                self.alpha * \
                (reward_next + self.gamma * np.max(self.Q[state_next]))

        elif self.algorithm == 'expected_sarsa':
            policy_state_t = self._epsilon_greedy_probabilities(
                self.Q[state_next])
            reward_expected = np.dot(
                policy_state_t,
                self.Q[state_next])
            q = (1 - self.alpha) * q_current + self.alpha * (
                reward_next + self.gamma * reward_expected)
        return q

    def select_action(self, state):
        """ Selects an action given the state.

        Args:
            state: the current state of the environment

        Returns:
            integer: action compatible with the task's action space

        """
        policy_s = self._policy_epsilon_greedy(self.Q[state])
        action_t = np.random.choice(self.num_states, p=policy_s)
        return action_t

    def step(self, state, action, reward, next_state, done):
        """Update the agent's knowledge, using the most recently sampled tuple.

        Args:
            state: the previous state of the environment
            action: the agent's previous choice of action
            reward: last reward received
            next_state: the current state of the environment
            done: whether the episode is complete (True or False)

        """
        q = self._updated_reward(
            state,
            action,
            reward,
            next_state)

        self.Q[state][action] = q

        if done:
            self.num_episodes += 1
            if self.num_episodes % 100 == 0:
                self.epsilon = max(self.epsilon * 0.9, 0.01)

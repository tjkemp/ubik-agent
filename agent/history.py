import copy
from collections import deque

import numpy as np

class TrainingHistory:
    """Class to record training history."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets internal training history."""

        self._scores_window = None
        self._training_history = {
            'episode_length': [],
            'reward_max': [],
            'reward_min': [],
            'reward_mean': [],
            'reward_std': [],
            'score': []}

    @property
    def num_episodes(self):
        """Return the amount of episodes trained."""
        return len(self._training_history['episode_length'])

    @property
    def prev_episode_length(self):
        """Previous episode's length."""
        return self._training_history['episode_length'][-1]

    @property
    def prev_reward_max(self):
        """Previous episode's max total reward collected by an agent."""
        return self._training_history['reward_max'][-1]

    @property
    def prev_reward_min(self):
        """Previous episode's min total reward collected by an agent."""
        return self._training_history['reward_min'][-1]

    @property
    def prev_reward_mean(self):
        """Previous episode's mean total reward collected by the agent(s)."""
        return self._training_history['reward_mean'][-1]

    @property
    def prev_reward_std(self):
        """Previous episode's standard deviation for total rewards collected
        by the agent(s)."""
        return self._training_history['reward_std'][-1]

    @property
    def prev_score(self):
        """Previous episode's score by the agent(s)."""
        return self._training_history['score'][-1]

    def as_dict(self):
        """Returns the training history as a dictionary."""
        return copy.deepcopy(self._training_history)

    def start_training(self, score_window_size=100):
        """Called when new training session starts so that training
        score can be calculated correctly.

        Args:
            scores_window_size (int): window size for calculating score

        """
        self._scores_window = deque(maxlen=score_window_size)

    def update(self, episode_length, episode_rewards):
        """Updates internal training history.

        Args:
            episode_length (int): length of the episode
            episode_rewards (list): total rewards collected during the episode
                by each agent

        """
        reward_episode_max = np.max(episode_rewards)
        reward_episode_min = np.min(episode_rewards)
        reward_episode_mean = np.mean(episode_rewards)
        reward_episode_std = np.std(episode_rewards)

        self._training_history['episode_length'].append(episode_length)
        self._training_history['reward_max'].append(reward_episode_max)
        self._training_history['reward_min'].append(reward_episode_min)
        self._training_history['reward_mean'].append(reward_episode_mean)
        self._training_history['reward_std'].append(reward_episode_std)

        score = self._calculate_score(episode_rewards)
        self._training_history['score'].append(score)

    def _calculate_score(self, episode_rewards):
        """Calculates training score for `update()`."""

        reward_episode_max = np.max(episode_rewards)
        self._scores_window.append(reward_episode_max)
        score_window_mean = np.mean(self._scores_window)
        return score_window_mean

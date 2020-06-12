import copy
from collections import deque

import numpy as np


class History:
    """Class to store episode training or run statistics."""

    def __init__(self, score_window_size=100):
        """Creates an instance of episode history class.

        Args:
            scores_window_size (int): window size for calculating score

        """
        self.score_window_size = score_window_size
        self.reset()

    def reset(self):
        """Resets internal training history."""

        self._scores_window = deque(maxlen=self.score_window_size)
        self._history = {
            'episode_length': [],
            'score': []
        }

    @property
    def num_episodes(self):
        """Return the amount of episodes trained."""
        return len(self._history['episode_length'])

    def keys(self):
        """Return keys to all the metrics in the history."""
        return self._history.keys()

    def __getattr__(self, key):
        """Syntactic sugar for `get_latest()` so that latest items can be
        printed more cleanly in f-strings."""
        return self.get_latest(key)

    def get_latest(self, key):
        """Return last item in the history for a given key.

        If the key does not exist, returns None.

        Args:
            key (str): metric name

        """
        try:
            return self._history[key][-1]
        except KeyError:
            pass
        except IndexError:
            pass
        return None

    def as_dict(self):
        """Returns the training history as a dictionary."""
        return copy.deepcopy(self._history)

    def update(self, episode_length, episode_rewards):
        """Updates internal training history.

        Args:
            episode_length (int): length of the episode
            episode_rewards (list): total rewards collected during the episode
                by agent(s)

        """
        self._history['episode_length'].append(episode_length)

        if isinstance(episode_rewards, list) and len(episode_rewards) > 1:
            self._update_multi_agent_history(episode_rewards)
        elif isinstance(episode_rewards, list) and len(episode_rewards) == 1:
            self._update_single_agent_history(episode_rewards[0])
        else:
            self._update_single_agent_history(episode_rewards)

        score = self._calculate_score(episode_rewards)
        self._history['score'].append(score)

    def _update_single_agent_history(self, episode_reward):

        if 'reward' not in self._history:
            self._history['reward'] = [episode_reward]
        else:
            self._history['reward'].append(episode_reward)

    def _update_multi_agent_history(self, episode_rewards):

        keys = [
            'reward_max',
            'reward_min',
            'reward_mean',
            'reward_std',
        ]

        for key in keys:
            if key not in self._history:
                self._history[key] = []

        reward_episode_max = np.max(episode_rewards)
        reward_episode_min = np.min(episode_rewards)
        reward_episode_mean = np.mean(episode_rewards)
        reward_episode_std = np.std(episode_rewards)

        self._history['reward_max'].append(reward_episode_max)
        self._history['reward_min'].append(reward_episode_min)
        self._history['reward_mean'].append(reward_episode_mean)
        self._history['reward_std'].append(reward_episode_std)

    def add_from(self, metrics):
        """Adds episode related metrics into history from a dictionary.

        This function should be called at the beginning or at the end
        of each episode to store new value for that episode.

        The use case for this is that Interaction class can get more
        insight into training from the Agent itself.

        If metrics is None, then nothing is done. This is so that the
        Interaction class History does not need to care what Agent returns.

        Args:
            metrics (dict): string key with int/float value, or None

        """
        if metrics is None:
            return

        for key, value in metrics.items():
            try:
                self._history[key].append(value)
            except KeyError:
                self._history[key] = [value]

    def _calculate_score(self, episode_rewards):
        """Calculates training score for `update()`.

        To calculate the score, for each episode, the total reward
        received by the best agent is stored. The score is the
        mean of those rewards received during the last 100 episodes.

        """
        reward_episode_max = np.max(episode_rewards)
        self._scores_window.append(reward_episode_max)
        score_window_mean = np.mean(self._scores_window)
        return score_window_mean

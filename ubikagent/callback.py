from collections import deque

import numpy as np

from ubikagent.helper import print_episode_statistics, print_target_reached


class Callback:

    def __init__(self, *args, **kwargs):
        pass

    def begin_training(self, interaction, agent, env, history):
        pass

    def begin_episode(self, interaction, agent, env, history):
        pass

    def end_episode(self, interaction, agent, env, history):
        pass

    def end_training(self, interaction, agent, env, history):
        pass


class BaseCallback(Callback):

    def end_episode(self, interaction, agent, env, history):
        """Prints a single row of statistics on episode performance."""

        history.add('episode_length', interaction.episode_timestep)
        history.add('reward', interaction.episode_rewards)

        if interaction.verbose:
            print_episode_statistics(history)


class TargetScore(Callback):

    def __init__(self, score_window_size, score_target=None):
        """Stops training when target_score is reached.

        To calculate the score, for each episode, the total reward
        received by the best agent is stored. The score is the
        mean of those rewards received during the last 100 episodes.

        Args:
            score_window_size (int): the number of latest steps which are
                considered when calculating the score
            score_target (float or None): mean total rewards collected,
                when reached, the training is ended

        """
        self._scores_window_size = score_window_size
        self._scores_window = deque(maxlen=score_window_size)

        if score_target is None:
            self.score_target = float('inf')
        else:
            self.score_target = score_target

    def end_episode(self, interaction, agent, env, history):

        reward_episode_max = np.max(interaction.episode_rewards)
        self._scores_window.append(reward_episode_max)
        current_score = np.mean(self._scores_window)

        history.add('score', current_score)

        if current_score >= self.score_target:
            interaction.stop_training = True

            if interaction.verbose:
                self._print_target_reached(history)

    def _print_target_reached(self, history):
        """Prints a notification that target score has been reached."""
        print_target_reached(history)

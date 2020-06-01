from collections import namedtuple

import pytest

from agent.history import TrainingHistory

class TestHistory:
    """Tests the class `History` which provides training history for the
    `Interaction` class."""

    def test_properties(self):

        hist = TrainingHistory()
        hist.start_training()

        episode_length = [1, 2]
        episode_rewards = [[4.0, 5.0], [2.0, 4.0]]
        for length, rewards in zip(episode_length, episode_rewards):
            hist.update(length, rewards)

        assert hist.num_episodes == 2
        assert hist.prev_episode_length == 2
        assert hist.prev_reward_max == 4.0
        assert hist.prev_reward_min == 2.0
        assert hist.prev_reward_mean == 3.0
        assert hist.prev_reward_std == 1.0

    def test_score_calculation(self):
        assert False

from collections import namedtuple

import pytest

from agent.history import History


class TestHistory:
    """Tests the class `History` which provides training history for the
    `Interaction` class."""

    def test_properties(self):
        hist = History()

        episode_length = [1, 2]
        episode_rewards = [[4.0, 5.0], [2.0, 4.0]]
        for length, rewards in zip(episode_length, episode_rewards):
            hist.update(length, rewards)

        assert hist.num_episodes == 2
        assert hist.episode_length == 2
        assert hist.reward_max == 4.0
        assert hist.reward_min == 2.0
        assert hist.reward_mean == 3.0
        assert hist.reward_std == 1.0

    def test_add_and_get_key(self):
        hist = History()

        expected_output = 2.0
        hist.add_from({'test_key': 1.0})
        hist.add_from({'test_key': expected_output})
        output = hist.get_latest('test_key')

        assert output == expected_output

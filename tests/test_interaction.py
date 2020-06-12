from collections import namedtuple

import pytest

from agent.mock import UnityEnvironmentMock as UnityEnvironment
from agent.interaction import UnityInteraction
from agent.agent import UnityRandomAgent


class TestInteraction:
    """Tests Interaction class with a randomly behaving agent and a mock Unity environment.

    These tests are integration type of tests (testing several components at once). Aim is
    to not check for specific outputs, but to check nothing crashes.

    """
    twenty_reachers_config = {
        'brain_name': 'ReacherBrain',
        'num_agents': 20,
        'num_stacked_vector_observations': 1,
        'vector_observation_space_size': 33,
        'vector_observation_space_type': 'continuous',
        'vector_action_space_size': 4,
        'vector_action_space_type': 'continuous',
    }

    banana_config = {
        'brain_name': 'BananaBrain',
        'num_agents': 1,
        'num_stacked_vector_observations': 1,
        'vector_observation_space_size': 37,
        'vector_observation_space_type': 'continuous',
        'vector_action_space_size': 4,
        'vector_action_space_type': 'discrete',
    }

    def test_stats_return_correct_info_about_env(self):

        config = self.twenty_reachers_config

        env = UnityEnvironment(**config)
        state_size_env, action_size_env, num_agents_env = UnityInteraction.stats(env)

        assert config['vector_observation_space_size'] == state_size_env
        assert config['vector_action_space_size'] == action_size_env
        assert config['num_agents'] == num_agents_env

    def test_running_reacher_returns_scores(self):

        config = self.twenty_reachers_config

        env = UnityEnvironment(**config)

        agent = UnityRandomAgent(
            config['vector_observation_space_size'],
            config['vector_action_space_size'],
            action_type=config['vector_action_space_type'],
            num_agents=config['num_agents'])

        sim = UnityInteraction(agent, env)

        num_episodes = 2
        max_time_steps = 2
        history = sim.run(
            num_episodes=num_episodes,
            max_time_steps=max_time_steps)
        env.close()

        assert len(history['episode_length']) == num_episodes
        assert history['episode_length'][0] == max_time_steps

    def test_training_reacher_returns_scores(self):

        config = self.twenty_reachers_config

        env = UnityEnvironment(**config)

        agent = UnityRandomAgent(
            config['vector_observation_space_size'],
            config['vector_action_space_size'],
            action_type=config['vector_action_space_type'],
            num_agents=config['num_agents'])

        sim = UnityInteraction(agent, env)

        num_episodes = 2
        max_time_steps = 2
        history = sim.train(
            num_episodes=num_episodes,
            max_time_steps=max_time_steps)
        env.close()

        assert len(history['episode_length']) == num_episodes
        assert history['episode_length'][0] == max_time_steps

    def test_running_banana_returns_scores(self):

        config = self.banana_config

        env = UnityEnvironment(**config)

        agent = UnityRandomAgent(
            config['vector_observation_space_size'],
            config['vector_action_space_size'],
            action_type=config['vector_action_space_type'],
            num_agents=config['num_agents'])

        sim = UnityInteraction(agent, env)

        num_episodes = 2
        max_time_steps = 2
        history = sim.run(
            num_episodes=num_episodes,
            max_time_steps=max_time_steps)
        env.close()

        assert len(history['episode_length']) == num_episodes
        assert history['episode_length'][0] == max_time_steps

    def test_training_banana_returns_scores(self):

        config = self.banana_config

        env = UnityEnvironment(**config)

        agent = UnityRandomAgent(
            config['vector_observation_space_size'],
            config['vector_action_space_size'],
            action_type=config['vector_action_space_type'],
            num_agents=config['num_agents'])

        sim = UnityInteraction(agent, env)

        num_episodes = 2
        max_time_steps = 2
        history = sim.train(
            num_episodes=num_episodes,
            max_time_steps=max_time_steps)
        env.close()

        assert len(history['episode_length']) == num_episodes
        assert history['episode_length'][0] == max_time_steps

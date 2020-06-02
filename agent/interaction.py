import numpy as np

from .agent import Agent
from .history import History

class UnityInteraction:
    """Class facilitates the interaction between an agent and a UnityEnvironment."""

    def __init__(self, agent, env):
        """Creates an instance of simulation.

        Args:
            agent: an instance of class implementing Agent
            env: an instance of UnityEnvironment environment

        """
        self._agent = agent
        self._env = env
        self._brain_name = env.brain_names[0]
        info = self.__class__.stats(env)
        self._state_size, self._action_size, self._num_agents = info
        self.history = History()

    @staticmethod
    def stats(env, brain_name=None):

        if brain_name is None:
            brain_name = env.brain_names[0]

        brain = env.brains[brain_name]
        action_size = brain.vector_action_space_size

        env_info = env.reset(train_mode=False)[brain_name]
        num_agents = len(env_info.agents)
        state_size = env_info.vector_observations.shape[1]

        return state_size, action_size, num_agents

    @property
    def action_size(self):
        return self._action_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def num_agents(self):
        return self._num_agents

    def run(self, num_episodes=1, max_time_steps=100):
        """Simulates the agent in the environment for given numbers of episodes.

        Args:
            num_episodes (int): number of episodes to simulate
            max_time_steps (int): maximum number of timesteps to play

        Returns:
            dict: episode lengths and rewards for each episode

        """
        self._agent.exploration(False)

        history = History()

        for i_episode in range(1, num_episodes + 1):

            env_info = self._env.reset(train_mode=False)[self._brain_name]
            states = env_info.vector_observations

            episode_rewards = np.zeros(self.num_agents)

            for timestep in range(1, max_time_steps + 1):

                # choose and execute an action in the environment
                actions = self._agent.act(states)
                env_info = self._env.step(actions)[self._brain_name]

                # observe the state and the reward
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                # score += reward
                episode_rewards += rewards

                states = next_states

                if np.any(dones):
                    break

            history.update(timestep, episode_rewards)

        return history.as_dict()

    def train(
            self,
            num_episodes=1,
            max_time_steps=100,
            score_target=None,
            score_window_size=100,
            verbose=1):
        """Trains the agent in the environment for a given number of episodes.

        Args:
            num_episodes (int): maximum number of training episodes
            max_time_steps (int): maximum number of timesteps per episode
            score_target (float): max total rewards collected during an episode
                at which to end training
            score_window_size (int): moving window size to calculate `score_target`
            verbose (bool): amount of printed output, if > 0 print progress bar

        Side effects:
            Alters the state of `agent` and `env`.

        Returns:
            dict: episode lengths and rewards for each episode

        """

        self._agent.exploration(True)

        if score_target is None:
            score_target = float('inf')

        for i_episode in range(1, num_episodes + 1):

            # prepare an agent, environment and reward calculations for a new episode
            self._agent.new_episode()

            env_info = self._env.reset(train_mode=True)[self._brain_name]
            states = env_info.vector_observations

            episode_rewards = np.zeros(self.num_agents)

            for timestep in range(1, max_time_steps + 1):

                # choose and execute actions
                actions = self._agent.act(states)
                env_info = self._env.step(actions)[self._brain_name]

                # observe state and reward
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                # save action, observation and reward for learning
                # TODO: DQNAgent only saves the first
                self._agent.step(states, actions, rewards, next_states, dones)
                states = next_states

                episode_rewards += rewards

                if np.any(dones):
                    break

            self.history.update(timestep, episode_rewards)

            if verbose:
                self._print_episode_statistics()

            if self.history.prev_score >= score_target:
                if verbose:
                    self._print_target_reached()
                break

        return self.history.as_dict()

    def _print_episode_statistics(self):
        """Prints a single row of statistics on episode performance."""

        if self.num_agents > 1:
            print(
                f"\rEpisode {self.history.num_episodes}"
                f" \tSteps: {self.history.prev_episode_length}"
                f" \tMax: {self.history.prev_reward_max:.2f}"
                f" \tMin: {self.history.prev_reward_min:.2f}"
                f" \tMean: {self.history.prev_reward_mean:.2f}"
                f" \tStd: {self.history.prev_reward_std:.2f}"
                f" \tScore: {self.history.prev_score:.2f}")
        else:
            print(
                f"\rEpisode {self.history.num_episodes}"
                f" \tSteps: {self.history.prev_episode_length}"
                f" \tReward: {self.history.prev_reward_max:.2f}"
                f" \tScore: {self.history.prev_score:.2f}")

    def _print_target_reached(self):
        """Prints a notification that target score has been reached."""
        print(f"\nTarget score reached in {self.history.num_episodes:d} episodes!")

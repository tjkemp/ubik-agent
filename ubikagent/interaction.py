import numpy as np

from .agent.agent import Agent
from .history import History
from .helper import print_episode_statistics, print_target_reached

class Interaction:
    """Class facilitates the interaction between an agent and a OpenAI Gym."""

    def __init__(self, agent, env, history=None):
        """Creates an instance of simulation.

        Args:
            agent: an instance of class implementing Agent
            env: an instance of Gym environment
            history: an instance of History to use to record the interaction

        """
        self._agent = agent
        self._env = env

        info = self.__class__.stats(env)
        self._state_size, self._action_size, self._num_agents = info

        if history is None:
            self.history = History()
        else:
            self.history = history

    @staticmethod
    def stats(env):
        """Function to reach state and action space sizes and number of agents."""
        raise NotImplementedError("stats() not implemented")

    @property
    def action_size(self):
        return self._action_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def num_agents(self):
        return self._num_agents

    def run(self, num_episodes=1, max_time_steps=100, verbose=1):
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

            state = self._env.reset()

            episode_rewards = 0.

            for timestep in range(1, max_time_steps + 1):

                # choose and execute an action in the environment
                action = self._agent.act(state)
                env_info = self._env.step(action)

                # observe the state and the reward
                next_state, reward, done, _ = env_info
                episode_rewards += reward
                state = next_state

                if done:
                    break

            agent_metrics = self._agent.new_episode()
            history.add_from(agent_metrics)
            history.update(timestep, episode_rewards)

            if verbose:
                self._print_episode_statistics(history)

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
        self._agent.new_episode()

        if score_target is None:
            score_target = float('inf')

        for i_episode in range(1, num_episodes + 1):

            state = self._env.reset()

            episode_rewards = 0.

            for timestep in range(1, max_time_steps + 1):

                # choose and execute an action in the environment
                action = self._agent.act(state)
                env_info = self._env.step(action)

                # observe the state and the reward
                next_state, reward, done, _ = env_info

                # save action, observation and reward for learning
                self._agent.step(state, action, reward, next_state, done)
                state = next_state

                episode_rewards += reward

                if done:
                    break

            agent_metrics = self._agent.new_episode()

            self.history.add_from(agent_metrics)
            self.history.update(timestep, episode_rewards)

            if verbose:
                self._print_episode_statistics(self.history)

            if self.history.score >= score_target:
                if verbose:
                    self._print_target_reached(self.history)
                break

        return self.history.as_dict()

    def _print_episode_statistics(self, history):
        """Prints a single row of statistics on episode performance."""
        print_episode_statistics(history)

    def _print_target_reached(self, history):
        """Prints a notification that target score has been reached."""
        print_target_reached(history)


class UnityInteraction(Interaction):
    """Class facilitates the interaction between an agent and a UnityEnvironment."""

    def __init__(self, agent, env, history=None):
        """Creates an instance of simulation.

        Args:
            agent: an instance of class implementing Agent
            env: an instance of UnityEnvironment environment

        """
        super().__init__(agent, env, history=history)
        self._brain_name = env.brain_names[0]

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

            history.update(timestep, episode_rewards.tolist())

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
        self._agent.new_episode()

        if score_target is None:
            score_target = float('inf')

        for i_episode in range(1, num_episodes + 1):

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

            agent_metrics = self._agent.new_episode()
            if isinstance(agent_metrics, dict):
                self.history.add_from(agent_metrics)

            self.history.update(timestep, episode_rewards.tolist())

            if verbose:
                self._print_episode_statistics(self.history)

            if self.history.score >= score_target:
                if verbose:
                    self._print_target_reached(self.history)
                break

        return self.history.as_dict()

class GymInteraction(Interaction):
    """Class facilitates the interaction between an agent and a OpenAI Gym."""

    def __init__(self, agent, env, history=None):
        """Creates an instance of simulation.

        Args:
            agent: an instance of class implementing Agent
            env: an instance of Gym environment

        """
        super().__init__(agent, env, history=history)

    @staticmethod
    def stats(env):

        state_size = env.observation_space
        action_size = env.action_space
        num_agents = 1
        return state_size, action_size, num_agents

import numpy as np

from .agent.agent import Agent
from .history import History
from .helper import print_episode_statistics, print_target_reached


class BaseInteraction:
    """Base class for classes facilitating the interaction between an agent and an environment.

    This class should be extended with environment specific implementations of run() and stats().

    """
    def __init__(self, agent, env):
        """Creates an instance of an interaction between an agent and an environment.

        Args:
            agent: an instance of a class implementing abstract class Agent
            env: an instance of environment (specifics implemented by the class
                extending this abstract class)

        """
        self._agent = agent
        self._env = env

        info = self.__class__.stats(env)
        self._state_size, self._action_size, self._num_agents = info

    @property
    def action_size(self):
        return self._action_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def num_agents(self):
        return self._num_agents

    @staticmethod
    def stats(env):
        """Implements retrieving state and action space sizes and number of agents.

        The function is declared static so that environment details can be retrieved
        before an agent is instantiated.

        Args:
            See implementing class `Interaction`.

        Returns:
            See implementing class `Interaction`.

        """
        raise NotImplementedError("stats() not implemented")

    def run(self, num_episodes, max_time_steps, score_target, score_window_size, verbose):
        """Implements the loop to make the agent interact in the environment.

        The function is declared static so that environment details can be retrieved
        before an agent is instantiated.

        Args:
            See implementing class `Interaction`.

        Returns:
            See implementing class `Interaction`.

        """
        raise NotImplementedError("run() not implemented")

    def _print_episode_statistics(self, history):
        """Prints a single row of statistics on episode performance."""
        print_episode_statistics(history)

    def _print_target_reached(self, history):
        """Prints a notification that target score has been reached."""
        print_target_reached(history)


class Interaction(BaseInteraction):
    """Class facilitates the interaction between an agent and OpenAI Gym environment."""

    def __init__(self, agent, env):
        """Creates an instance of simulation.

        Args:
            agent: an instance of class implementing Agent
            env: an instance of Gym environment

        """
        super().__init__(agent, env)

    @staticmethod
    def stats(env):
        """Returns important environment details needed to instantiate an agent.

        Args:
            env: Gym environment

        Returns:
            state_size (int): size of the state space
            action_size (int): size of the action space
            num_agents (int): number of agents in the environment

        """
        state_size = env.observation_space
        action_size = env.action_space
        num_agents = 1
        return state_size, action_size, num_agents

    def run(
            self,
            num_episodes=1,
            max_time_steps=100,
            score_target=None,
            score_window_size=100,
            verbose=1):
        """Runs the agent in the environment for a given number of episodes.

        Args:
            num_episodes (int): number of episodes to run, >= 1
            max_time_steps (int): maximum number of timesteps per episode
            score_target (float): mean total rewards collected (within a window)
                at which to end training
            score_window_size (int): moving window size to calculate `score_target`
            verbose (bool): amount of printed output, if > 0 print progress bar

        Side effects:
            Alters the state of `agent` and `env`.

        Returns:
            dict: episode lengths, rewards collected and any information
            the agent provides after each episode

        """

        history = History()
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
            history.add_from(agent_metrics)
            history.update(timestep, episode_rewards)

            if verbose:
                self._print_episode_statistics(history)

            if history.score >= score_target:
                if verbose:
                    self._print_target_reached(history)
                break

        return history.as_dict()


class UnityInteraction(Interaction):
    """Class facilitates the interaction between an agent and a UnityEnvironment."""

    def __init__(self, agent, env):
        """Creates an instance of simulation.

        Args:
            agent: an instance of class implementing Agent
            env: an instance of UnityEnvironment environment

        """
        super().__init__(agent, env)
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

    def run(
            self,
            num_episodes=1,
            max_time_steps=100,
            learn=True,
            score_target=None,
            score_window_size=100,
            verbose=1):
        """Run the agent in the environment for a given number of episodes.

        Args:
            num_episodes (int): number of episodes to run, >= 1
            max_time_steps (int): maximum number of timesteps per episode
            score_target (float): max total rewards collected during an episode
                at which to end training
            score_window_size (int): moving window size to calculate `score_target`
            verbose (bool): amount of printed output, if > 0 print progress bar

        Side effects:
            Alters the state of `agent` and `env`.

        Returns:
            dict: episode lengths, rewards collected and any information
            the agent provides after each episode

        """

        history = History()
        self._agent.new_episode()

        if score_target is None:
            score_target = float('inf')

        for i_episode in range(1, num_episodes + 1):

            env_info = self._env.reset(train_mode=learn)[self._brain_name]
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
                self._agent.step(states, actions, rewards, next_states, dones)
                states = next_states
                episode_rewards += rewards

                if np.any(dones):
                    break

            agent_metrics = self._agent.new_episode()
            history.add_from(agent_metrics)
            history.update(timestep, episode_rewards.tolist())

            if verbose:
                self._print_episode_statistics(history)

            if history.score >= score_target:
                if verbose:
                    self._print_target_reached(history)
                break

        return history.as_dict()
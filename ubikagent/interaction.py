import numpy as np

from ubikagent.history import History
from ubikagent.helper import print_episode_statistics, print_target_reached


class BaseInteraction:
    """Base class facilitating the interaction between an agent and an environment.

    This class should be extended with environment specific implementations of
    run() and stats().

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

    def run(self, num_episodes, max_time_steps):
        """Implements the loop to make the agent interact in the environment.

        Args:
            See implementing class `Interaction`.

        Returns:
            See implementing class `Interaction`.

        """
        raise NotImplementedError("run() not implemented")

    def _callback(
            self,
            event,
            callbacks,
            history=None,
            state=None,
            action=None,
            env_info=None
    ):
        """Iterates over callbacks and calls a function appropriate to the
        callback event.

        This function is used within `run()` to handle callbacks.

        Callback events with 'process_' prefix observe and possibly alter
        output received from and agent or an environment to make it easier
        to work with different kinds of outputs.

        Other events, start with either 'begin_' or 'end_' can be used
        similarly to observe and possibly alter states of either the agent
        or the environment.

        """
        for callback in callbacks:
            try:
                method = getattr(callback, event)
            except AttributeError as err:
                print(f"Error while calling event '{event}' in callback {callback}: {err}")
            if event == 'process_state':
                return method(state)
            elif event == 'process_action':
                return method(action)
            elif event == 'process_env_info':
                return method(env_info)
            else:
                method(self._agent, self._env, history)

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

    def run(
            self,
            num_episodes=1,
            max_time_steps=100,
            score_target=None,
            score_window_size=100,
            callbacks=[],
            verbose=1,
    ):
        """Runs the agent in the environment for a given number of episodes.

        Args:
            num_episodes (int): number of episodes to run, >= 1
            max_time_steps (int): maximum number of timesteps per episode
            score_target (float): mean total rewards collected (within a window)
                at which to end training
            score_window_size (int): moving window size to calculate `score_target`
            callbacks (list): list of instances of `Callback` which are called
                during execution
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

        self._callback('begin_training', callbacks)

        for i_episode in range(1, num_episodes + 1):

            state = self._env.reset()
            state = self._callback('process_state', callbacks)

            self._callback('begin_episode', callbacks)

            episode_rewards = 0.

            for timestep in range(1, max_time_steps + 1):

                # choose and execute an action in the environment
                action = self._agent.act(state)
                action = self._callback(
                    'process_action', callbacks, action=action)
                env_info = self._env.step(action)
                env_info = self._callback(
                    'process_env_info', callbacks, env_info=env_info)

                # observe the state and the reward
                next_state, reward, done, _ = env_info
                next_state = self._callback(
                    'process_state', callbacks, state=next_state)

                # save action, observation and reward for learning
                self._agent.step(state, action, reward, next_state, done)

                state = next_state
                episode_rewards += reward
                if done:
                    break

            agent_metrics = self._agent.new_episode()
            history.add_from(agent_metrics)
            history.update(timestep, episode_rewards)

            self._callback('end_episode', callbacks)

            if verbose:
                self._print_episode_statistics(history)

            if history.score >= score_target:
                if verbose:
                    self._print_target_reached(history)
                break

        self._callback('end_training', callbacks)

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
        self.state_size, self.action_size, self.num_agents = self.stats(env)
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
            learn (bool): whether to send observations to the agent
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
                if learn:
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

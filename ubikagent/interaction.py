import numpy as np

from ubikagent.history import History
from ubikagent.helper import print_episode_statistics, print_target_reached
from ubikagent.callback import BaseCallback, InteractionAdapter

class BaseInteraction:
    """Base class facilitating the interaction between an agent and an environment.

    This class can be extended with environment specific implementations.

    """
    def __init__(self, agent, env, adapter=None, base_callbacks=None):
        """Creates an instance of an interaction between an agent and an environment.

        Args:
            agent: an instance of a class implementing abstract class Agent
            env: an instance of environment (specifics implemented by the class
                extending this abstract class)

        """
        self._agent = agent
        self._env = env

        if adapter is None:
            self._adapter = InteractionAdapter()
        else:
            self._adapter = adapter

        if base_callbacks is None:
            self._default_callbacks = [BaseCallback()]
        else:
            self._default_callbacks = base_callbacks

    def run(self, num_episodes, max_time_steps):
        """Implements the loop to make the agent interact in the environment.

        Args:
            See implementing class `Interaction`.

        Returns:
            See implementing class `Interaction`.

        """
        raise NotImplementedError("run() not implemented")

    def _hook(
            self,
            event,
            data=None,
    ):
        """Calls an adapter function appropriate to the event.

        This function is used by `run()` to run methods of InteractionAdapter
        and extend Interaction class to handle non-standard environments and
        agents. See InteractionAdapter for methods that can be used to extend
        Interaction.

        """
        try:
            method = getattr(self._adapter, event)
        except AttributeError as err:
            print(f"Error while calling event '{event}' in adapter: {err}")
            raise

        if event.startswith('agent_'):
            return method(self._agent, data)
        elif event.startswith('env_'):
            return method(self._env, data)

        raise NotImplementedError(
            f"adapter does not implement method {event}")

    def _callback(
            self,
            event,
            data=None,
    ):
        """Iterates over callbacks and for each calls a function appropriate
        to the given event.

        This function is used within `run()` to handle callbacks. Callbacks
        can be used to observe and possibly alter states of either the agent
        or the environment. Events start with prefix 'begin_' or 'end_'.

        """
        for callback in self._callbacks:
            try:
                method = getattr(callback, event)
            except AttributeError as err:
                print(f"Error while calling event '{event}' in callback {callback}: {err}")
                raise

            method(self, self._agent, self._env, self.history)


class Interaction(BaseInteraction):
    """Class facilitates the interaction between an agent and OpenAI Gym environment."""

    def __init__(self, agent, env, history=None):
        """Creates an instance of simulation.

        Args:
            agent (ubikagent.agent.Agent): an instance of Agent
            env (gym.Env): an instance of Gym environment
            history (ubikagent.History): if None, a new instance is created

        """
        super().__init__(agent, env)
        self.history = History()
        self.i_episode = 0
        self.timestep = 1
        self.episode_rewards = 0.
        self.stop_training = False
        self.verbose = 0

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
            callbacks (list): list of instances of `Callback` which are called
                during execution
            verbose (bool): amount of printed output, if > 0 print progress bar

        Side effects:
            Alters the state of `agent` and `env`.

        Returns:
            dict: episode lengths, rewards collected and any information
            the agent provides after each episode

        """

        self.verbose = verbose
        self._callbacks = self._default_callbacks + callbacks

        self._callback('begin_training')

        for _ in range(1, num_episodes + 1):

            self.i_episode += 1
            self.episode_rewards = 0.

            self._hook('agent_reset')
            state = self._hook('env_reset')

            self._callback('begin_episode', callbacks)

            for idx in range(1, max_time_steps + 1):

                self.episode_timestep = idx

                # choose and execute an action in the environment
                action = self._hook('agent_act', state)
                env_info = self._hook('env_step', action)

                # observe the new state and the reward
                next_state, reward, done, info = env_info
                observation = (state, action, reward, next_state, done)

                # save action, observation and reward for learning
                self._hook('agent_observe', observation)

                self.episode_rewards += reward
                state = next_state

                if done:
                    break

            agent_state = self._hook('agent_reset')
            self.history.add_from(agent_state)

            self._callback('end_episode')

            if self.stop_training:
                break

        self._callback('end_training')

        return self.history.as_dict()


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

    def _print_episode_statistics(self, history):
        """Prints a single row of statistics on episode performance."""
        print_episode_statistics(history)

    def _print_target_reached(self, history):
        """Prints a notification that target score has been reached."""
        print_target_reached(history)

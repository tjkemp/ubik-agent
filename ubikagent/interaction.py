import os
import random

import numpy as np

from ubikagent.history import History
from ubikagent.helper import print_episode_statistics, print_target_reached
from ubikagent.callback import BaseCallback
from ubikagent.adapter import InteractionAdapter
from ubikagent import exception


class BaseInteraction:
    """Base class facilitating the interaction between an agent and an environment.

    This class should be extended with specific implementation.

    """
    def __init__(self, agent, env, seed, *args, **kwargs):
        """Creates an instance of an interaction between an agent and an environment.

        Sets the random seed for Python, Numpy and environment.

        Args:
            agent: an instance of a class implementing abstract class Agent
            env: an instance of environment (specifics implemented by the class
                extending this abstract class)

        """
        self._agent = agent
        self._env = env
        self._seed = seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)

    def run(self, num_episodes, max_time_steps):
        """Implements the loop to make the agent interact in the environment.

        Args:
            See implementing class `Interaction`.

        Returns:
            See implementing class `Interaction`.

        """
        raise NotImplementedError("run() not implemented")


class Interaction(BaseInteraction):
    """Class facilitates the interaction between an agent and OpenAI Gym environment."""

    def __init__(self, agent, env, seed, history=None, adapter=None, base_callbacks=None):
        """Creates an instance of interaction between an agent and an environment.

        Args:
            agent (ubikagent.agent.Agent): an instance of Agent
            env (gym.Env): an instance of Gym environment
            seed (int): a seed for `random` and `numpy`
            history (ubikagent.History): if None, a new instance of training
                history is created
            adapter (ubikagent.InteractionAdapter): class to adapt interaction
                to agent and environment, if they are not compliant with Gym
            base_callbacks (list of ubikagent.Callback): list of callbacks
                which are called wihtin the `run()` to observe

        """
        super().__init__(agent, env, seed)
        env.seed(seed)

        if history is None:
            self.history = History()
        else:
            self.history = history

        if adapter is None:
            self._adapter = InteractionAdapter()
        else:
            if isinstance(adapter, InteractionAdapter):
                self._adapter = adapter
            else:
                raise exception.UbikTypeError(
                    "argument `adapter` should be an instance of `InteractionAdapter`")

        if base_callbacks is None:
            self._default_callbacks = [BaseCallback()]
        else:
            self._default_callbacks = base_callbacks

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

        `run()` can be ran multiple times with an instance of Interaction and
        the state is not lost inbetween.

        Args:
            num_episodes (int): number of episodes to run, >= 1
            max_time_steps (int): maximum number of timesteps per episode
            callbacks (list): list of instances of `Callback` which are called
                during execution in addition to `base_callbacks`
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

from collections import deque

import numpy as np

from .agent import Agent


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

    @staticmethod
    def stats(env):

        brain_name = env.brain_names[0]

        brain = env.brains[brain_name]
        action_size = brain.vector_action_space_size

        env_info = env.reset(train_mode=False)[brain_name]
        state_size = len(env_info.vector_observations[0])
        num_agents = len(env_info.agents)

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
            list: cumulative rewards for each episode

        """
        scores = []

        for idx in range(1, num_episodes + 1):

            env_info = self._env.reset(train_mode=False)[self._brain_name]
            state = env_info.vector_observations
            score = np.zeros(self._num_agents)

            for step in range(1, max_time_steps + 1):

                # choose and execute an action in the environment
                action = self._agent.act(state)
                action = np.random.choice(self._action_size)
                env_info = self._env.step(action)[self._brain_name]

                # observe the state and the reward
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                ended = env_info.local_done[0]

                score += reward
                state = next_state

                if ended:
                    break

        return scores

    def train(
            self,
            n_episodes=1,
            max_time_steps=1000,
            target_score=0.5):
        """Trains the agent in the environment for a given number of episodes.

        Args:
            n_episodes (int): maximum number of training episodes
            max_time_steps (int): maximum number of timesteps per episode
            target_score (float): target score at which to end training

        Side effects:
            Alters the state of `agent` and `env`.

        Returns:
            list: sum of all rewards per episode

        """
        scores = []
        scores_window = deque(maxlen=100)

        for i_episode in range(1, n_episodes + 1):

            env_info = self._env.reset(train_mode=True)[self._brain_name]
            state = env_info.vector_observations[0]

            score = 0

            # agent.reset()

            for timestep in range(max_time_steps):

                # choose and execute actions
                action = self._agent.act(state)  # noise=True)
                action = np.clip(action, -1, 1)
                env_info = self._env.step(action)[self._brain_name]

                # observe state and reward
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                # save action, obervation and reward for learning
                self._agent.step(state, action, reward, next_state, done)
                state = next_state

                score += reward  # max(reward)

                # agent.learn(timestep)

                if np.any(done):
                    break

            scores.append(score)
            scores_window.append(score)

            best_score = max(scores)
            window_mean = np.mean(scores_window)

            # memory_size = len(agent.memory)
            memory_size = 1
            num_learn = 1
            # num_learn = agent.learn_counter

            print(
                f"\rEpisode {i_episode}\tScore: {score:.2f}\tBest: {best_score:.2f}"
                f"\tMean: {window_mean:.2f}"
                f"\tTimesteps: {timestep}\tMem: {memory_size}\tModel updates: {num_learn}")

            if window_mean >= target_score:
                print(f"\nTarget score reached in {i_episode:d} episodes!")
                # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                break

        return scores

import argparse
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
from agent import DQNAgent

def train(
        agent,
        env,
        n_episodes=10,
        max_time_steps=300,
        target_score=None,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995):
    """Training loop for given reinforcement learning agent in given environment.

    Args:
        agent: instance of class implementing Agent
        env: environment
        n_episodes (int): maximum number of training episodes
        max_time_steps (int): maximum number of timesteps per episode
        target_score (float): target score at which to end training
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon

    Side effects:
        Alters the state of `agent` and `env`.

    Returns:
        list: sum of all rewards per episode
    """
    scores = []
    scores_window = deque(maxlen=100)

    epsilon = eps_start
    brain_name = env.brain_names[0]

    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        score = 0
        for t in range(max_time_steps):

            # choose and execute an action
            action = agent.act(state, epsilon)
            env_info = env.step(action)[brain_name]

            # observe state and reward
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            ended = env_info.local_done[0]

            # save action, obervation and reward for learning
            agent.step(state, action, reward, next_state, ended)
            state = next_state

            score += reward

            if ended:
                break

        epsilon = max(eps_end, eps_decay * epsilon)

        scores.append(score)
        scores_window.append(score)

        mean_score = np.mean(scores_window)

        print(f"\rEpisode {i_episode}\tAverage Score: {mean_score:.2f}\tEpsilon: {epsilon:.2f}", end="")

        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {mean_score:.2f}")

        if target_score is not None:
            if mean_score >= target_score:
                print(f"\nTarget score reached in {i_episode:d} episodes!\tAverage Score: {mean_score:.2f}")
                break

    return scores

def main(
        filename_env_unity,
        n_episodes,
        max_time_steps,
        target_score):
    """Main function creates the environment and and the agent and starts training.

    Saves agent modelfile and score graph as files.
    """

    env = UnityEnvironment(file_name=filename_env_unity)

    # NOTE: the environment only has one brain: 'BananaBrain'
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size

    print(f"Creating an agent (state size of {state_size}, and {action_size} actions).")

    agent = DQNAgent(
        state_size,
        action_size)

    scores = train(
        agent,
        env,
        n_episodes=n_episodes,
        max_time_steps=max_time_steps,
        target_score=target_score)

    env.close()

    savefile = 'checkpoint.pth'
    print(f"Saving agent model as {savefile}.")
    agent.save(savefile)

    scorefile = 'scores.png'
    print(f"Saving score graph as {scorefile}.")
    plt.plot(list(range(1, len(scores) + 1)), scores)
    plt.ylabel('score')
    plt.xlabel('episode')
    plt.savefig(scorefile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trains an agent.')
    parser.add_argument(
        '--unity_env',
        nargs='?',
        help='unity environment file, default Banana_Linux_Novis',
        default='./Banana_Linux_NoVis/Banana.x86_64')
    parser.add_argument(
        '--episodes',
        nargs='?',
        type=int,
        help='number of training episodes, default 800',
        default=800)
    parser.add_argument(
        '--time_steps',
        nargs='?',
        type=int,
        help='maximum number of time steps in episode, default 300',
        default=300)
    parser.add_argument(
        '--target_score',
        nargs='?',
        type=float,
        help='stops training when target score is reached',
        default=None)
    args = parser.parse_args()

    main(
        filename_env_unity=args.unity_env,
        n_episodes=args.episodes,
        max_time_steps=args.time_steps,
        target_score=args.target_score)

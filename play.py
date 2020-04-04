import argparse
from collections import deque

from unityagents import UnityEnvironment
from agent import DQNAgent

def play(
        agent,
        env,
        max_time_steps=300,
        epsilon=0.):
    """Plays one episode with given agent model in given unity environment.

    Args:
        agent: instance of class implementing Agent
        env: environment
        max_time_steps (int): maximum number of timesteps to play
        epsilon (float): probabily of choosing a random action

    Returns:
        float: sum of all rewards
    """

    brain_name = env.brain_names[0]

    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]

    score = 0
    for step in range(1, max_time_steps + 1):

        # choose and execute an action in the environment
        action = agent.act(state, epsilon)
        env_info = env.step(action)[brain_name]

        # observe the state and the reward
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        ended = env_info.local_done[0]

        score += reward
        state = next_state

        if ended:
            break

    print(f"\rEpisode ended after {step} time steps with the score: {score:.2f}")

    return score

def main(
        filename_env_unity,
        modelfile):
    """ Main function creates the environment and and the agent and starts the simulation."""

    env = UnityEnvironment(file_name=filename_env_unity)

    # NOTE: the environment only has one brain: 'BananaBrain'
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]

    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size

    agent = DQNAgent(
        state_size,
        action_size)

    print(f"Loading the agent from file {modelfile}.")

    # TODO: check for exception
    agent.load(modelfile)

    scores = play(agent, env)

    env.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plays the simulation with a given agent.')
    parser.add_argument(
        '--unity_env',
        nargs='?',
        help='unity environment file, default `Banana_Linux_Novis`',
        default='./Banana_Linux_NoVis/Banana.x86_64')
    parser.add_argument(
        'modelfile',
        nargs='?',
        type=str,
        help="name of the agent's model file to load, default `checkpoint.pth`",
        default='checkpoint.pth')
    args = parser.parse_args()

    main(
        filename_env_unity=args.unity_env,
        modelfile=args.modelfile)

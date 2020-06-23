#!/usr/bin/env python

import os

from unityagents import UnityEnvironment

from ubikagent import UnityInteraction
from ubikagent.agent import DQNAgent, UnityRandomAgent
from ubikagent.agent.dqn import DQNAgentWithPER
from ubikagent.helper import (
    get_model_dir, create_model_dir, save_graph, save_history, parse_and_run)

ENV_PATH = './environments/Banana_Linux/Banana.x86_64'

MODEL_PARAMS = {
    'learning_rate': 5e-4,  # 5e-4
    'batch_size': 64,
    'tau': 1e-3,
    'gamma': 0.9997,
    'update_interval': 4,
    'replay_buffer_size': 65536,  # 1e5
    'seed': 4,
}

TRAINING_PARAMS = {
    'num_episodes': 500,
    'max_time_steps': 300,
    'score_target': 13.0,
}


class BananaCollector:

    def train(self, modelname):

        # create environment
        env = UnityEnvironment(file_name=ENV_PATH, no_graphics=True)

        # create an agent
        state_size, action_size, num_agents = UnityInteraction.stats(env)
        agent = DQNAgentWithPER(state_size, action_size, num_agents, **MODEL_PARAMS)

        # and create an interaction between them
        sim = UnityInteraction(agent, env)

        if modelname is not None:
            create_model_dir(modelname)

        history = sim.run(**TRAINING_PARAMS)

        if modelname is not None:
            modeldir = os.path.join(get_model_dir(modelname))
            agent.save(modeldir)
            save_history(modelname, history)
            save_graph(modelname, history['score'])

        env.close()

    def run(self, modelname):

        # create environment
        env = UnityEnvironment(file_name=ENV_PATH, no_graphics=False)

        # create an agent
        state_size, action_size, num_agents = UnityInteraction.stats(env)
        agent = DQNAgent(state_size, action_size, num_agents, **MODEL_PARAMS)
        modelfile = os.path.join(get_model_dir(modelname))
        agent.load(modelfile)

        # run simulation
        sim = UnityInteraction(agent, env)
        sim.run(learn=False)
        env.close()

    def random(self, modelname):
        # create environment
        env = UnityEnvironment(file_name=ENV_PATH, no_graphics=False)

        # create an agent
        state_size, action_size, num_agents = UnityInteraction.stats(env)
        agent = UnityRandomAgent(
            state_size, action_size, action_type='discrete', num_agents=num_agents)

        # create train or run loop
        sim = UnityInteraction(agent, env)
        sim.run(learn=False)
        env.close()


if __name__ == "__main__":
    project = BananaCollector()
    args = parse_and_run(project)

#!/usr/bin/env python

import os
import argparse
from unityagents import UnityEnvironment

from ubikagent.interaction import UnityInteraction
from ubikagent.helper import (
    get_model_dir, create_model_dir, save_graph, save_history, parse_and_run)
from ubikagent.agent import DDPGAgent
from ubikagent.agent import RandomAgent

ENV_PATH = './environments/Crawler_Linux/Crawler.x86_64'

MODEL_PARAMS = {
    'lr_actor': 3e-3,
    'layers_actor': [128, 64],
    'lr_critic': 3e-3,
    'layers_critic': [64, 32, 32],
    'batch_size': 128,
    'tau': 2e-2,
    'gamma': 0.99,
    'replay_buffer_size': 1e5,
    'seed': 42,
}

TRAINING_PARAMS = {
    'num_episodes': 300,
    'max_time_steps': 1000,
    'score_target': 2000.0,
}


class Crawler:
    def train(self, modelname):

        # create environment
        env = UnityEnvironment(file_name=ENV_PATH, no_graphics=True)

        # create an agent
        state_size, action_size, num_agents = UnityInteraction.stats(env)
        agent = DDPGAgent(state_size, action_size, num_agents, **MODEL_PARAMS)

        # and create an interaction between them
        sim = UnityInteraction(agent, env)

        if modelname is not None:
            create_model_dir(modelname)

        history = sim.run(**TRAINING_PARAMS)

        if modelname is not None:
            modelfile = os.path.join(get_model_dir(modelname))
            agent.save(modelfile)
            save_history(modelname, history)
            save_graph(modelname, history['score'])

        env.close()

    def run(self, modelname):

        # create environment
        env = UnityEnvironment(file_name=ENV_PATH, no_graphics=False)

        # create an agent
        state_size, action_size, num_agents = UnityInteraction.stats(env)
        agent = DDPGAgent(state_size, action_size, num_agents, **MODEL_PARAMS)
        modelfile = os.path.join(get_model_dir(modelname))
        agent.load(modelfile)

        # create train or run loop
        sim = UnityInteraction(agent, env)
        sim.run(learn=False)
        env.close()

    def random_run(self, modelname):

        # create environment
        env = UnityEnvironment(file_name=ENV_PATH, no_graphics=False)

        # create an agent
        state_size, action_size, num_agents = UnityInteraction.stats(env)
        agent = RandomAgent(
            state_size, action_size, action_type='continuous', num_agents=num_agents)

        # create train or run loop
        sim = UnityInteraction(agent, env)
        sim.run(learn=False)
        env.close()

if __name__ == "__main__":
    project = Crawler()
    args = parse_and_run(project)

class InteractionAdapter:

    def agent_reset(self, agent, data):
        return agent.new_episode()

    def env_reset(self, env, state):
        return env.reset()

    def agent_act(self, agent, state):
        return agent.act(state)

    def env_step(self, env, action):
        return env.step(action)

    def agent_observe(self, agent, observation):
        return agent.step(*observation)

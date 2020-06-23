class Callback:

    def __init__(self, *args, **kwargs):
        pass

    def begin_training(self, agent, env, history):
        pass

    def begin_episode(self, agent, env, history):
        pass

    def process_state(self, state):
        return state

    def process_action(self, action):
        return action

    def process_env_info(self, action):
        return action

    def end_episode(self, agent, env, history):
        pass

    def end_training(self, agent, env, history):
        pass

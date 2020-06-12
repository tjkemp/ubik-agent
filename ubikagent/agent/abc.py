import abc


class Agent(abc.ABC):

    @abc.abstractmethod
    def new_episode(self):
        pass

    @abc.abstractmethod
    def act(self, state):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def load(self, directory):
        pass

    @abc.abstractmethod
    def save(self, directory):
        pass

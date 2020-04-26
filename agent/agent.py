import abc


class Agent(abc.ABC):

    @abc.abstractmethod
    def act(self, state, eps=0.):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def load(self, filename):
        pass

    @abc.abstractmethod
    def save(self, filename):
        pass

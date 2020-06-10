import random
from collections import namedtuple, deque


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed

        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.Experience = namedtuple(
            "Experience",
            ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience into buffer."""

        exp = self.Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

    def sample(self):
        """Sample a random batch of experiences from buffer."""

        experiences = random.sample(self.buffer, k=self.batch_size)

        states = [exp.state for exp in experiences]
        actions = [exp.action for exp in experiences]
        rewards = [exp.reward for exp in experiences]
        next_states = [exp.next_state for exp in experiences]
        dones = [exp.done for exp in experiences if exp]

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self.buffer)
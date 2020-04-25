import random
from collections import namedtuple, deque

import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Adds a new experience into buffer."""

        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        """Randomly samples a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack(
            [exp.state for exp in experiences if exp is not None]
        )
        actions = np.vstack(
            [exp.action for exp in experiences if exp is not None]
        )
        rewards = np.vstack(
            [exp.reward for exp in experiences if exp is not None]
        )
        next_states = np.vstack(
            [exp.next_state for exp in experiences if exp is not None]
        )
        dones = np.vstack(
            [exp.done for exp in experiences if exp is not None]
        ).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

import random
from collections import namedtuple, deque

import numpy as np

from ubikagent.data_structure import SumTree


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed

        """
        self._storage = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.Experience = namedtuple(
            "Experience",
            ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience into buffer."""

        exp = self.Experience(state, action, reward, next_state, done)
        self._storage.append(exp)

    def sample(self):
        """Sample a random batch of experiences from buffer."""

        experiences = random.sample(self._storage, k=self.batch_size)

        states = [exp.state for exp in experiences]
        actions = [exp.action for exp in experiences]
        rewards = [exp.reward for exp in experiences]
        next_states = [exp.next_state for exp in experiences]
        dones = [exp.done for exp in experiences]

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self._storage)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with Proportional Prioritization.

    In reinforcement learning, prioritizing which transitions are replayed
    can make experience replay more effective compared to if all transitions
    are replayed uniformly.

    Related paper: https://arxiv.org/pdf/1511.05952.pdf

    """
    def __init__(
            self,
            buffer_size,
            batch_size,
            epsilon=0.001,
            alpha=0.6,
            beta=0.6,
            seed=None,
    ):
        """Initialize an instance of Prioritized Experience Replay.

        Raises:
            ValueError: max_buffer must be a power of two

        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of training batches to sample
            epsilon (float): value that is added to all priorities
            alpha (float): exponent which determines how much priorization is
                used, with alpha == 0 corresponding to the uniform case
            beta (float): important sampling bias correction exponent, where
                beta == 1 corresponds to full bias correction
            seed (int): optional, seed for randomness

        """
        try:
            self._storage = SumTree(buffer_size)
        except ValueError:
            raise
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta

        self.Experience = namedtuple(
            "Experience",
            ["state", "action", "reward", "next_state", "done"])
        self.highest_priority = 0.1
        self.highest_isweight = 0.
        self._sampled_indices = deque()

        if seed is not None:
            self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, priority=None):
        """Adds a new experience into buffer."""

        exp = self.Experience(state, action, reward, next_state, done)

        if priority is not None:
            priority = pow(abs(priority) + self.epsilon, self.alpha)
            if priority > self.highest_priority:
                self.highest_priority = priority
        else:
            priority = self.highest_priority

        self._storage.append(priority, exp)

    def update_priorities(self, new_priorities):
        """Updates priorities for previously sampled batch of experience."""

        if len(new_priorities) != len(self._sampled_indices):
            raise ValueError(
                "sample() should be called before called right before "
                "calling this method, and length of argument "
                "'new_priorities' should match batch_size")

        new_priorities = np.power(np.abs(new_priorities) + self.epsilon, self.alpha)

        max_priority = np.max(new_priorities)
        if max_priority > self.highest_priority:
            self.highest_priority = max_priority

        for index, new_priority in zip(self._sampled_indices, new_priorities):
            self._storage.update_priority(index, new_priority.item())

    def sample(self):
        """Randomly samples a batch of experiences from buffer.

        Raises:
            IndexError: if not enough items in buffer, len() needs to be at
                least size of batch_size

        Returns:
            list: a randomly sampled list of objects stored in the buffer

        """
        if len(self._storage) < self.batch_size:
            raise IndexError("not enough items in buffer to sample(), try again later")

        self._sampled_indices.clear()
        range_size = self.sum() / self.batch_size
        sample_priorities = deque()

        for range_start, range_end in zip(
                range(self.batch_size), range(1, self.batch_size + 1)):

            priority = random.uniform(
                range_size * range_start, range_size * range_end)

            index = self._storage.retrieve(priority)
            self._sampled_indices.append(index)
            priority = self._storage.get_priority(index)
            sample_priorities.append(priority)

        experiences = [self._storage[idx] for idx in self._sampled_indices]

        importance_sampling_weights = np.power(
            self.sum() * np.array(sample_priorities), -self.beta)

        max_weight = np.max(importance_sampling_weights)
        if max_weight > self.highest_isweight:
            self.highest_isweight = max_weight
        importance_sampling_weights /= self.highest_isweight

        return (importance_sampling_weights,) + self._unpack_samples(experiences)

    def sum(self):
        """Returns the total sum of priorities within the buffer."""
        return self._storage.sum

    def _unpack_samples(self, samples):
        """Unpacks a list of Experience samples into tuples."""

        states = [exp.state for exp in samples]
        actions = [exp.action for exp in samples]
        rewards = [exp.reward for exp in samples]
        next_states = [exp.next_state for exp in samples]
        dones = [exp.done for exp in samples]

        return (states, actions, rewards, next_states, dones)

    def __getitem__(self, index):
        """Returns the item from the Replay Buffer at the given `index`.

        Args:
            index (int): index of the item

        Raises:
            IndexError: if index is out of range

        Returns:
            object: the object at location `index`

        """
        try:
            item = self._storage[index]
        except IndexError:
            raise
        return item

    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self._storage)

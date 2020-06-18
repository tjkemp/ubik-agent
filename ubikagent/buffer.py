import random
from collections import namedtuple, deque

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
        dones = [exp.done for exp in experiences]

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with Proportional Prioritization.

    In reinforcement learning, prioritizing which transitions are replayed
    can make experience replay more effective compared to if all transitions
    are replayed uniformly.

    Related paper: https://arxiv.org/pdf/1511.05952.pdf

    """
    def __init__(self, buffer_size, batch_size, seed=None):
        """Initialize an instance.

        Raises:
            ValueError: max_buffer must be a power of two

        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of training batches to sample
            seed (int): optional, seed for randomness

        """
        try:
            self.sumtree = SumTree(buffer_size)
        except ValueError:
            raise
        self.batch_size = batch_size
        self.Experience = namedtuple(
            "Experience",
            ["state", "action", "reward", "next_state", "done"])
        self.sampled_indices = deque()

        if seed is not None:
            self.seed = random.seed(seed)

    def add(self, priority, state, action, reward, next_state, done):
        """Adds a new experience into buffer."""
        exp = self.Experience(state, action, reward, next_state, done)
        self.sumtree.append(abs(priority), exp)

    def update_priorities(self, new_priorities):
        """Updates priorities for previously sampled batch of experience."""

        if len(new_priorities) != len(self.sampled_indices):
            raise ValueError(
                "sample() should be called before called right before calling this method, "
                "and length of argument 'new_priorities' should match batch_size")

        for index, new_priority in zip(self.sampled_indices, new_priorities):
            self.sumtree.update_priority(index, new_priority)

    def sample(self):
        """Randomly samples a batch of experiences from buffer.

        Raises:
            IndexError: if not enough items in buffer, len() needs to be at
                least size of batch_size

        Returns:
            list: a randomly sampled list of objects stored in the buffer

        """
        if len(self.sumtree) < self.batch_size:
            raise IndexError("not enough items in buffer to sample(), try again later")

        self.sampled_indices.clear()
        range_size = self.sumtree.sum / self.batch_size

        for batch, batch_next in zip(
                range(self.batch_size), range(1, self.batch_size + 1)):
            priority = random.uniform(
                range_size * batch,
                range_size * batch_next)
            index = self.sumtree.retrieve(priority)
            self.sampled_indices.append(index)

        experiences = [self.sumtree[idx] for idx in self.sampled_indices]

        return self._unpack_samples(experiences)

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
            item = self.sumtree[index]
        except IndexError:
            raise
        return item

    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self.sumtree)

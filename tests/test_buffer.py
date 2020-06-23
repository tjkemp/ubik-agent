from collections import namedtuple

from ubikagent.buffer import PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer:

    def test_add_adds_items(self):
        buffer_size = 16
        batch_size = 4
        buffer = PrioritizedReplayBuffer(buffer_size, batch_size)

        experience = (0.1, 100, 2, 3., 200, False)
        for _ in range(buffer_size):
            buffer.add(*experience)

        assert len(buffer) == buffer_size

    def test_items_are_namedtuples(self):
        buffer_size = 16
        batch_size = 4
        buffer = PrioritizedReplayBuffer(buffer_size, batch_size)

        Experience = namedtuple(
            "Experience",
            ["state", "action", "reward", "next_state", "done"])
        experience = Experience(100, 2, 3., 200, False)
        priority = 1.0

        for _ in range(buffer_size):
            buffer.add(priority, *experience)

        for idx in range(buffer_size):
            assert buffer[idx] == experience

    def test_sample_priorities_are_updated(self):
        buffer_size = 16
        batch_size = 4
        buffer = PrioritizedReplayBuffer(buffer_size, batch_size)

        priority = 1.0
        experience = (priority, 100, 2, 3., 200, False)
        for _ in range(buffer_size):
            buffer.add(*experience)

        assert buffer.sum() == priority * buffer_size + buffer.min_priority * buffer_size

        _ = buffer.sample()
        new_priority = 100.0
        new_priorities_for_samples = [new_priority for _ in range(batch_size)]
        buffer.update_priorities(new_priorities_for_samples)

        assert buffer.sum() > 100.0

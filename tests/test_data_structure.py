import pytest

from ubikagent.data_structure import SumTree


class TestSumTree:

    def test_max_length_can_only_be_power_of_two(self):

        with pytest.raises(ValueError):
            _ = SumTree(3)

        with pytest.raises(ValueError):
            _ = SumTree(5)

    def test_doctring_example_works_as_described(self):

        tree = SumTree(2)
        tree.append(0.1, "item object")
        assert tree[0] == "item object"

    def test_append_adds_items(self):

        tree = SumTree(2)
        tree.append(0.1, 10)
        tree.append(0.1, 20)

        assert tree[0] == 10
        assert tree[1] == 20

    def test_append_overflow_replaces_oldest_item(self):

        tree = SumTree(2)
        tree.append(0.1, 10)
        tree.append(0.1, 20)
        tree.append(0.1, 30)

        assert tree[0] == 30
        assert tree[1] == 20

    def test_total_priority_is_correct(self):

        tree = SumTree(2)
        tree.append(0.1, 10)
        tree.append(0.2, 20)

        assert tree.total_priority == pytest.approx(0.3, 0.01)

    def test_total_priority_of_empty_tree_is_zero(self):

        tree = SumTree(2)
        assert tree.total_priority == 0.0

    def test_total_priority_of_incomplete_tree_is_correct(self):

        tree = SumTree(4)
        tree.append(0.1, 10)
        tree.append(0.2, 20)
        tree.append(0.3, 30)

        assert tree.total_priority == pytest.approx(0.6, 0.01)

    def test_total_priority_correct_after_overflow(self):

        tree = SumTree(2)
        tree.append(0.1, 10)
        tree.append(0.2, 20)
        tree.append(0.3, 20)

        assert tree.total_priority == pytest.approx(0.5, 0.01)

    def test_indexing_non_existing_item_raises_error(self):

        tree = SumTree(2)
        tree.append(0.1, 10)

        with pytest.raises(IndexError):
            _ = tree[1]

    def test_retrieve_returns_object_correctly(self):

        tree = SumTree(2)
        tree.append(0.1, 10)
        tree.append(0.2, 10)

        assert tree.retrieve(0.05) == 0
        assert tree.retrieve(0.15) == 1

    def test_retrieve_correct_on_two_level_tree(self):

        tree = SumTree(4)
        tree.append(0.1, 10)
        tree.append(0.1, 10)
        tree.append(0.1, 10)

        assert tree.retrieve(0.25) == 2

    def test_retrieve_correct_at_priority_boundaries(self):

        tree = SumTree(2)
        tree.append(0.1, 10)
        tree.append(0.2, 20)

        assert tree.retrieve(0.1) == 0
        assert tree.retrieve(0.2) == 1

    def test_retrieve_return_last_item_when_priority_gt_total_priority(self):

        expected_item = 20
        tree = SumTree(4)
        tree.append(0.1, 10)
        tree.append(0.1, expected_item)

        priority = tree.total_priority + 10.0
        index = tree.retrieve(priority)
        assert tree[index] == expected_item

import math


class SumTree(object):
    """Implementation of sum-tree data structure as a binary heap.

    Sum-tree provides an efficient way of calculating the cumulative
    sum of priorities. Leaf nodes store priorities and payloads, and
    the internal nodes are intermediate sums of priorities, with the
    parent node containing the sum over all priorities.

    An instance of SumTree can be used almost like a Python list. The
    class implements `append()` for adding items and getting items by
    indices. For example:

    >>> tree = SumTree(max_length=2)
    >>> tree.append(0.1, "item object")
    >>> tree[0]
    "item object"

    Note that the class implements only features required to implement
    Prioritized Replay Buffer.

    The major differences compared to a list are that
        - the `append()` is the only way too add items,
        - each item in the list will have a priority (positive float),
        - when the `max_length` has been reached, old items will be
        replaced last in first out basis, and
        - the sum of items' priorities is constantly maintained.

    Insertion time complexity is O(log n).

    """
    def __init__(self, max_length):
        """Initialize a Sum-Tree.

        Args:
            max_length (int): maximum number of items that are
                remembered, has to be a power of two

        """
        power_of_two = (max_length & (max_length - 1) == 0) and max_length > 1
        if not power_of_two:
            raise ValueError("max_length must be power of two")
        self._num_leaves = max_length
        self.clear()

    @property
    def total_priority(self):
        """Return the combined total priority of items in the queue."""
        return self._heap[0]

    def clear(self):
        """Reset the internal data structure."""

        self._heap = [0.] * (self._num_leaves - 1) + [None] * self._num_leaves
        self._num_appends = 0
        self._index = self._num_leaves - 1

    def append(self, priority, item):
        """Append an item and a priority into the Sum-Tree.

        Args:
            priority (float): the priority of item
            item (object): the item to be stored

        """
        valid_priority = isinstance(priority, float)
        if not valid_priority:
            raise ValueError("priority must be a float")

        self._heap[self._index] = (abs(priority), item)
        self._propagate_sum(self._index)
        self._num_appends += 1
        self._index = \
            self._first_index() + \
            self._num_appends % self._num_leaves

    def update_priority(self, index, new_priority):
        """Updates the priority for the item at given index."""
        try:
            item = self._heap[index]
        except IndexError:
            raise
        self._heap[index] = (abs(new_priority), item)
        # self._propagate_sum(index)

    def retrieve(self, priority):
        """Traverse through the list, sum priorities and return the index of
        the item at which the summed priority matches the argument `priority`.

        That is the naive and the easy to explain description of what this
        function does, but actually in this implmentation SumTree is used to
        retrieve the item in time O(log n) compared to time O(N) of the naive
        solution.

        Args:
            priority (float): the priority at which to end traversing
                and to return the index of an item

        Returns:
            index (int): the index of the item at which the sum of
                priorities was greater or equal to given `priority`

        """
        if self._num_appends == 0:
            raise IndexError("To retrieve the tree needs to have at least one item")

        if priority > self.total_priority:
            return self._last_index()

        node = self._retrieve(priority, 0)
        if self._is_leaf(node):
            return node - (self._num_leaves - 1)
        else:
            return self._last_index()

    def _retrieve(self, priority, node):
        """Tree search for `retrieve()` as a recursive function."""

        if self._is_leaf(node):
            return node

        left_node = self._left_child(node)
        left_priority = self._priority(left_node)

        if left_priority >= priority:
            return self._retrieve(priority, left_node)
        else:
            right_node = self._right_child(node)
            return self._retrieve(priority - left_priority, right_node)

    def _priority(self, index):
        """Return the priority of the node at given `index`."""

        if isinstance(self._heap[index], tuple):
            return self._heap[index][0]
        elif self._heap[index] is None:
            return 0.
        else:
            return self._heap[index]

    def _propagate_sum(self, index):
        """Propagate priority changes up along the tree."""

        index_parent = self._parent(index)
        left_priority = self._priority(self._left_child(index_parent))
        right_priority = self._priority(self._right_child(index_parent))
        self._heap[index_parent] = left_priority + right_priority

        if not self._is_root(index_parent):
            self._propagate_sum(index_parent)

    def _is_root(self, index):
        """Return True, if node at `index` is root, False otherwise."""
        return (index == 0)

    def _parent(self, index):
        """Return the parent of node at `index`."""
        return math.floor((index - 1) / 2)

    def _left_child(self, index):
        """Return the left child of node at `index`."""
        return 2 * index + 1

    def _right_child(self, index):
        """Return the right child of node at `index`."""
        return 2 * index + 2

    def _is_leaf(self, index):
        """Return True, if node at `index` is a leaf, False otherwise."""
        return index >= self._first_index()

    def _first_index(self):
        """Return index of the first item in the internal storage."""
        return self._num_leaves - 1

    def _last_index(self):
        """Return index of the last item in the internal storage."""
        if self._num_appends == 0:
            raise IndexError("list is empty")
        return min(self._num_appends, self._num_leaves) - 1

    def __getitem__(self, index):
        index_ = index + self._first_index()
        try:
            item = self._heap[index_][1]
        except IndexError:
            raise IndexError("index out of range")
        except TypeError:
            raise IndexError("index out of range")
        return item

    def __len__(self):
        return min(self._num_leaves, self._num_appends)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._num_leaves})'

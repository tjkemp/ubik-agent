import math


class SumTree:
    """An implementation of a Sum Tree data structure.

    Sum Tree is like deque (collections.deque) with a set max length where
    each item has an attached priority, and the total sum of priorities can be
    efficiently summed.

    Usage:

    In practice, an instance of Sum Tree can be used almost like a Python list.

    Sum Tree implements appending new items and retrieval of added items by
    indices.

    >>> tree = SumTree(16)
    >>> tree.append(0.1, "item object")
    >>> tree[0]
    "item object"

    The unique property of Sum Tree is that it efficiently keeps count of the
    total sum of priorities.

    >>> tree.append(0.1, "another object")
    >>> tree.sum
    0.2

    Complexities:

    Insertion time complexity is O(log n).
    Calculating sum of priorities time complexity is O(1).

    Implementation details:

    This Sum Tree implementation combines qualities from
        - deques (fast inserts, automatically discarding the oldest item after
        max length has been surpassed),
        - priority queues (each item has a "priority" associated with it), and
        - complete binary trees (as an internal data structure).

    All tree nodes are kept in a list. Leaf nodes store item priorities and
    payloads as tuples (priority, object). All non-leaf nodes store the the
    sum of their childrens priorities as floats. The root node stores the
    sum of all priorities as a float.

    Things to note:
        - in the code `leaf_node` refers to indices from the point of view of
        the user of the class, and `node_index` refers to indices in the
        internal tree data stucture
        - only features required to implement Prioritized Replay Buffer are
        implemented.

    """
    def __init__(self, max_length):
        """Initialize a Sum Tree.

        Raises:
            ValueError: if max_length is not a power of two

        Args:
            max_length (int): maximum number of items that are
                remembered, has to be a power of two

        """
        power_of_two = (max_length & (max_length - 1) == 0) and max_length > 1
        if not power_of_two:
            raise ValueError("max_length must be a power of two")
        self._num_leaves = max_length
        self.clear()

    @property
    def sum(self):
        """Return the combined total priority of items in the queue.

        Returns:
            float:  sum of all items' priorities

        """
        return self._tree[0]

    def clear(self):
        """Reset the internal data structure."""

        self._tree = [0.] * (self._num_leaves - 1) + [None] * self._num_leaves
        self._num_appends = 0
        self._current_index = self._leaf_to_node_index(0)

    def append(self, priority, item):
        """Append an item and a priority into the Sum Tree.

        Negative priorities are accepted, but are turned into absolute
        values.

        Args:
            priority (float): the priority of the appended item
            item (object): the item to be stored

        """
        valid_priority = isinstance(priority, float)
        if not valid_priority:
            raise ValueError("priority must be a float")

        self._tree[self._current_index] = (abs(priority), item)
        self._propagate_sum(self._current_index)
        self._num_appends += 1
        self._current_index = \
            self._leaf_to_node_index(0) + \
            self._num_appends % self._num_leaves

    def update_priority(self, index, new_priority):
        """Updates priority for the item at given index.

        Args:
            index (int): index of the item of which priority is updated
            new_priority (float): new priority given to the item

        Raises:
            IndexError: if there's no item at the given index

        """
        try:
            item = self._tree[index]
        except IndexError:
            raise

        node_index = self._leaf_to_node_index(index)
        self._tree[node_index] = (abs(new_priority), item)
        self._propagate_sum(node_index)

    def retrieve(self, priority):
        """Returns the leaf node index (?) at which the leaf node priorities
        sum up to greater than `priority` when traversing them from left to
        right.

        Sum Tree retrieves the item in time O(log n) by using the internal
        binary tree data strucutre, compared to time O(N) of the naive
        solution of traversing the leaf nodes from left to right.

        Args:
            priority (float): the priority at which to end traversing
                and to return the index of an item

        Raises:
            IndexError: if there are no items in the tree

        Returns:
            index (int): the index of the item at which the sum of
                priorities was greater or equal to given `priority`

        """
        if self._num_appends == 0:
            raise IndexError("the tree needs to have at least one item")

        if priority > self.sum:
            return self._rightmost_leaf_index()

        node_index = self._retrieve(priority, 0)
        if self._is_leaf(node_index):
            return self._node_to_leaf_index(node_index)
        else:
            return self._rightmost_leaf_index()

    def _retrieve(self, priority, node_index):
        """Tree search for `retrieve()` as a recursive function."""

        if self._is_leaf(node_index):
            return node_index

        left_node = self._left_child(node_index)
        left_priority = self._priority(left_node)

        if left_priority >= priority:
            return self._retrieve(priority, left_node)
        else:
            right_node = self._right_child(node_index)
            return self._retrieve(priority - left_priority, right_node)

    def _priority(self, node_index):
        """Return the priority of the node at given `node_index`."""

        if isinstance(self._tree[node_index], tuple):
            return self._tree[node_index][0]
        elif self._tree[node_index] is None:
            return 0.
        else:
            return self._tree[node_index]

    def _propagate_sum(self, node_index):
        """Propagate priority changes up along the tree."""

        index_parent = self._parent(node_index)
        left_priority = self._priority(self._left_child(index_parent))
        right_priority = self._priority(self._right_child(index_parent))
        self._tree[index_parent] = left_priority + right_priority

        if not self._is_root(index_parent):
            self._propagate_sum(index_parent)

    def _is_root(self, node_index):
        """Return True, if node at `node_index` is root, False otherwise."""
        return (node_index == 0)

    def _parent(self, node_index):
        """Return the index of the parent of a node at `node_index`."""
        return math.floor((node_index - 1) / 2)

    def _left_child(self, node_index):
        """Return the index of left child of a node at `node_index`."""
        return 2 * node_index + 1

    def _right_child(self, node_index):
        """Return the index of right child of a node at `node_index`."""
        return 2 * node_index + 2

    def _is_leaf(self, node_index):
        """Return True, if node at `node_index` is a leaf, False otherwise."""
        return node_index >= self._leaf_to_node_index(0)

    def _rightmost_leaf_index(self):
        """Return the leaf index of the last item in tree.

        Raises:
            IndexError: if there are no items in the tree

        Returns:
            integer: leaf index of rightmost leaf in the tree

        """
        if self._num_appends == 0:
            raise IndexError("function is not defined on empty trees")

        return len(self) - 1

    def _leaf_to_node_index(self, leaf_index):
        """Translates a leaf node index into an internal tree storage index."""
        return leaf_index + (self._num_leaves - 1)

    def _node_to_leaf_index(self, node_index):
        """Translates an internal tree storage index into a leaf node."""
        return node_index - (self._num_leaves - 1)

    def __getitem__(self, index):
        """Return a leaf item with given `index`.

        Args:
            index (int): index of the item in the list

        Raises:
            IndexError: if index is out of range

        Returns:
            object: the object at location `index`

        """
        node_index = self._leaf_to_node_index(index)
        try:
            item = self._tree[node_index][1]
        except IndexError:
            raise IndexError("index out of range")
        except TypeError:
            raise IndexError("index out of range")
        return item

    def __len__(self):
        """Return the number of items in the Sum Tree."""
        return min(self._num_leaves, self._num_appends)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._num_leaves})'

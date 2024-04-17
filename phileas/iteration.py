"""
This modules contains the trees used for data iteration.

The `DataTree` stores an actual data point, which consists of nested `dict` and
`list` objects, with `DataLiteral` leaves.

Then, the `IterationTree` provides a framework to build complex searches over
data trees. Its leaves consist of literal values, or data iterators. Those
leaves can simply be iterated over, but they can also be used to build more
complex iteration trees.

First, they can be combined with `IterationMethod` nodes, which provide a way to
iterate over multiple data sources. Then, `Transform` nodes can be inserted in
those trees in order to modify the data trees generated while iterating.
"""

import collections.abc
import dataclasses
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from math import exp, log
from typing import Callable, Generic, Iterator, TypeVar

from .utility import Sentinel

#################
### Data tree ###
#################

#: Data values that can be used
DataLiteral = typing.Union[None, bool, str, int, float, "_NoDefault"]

#: Dictionary keys
Key = None | bool | str | int | float

#: A data tree consists of literal leaves, and dictionary or list nodes
DataTree = DataLiteral | dict[Key, "DataTree"] | list["DataTree"]

########################
### Pseudo data tree ###
########################

# Note: a DataTree is a PseudoDataTree, but mypy seems not to understand it.

#: A leave of a pseudo data tree is either a data tree leave, or a non-trivial
#: iteration tree leave.
PseudoDataLiteral = typing.Union[DataLiteral, "NumericRange", "Sequence"]

#: A pseudo data tree is a data tree whose leaves can be non literal iteration
#: leaves.
PseudoDataTree = (
    PseudoDataLiteral | dict[Key, "PseudoDataTree"] | list["PseudoDataTree"]
)

######################
### Iteration tree ###
######################


### Iteration ###


class TreeIterator(ABC):
    """
    Iteration tree iterator. Compared to a usual iterator, it supports forward
    and backward iteration, and is endlessly usable. This means that, whenever
    it is "exhausted" (`__next__()` raise `StopIteration`), it can either be
    reset to its starting position with `reset()`, or its iteration direction
    can be switched using `reverse()`.
    """

    #: Current iteration direction of the iterator.
    forward: bool

    def __init__(self) -> None:
        self.forward = True

    def __iter__(self) -> Iterator[DataTree]:
        return self

    def reverse(self):
        """
        Reverse the iteration direction of the iterator, but stay at the same
        position. Thus, if `it` is any `TreeIterator`, the following behavior
        is expected:

        >>> it.reset()
        >>> it.reverse()
        >>> list(it)
        []

        If you want to iterate over an iteration tree `tree` backward, you have
        to reset the iterator after having reversed it:

        >>> it = iter(tree)
        >>> it.reverse()
        >>> it.reset()

        """
        self.forward = not self.forward

    @abstractmethod
    def reset(self):
        """
        Reset the internal state of the iterator, so that its next value will be
        the first iterated value in the current direction. It takes into
        account the value of forward, going either to the start or the end of
        the iterated collection.
        """
        raise NotImplementedError()

    @abstractmethod
    def __next__(self) -> DataTree:
        """
        Return the next iterated value in the current iteration direction.
        """
        raise NotImplementedError()


@dataclass
class ListIterator(TreeIterator):
    """
    Iterator over a list of data trees.

    It is to be replaced by dedicated iterators dedicated to each concrete
    numeric range, to reduce memory use.
    """

    #: Values being iterated
    sequence: list[DataTree]

    #: Position of the last yielded element.
    position: int = field(init=False)

    #: Cached length of the sequence
    length: int = field(init=False)

    def __post_init__(self):
        super().__init__()
        self.length = len(self.sequence)
        self.reset()

    def reset(self):
        if self.forward:
            self.position = -1
        else:
            self.position = self.length

    def reverse(self):
        super().reverse()

    def __next__(self) -> DataTree:
        if self.forward:
            self.position += 1

            if self.position >= self.length:
                self.position = self.length
                raise StopIteration
        else:
            self.position -= 1

            if self.position < 0:
                self.position = -1
                raise StopIteration

        return self.sequence[self.position]


### Indexing ###


class _Child(Sentinel):
    """
    Sentinel representing the index of the only child of a 1-ary iteration node.
    """

    pass


child = _Child()

ChildPath = list[Key | _Child]

### Default related objects ###


class NoDefaultPolicy(Enum):
    """
    Behavior of `default` for trees not having a default value.
    """

    #: Raise a `NoDefaultError` if any of the nodes in the tree does not have a
    #: default value.
    ERROR = "ERROR"

    #: Return the `_NoDefault` sentinel if any of the nodes in the tree does not
    #: have a default value.
    SENTINEL = "SENTINEL"

    #: Skip elements without a default element. If the root of the tree does
    #: not have a default value, return a `_NoDefault` sentinel.
    #:
    #: Note that this is not supported by iteration method nodes with list
    #: children.
    SKIP = "SKIP"


class NoDefaultError(Exception):
    """
    Indicates that `default()` has been called on an iteration tree where a node
    does not have a default value.
    """

    #: Path of a child without a default value.
    path: ChildPath

    def __init__(self, message: str, path: ChildPath) -> None:
        super().__init__(message)
        self.path = path

    def __str__(self):
        path_msg = (
            f" (path {'/'.join(map(str, self.path))})" if len(self.path) > 0 else ""
        )
        return f"{super().__str__()}{path_msg}"


class _NoDefault(Sentinel):
    """
    Sentinel representing a default value which is not set.
    """

    pass


#: You can store this value - instead of an actual default value - in instances
#: of classes that can have a default value, but don't.
no_default = _NoDefault()


### Iteration tree API ###


class IterationTree(ABC):
    """
    Represents a set of data trees, as well as the way to iterate over them. In
    order to be able to get a single data tree from an iteration tree, they are
    able to build a default data tree, which (usually) has the same shape as the
    generated data tree.
    """

    @abstractmethod
    def __iter__(self) -> TreeIterator:
        """
        Return a two-way resetable tree iterator.
        """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of data trees represented by the iteration tree. If it
        is finite, it should be the same as the number of elements yielded by
        `__iter__`. Otherwise, a `TypeError` is raised.
        """
        raise NotImplementedError()

    def iterate(self) -> Iterator[DataTree]:
        """
        Other name of `__iter__`, which can be more explicit in for example
        `list(tree.iterate())`.
        """
        return self.__iter__()

    @abstractmethod
    def to_pseudo_data_tree(self) -> PseudoDataTree:
        """
        Converts the iteration tree to a pseudo data tree.
        """
        raise NotImplementedError()

    @abstractmethod
    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        """
        Returns a default data tree. If the tree does not have a default value,
        follows the behavior dictated by `no_default_policy`.
        """
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, key: Key) -> "IterationTree":
        """
        It is implemented to allow working with an iteration tree as if it
        consisted of nested list and dict objects.
        """
        raise NotImplementedError()

    # Path API
    #
    # Internal nodes, and the structure, of iteration trees are to be modified
    # using a path-based API.
    #
    # After using a modification function, only the output of the function
    # should be used, and `self` should be discarded.

    def get(self, path: ChildPath) -> "IterationTree":
        """
        Get a node inside a tree. It should not be used to modify the tree.
        """
        current = self
        for key in path:
            current = current._get(key)

        return current

    @abstractmethod
    def _get(self, child_key: Key | _Child) -> "IterationTree":
        """
        Returns the root child with the given key, or raises a `KeyError` if
        there is no child with this key.
        """
        raise NotImplementedError()

    @abstractmethod
    def _insert_child(
        self, child_key: Key | _Child, child: "IterationTree"
    ) -> "IterationTree":
        """
        Insert a new child tree to the root and return the newly created tree.
        """
        raise NotImplementedError()

    @abstractmethod
    def _remove_child(self, child_key: Key | _Child) -> "IterationTree":
        """
        Remove a child of the root, and return the newly created tree. Raises a
        `KeyError` if there is no child with this key.
        """
        raise NotImplementedError()

    @abstractmethod
    def _replace_root(
        self, Node: type["IterationTree"], *args, **kwargs
    ) -> "IterationTree":
        """
        Change the root node of a tree, keeping the remaining of the tree
        unmodified. In order to keep a valid tree structure, `Node` must have
        be of the same type as the current root, otherwise a `TypeError` is
        raised.
        """
        raise NotImplementedError()

    def insert_child(
        self, path: ChildPath, tree: typing.Union["IterationTree", None]
    ) -> "IterationTree":
        """
        Insert a child anywhere in the tree, whose location is specified by path
        argument. Return the newly created tree.

        If there is already a node at this location, it will be replaced.

        If the specified tree is `None`, a node is supposed to exist at this
        location (otherwise, a `KeyError` is raised), and will be removed if
        possible. Only iteration methods nodes will support child removal, see
        their implementation of `_remove_child`. In any other case, a
        `TypeError` is raised.

        Note that the root of a tree cannot be removed, so specifying an empty
        path with a `None` tree will raise a `KeyError`.
        """
        if tree is None and len(path) == 1:
            return self._remove_child(path[0])
        if len(path) == 0:
            if tree is None:
                raise KeyError("Cannot remove the root of an iteration tree.")

            return tree
        else:
            key = path[0]
            new_child = self._get(key).insert_child(path[1:], tree)
            return self._insert_child(key, new_child)

    def remove_child(self, path: ChildPath) -> "IterationTree":
        """
        Remove a node in a tree. It is equivalent to `insert_child(path, None)`.
        """
        return self.insert_child(path, None)

    def insert_transform(
        self, path: ChildPath, Parent: type["Transform"], *args, **kwargs
    ) -> "IterationTree":
        """
        Insert a parent to the node at the given path, parent which is
        necessarily a transform node, as it will only have a single child. The
        parent is built using the `Parent` class, and the supplied arguments.

        The newly created tree is returned.
        """
        if not issubclass(Parent, Transform):
            raise TypeError("Cannot insert a non-transform node as a parent.")

        if len(path) == 0:
            return Parent(self, *args, **kwargs)

        key = path[0]
        new_child = self._get(key).insert_transform(path[1:], Parent, *args, **kwargs)
        new_me = self._insert_child(key, new_child)

        return new_me

    def replace_node(
        self, path: ChildPath, Node: type["IterationTree"], *args, **kwargs
    ) -> "IterationTree":
        """
        Replace the node at the given path with another one. The other node is
        built using its type, `Node`, and the `args` and `kwargs` arguments.
        Note that the sub-tree of the replaced node is not modified.

        This requires `Node` to be of the same kind of the node that is being
        replaced: a transform for a transform, an iteration method for an
        iteration method, a leaf for a leaf.
        """
        if len(path) == 0:
            return self._replace_root(Node, *args, **kwargs)
        else:
            key = path[0]
            new_child = self._get(key).replace_node(path[1:], Node, *args, **kwargs)
            new_me = self._insert_child(key, new_child)

            return new_me

    def depth_first_modify(
        self, modifier: Callable[["IterationTree", ChildPath], "IterationTree"]
    ) -> "IterationTree":
        """
        Using a post-fix depth-first search, replace each `node` of the tree,
        located at `path`, with `modifier(node, path)`.
        """
        return self._depth_first_modify(modifier, [])

    @abstractmethod
    def _depth_first_modify(
        self,
        modifier: Callable[["IterationTree", ChildPath], "IterationTree"],
        path: ChildPath,
    ) -> "IterationTree":
        """
        Recursive implementation of `depth_first_modify`, implemented by the
        different node types.
        """
        raise NotImplementedError()


### Nodes ###


@dataclass(frozen=True)
class IterationMethod(IterationTree):
    """
    Iteration node having multiple children, supplied either as a list or
    dictionary.

    In order to implement a concrete iteration method, you should sub-class
    `IterationMethod` and implement a corresponding `IterationMethodIterator`,
    which is returned by `__iter__()`.

    This should remain the only node in an iteration tree that can hold `dict`
    and `list` children. If you are tempted to create another node doing so,
    you should verify that it cannot be done by sub-classing `IterationMethod`
    instead.
    """

    #: The children of the node. It must not be empty.
    children: list[IterationTree] | dict[Key, IterationTree]

    #: Order of iteration over the children. How it is used depends on the
    #: concrete iteration method implementation. It must be a permutation of the
    #: set of keys of `children`.
    order: list[Key] | None = None

    #: Notify the iteration method to be lazy. For now, this feature is only
    #: supported for `dict` children. In this case, lazy iteration will just
    #: yield the keys that have changed at each step.
    #:
    #: Note that concrete iteration method classes are not required to actually
    #: implement lazy iteration, and if they don't, they will probably do so
    #: silently. Refer to their documentation or their implementation.
    lazy: bool = field(default=False)

    def __post_init__(self):
        if len(self.children) == 0:
            raise ValueError("Empty children are forbidden.")

        if not isinstance(self.children, (list, dict)):
            raise TypeError("IterationMethod must have list or dict children.")

        if self.order is not None:
            keys: set[Key]
            if isinstance(self.children, list):
                keys = set(range(len(self.children)))
            else:
                keys = set(self.children.keys())

            if len(self.order) != len(set(self.order)):
                raise ValueError("The iteration order must not have repetitions.")

            if set(self.order) != keys:
                msg = "The iteration order must be a permutation of the children keys."
                raise ValueError(msg)

        if self.lazy and not isinstance(self.children, dict):
            raise TypeError("Lazy iteration is only supported for dictionary children.")

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        if isinstance(self.children, list):
            return [child.to_pseudo_data_tree() for child in self.children]
        else:  # isinstance(self.children, dict)
            return {
                key: value.to_pseudo_data_tree() for key, value in self.children.items()
            }

    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        # List children
        if isinstance(self.children, list):
            if no_default_policy == NoDefaultPolicy.SKIP:
                msg = "NoDefaultPolicy.SKIP is not supported by iteration "
                msg += "methods with list children."
                raise TypeError(msg)

            default_list: list[DataTree | _NoDefault] = [None] * len(self.children)
            for position, child in enumerate(self.children):
                error: None | NoDefaultError = None
                try:
                    default_list[position] = child.default(no_default_policy)
                except NoDefaultError as err:
                    error = NoDefaultError(err.args[0], [position] + err.path)

                if error is not None:
                    raise error

            return default_list

        # Dict children
        default_dict: dict[Key, DataTree] = {}
        for key, child in self.children.items():
            error = None
            try:
                default_dict[key] = child.default(no_default_policy)
            except NoDefaultError as err:
                own_path: ChildPath = [key]
                error = NoDefaultError(err.args[0], own_path + err.path)

            if error is not None:
                raise error

        if no_default_policy == NoDefaultPolicy.SKIP:
            return {
                key: child
                for key, child in default_dict.items()
                if not child == no_default
            }
        else:
            return default_dict

    def __getitem__(self, Key) -> IterationTree:
        return self.children[Key]

    # Path API

    # Note: self.children[child_key] is not properly typed. Indeed, child_key
    # can be a dict or a list key, and there is no guarantee that self.children
    # is of the corresponding type. However, if the key does not exist, a
    # `KeyError` will be raised, which is the expected behavior. For those
    # reasons, it is valid to ignore[index] all the self.children
    # [child_key] expressions.

    def _get(self, child_key: Key | _Child) -> IterationTree:
        if isinstance(child_key, _Child):
            raise KeyError("Iteration method does not support Child() index.")

        return self.children[child_key]  # type: ignore[index]

    def _insert_child(
        self, child_key: Key | _Child, child: IterationTree
    ) -> IterationTree:
        if isinstance(child_key, _Child):
            raise KeyError("Iteration method does not support Child() index.")

        self.children[child_key] = child  # type: ignore[index]
        return self

    def _remove_child(self, child_key: Key | _Child) -> IterationTree:
        _ = self.children.pop(child_key)  # type: ignore[arg-type]
        return self

    def _replace_root(
        self, Node: type[IterationTree], *args, **kwargs
    ) -> IterationTree:
        if not issubclass(Node, IterationMethod):
            raise TypeError(f"Cannot replace an iteration method with a {Node}")

        return Node(self.children, *args, **kwargs)

    def _depth_first_modify(
        self,
        modifier: Callable[[IterationTree, ChildPath], IterationTree],
        path: ChildPath,
    ) -> IterationTree:
        new_children: dict[Key, IterationTree] | list[IterationTree]
        if isinstance(self.children, list):
            new_children = [
                child._depth_first_modify(modifier, path + [position])
                for position, child in enumerate(self.children)
            ]
        else:
            new_children = {
                position: child._depth_first_modify(modifier, path + [position])
                for position, child in self.children.items()
            }

        return modifier(dataclasses.replace(self, children=new_children), path)


class IterationMethodIterator(TreeIterator):
    """
    Base class used to implement concrete `IterationMethod` nodes iterators, and
    providing helper attributes to do so.
    """

    tree: IterationMethod

    #: Children iterators stored in a list. This, with `keys`, allows
    #: child-class to only implement iteration over lists.
    #:
    #: If the iteration tree does specify an order, use it. Otherwise,
    #: dictionaries are sorted by their key value, and lists keep the same
    #: order.
    iterators: list[TreeIterator]

    #: Access keys for the children iterators, such that
    #: `iterators[i]` is an iterator of `tree.children[keys[i]]`.
    #:
    #: Note that the previous statement is not properly typed, as the current
    #: type hints do not state that `keys` contains valid keys for `children`.
    #: However, the constructor takes care of the validity of the runtime
    #: types. Because of that, ignoring type checks can be required when
    #: implementing concrete iteration methods.
    keys: list[Key | int]

    def __init__(self, tree: IterationMethod) -> None:
        super().__init__()
        self.tree = tree

        if tree.order is not None:
            self.keys = tree.order
        elif isinstance(tree.children, list):
            self.keys = list(range(len(tree.children)))
        else:  # isinstance(tree.children, dict)
            # An exception will be raised if comparison is not possible
            self.keys = sorted(tree.children.keys())  # type: ignore[type-var]

        self.iterators = [iter(tree.children[key]) for key in self.keys]  # type: ignore[index]

    def reset(self):
        for iterator in self.iterators:
            iterator.reset()

    def reverse(self):
        super().reverse()
        for iterator in self.iterators:
            iterator.reverse()


@dataclass(frozen=True)
class CartesianProduct(IterationMethod):
    """
    Iteration over the cartesian product of the children. The iteration order is
    the same as `itertools.product`. In other words, iteration will behave
    roughly as

    ```py
    for v1 in c1:
        for v2 in c2:
            for v3 in c3:
                yield [v1, v2, v3]
    ```

    If an order is specified, its first element will correspond to the outermost
    loop, and its last to the innermost one.
    """

    #: Enable snake iteration, which guarantees that successive yielded elements
    #: differ by only one key at most (a la Gray code).
    snake: bool = False

    def __iter__(self) -> TreeIterator:
        return CartesianProductIterator(self)

    def __len__(self) -> int:
        children: collections.abc.Sized
        if isinstance(self.children, list):
            children = self.children
        else:  # isinstance(self.chidren, dict)
            children = self.children.values()

        return reduce(int.__mul__, map(len, children), 1)


class CartesianProductIterator(IterationMethodIterator):
    base: list | dict

    #: Indicates if the iterator is exhausted, in the forward and backward
    #: directions.
    done: tuple[bool, bool]

    #: Contains, at `position`, whether the tree children at the position
    #: `key[position]` is 1 element long.
    trivial_length_child: list[bool]

    def __init__(self, product: CartesianProduct) -> None:
        super().__init__(product)
        self.trivial_length_child = [
            len(self.tree.children[key]) == 1 for key in self.keys  # type: ignore[index]
        ]
        self.reset()

    def __base(self):
        if isinstance(self.tree.children, list):
            return [next(iterator) for iterator in self.iterators]
        else:
            return {
                key: next(iterator) for key, iterator in zip(self.keys, self.iterators)
            }

    def reset(self):
        super().reset()
        self.done = (False, True)

    def reverse(self):
        super().reverse()
        self.done = (self.done[1], self.done[0])

    def __next__(self) -> DataTree:
        """
        Lazy iteration is supported, and works except for the case where a child
        has only one element.
        """
        # Last iteration in the current direction
        if self.done[0]:
            raise StopIteration

        # First iteration in the current direction
        if self.done[1]:
            self.base = self.__base()
            self.done = (False, False)
            return self.base.copy()

        if self.tree.lazy:
            self.base = {}

        # First, find all the children iterators that are exhausted.
        position = len(self.iterators) - 1
        while position >= 0:
            iterator = self.iterators[position]
            try:
                self.base[self.keys[position]] = next(iterator)  # type: ignore[index]
                break
            except StopIteration:
                position -= 1

        # At this point, all the children at positions > `position` are
        # exhausted in their current direction.

        # Then,
        #   - register that the iterator is exhausted if all of them are
        #     exhausted;
        #   - or reset/reverse them, and yield the current value.
        if position == -1:
            self.done = (True, False)
            raise StopIteration
        else:
            position += 1
            while position < len(self.iterators):
                iterator = self.iterators[position]

                assert isinstance(self.tree, CartesianProduct)
                if self.tree.snake:
                    iterator.reverse()
                else:
                    iterator.reset()

                if self.tree.snake and self.tree.lazy:
                    # Just consume the former last, now first, value, as it does
                    # not require to be updated
                    next(iterator)
                elif self.tree.lazy and self.trivial_length_child[position]:
                    # This "new" value is the same as before, as the child has
                    # only one element. Just consume it.
                    next(iterator)
                else:
                    self.base[self.keys[position]] = next(iterator)  # type: ignore[index]

                position += 1

            self.done = (False, False)

            return self.base.copy()


@dataclass(frozen=True)
class Union(IterationMethod):
    """
    Iteration over one child at a time, starting with the first one (or the
    first one of the order, if specified).

    For children that have a default value, they will be reset to it when they
    are not being iterated. However, there will be no complain about children
    not implementing a default value.
    """

    #: Policy used while iterating, to generate the values returned for the
    #: children that are not being iterated over.
    no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR

    def __iter__(self) -> TreeIterator:
        return UnionIterator(self)

    def __len__(self) -> int:
        children: collections.abc.Sized
        if isinstance(self.children, list):
            children = self.children
        else:  # isinstance(self.children, dict)
            children = self.children.values()
        return sum(map(len, children))


@dataclass
class UnionIterator(IterationMethodIterator):
    #: Iterator over the children indices, whose values are stored in
    #: `current_index`, and used to know which child is currently iterated
    #: over.
    indices: ListIterator = field(init=False)

    #: Special value -1 just after reset. Even for lazy iteration, returns all
    #: the values.
    current_index: int = field(init=False)

    #: Indicates that, at the next iteration, a new `current_index` will be
    #: drawn.
    change_index: bool = field(init=False)

    #: Cached children default values.
    default_values: list | dict = field(init=False)

    def __init__(self, union: Union) -> None:
        super().__init__(union)

        self.indices = ListIterator(list(range(len(self.iterators))))
        self.reset()

        self.yield_all_children = True
        self.default_values = self.__default_values()

    def __default_values(self) -> dict | list:
        assert isinstance(self.tree, Union)
        default_values = self.tree.default(self.tree.no_default_policy)
        assert isinstance(default_values, (dict, list))

        return default_values

    def reset(self):
        super().reset()
        self.indices.reset()
        self.current_index = -1
        self.change_index = True

    def reverse(self):
        super().reverse()
        self.indices.reverse()

    def __next__(self) -> DataTree:
        return self.__next()

    def __next(self, children: None | list | dict = None) -> DataTree:
        # First iteration after reset
        if self.current_index == -1:
            assert self.change_index
            children = self.__default_values()

        # First iteration with a new index
        if self.change_index:
            index = next(self.indices)
            assert isinstance(index, int)
            self.current_index = index
            self.change_index = False

        # Children is not None just after a child iterator has been exhausted.
        if children is None:
            if self.tree.lazy:
                children = {}
            else:
                children = self.default_values.copy()

        current_key = self.keys[self.current_index]
        try:
            new_value = next(self.iterators[self.current_index])
        except StopIteration:
            # A child iterator has been exhausted
            self.change_index = True

            try:
                children[current_key] = self.default_values[current_key]  # type: ignore[index]
            except KeyError:
                assert isinstance(self.tree, Union)
                if self.tree.no_default_policy != NoDefaultPolicy.SKIP:
                    raise

            return self.__next(children)

        children[current_key] = new_value  # type: ignore[index]

        return children


@dataclass(frozen=True)
class Transform(IterationTree):
    """
    Node that modifies the data trees generated by its child during iteration.

    If you want to transform a `list` or `dict` of iteration trees, you should
    wrap them in an `IterationMethod` object first.
    """

    child: IterationTree

    @abstractmethod
    def transform(self, data_tree: DataTree) -> DataTree:
        """
        Method implemented by concrete sub-classes to modify the data tree
        generated by the `child` tree.
        """
        raise NotImplementedError()

    def __iter__(self) -> TreeIterator:
        return TransformIterator(self)

    def __len__(self) -> int:
        return len(self.child)

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self.child.to_pseudo_data_tree()

    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        error: None | NoDefaultError = None
        try:
            # This is the only statement that can raise an exception.
            child_default = self.child.default(no_default_policy)

            # We keep those lines here so that `child_default` is explicitly
            # bound.
            if isinstance(child_default, _NoDefault):
                return no_default

            return self.transform(child_default)
        except NoDefaultError as err:
            own_path: ChildPath = [child]
            error = NoDefaultError(err.args[0], own_path + err.path)

        assert error is not None
        raise error

    def __getitem__(self, key: Key) -> IterationTree:
        return self.child[key]

    # Path API

    def _get(self, child_key: Key | _Child) -> IterationTree:
        if not isinstance(child_key, _Child):
            raise KeyError("Transform node child is only accessible with Child().")

        return self.child

    def _insert_child(
        self, child_key: Key | _Child, child: IterationTree
    ) -> IterationTree:
        if not isinstance(child_key, _Child):
            raise KeyError("Transform node child is only accessible with Child().")

        return dataclasses.replace(self, child=child)

    def _remove_child(self, child_key: Key | _Child) -> IterationTree:
        raise TypeError("Transform node does not support child removal.")

    def _replace_root(
        self, Node: type[IterationTree], *args, **kwargs
    ) -> IterationTree:
        # The signature of `Node` is statically unknown
        return Node(self.child, *args, **kwargs)  # type: ignore[call-arg]

    def _depth_first_modify(
        self,
        modifier: Callable[["IterationTree", ChildPath], "IterationTree"],
        path: ChildPath,
    ) -> "IterationTree":
        new_child = self.child._depth_first_modify(modifier, path + [child])
        return modifier(dataclasses.replace(self, child=new_child), path)


@dataclass
class TransformIterator(TreeIterator):
    transform_node: Transform
    child_iterator: TreeIterator = field(init=False)

    def __post_init__(self):
        super().__init__()
        self.child_iterator = iter(self.transform_node.child)

    def reset(self):
        self.child_iterator.reset()

    def reverse(self):
        super().reverse()
        self.child_iterator.reverse()

    def __next__(self) -> DataTree:
        return self.transform_node.transform(next(self.child_iterator))


@dataclass(frozen=True)
class FunctionalTranform(Transform):
    """
    Transform node using its function attribute to modify its child.
    """

    f: Callable[[DataTree], DataTree]

    def transform(self, data_tree: DataTree) -> DataTree:
        return self.f(data_tree)


@dataclass(frozen=True)
class AccumulatorTransform(Transform):
    """
    Transform node that accumulates its inputs, as a kind of *unlazifying*
    transform:
      - if its successive inputs are dictionaries, merge them using the union
        operator, and return the results;
      - else, leave its inputs untouched.
    """

    #: Start value of the accumulator, which must either be a dictionary, or
    #: `None`.
    start_value: dict[Key, DataTree] | None = None

    def __iter__(self) -> TreeIterator:
        return AccumulatorTransformIterator(self, last_value=self.start_value)

    def transform(self, data_tree: DataTree) -> DataTree:
        return data_tree


@dataclass
class AccumulatorTransformIterator(TransformIterator):
    last_value: dict | None

    def __next__(self) -> DataTree:
        data_tree = super().__next__()
        if isinstance(data_tree, dict):
            if isinstance(self.last_value, dict):
                self.last_value |= data_tree
            else:
                self.last_value = data_tree

            return self.last_value.copy()
        else:
            return data_tree


### Leaves ###


class IterationLeaf(IterationTree):
    def _get(self, child_key: Key | _Child) -> IterationTree:
        raise TypeError("Iteration leaves do not support indexing.")

    def _insert_child(
        self, child_key: Key | _Child, child: IterationTree
    ) -> IterationTree:
        raise TypeError("Iteration leaves do not support indexing.")

    def _remove_child(self, child_key: Key | _Child) -> IterationTree:
        raise TypeError("Iteration leaves do not support indexing.")

    def _replace_root(
        self, Node: type[IterationTree], *args, **kwargs
    ) -> IterationTree:
        if not issubclass(Node, IterationLeaf):
            raise TypeError(f"Cannot replace an iteration leaf with a {Node}")

        return Node(*args, **kwargs)

    def _depth_first_modify(
        self,
        modifier: Callable[["IterationTree", ChildPath], "IterationTree"],
        path: ChildPath,
    ) -> "IterationTree":
        return modifier(self, path)


DT = TypeVar("DT", bound=DataTree)


@dataclass(frozen=True)
class IterationLiteral(IterationLeaf, Generic[DT]):
    """
    Wrapper around a data tree.
    """

    value: DT

    def __iter__(self) -> TreeIterator:
        return ListIterator([self.value])

    def __len__(self) -> int:
        return 1

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self.value  # type: ignore[return-value]

    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        return self.value

    def __getitem__(self, key: Key) -> DataTree:  # type: ignore[override]
        """
        Work as if the iteration literal was the literal it contains.

        It does not respect the API of `IterationTree.__getitem__`, as it
        returns a `DataTree`, but this is so convenient that we accept this
        compromise.
        """
        # A `KeyError` is raised in case of an improper index.
        return self.value[key]  # type: ignore[index]


T = TypeVar("T", bound=int | float)


@dataclass(frozen=True)
class NumericRange(IterationLeaf, Generic[T]):
    """
    Represents a range of numeric values.
    """

    start: T
    end: T
    default_value: T | _NoDefault = field(default_factory=_NoDefault)

    def __iter__(self) -> TreeIterator:
        raise TypeError("Cannot iterate over a numeric range.")

    def __len__(self) -> int:
        raise TypeError("A numeric range does not have a length.")

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self

    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        if isinstance(self.default_value, _NoDefault):
            if no_default_policy == NoDefaultPolicy.ERROR:
                raise NoDefaultError("Numeric range without a default value", [])
            elif no_default_policy == NoDefaultPolicy.SENTINEL:
                return no_default
            else:  # no_default_policy == NoDefaultPolicy.SKIP:
                return no_default

        return self.default_value

    def __getitem__(self, key: Key) -> IterationTree:
        raise TypeError("Numeric ranges do not support indexing.")


@dataclass(frozen=True)
class LinearRange(NumericRange[float]):
    """
    Generate `steps` values linearly spaced between `start` and `end`, both
    included.
    """

    # Have to specify a default value because `default_value` has one
    steps: int = field(default=2)

    def __post_init__(self):
        if self.steps < 1 or (self.start != self.end and self.steps < 2):
            raise ValueError("Invalid number of steps.")

    def __iter__(self) -> TreeIterator:
        sequence: list
        if self.steps == 1:
            sequence = [self.start]
        else:
            delta = self.end - self.start
            sequence = [
                self.start + delta * step / (self.steps - 1)
                for step in range(self.steps)
            ]

        return ListIterator(sequence)

    def __len__(self) -> int:
        return self.steps


@dataclass(frozen=True)
class GeometricRange(NumericRange[float]):
    """
    Generate `steps` values geometrically spaced between `start` and `end`, both
    included.
    """

    # Have to specify a default value because `default_value` has one
    steps: int = field(default=2)

    def __post_init__(self):
        if self.start * self.end <= 0:
            raise ValueError("Range limits must be non-zero and with the same sign.")
        if self.steps < 1 or (self.start != self.end and self.steps < 2):
            raise ValueError("Invalid number of steps.")

    def __iter__(self) -> TreeIterator:
        sequence: list
        if self.steps == 1:
            sequence = [self.start]
        else:
            sign = 1 if self.start > 0 else -1
            start = self.start * sign
            end = self.end * sign
            ratio = exp(log(end / start) / (self.steps - 1))

            sequence = [sign * start * (ratio**e) for e in range(self.steps)]

        return ListIterator(sequence)

    def __len__(self) -> int:
        return self.steps


@dataclass(frozen=True)
class IntegerRange(NumericRange[int]):
    """
    Generate integer values `step` spaced, between `start` and `end`, both
    included.
    """

    step: int = field(default=1)

    def __post_init__(self):
        if self.step < 0 or (self.start != self.end and self.step < 1):
            raise ValueError("Invalid step size")

    def __iter__(self) -> TreeIterator:
        sequence: list
        if self.step == 0:
            sequence = [self.start]
        else:
            direction = 1 if self.end > self.start else -1
            positions = range(1 + abs(self.end - self.start) // self.step)
            sequence = [self.start + direction * m * self.step for m in positions]

        return ListIterator(sequence)

    def __len__(self) -> int:
        return 1 + abs(self.end - self.start) // self.step


@dataclass(frozen=True)
class Sequence(IterationLeaf):
    """
    Non-empty sequence of data trees.
    """

    elements: list[DataTree]
    default_value: DataTree | _NoDefault = field(default_factory=_NoDefault)

    def __post_init__(self):
        if len(self.elements) == 0:
            raise ValueError("Empty elements are forbidden.")

    def __iter__(self) -> TreeIterator:
        return ListIterator(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self

    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        if isinstance(self.default_value, _NoDefault):
            if no_default_policy == NoDefaultPolicy.ERROR:
                raise NoDefaultError("Sequence without a default value", [])
            elif no_default_policy == NoDefaultPolicy.SENTINEL:
                return no_default
            else:  # no_default_policy == NoDefaultPolicy.SKIP:
                return no_default

        return self.default_value

    def __getitem__(self, key: Key) -> DataTree:  # type: ignore[override]
        """
        Work as if the iteration sequence was the sequence it contains.

        It does not respect the API of `IterationTree.__getitem__`, as it
        returns a `DataTree`, but this is so convenient that we accept this
        compromise.
        """
        return self.elements[key]  # type: ignore[index]


###########
# Utility #
###########


class RestrictionPolicy(Enum):
    #: Iteration leaves only keep their first and last values.
    FIRST_LAST = "FIRST_LAST"

    #: Iteration leaves only keep their two first values.
    FIRST_SECOND = "FIRST_SECOND"

    #: Iteration leaves keep their first, second and last values.
    COMBINED = "COMBINED"


def restrict_leaves_sizes(
    tree: IterationTree,
    policy: RestrictionPolicy = RestrictionPolicy.FIRST_LAST,
) -> IterationTree:
    """
    Restrict the size of the iteration leaves of the tree, depending on the
    restriction policy. This is useful for troubleshooting, or to verify that
    the full range of the leaves is supported by *something*.
    """

    def _restrict(tree: IterationTree, _: ChildPath) -> IterationTree:
        if not isinstance(tree, IterationLeaf):
            return tree

        if policy == RestrictionPolicy.FIRST_LAST:
            if len(tree) <= 2:
                return tree

            tree_iter = iter(tree)
            first = next(tree_iter)
            tree_iter.reverse()
            tree_iter.reset()
            last = next(tree_iter)
            return Sequence([first, last])
        elif policy == RestrictionPolicy.FIRST_SECOND:
            if len(tree) <= 2:
                return tree

            tree_iter = iter(tree)
            return Sequence([next(tree_iter), next(tree_iter)])
        else:
            if len(tree) <= 3:
                return tree

            tree_iter = iter(tree)
            first, second = next(tree_iter), next(tree_iter)
            tree_iter.reverse()
            tree_iter.reset()
            last = next(tree_iter)
            return Sequence([first, second, last])

    return tree.depth_first_modify(_restrict)

"""
This module contains the definition of the base type and classes used for
iteration (data tree, pseudo data tree and iteration tree).
"""

import typing
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from _typeshed import Self
else:
    Self = Any

from ..utility import Sentinel

#################
### Data tree ###
#################

#: Data values that can be used
DataLiteral = typing.Union[None, bool, str, int, float, "_NoDefault"]

#: Dictionary keys
Key = bool | str | int | float

#: A data tree consists of literal leaves, and dictionary or list nodes
DataTree = DataLiteral | dict[Key, "DataTree"] | list["DataTree"]

########################
### Pseudo data tree ###
########################

# Note: a DataTree is a PseudoDataTree, but mypy seems not to understand it.

#: A leave of a pseudo data tree is either a data tree leave, or a non-trivial
#: iteration tree leave.
PseudoDataLiteral = typing.Union[DataLiteral, "IterationLeaf"]

#: A pseudo data tree is a data tree whose leaves can be non literal iteration
#: leaves.
PseudoDataTree = (
    PseudoDataLiteral | dict[Key, "PseudoDataTree"] | list["PseudoDataTree"]
)

######################
### Iteration tree ###
######################


### Iteration ###


class DefaultIndex(Sentinel):
    """
    Index of the default value of an iteration tree.
    """

    pass


T = TypeVar("T", bound="IterationTree", covariant=True)


class TreeIterator(ABC, Generic[T]):
    """
    Iteration tree iterator.

    Compared to a usual iterator, it supports forward and backward iteration,
    and is endlessly usable. This means that, whenever it is "exhausted"
    (`__next__()` raises `StopIteration`), it can either be reset to its
    starting position with `reset()`, or its iteration direction can be
    switched using `reverse()`.

    Additionally, it supports random access with the `__getitem__` method, which
    uses `update` under the hood.
    """

    #: Reference to the tree being iterated over.
    tree: T

    #: Current iteration direction of the iterator.
    #:
    #: This attribute is managed by `TreeIterator`, and should thus not be
    #: modified by sub-classes. However, it can be read.
    __forward: bool

    #: Position of the last value that was yielded. Valid values start at -1
    #: (backward-exhausted iterator, or forward iteration start), and go up to
    #: `size`(forward-exhausted iterator, or backward iteration start).
    #:
    #: It can be directly modified by `update`.
    #:
    #: This attribute is managed by `TreeIterator`, thus it must thus not be
    #: modified by sub-classes. However, it can be read.
    position: int

    def __init__(self, tree: T) -> None:
        self.__forward = True
        self.position = -1
        self.tree = tree

    def __iter__(self: Self) -> Self:
        return self

    def reset(self):
        """
        Reset the internal state of the iterator, so that its next value will be
        the first iterated value in the current direction. It takes into
        account the value of forward, going either to the start or the end of
        the iterated collection.
        """
        if self.__forward:
            self.position = -1
        else:
            try:
                self.position = len(self.tree)
            except TypeError as error:
                raise Exception(
                    "Cannot backward-reset an infinite iterator."
                ) from error

    def is_forward(self) -> bool:
        """
        Returns whether the iterator is going forward or not.
        """
        return self.__forward

    def reverse(self):
        """
        Reverse the iteration direction of the iterator, but stay at the same
        position. Thus, if `it` is any `TreeIterator`, the following behaviour
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
        self.__forward = not self.__forward

    def update(self, position: int):
        """
        Update the position of the iterator to any supported position.

        If an invalid position is requested, an `IndexError` is raised, and the
        state of the iterator remains unchanged.
        """
        if position < -1:
            raise Exception("Cannot update to a position < -1.")
        try:
            if position > len(self.tree):
                raise Exception("Cannot update to a position > size.")
        except TypeError:
            # There is no upper bound to the indices of an infinite tree
            pass

        self.position = position

    def __getitem__(self, position: int | DefaultIndex) -> DataTree:
        """
        Return the element at `position`. If it is a `DefaultIndex`, return the
        default value of the iteration tree.
        """
        if isinstance(position, DefaultIndex):
            return self.tree.default(NoDefaultPolicy.ERROR)

        self.update(position)
        return self._current_value()

    @abstractmethod
    def _current_value(self) -> DataTree:
        """
        Return the value at `position`.

        It is assumed that whenever it is called, the return value is given to
        the user.
        """
        raise NotImplementedError()

    def __next__(self) -> DataTree:
        """
        Return the next value in the current iteration direction.
        """
        if self.__forward:
            self.position += 1

            try:
                if self.position >= len(self.tree):
                    self.position = len(self.tree)
                    raise StopIteration
            except TypeError:
                pass
        else:
            self.position -= 1

            if self.position < 0:
                self.position = -1
                raise StopIteration

        return self._current_value()


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

if typing.TYPE_CHECKING:
    from .node import Transform


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

    def iterate(self) -> TreeIterator:
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
        from .node import Transform

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

    def __getitem__(self, key: Key) -> "IterationTree":
        raise TypeError(f"{self.__class__.__name__} does not support indexing.")

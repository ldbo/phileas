"""
This module contains the definition of the base type and classes used for
iteration (data tree, pseudo data tree and iteration tree).
"""

from __future__ import annotations

import dataclasses
import math
import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from typing_extensions import assert_never

if TYPE_CHECKING:  # pragma: no cover
    from _typeshed import Self  # pragma: no cover
else:
    Self = Any

from ..logging import logger
from ..utility import Sentinel

#################
### Data tree ###
#################

#: Data values that can be used as data tree leaves
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
    (:py:meth:`__next__` raises ``StopIteration``), it can either be reset to
    its starting position with :py:meth:`reset`, or its iteration direction can
    be switched using :py:meth:`reverse`.

    Additionally, it supports random access with the :py:meth:`__getitem__`
    method, which uses :py:meth:`update` under the hood.
    """

    #: Reference to the tree being iterated over.
    tree: T

    #: Current iteration direction of the iterator.
    #:
    #: .. warning
    #:     This attribute is managed by :py:class:`TreeIterator`, and should
    #:     thus not be modified by sub-classes. However, it can be read.
    __forward: bool

    #: Position of the last value that was yielded. Valid values start at -1
    #: (backward-exhausted iterator, or forward iteration start), and go up to
    #: :py:attr:`size` (forward-exhausted iterator, or backward iteration
    #: start).
    #:
    #: It can be directly modified by :py:meth:`update`.
    #:
    #: .. warning
    #:     This attribute is managed by :py:class:`TreeIterator`, thus it must
    #:     not be modified by sub-classes. However, it can be read.
    position: int

    #: Cached size of the iterated tree. Consider using this instead of ``len
    #: (self.tree)``, as computing the length of a tree can be an expensive
    #: operation.
    size: int | None

    def __init__(self, tree: T) -> None:
        self.__forward = True
        self.position = -1
        self.tree = tree
        self.size = tree.safe_len()

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
            if self.size is None:
                raise ValueError("Cannot reset a backward infinite iterator.")
            self.position = self.size

    def is_forward(self) -> bool:
        """
        Returns whether the iterator is going forward or not.
        """
        return self.__forward

    def reverse(self):
        """
        Reverse the iteration direction of the iterator, but stay at the same
        position. Thus, if ``it`` is any :py:class:`TreeIterator`, the
        following behavior is expected:

        >>> it.reset()
        >>> it.reverse()
        >>> list(it)
        []

        If you want to iterate over an iteration tree ``tree`` backward, you have
        to reset the iterator after having reversed it:

        >>> it = iter(tree)
        >>> it.reverse()
        >>> it.reset()

        """
        self.__forward = not self.__forward

    def update(self, position: int):
        """
        Update the position of the iterator to any supported position. This
        includes the positions ranging from ``0`` to ``self.size - 1``, included, as
        well as
        - ``-1``, which represents a reset forward iterator and
        - ``self.size``, which represents a reset backward iterator.

        If an invalid position is requested, an :py:class:`IndexError` is
        raised, and the state of the iterator remains unchanged.
        """
        if position < -1:
            raise IndexError("Cannot update to a position < -1.")
        try:
            if self.size is not None and position > self.size:
                raise IndexError("Cannot update to a position > size.")
        except InfiniteLength:
            # There is no upper bound to the indices of an infinite tree
            pass

        self.position = position

    def __getitem__(self, position: int | DefaultIndex) -> DataTree:
        """
        Return the element at ``position``. If it is a :py:class:`DefaultIndex`,
        return the default value of the iteration tree, using the
        :py:attr:`NoDefaultPolicy.FIRST_ELEMENT policy`.

        Raises:
            IndexError: if the index is invalid.
        """
        if isinstance(position, DefaultIndex):
            return self.tree.default(NoDefaultPolicy.FIRST_ELEMENT)

        size = math.inf if self.size is None else self.size
        if position < 0:
            if size < math.inf:
                assert isinstance(size, int)
                position = size + position
            else:
                raise IndexError(
                    f"Negative index {position} is not supported with infinite "
                    + "tree."
                )
        elif position >= size:
            raise IndexError(f"Index {position} larger than the tree size.")

        self.update(position)
        return self._current_value()

    @abstractmethod
    def _current_value(self) -> DataTree:
        """
        Return the value at :py:attr:`position`.

        Assumptions:

         - The ownership of the return value is given to the user.
         - ``self._position`` is a valid value, between ``0`` and ``self.size -
           1``. Thus, it should never raise a :py:class:`StopIteration`.
        """
        raise NotImplementedError()

    def __next__(self) -> DataTree:
        """
        Return the next value in the current iteration direction.
        """
        if self.__forward:
            self.position += 1

            if self.size is not None and self.position >= self.size:
                self.position = self.size
                raise StopIteration
        else:
            self.position -= 1

            if self.position < 0:
                self.position = -1
                raise StopIteration

        return self._current_value()


class OneWayTreeIterator:
    """
    Iterator used in cases where random access iteration is too cumbersome to
    implement. It only requires implementing the :py:meth:`_next` method.

    Random access is obtained by using a cache. Thus, this method is usually
    more time and memory expensive than classical random access iterators, so
    it should only be used as a last resort. For now, the cache size is unbound,
    as it stores all the iterated values. This might change in the future.

    This is not a subclass of `TreeIterator` in order to prevent diamond
    inheritance issues. Child classes must thus inherit from this, alongside
    :py:class:`TreeIterator`. It is recommended to place
    :py:meth:`OneWayTreeIterator` first, so that its :py:meth:`_current_value`
    implementation is used.
    """

    # Attributes defined in `TreeIterator` and used here
    position: int
    size: int | None

    #: Cache of the already generated values.
    __cache: list[DataTree]

    #: Last generated position.
    __last_position: int

    def __init__(self) -> None:
        self.__cache = []
        self.__last_position = -1

    @abstractmethod
    def _next(self) -> DataTree:
        """Next iteration value."""
        raise NotImplementedError()

    def _current_value(self) -> DataTree:
        if self.position >= len(self.__cache):
            position = self.__last_position
            while position < self.position:
                try:
                    value = self._next()
                    self.__cache.append(value)
                    position += 1
                except StopIteration as e:
                    tree_size = self.size if self.size is not None else "infinite"
                    raise ValueError(
                        f"Iterator exhausted at position {self.position}, "
                        f"whereas the tree size is {tree_size}."
                    ) from e

            self.__last_position = position

        return deepcopy(self.__cache[self.position])


### Indexing ###


class _Child(Sentinel):
    """
    Sentinel representing the index of the only child of a unary nodes.
    """

    pass


child = _Child()

ChildPath = list[Key | _Child]

### Default related objects ###


class NoDefaultPolicy(Enum):
    """
    Behavior of :py:meth:`IterationTree.default` for trees not having a default
    value.
    """

    #: Raise a :py:class:`NoDefaultError` if any of the nodes in the tree does
    #: not have a default value.
    ERROR = "ERROR"

    #: Return the :py:class:`_NoDefault` sentinel if any of the nodes in the
    #: tree does not have a default value.
    SENTINEL = "SENTINEL"

    #: Skip elements without a default element. If the root of the tree does
    #: not have a default value, return a :py:class:`_NoDefault` sentinel.
    #:
    #: Note that this is not supported by iteration method nodes with list
    #: children.
    SKIP = "SKIP"

    #: Return the first element of iteration leaves without a default value.
    FIRST_ELEMENT = "FIRST_ELEMENT"


class NoDefaultError(Exception):
    """
    Indicates that :py:meth:`IterationTree.default` has been called on an
    iteration tree where a node does not have a default value.
    """

    #: Path of a child without a default value.
    path: ChildPath

    def __init__(self, message: str | None, path: ChildPath) -> None:
        super().__init__(message)
        self.path = path

    @staticmethod
    def build_from(tree: IterationTree) -> NoDefaultError:
        return NoDefaultError(f"{tree.__class__.__name__} without a default value.", [])

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
    from .node import UnaryNode


class InfiniteLength(BaseException):
    """
    Exception raised when requesting the length of an infinite collection.
    """

    pass


@dataclass(frozen=True)
class IterationTree(ABC):
    """
    Represents a set of data trees, as well as the way to iterate over them. In
    order to be able to get a single data tree from an iteration tree, they are
    able to build a default data tree, which (usually) has the same shape as the
    generated data tree.

    An iteration tree cannot be modified, as it is a frozen
    :py:class:`dataclass`. Instead, a new one must be created from this one.
    """

    #: Names of the available configurations.
    configurations: frozenset[Key] = field(
        default_factory=frozenset, init=False, repr=False
    )

    def get_configuration(self, key: Key) -> IterationTree:
        """
        Returns a given configuration of the tree, if it exists. Otherwise,
        raises a :py:class:`KeyError`. See
        :py:class:`~phileas.iteration.Configurations` node for more details on
        the behavior of this method.
        """
        if key not in self.configurations:
            raise KeyError(f"Configuration {key} does not exist.")

        return self._get_configuration(key)

    @abstractmethod
    def _get_configuration(self, config_name: Key) -> IterationTree:
        """
        Actual implementation of the configuration access in sub-classes.

        ``config_name`` is assumed to be a valid configuration of the tree, so
        recursive calls either return an actual configuration, or the
        unmodified tree.
        """
        raise NotImplementedError()

    def unroll_configurations(self) -> IterationTree:
        """
        Transforms a configurable tree, *ie*. one with ``configurations !=
        {}``, to a non-configurable tree. It requests all the available
        configurations, and gather them into
        a :py:class:`~phileas.iteration.Union` node with ``reset=False``. A
        :py:class:`~phileas.iteration.MoveUpTransform` is used to get rid of the
        name of the configurations.

        If the tree is not configurable, return it.
        """
        if len(self.configurations) == 0:
            return self

        from phileas.iteration import Union
        from phileas.iteration.node import MoveUpTransform

        children = {name: self.get_configuration(name) for name in self.configurations}
        configurations = Union(
            children, lazy=True, preset=None, common_preset=False, reset=None
        )

        return MoveUpTransform(configurations)

    def __iter__(self) -> TreeIterator:
        """
        Return a two-way resetable tree iterator.
        """
        if len(self.configurations) == 0:
            return self._iter()

        logger.warning(
            "Iterating through a configurable tree. It is automatically "
            "unrolled, but it is advised to explicitly call "
            "IterationTree.unroll_configurations before iteration."
        )

        return iter(self.unroll_configurations())

    @abstractmethod
    def _iter(self) -> TreeIterator:
        """
        Implementation of :py:meth:`__iter__` which assumes that ``self`` does
        not have any configuration.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """
        Return the number of data trees represented by the iteration tree. If it
        is finite, it is the same as the number of elements yielded by
        :py:meth:`__iter__`. Otherwise, an :py:class:`InfiniteLength` is raised.

        See :py:meth:`safe_len` for a method that does not raise errors.
        """
        if len(self.configurations) == 0:
            return self._len()

        return sum(
            len(self.get_configuration(config)) for config in self.configurations
        )

    def safe_len(self) -> int | None:
        """
        Return the number of data trees represented by the iteration tree. If it
        is finite, it is the same as the number of elements yielded by
        :py:meth:`__iter__`. Otherwise, return ``None``.
        """
        try:
            return len(self)
        except InfiniteLength:
            return None

    @abstractmethod
    def _len(self) -> int:
        """
        Implementation of :py:meth:`__len__` which assumes that ``self`` does
        not have any configuration.
        """
        raise NotImplementedError()

    def iterate(self) -> TreeIterator:
        """
        Implementation of :py:meth:`__len__` which assumes that ``self`` does
        not have any configuration.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_pseudo_data_tree(self) -> PseudoDataTree:
        """
        Converts the iteration tree to a pseudo data tree.
        """
        raise NotImplementedError()

    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        """
        Returns a default data tree. If the tree does not have a default value,
        follows the behavior dictated by ``no_default_policy``.
        """
        try:
            return self._default(no_default_policy)
        except NoDefaultError:
            if no_default_policy == NoDefaultPolicy.ERROR:
                raise
            elif no_default_policy == NoDefaultPolicy.SENTINEL:
                return no_default
            elif no_default_policy == NoDefaultPolicy.FIRST_ELEMENT:
                return next(iter(self))
            elif no_default_policy == NoDefaultPolicy.SKIP:
                return no_default
            else:
                assert_never(no_default_policy)

    @abstractmethod
    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree | _NoDefault:
        """
        Concrete implementation of :py:meth:`default`. It returns the default
        value, or raises a :py:class:`NoDefaultError` if it is not defined.
        """
        raise NotImplementedError()

    def __getitem__(self, key: Key) -> IterationTree:
        """
        It is implemented to allow working with an iteration tree as if it
        consisted of nested list and dict objects.
        """
        return self.get([key])

    # Path API
    #
    # Internal nodes and the structure of iteration trees are to be modified
    # using a path-based API.
    #
    # After using a modification function, only the output of the function
    # should be used, and `self` should be discarded.

    def with_params(self, path=ChildPath | None, **kwargs) -> IterationTree:
        """
        Returns a similar iteration tree, where the node at ``path`` is assigned
        the given keyword parameters. If ``path`` is not specified, modifies
        the root of the tree directly.
        """
        valid_path = []
        if path is not None:
            valid_path = path

        def modifier(tree: IterationTree, path: ChildPath) -> IterationTree:
            if path == valid_path:
                return dataclasses.replace(tree, **kwargs)

            return tree

        return self.depth_first_modify(modifier)

    def get(self, path: ChildPath) -> IterationTree:
        """
        Get a node inside a tree. It should not be used to modify the tree.
        """
        current = self
        for key in path:
            current = current._get(key)

        return current

    @abstractmethod
    def _get(self, child_key: Key | _Child) -> IterationTree:
        """
        Returns the root child with the given key, or raises a `KeyError` if
        there is no child with this key.

        The output might be modified by the user.
        """
        raise NotImplementedError()

    @abstractmethod
    def _insert_child(
        self, child_key: Key | _Child, child: IterationTree
    ) -> IterationTree:
        """
        Insert a new child tree to the root and return the newly created tree.

        The initial object should not be modified. It is preferable to work with
        a copy.
        """
        raise NotImplementedError()

    @abstractmethod
    def _remove_child(self, child_key: Key | _Child) -> IterationTree:
        """
        Remove a child of the root, and return the newly created tree. Raises a
        :py:class:`KeyError` if there is no child with this key.

        The initial object should not be modified. It is preferable to work with
        a copy.
        """
        raise NotImplementedError()

    @abstractmethod
    def _replace_root(
        self, Node: type[IterationTree], *args, **kwargs
    ) -> IterationTree:
        """
        Change the root node of a tree, keeping the remaining of the tree
        unmodified. In order to keep a valid tree structure, `Node` must have
        the same type as the current root, otherwise a :py:class:`TypeError` is
        raised.

        The initial object should not be modified. It is preferable to work with
        a copy.
        """
        raise NotImplementedError()

    def insert_child(
        self, path: ChildPath, tree: IterationTree | None
    ) -> IterationTree:
        """
        Insert a child anywhere in the tree, whose location is specified by path
        argument. Return the newly created tree.

        If there is already a node at this location, it will be replaced.

        If the specified tree is ``None``, a node is supposed to exist at this
        location (otherwise, a `KeyError` is raised), and will be removed if
        possible. Only iteration methods nodes will support child removal, see
        their implementation of :py:meth:`_remove_child`. In any other case, a
        :py:class:`TypeError` is raised.

        Note that the root of a tree cannot be removed, so specifying an empty
        path with a ``None`` tree will raise a :py:class:`KeyError`.
        """
        if tree is None and len(path) == 1:
            return self._remove_child(path[0])

        if len(path) == 0:
            if tree is None:
                raise KeyError("Cannot remove the root of an iteration tree.")

            return tree
        else:
            key = path[0]
            new_child = tree

            if len(path) > 1:
                new_child = self._get(key).insert_child(path[1:], tree)

            assert new_child is not None
            return self._insert_child(key, new_child)

    def remove_child(self, path: ChildPath) -> IterationTree:
        """
        Remove a node in a tree. It is equivalent to ``insert_child(path, None)``.
        """
        return self.insert_child(path, None)

    def insert_unary(
        self, path: ChildPath, Parent: type[UnaryNode], *args, **kwargs
    ) -> IterationTree:
        """
        Insert a parent to the node at the given path, parent which is
        necessarily a transform node, as it will only have a single child. The
        parent is built using the ``Parent`` class, and the supplied arguments.

        The newly created tree is returned.
        """
        from .node import UnaryNode

        if not issubclass(Parent, UnaryNode):
            raise TypeError("Cannot insert a non-unary node as a parent.")

        if len(path) == 0:
            return Parent(self, *args, **kwargs)

        key = path[0]
        new_child = self._get(key).insert_unary(path[1:], Parent, *args, **kwargs)
        new_me = self._insert_child(key, new_child)

        return new_me

    def replace_node(
        self, path: ChildPath, Node: type[IterationTree], *args, **kwargs
    ) -> IterationTree:
        """
        Replace the node at the given path with another one. The other node is
        built using its type, ``Node``, and the ``args`` and ``kwargs``
        arguments. Note that the sub-tree of the replaced node is not
        modified.

        This requires ``Node`` to be of the same kind of the node that is being
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
        self, modifier: Callable[[IterationTree, ChildPath], IterationTree]
    ) -> IterationTree:
        """
        Using a post-fix depth-first search, replace each ``node`` of the tree,
        located at ``path``, with ``modifier(node, path)``.
        """
        return self._depth_first_modify(modifier, [])

    @abstractmethod
    def _depth_first_modify(
        self,
        modifier: Callable[[IterationTree, ChildPath], IterationTree],
        path: ChildPath,
    ) -> IterationTree:
        """
        Recursive implementation of :py:meth:`depth_first_modify`, implemented
        by the different node types.

        In order to prevent side-effects, ``modifier`` should not modify its
        input objects. It should rather returns modified copies.
        """
        raise NotImplementedError()


@dataclass(frozen=True)
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
    ) -> IterationTree:
        return modifier(self, path)

    def __getitem__(self, key: Key) -> IterationTree:
        raise TypeError(f"{self.__class__.__name__} does not support indexing.")

    def _get_configuration(self, config_name: Key) -> IterationTree:
        return self

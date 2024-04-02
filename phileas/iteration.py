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

import dataclasses
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from math import exp, log
from typing import Callable, Generic, Iterator, TypeVar

#################
### Data tree ###
#################

#: Data values that can be used
DataLiteral = None | bool | str | int | float

#: Dictionary keys
Key = DataLiteral

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


class _Child:
    """
    Utility sentinel class used to represent the index of the only child of a
    1-ary iteration node.
    """

    def __repr__(self) -> str:
        return "Child()"


child = _Child()

ChildPath = list[Key | _Child]


class IterationTree(ABC):
    """
    Represents a set of data trees, as well as the way to iterate over them. In
    order to be able to get a single data tree from an iteration tree, they are
    able to build a default data tree, which (usually) has the same shape as the
    generated data tree.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[DataTree]:
        """
        Yields all the data trees represented by the iteration tree.
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
    def default(self) -> DataTree:
        """
        Returns a default data tree. If the node does not have a default value,
        it should raise a `TypeError`.
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


class _NoDefault:
    """
    Utility sentinel class used to store a default value which is not set.
    """

    def __repr__(self) -> str:
        return "NoDefault()"


#: You can store this value - instead of an actual default value - in instances
#: of classes that can have a default value, but don't.
no_default = _NoDefault()


### Nodes ###


@dataclass(frozen=True)
class IterationMethod(IterationTree):
    """
    Node which knows how to iterate through a `list` or `dict` of iteration
    trees.

    In order to implement a concrete iteration method, you should sub-class
    `IterationMethod` and implement the `__iter__` method.

    This should remain the only node in an iteration tree that can hold `dict`
    and `list`. If you are tempted to create another node doing so, you should
    verify that it cannot be done by sub-classing `IterationMethod` instead.
    """

    #: The children of the node. It must not be empty.
    children: list[IterationTree] | dict[Key, IterationTree]

    #: Notify the iteration method to be lazy. For now, this feature is only
    #: supported for `dict` children. In this case, lazy iteration will just
    #: yield the keys that have changed at each step.
    #:
    #: Note that concrete iteration method classes are not required to actually
    #: implement lazy iteration, and if they don't, they will probably do so
    #: silently. Refer to their documentation or their implementation.
    lazy: bool = field(default=False)

    #: Children trees stored in a list. This allows child-class to only
    #: implement iteration over lists.
    #:
    #: For now, dictionaries are sorted by their key value.
    _iterated_trees: list[IterationTree] = field(
        init=False, default_factory=list, repr=False, compare=False, hash=False
    )

    #: Access keys for the children trees, such that
    #: `children[_keys[i]] = _iterated_trees[i]`.
    #:
    #: Note that the previous statement is not properly typed, as the current
    #: type hints do not state that `_keys` contains valid keys for `children`.
    #: However, `IterationMethod.__post_init__` takes care of the validity of
    #: the runtime types. Because of that, ignoring type checks can be required
    #: when implementig concrete iteration methods.
    _keys: list[Key | int] = field(
        init=False, default_factory=list, repr=False, compare=False, hash=False
    )

    def __post_init__(self):
        if len(self.children) == 0:
            raise ValueError("Empty children are forbidden.")

        if isinstance(self.children, list):
            self._iterated_trees.extend(self.children)
            self._keys.extend(range(len(self.children)))
        elif isinstance(self.children, dict):
            keys = sorted(self.children.keys())  # type: ignore[type-var]
            self._keys.extend(keys)
            trees = (self.children[key] for key in keys)
            self._iterated_trees.extend(trees)
        else:
            raise TypeError("The children don't have a supported type.")

        if self.lazy and not isinstance(self.children, dict):
            raise TypeError("Lazy iteration is only supported for dictionary children")

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        if isinstance(self.children, list):
            return [child.to_pseudo_data_tree() for child in self.children]
        else:  # isinstance(self.children, dict)
            return {
                key: value.to_pseudo_data_tree() for key, value in self.children.items()
            }

    def default(self) -> DataTree:
        if isinstance(self.children, list):
            return [child.default() for child in self.children]
        else:  # isinstance(self.children, dict)
            return {key: value.default() for key, value in self.children.items()}

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
    """

    def __base_value(self) -> list[DataTree] | dict[Key, DataTree]:
        """
        Method used instead of `default`, allowing to build the shape of the
        elements yielded by `__iter__` with children not having a default value.
        """
        if isinstance(self.children, list):
            return [None for _ in self.children]
        else:
            return {key: None for key in self.children}

    def __iter__(self) -> Iterator[DataTree]:
        """
        Does not require the children iteration trees to have a default value.

        Implements lazy iteration.
        """
        n = len(self._iterated_trees)
        index = n - 1
        iterators = [iter(tree) for tree in self._iterated_trees]
        base = self.__base_value()
        for key, iterator in zip(self._keys, iterators):
            base[key] = next(iterator)  # type: ignore[index]

        done = False
        while not done:
            yield base.copy()

            if self.lazy:
                base = {}

            iterator_exhausted = True
            while iterator_exhausted:
                try:
                    base[self._keys[index]] = next(iterators[index])  # type: ignore[index]
                    index = n - 1
                    iterator_exhausted = False
                except StopIteration:
                    iterators[index] = iter(self._iterated_trees[index])
                    base[self._keys[index]] = next(iterators[index])  # type: ignore[index]
                    index -= 1

                    if index == -1:
                        done = True
                        break

    def __len__(self) -> int:
        return reduce(int.__mul__, map(len, self._iterated_trees), 1)


@dataclass(frozen=True)
class Union(IterationMethod):
    """
    Iteration over one child at a time, starting with the first one.

    For children that have a default value, they will be reset to it when they
    are not being iterated. However, there will be no complain about children
    not implementing a default value.
    """

    def __base_value(self) -> list[DataTree] | dict[Key, DataTree]:
        """
        Method used instead of `default`, implementing a best-effort default
        value if the children are specified as a dictionary.
        """
        if isinstance(self.children, dict):
            base = {}
            for key, tree in self.children.items():
                try:
                    base[key] = tree.default()
                except TypeError:
                    pass

            return base
        else:
            return self.default()  # type: ignore[return-value]

    def __iter__(self) -> Iterator[DataTree]:
        """
        Does not require the children trees to have a default value.

        Implements lazy iteration.
        """
        iterators = [iter(tree) for tree in self._iterated_trees]
        base = self.__base_value()
        current = base.copy()

        for index, iterator in enumerate(iterators):
            for value in iterator:
                current[self._keys[index]] = value  # type: ignore[index]
                yield current

                if self.lazy:
                    current = {}
                else:
                    current = base.copy()

            key = self._keys[index]
            if key in base:
                current[key] = base[key]  # type: ignore[index]

    def __len__(self) -> int:
        return sum(map(len, self._iterated_trees))


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

    def __iter__(self) -> Iterator[DataTree]:
        for data_child in self.child:
            yield self.transform(data_child)

    def __len__(self) -> int:
        return len(self.child)

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self.child.to_pseudo_data_tree()

    def default(self) -> DataTree:
        return self.transform(self.child.default())

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


@dataclass(frozen=True)
class FunctionalTranform(Transform):
    """
    Transform node using its function attribute to modify its child.
    """

    f: Callable[[DataTree], DataTree]

    def transform(self, data_tree: DataTree) -> DataTree:
        return self.f(data_tree)


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


DT = TypeVar("DT", bound=DataTree)


@dataclass(frozen=True)
class IterationLiteral(IterationLeaf, Generic[DT]):
    """
    Wrapper around a data tree.
    """

    value: DT

    def __iter__(self) -> Iterator[DT]:
        yield self.value

    def __len__(self) -> int:
        return 1

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self.value  # type: ignore[return-value]

    def default(self) -> DT:
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
    default_value: T | _NoDefault = field(default=no_default)

    def __iter__(self) -> Iterator[T]:
        raise TypeError("Cannot iterate over a numeric range.")

    def __len__(self) -> int:
        raise TypeError("A numeric range does not have a length.")

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self

    def default(self) -> T:
        if isinstance(self.default_value, _NoDefault):
            raise TypeError("This range does not have a default value.")
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

    def __iter__(self) -> Iterator[float]:
        if self.steps == 1:
            yield self.start
        else:
            delta = self.end - self.start
            for step in range(self.steps):
                yield self.start + delta * step / (self.steps - 1)

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

    def __iter__(self) -> Iterator[float]:
        if self.steps == 1:
            yield self.start
        else:
            sign = 1 if self.start > 0 else -1
            start = self.start * sign
            end = self.end * sign
            ratio = exp(log(end / start) / (self.steps - 1))

            for e in range(self.steps):
                yield sign * start * (ratio**e)

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

    def __iter__(self) -> Iterator[int]:
        if self.step == 0:
            yield self.start
        else:
            direction = 1 if self.end > self.start else -1
            for m in range(1 + abs(self.end - self.start) // self.step):
                yield self.start + direction * m * self.step

    def __len__(self) -> int:
        return 1 + abs(self.end - self.start) // self.step


@dataclass(frozen=True)
class Sequence(IterationLeaf):
    """
    Non-empty sequence of data trees.
    """

    elements: list[DataTree]
    default_value: DataTree | _NoDefault = field(default=no_default)

    def __post_init__(self):
        if len(self.elements) == 0:
            raise ValueError("Empty elements are forbidden.")

    def __iter__(self) -> Iterator[DataTree]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self

    def default(self) -> DataTree:
        if isinstance(self.default_value, _NoDefault):
            raise TypeError("This sequence does not have a default value")
        return self.elements[0]

    def __getitem__(self, key: Key) -> DataTree:  # type: ignore[override]
        """
        Work as if the iteration sequence was the sequence it contains.

        It does not respect the API of `IterationTree.__getitem__`, as it
        returns a `DataTree`, but this is so convenient that we accept this
        compromise.
        """
        return self.elements[key]  # type: ignore[index]

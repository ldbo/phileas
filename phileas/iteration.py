"""
This modules contains the trees used for data iteration.

The `DataTree` stores an actual data point, which consists of nested `dict` and
`list` objects, with `DataLiteral` leaves.

Then, the `IterationTree` provides a framework to build complex searches over
data trees. Its leaves consist of literal values, or data iterators. Those
leaves can simply be iterated over using `IterationTree.iterate`, but they can
also be used to build more complex iteration trees.

First, they can be combined with `IterationMethod` nodes, which provide a way to
iterate over multiple data sources. Then, `Transform` nodes can be inserted in
those trees in order to modify the data trees generated while iterating.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from math import exp, log
from typing import Generic, Iterator, TypeVar


#################
### Data tree ###
#################

#: Data values that can be used
DataLiteral = None | bool | str | int | float

#: Dictionary keys
Key = DataLiteral

#: A data tree consists of literal leaves, and dictionary or list nodes
DataTree = DataLiteral | dict[Key, "DataTree"] | list["DataTree"]

######################
### Iteration tree ###
######################


class IterationTree(ABC):
    """
    Represents a set of data trees, as well as the way to iterate over them. In
    order to be able to get a single data tree from an iteration tree, they are
    able to build a default data tree, which (usually) has the same shape as the
    generated data tree.
    """

    @abstractmethod
    def iterate(self) -> Iterator[DataTree]:
        """
        Yields all the data trees represented by the iteration tree.
        """
        raise NotImplementedError()

    @abstractmethod
    def default(self) -> DataTree:
        """
        Returns a default data tree.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        raise ValueError("This tree does not have a length.")


class _NoDefault:
    """
    Utility sentinel class used to store a default value which is not set.
    """

    pass

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
    `IterationMethod` and implement the `iterate` method.

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

    def default(self) -> DataTree:
        if isinstance(self.children, IterationTree):
            return self.children.default()
        elif isinstance(self.children, list):
            return [child.default() for child in self.children]
        else:  # isinstance(self.children, dict)
            return {key: value.default() for key, value in self.children.items()}


@dataclass(frozen=True)
class CartesianProduct(IterationMethod):
    """
    Iteration over the cartesian product of the children. The iteration order
    is the same as `itertools.product`. In other words, `iterate` will behave
    roughly as

    ```py
    for v1 in c1.iterate():
        for v2 in c2.iterate():
            for v3 in c3.iterate():
                yield [v1, v2, v3]
    ```
    """

    def __base_value(self) -> list[DataTree] | dict[Key, DataTree]:
        """
        Method used instead of `default`, allowing to build the shape of the
        elements yielded by `iterate` with children not having a default value.
        """
        if isinstance(self.children, list):
            return [None for _ in self.children]
        else:
            return {key: None for key in self.children}

    def iterate(self) -> Iterator[DataTree]:
        """
        Does not require the children iteration trees to have a default value.

        Implements lazy iteration.
        """
        n = len(self._iterated_trees)
        index = n - 1
        iterators = [tree.iterate() for tree in self._iterated_trees]
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
                    iterators[index] = self._iterated_trees[index].iterate()
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
                except ValueError:
                    pass

            return base
        else:
            return self.default()  # type: ignore[return-value]

    def iterate(self) -> Iterator[DataTree]:
        """
        Does not require the children trees to have a default value.

        Implements lazy iteration.
        """
        iterators = [tree.iterate() for tree in self._iterated_trees]
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

    def iterate(self) -> Iterator[DataTree]:
        for data_child in self.child.iterate():
            yield self.transform(data_child)

    def __len__(self) -> int:
        return len(self.child)

    def default(self) -> DataTree:
        return self.transform(self.child.default())


### Leaves ###


DT = TypeVar("DT", bound=DataTree)


@dataclass(frozen=True)
class IterationLiteral(IterationTree, Generic[DT]):
    """
    Wrapper around a data tree.
    """

    value: DT

    def iterate(self) -> Iterator[DT]:
        yield self.value

    def __len__(self) -> int:
        return 1

    def default(self) -> DT:
        return self.value


T = TypeVar("T", bound=int | float)


@dataclass(frozen=True)
class NumericRange(IterationTree, Generic[T]):
    """
    Represents a range of numeric values.
    """

    start: T
    end: T
    default_value: T | _NoDefault = field(default=no_default)

    def iterate(self) -> Iterator[T]:
        raise TypeError("Cannot iterate over a numeric range.")

    def default(self) -> T:
        if isinstance(self.default_value, _NoDefault):
            raise ValueError("This range does not have a default value.")
        return self.default_value


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

    def iterate(self) -> Iterator[float]:
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

    def iterate(self) -> Iterator[float]:
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

    def iterate(self) -> Iterator[int]:
        if self.step == 0:
            yield self.start
        else:
            direction = 1 if self.end > self.start else -1
            for m in range(1 + abs(self.end - self.start) // self.step):
                yield self.start + direction * m * self.step

    def __len__(self) -> int:
        return 1 + abs(self.end - self.start) // self.step


@dataclass(frozen=True)
class Sequence(IterationTree):
    """
    Non-empty sequence of data trees.
    """

    elements: list[DataTree]
    default_value: DataTree | _NoDefault = field(default=no_default)

    def __post_init__(self):
        if len(self.elements) == 0:
            raise ValueError("Empty elements are forbidden.")

    def iterate(self) -> Iterator[DataTree]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def default(self) -> DataTree:
        if isinstance(self.default_value, _NoDefault):
            raise TypeError("This sequence does not have a default value")
        return self.elements[0]

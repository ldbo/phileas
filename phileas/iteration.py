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
from math import exp, log
from typing import Generic, Iterator, TypeVar

#: Data values that can be used
DataLiteral = None | bool | str | int | float

#: Dictionary keys
Key = DataLiteral

#: A data tree consists of literal leaves, and dictionary or list nodes
DataTree = DataLiteral | dict[Key, "DataTree"] | list["DataTree"]


class IterationTree(ABC):
    """
    Represents a set of data trees, as well as the way to iterate over them.
    """

    @abstractmethod
    def iterate(self) -> Iterator[DataTree]:
        """
        Yields all the data trees represented by the iteration tree.
        """
        raise NotImplementedError()


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

    children: IterationTree | list[IterationTree] | dict[DataLiteral, IterationTree]


@dataclass(frozen=True)
class CartesianProduct(IterationMethod):
    """
    Iteration over the cartesian product of its children, starting iterating
    over the first element first.
    """

    #: Children trees stored in a list
    _iterated_trees: list[IterationTree] = field(
        init=False, default_factory=list, repr=False, compare=False, hash=False
    )

    def __post_init__(self):
        if isinstance(self.children, IterationTree):
            self._iterated_trees.append(self.children)
        elif isinstance(self.children, list):
            self._iterated_trees.extend(self.children)
        elif isinstance(self.children, dict):
            self._iterated_trees.extend(list(self.children.values()))
        else:
            raise TypeError("The children don't have a supported type.")

    def iterate(self) -> Iterator[DataTree]:
        n = len(self._iterated_trees)
        index = 0
        iterators = [tree.iterate() for tree in self._iterated_trees]
        current = [next(iterator) for iterator in iterators]

        done = False
        while not done:
            yield current.copy()

            iterator_exhausted = True
            while iterator_exhausted:
                try:
                    current[index] = next(iterators[index])
                    index = 0
                    iterator_exhausted = False
                except StopIteration:
                    iterators[index] = self._iterated_trees[index].iterate()
                    current[index] = next(iterators[index])
                    index += 1

                    if index == n:
                        done = True
                        break


@dataclass(frozen=True)
class Union(IterationMethod):
    """
    Iteration over one varying field at a time, starting with the first one.
    """

    #: Children trees stored in a list
    _iterated_trees: list[IterationTree] = field(
        init=False, default_factory=list, repr=False, compare=False, hash=False
    )

    def __post_init__(self):
        if isinstance(self.children, IterationTree):
            self._iterated_trees.append(self.children)
        elif isinstance(self.children, list):
            self._iterated_trees.extend(self.children)
        elif isinstance(self.children, dict):
            self._iterated_trees.extend(list(self.children.values()))
        else:
            raise TypeError("The children don't have a supported type.")

    def iterate(self) -> Iterator[DataTree]:
        iterators = [tree.iterate() for tree in self._iterated_trees]
        base = [next(iterator) for iterator in iterators]

        yield base.copy()

        for index, iterator in enumerate(iterators):
            for value in iterator:
                current = base.copy()
                current[index] = value
                yield current


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


### Leaves ###


@dataclass(frozen=True)
class IterationLiteral(IterationTree):
    """
    Wrapper around a `DataLiteral`.
    """

    value: DataLiteral

    def iterate(self) -> Iterator[DataLiteral]:
        yield self.value


T = TypeVar("T", bound=int | float)


@dataclass(frozen=True)
class NumericRange(IterationTree, Generic[T]):
    """
    Represents a range of numeric values.
    """

    start: T
    end: T

    @abstractmethod
    def iterate(self) -> Iterator[T]:
        raise NotImplementedError()


@dataclass(frozen=True)
class LinearRange(NumericRange[float]):
    """
    Generate `steps` values linearly spaced between `start` and `end`, both
    included.
    """

    steps: int

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


@dataclass(frozen=True)
class GeometricRange(NumericRange[float]):
    """
    Generate `steps` values geometrically spaced between `start` and `end`, both
    included.
    """

    steps: int

    def __post_init__(self):
        if self.start * self.end <= 0:
            raise ValueError("Range limits must be non-zero and with the same sign")
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


@dataclass(frozen=True)
class Sequence(IterationTree):
    """
    Sequence of data trees.
    """

    elements: list[DataTree]

    def iterate(self) -> Iterator[DataTree]:
        return iter(self.elements)

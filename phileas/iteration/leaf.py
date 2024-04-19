"""
This module defines abstract and concrete iteration leaves, which are the
actual data sources of an iteration tree, alongside their iterators.
"""

from dataclasses import dataclass, field
from math import exp, log
from typing import Any, Callable, Generic, Iterator, TypeVar

import numpy as np

from .base import (
    ChildPath,
    DataTree,
    IterationLeaf,
    IterationTree,
    Key,
    ListIterator,
    NoDefaultError,
    NoDefaultPolicy,
    PseudoDataTree,
    TreeIterator,
    _NoDefault,
    no_default,
)

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


@dataclass(frozen=True)
class GeneratorWrapper(IterationLeaf):
    """
    Wrapper around a generator function, which can be used in order not to have
    to implement an new iteration leave, and its iterator. Not that only forward
    iteration is supported by the node.
    """

    generator_function: Callable[[Any], Iterator[DataTree]]
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    size: int | None = None
    default_value: DataTree = field(default_factory=_NoDefault)

    def __len__(self) -> int:
        if self.size is not None:
            return self.size

        raise ValueError("Generator wrapper does not have a size.")

    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        if self.default_value != no_default:
            return self.default_value

        if no_default_policy == NoDefaultPolicy.ERROR:
            raise NoDefaultError("No default value.", [])
        elif no_default_policy == NoDefaultPolicy.SENTINEL:
            return no_default
        else:  # no_default_policy == NoDefaultPolicy.SKIP
            return no_default

    def __iter__(self) -> TreeIterator:
        return GeneratorWrapperIterator(self)

    def __getitem__(self, key: Key) -> IterationTree:
        raise TypeError("Generator wrapper does not support indexing.")

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self


class GeneratorWrapperIterator(TreeIterator):
    tree: GeneratorWrapper
    generator: Iterator[DataTree]

    def __init__(self, tree: GeneratorWrapper):
        super().__init__()
        self.tree = tree
        self.reset()

    def __next__(self) -> DataTree:
        return next(self.generator)

    def reset(self):
        self.generator = self.tree.generator_function(
            *self.tree.args, **self.tree.kwargs
        )

    def reverse(self):
        raise TypeError("Generator wrapper iterator does not support reverse.")


## Numeric ranges


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


## Sequence


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


## Random leaves


@dataclass(frozen=True)
class Seed:
    """
    Seed of a random iteration node, used for its RNG.
    """

    #: Path of the node in the biggest tree that used it.
    path: ChildPath

    #: Salt value, to customize iteration independently of the shape of the
    #: iteration tree.
    salt: DataTree | None

    def to_bytes(self) -> bytes:
        """
        Convert the seed to bytes, for RNG seeding.
        """
        salt = f":{self.salt}" if self.salt is not None else ""
        return ("/".join(map(str, self.path)) + salt).encode("utf-8")


@dataclass(frozen=True)
class RandomIterationLeaf(IterationLeaf):
    """
    Deterministic pseudo-random elements generator.


    Concrete leaves must be dataclasses, in order to be able to modify their
    seed.
    """

    #: Seed of the generator. It must be guaranteed that successive iteration
    #: values only depend on the seed value.
    #:
    #: For iteration to be possible, the seed must be set.
    seed: Seed | None = None

    #: Optional number of elements generated by the leaf.
    size: None | int = None

    def __len__(self) -> int:
        if self.size is not None:
            return self.size

        raise TypeError("This random iteration leaf does not have a length.")

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self.default()  # type: ignore[return-value]


@dataclass(frozen=True)
class NumpyRNG(RandomIterationLeaf):
    """
    Random iteration leaf based on the RNG of numpy. In order to be
    """

    #: Which distribution to use for the node. It must be a distribution method
    #: of `np.random.Generator`.
    distribution: Callable = np.random.Generator.random

    #: Arguments list to pass to the distribution.
    args: list = field(default_factory=list)

    #: Keyword arguments to pass to the distribution.
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> TreeIterator:
        if self.seed is None:
            raise ValueError("Cannot iterate over a non seeded random leaf.")

        return NumpyRNGIterator(
            list(self.seed.to_bytes()),
            self.distribution,
            self.args,
            self.kwargs,
            self.size,
        )

    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        if no_default_policy == NoDefaultPolicy.ERROR:
            raise NoDefaultError("Numpy RNG does not have a default value.", [])
        elif no_default_policy == NoDefaultPolicy.SENTINEL:
            return no_default
        else:  # no_default_policy == NoDefaultPolicy.SKIP:
            return no_default

    def __getitem__(self, key: Key) -> IterationTree:
        raise TypeError("Numpy RNG does not support indexing.")


@dataclass
class NumpyRNGIterator(TreeIterator):
    """
    Iterator that generates random numbers by reseeding a numpy bit generator,
    and getting its first returned values.
    """

    seed: list[int]
    distribution: Callable
    args: list
    kwargs: dict
    size: int | None
    last_position: int = field(init=False)

    def __post_init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        if self.forward:
            self.position = -1
        elif self.size is not None:
            self.position = self.size
        else:
            raise ValueError("Cannot reset a reversed unsized iterator.")

    def __next__(self) -> DataTree:
        if self.forward:
            self.position += 1

            if self.size is not None and self.position >= self.size:
                raise StopIteration
        else:
            self.position -= 1

            if self.position < 0:
                raise StopIteration

        generator = np.random.Generator(
            np.random.PCG64(self.seed + list(f"%{self.position}".encode("utf-8")))
        )
        random = self.distribution(generator, *self.args, **self.kwargs)

        return random

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
    InfiniteLength,
    IterationLeaf,
    Key,
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
        return LiteralIterator(self)

    def __len__(self) -> int:
        return 1

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self.value  # type: ignore[return-value]

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree:
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


class LiteralIterator(TreeIterator[IterationLiteral]):
    def _current_value(self) -> DataTree:
        return self.tree.value


@dataclass(frozen=True)
class GeneratorWrapper(IterationLeaf):
    """
    Wrapper around a generator function, which can be used in order not to have
    to implement a new iteration leave, and its iterator. Note that only
    continuous forward iteration is supported by the node.
    """

    generator_function: Callable[..., Iterator[DataTree]]
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)

    #: Size of the tree. If the generator can provide more elements, only the
    #: first `size` ones are returned. If it cannot generate enough, a
    #: `StopIteration` is raised during iteration. `None` represents an
    #: infinite generator.
    size: int | None = None

    default_value: DataTree = field(default_factory=_NoDefault)

    def __len__(self) -> int:
        if self.size is None:
            raise InfiniteLength

        return self.size

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree:
        if self.default_value == no_default:
            raise NoDefaultError.build_from(self)

        return self.default_value

    def __iter__(self) -> TreeIterator:
        return GeneratorWrapperIterator(self)

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self


class GeneratorWrapperIterator(TreeIterator[GeneratorWrapper]):
    generator: Iterator[DataTree]
    last_position: int
    size: int | None

    def __init__(self, tree: GeneratorWrapper):
        super().__init__(tree)
        self.last_position = -1
        self.generator = self.tree.generator_function(
            *self.tree.args, **self.tree.kwargs
        )
        self.size = tree.size

    def _current_value(self) -> DataTree:
        if self.position != self.last_position + 1:
            msg = (
                f"{self.__class__.__name__} can only be used for "
                "continuous forward iteration."
            )
            raise Exception(msg)

        if self.position == self.size:
            raise StopIteration

        self.last_position = self.position

        value = None
        try:
            value = next(self.generator)
        except StopIteration as e:
            tree_size = self.size if self.size is not None else "infinite"
            raise ValueError(
                f"Generator exhausted at position {self.position}, whereas the "
                f"tree size is {tree_size}."
            ) from e

        return value

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

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree:
        if isinstance(self.default_value, _NoDefault):
            raise NoDefaultError.build_from(self)

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

        # TBD do we want to use a SequenceIterator for those?
        return SequenceIterator(Sequence(sequence, default_value=self.default_value))

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

        return SequenceIterator(Sequence(sequence, default_value=self.default_value))

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

        return SequenceIterator(Sequence(sequence, default_value=self.default_value))

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
        return SequenceIterator(self)

    def __len__(self) -> int:
        return len(self.elements)

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree | _NoDefault:
        if self.default_value == no_default:
            raise NoDefaultError.build_from(self)

        return self.default_value

    def __getitem__(self, key: Key) -> DataTree:  # type: ignore[override]
        """
        Work as if the iteration sequence was the sequence it contains.

        It does not respect the API of `IterationTree.__getitem__`, as it
        returns a `DataTree`, but this is so convenient that we accept this
        compromise.
        """
        return self.elements[key]  # type: ignore[index]


class SequenceIterator(TreeIterator[Sequence]):
    def _current_value(self) -> DataTree:
        return self.tree.elements[self.position]


## Random leaves


@dataclass(frozen=True)
class Seed:
    """
    Seed of a random iteration node, used by its RNG.
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
    #: For iteration to be possible, the seed must be set. If you don't want to
    #: manually specify the seed, you can use
    #: `phileas.iteration.utility.generate_seeds`.
    seed: Seed | None = None

    #: Number of elements generated by the leaf. If `None`, the leaf is
    #: infinite.
    size: None | int = None

    default_value: DataTree | _NoDefault = field(default_factory=_NoDefault)

    def __len__(self) -> int:
        if self.size is None:
            raise InfiniteLength

        return self.size

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self


@dataclass(frozen=True)
class NumpyRNG(RandomIterationLeaf):
    """
    Random iteration leaf based on the RNG of numpy.
    """

    #: Which distribution to use for the node. It must be a distribution method
    #: of `np.random.Generator`.
    distribution: Callable = np.random.Generator.random

    #: Arguments list to pass to the distribution.
    args: list = field(default_factory=list)

    #: Keyword arguments to pass to the distribution.
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> TreeIterator:
        return NumpyRNGIterator(self)

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree:
        if self.default_value == no_default:
            raise NoDefaultError.build_from(self)

        return self.default_value


@dataclass
class NumpyRNGIterator(TreeIterator[NumpyRNG]):
    """
    Iterator that generates random numbers by reseeding a numpy bit generator,
    and getting its first returned values.
    """

    seed: list[int]
    size: int | None

    def __init__(self, tree: NumpyRNG) -> None:
        super().__init__(tree)

        if tree.seed is None:
            raise ValueError("Cannot iterate over a non seeded random leaf.")

        self.seed = list(tree.seed.to_bytes())

    def _current_value(self) -> DataTree:
        generator = np.random.Generator(
            np.random.PCG64(self.seed + list(f"%{self.position}".encode("utf-8")))
        )
        random = self.tree.distribution(generator, *self.tree.args, **self.tree.kwargs)

        return random

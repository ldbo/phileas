"""
This module defines abstract and concrete iteration leaves, which are the
actual data sources of an iteration tree, alongside their iterators.
"""

from dataclasses import dataclass, field
from math import exp, inf, log
from typing import Any, Callable, Generic, Iterator, TypeVar

import numpy as np

from .base import (
    DataTree,
    InfiniteLength,
    IterationLeaf,
    Key,
    NoDefaultError,
    NoDefaultPolicy,
    OneWayTreeIterator,
    PseudoDataTree,
    TreeIterator,
    _NoDefault,
    no_default,
)
from .random import RandomTree

DT = TypeVar("DT", bound=DataTree)


@dataclass(frozen=True)
class IterationLiteral(IterationLeaf, Generic[DT]):
    """
    Wrapper around a data tree.
    """

    value: DT

    def _iter(self) -> TreeIterator:
        return LiteralIterator(self)

    def _len(self) -> int:
        return 1

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self.value  # type: ignore[return-value]

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree:
        return self.value

    def __getitem__(self, key: Key) -> DataTree:  # type: ignore[override]
        """
        Work as if the iteration literal was the literal it contains.

        It does not respect the API of :py:meth:`IterationTree.__getitem__`, as
        it returns a :py:class:`~phileas.iteration.DataTree`, but this is so
        convenient that we accept this compromise.

        Raises:
            KeyError: if the key does not exist.
        """
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
    #: first :py:attr:`size` ones are returned. If it cannot generate enough, a
    #: :py:class:`StopIteration` is raised during iteration. ``None`` represents
    #: an infinite generator.
    size: int | None = None

    default_value: DataTree = field(default_factory=_NoDefault)

    def _len(self) -> int:
        if self.size is None:
            raise InfiniteLength

        return self.size

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree:
        if self.default_value == no_default:
            raise NoDefaultError.build_from(self)

        return self.default_value

    def _iter(self) -> TreeIterator:
        return GeneratorWrapperIterator(self)

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self


class GeneratorWrapperIterator(OneWayTreeIterator, TreeIterator[GeneratorWrapper]):
    generator: Iterator[DataTree]

    def __init__(self, tree: GeneratorWrapper):
        OneWayTreeIterator.__init__(self)
        TreeIterator.__init__(self, tree)

        self.generator = self.tree.generator_function(
            *self.tree.args, **self.tree.kwargs
        )

    def _next(self) -> DataTree:
        return next(self.generator)


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

    def _iter(self) -> TreeIterator:
        raise TypeError("Cannot iterate over a numeric range.")

    def _len(self) -> int:
        msg = "A numeric range does not have a length. "
        msg += "You can instead use a geometric, linear or integer range."
        raise TypeError(msg)

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree:
        if isinstance(self.default_value, _NoDefault):
            raise NoDefaultError.build_from(self)

        return self.default_value


@dataclass(frozen=True)
class LinearRange(NumericRange[float]):
    """
    Generate :py:attr:`steps` values linearly spaced between :py:attr:`start`
    and :py:attr:`end`, both included.
    """

    # Have to specify a default value because :py:attr:`default_value` has one.
    steps: int = field(default=2)

    def __post_init__(self):
        if self.steps < 1 or (self.start != self.end and self.steps < 2):
            raise ValueError("Invalid number of steps.")

    def _iter(self) -> TreeIterator:
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

    def _len(self) -> int:
        return self.steps


@dataclass(frozen=True)
class GeometricRange(NumericRange[float]):
    """
    Generate :py:attr:`steps` values geometrically spaced
    between :py:attr:`start` and :py:attr:`end`, both included.
    """

    # Have to specify a default value because :py:attr:`default_value` has one.
    steps: int = field(default=2)

    def __post_init__(self):
        if self.start * self.end <= 0:
            raise ValueError("Range limits must be non-zero and with the same sign.")
        if self.steps < 1 or (self.start != self.end and self.steps < 2):
            raise ValueError("Invalid number of steps.")

    def _iter(self) -> TreeIterator:
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

    def _len(self) -> int:
        return self.steps


@dataclass(frozen=True)
class IntegerRange(NumericRange[int | float]):
    """
    Generate integer values :py:attr:`step` spaced, between :py:attr:`start`
    and :py:attr:`end`, both included. :py:attr:`start` must be an ``int``,
    but :py:attr:`end` can also be ``math.inf`` or ``-math.inf``. In these
    cases, the range is infinite.
    """

    step: int = field(default=1)

    def __post_init__(self):
        if self.step < 0 or (self.start != self.end and self.step == 0):
            raise ValueError("Invalid step size.")

        if not isinstance(self.step, int):
            raise ValueError("step must be an int.")

        if not isinstance(self.start, int):
            raise ValueError(f"start must be an int, {self.start} is not supported.")

        if not isinstance(self.end, int) and self.end not in {inf, -inf}:
            raise ValueError(
                "end must be an int or +/-math.inf, {self.end} is not supported"
            )

    def _iter(self) -> TreeIterator:
        return IntegerRangeIterator(self)

    def _len(self) -> int:
        if self.end in {inf, -inf}:
            raise InfiniteLength

        if self.end == self.start:
            return 1

        assert isinstance(self.start, int)
        assert isinstance(self.end, int)
        assert self.step >= 1
        return 1 + abs(self.end - self.start) // self.step


class IntegerRangeIterator(TreeIterator[IntegerRange]):
    length: int | float
    direction: int

    def __init__(self, tree: IntegerRange) -> None:
        super().__init__(tree)
        length: int | float | None = tree.safe_len()
        if length is None:
            length = inf
        self.length = length
        self.direction = 1 if tree.end > tree.start else -1

    def _current_value(self) -> DataTree:
        return self.tree.start + self.direction * self.position * self.tree.step


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

    def _iter(self) -> TreeIterator:
        return SequenceIterator(self)

    def _len(self) -> int:
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

        It does not respect the API of :py:meth:`IterationTree.__getitem__`, as
        it returns a :py:attr:`~phileas.iteration.base.DataTree`, but this is
        so convenient that we accept this compromise.
        """
        return self.elements[key]  # type: ignore[index]


class SequenceIterator(TreeIterator[Sequence]):
    def _current_value(self) -> DataTree:
        return self.tree.elements[self.position]


## Random leaves


@dataclass(frozen=True)
class RandomIterationLeaf(IterationLeaf, RandomTree):
    """
    Deterministic pseudo-random elements generator.
    """

    #: Number of elements generated by the leaf. If ``None``, the leaf is
    #: infinite.
    size: None | int = None

    default_value: DataTree | _NoDefault = field(default_factory=_NoDefault)

    def _len(self) -> int:
        if self.size is None:
            raise InfiniteLength

        return self.size

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self


@dataclass(frozen=True)
class NumpyRNG(RandomIterationLeaf):
    """
    Random iteration leaf based on the RNG of numpy.

    Note:

    Iteration is based on :py:class:`NumpyRNGIterator`. It works by seeding a
    PRNG with a value which depends on the requested iteration position. This
    might introduce a bias on the generated random values. This is done as a
    way to provide a random-access PRNG, although an actual random-access PRNG
    would be preferable.

    Alternatively, the iterator could be built from a usual iterative PRNG, and
    caching would be used to provide access to previous values. This would
    impact iteration performances (especially for cache misses, which can
    induce unbound time delays).
    """

    #: Which distribution to use for the node. It must be a distribution method
    #: of :py:class:`numpy.random.Generator`.
    distribution: Callable = np.random.Generator.random

    #: Arguments list to pass to the distribution.
    args: list = field(default_factory=list)

    #: Keyword arguments to pass to the distribution.
    kwargs: dict[str, Any] = field(default_factory=dict)

    def _iter(self) -> TreeIterator:
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


@dataclass(frozen=True)
class UniformBigIntegerRng(RandomIterationLeaf):
    """
    Random iteration leaf generating arbitrarily big integers. It works by
    iteratively repeatedly using a Numpy RNG.

    By default returns a byte (integer larger or equal to 0 and smaller than 256).
    """

    #: The inclusively lowest value that may be taken by the output.
    low: int = 0

    #: The inclusively highest value that may be taken by the output.
    high: int = 255

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError("Lower bound must be smaller or equal to the upper bound.")

    def _iter(self) -> TreeIterator:
        return UniformBigIntegerRngIterator(self)

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree:
        if self.default_value == no_default:
            raise NoDefaultError.build_from(self)

        return self.default_value


@dataclass
class UniformBigIntegerRngIterator(TreeIterator[UniformBigIntegerRng]):
    """
    Iterator that generates random numbers by reseeding a numpy byte generator.

    Bytes are repeatedly sampled until the generated number fits in the required
    bounds.
    """

    seed: list[int]

    #: The number of bytes required to generate a new integer. It is positive.
    _required_bytes: int

    def __init__(self, tree: UniformBigIntegerRng) -> None:
        super().__init__(tree)

        if tree.seed is None:
            raise ValueError("Cannot iterate over a non seeded random leaf.")

        self.seed = list(tree.seed.to_bytes())
        self._required_bytes = 1 + (self.tree.high - self.tree.low) // 256

    def _current_value(self) -> DataTree:
        generator = np.random.Generator(
            np.random.PCG64(self.seed + list(f"%{self.position}".encode("utf-8")))
        )

        value = self.tree.high - self.tree.low + 1
        while value > (self.tree.high - self.tree.low):
            value = int.from_bytes(
                generator.bytes(self._required_bytes), byteorder="little"
            )

        return self.tree.low + value

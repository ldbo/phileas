"""
This package contains the trees used for data iteration.

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

__all__ = [
    "Accumulator",
    "CartesianProduct",
    "ChildPath",
    "Configurations",
    "DataLiteral",
    "DataTree",
    "First",
    "FunctionalTranform",
    "GeneratorWrapper",
    "GeometricRange",
    "InfiniteLength",
    "IntegerRange",
    "IterationLeaf",
    "IterationLiteral",
    "IterationMethod",
    "IterationTree",
    "Key",
    "Lazify",
    "LinearRange",
    "NoDefaultError",
    "NoDefaultPolicy",
    "NumericRange",
    "NumpyRNG",
    "Pick",
    "PrimeRng",
    "PseudoDataLiteral",
    "PseudoDataTree",
    "RandomIterationLeaf",
    "Seed",
    "Sequence",
    "Shuffle",
    "Transform",
    "TreeIterator",
    "UniformBigIntegerRng",
    "Union",
    "Zip",
    "_Child",
    "_NoDefault",
    "child",
    "generate_seeds",
    "no_default",
    "restrict_leaves_sizes",
]

from .base import (
    ChildPath,
    DataLiteral,
    DataTree,
    InfiniteLength,
    IterationLeaf,
    IterationTree,
    Key,
    NoDefaultError,
    NoDefaultPolicy,
    PseudoDataLiteral,
    PseudoDataTree,
    TreeIterator,
    _Child,
    _NoDefault,
    child,
    no_default,
)
from .leaf import (
    GeneratorWrapper,
    GeometricRange,
    IntegerRange,
    IterationLiteral,
    LinearRange,
    NumericRange,
    NumpyRNG,
    PrimeRng,
    RandomIterationLeaf,
    Sequence,
    UniformBigIntegerRng,
)
from .node import (
    Accumulator,
    CartesianProduct,
    Configurations,
    First,
    FunctionalTranform,
    IterationMethod,
    Lazify,
    Pick,
    Shuffle,
    Transform,
    Union,
    Zip,
)
from .random import Seed, generate_seeds
from .utility import restrict_leaves_sizes

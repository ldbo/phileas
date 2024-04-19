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
    "ChildPath",
    "DataLiteral",
    "DataTree",
    "IterationLeaf",
    "IterationTree",
    "Key",
    "ListIterator",
    "NoDefaultError",
    "NoDefaultPolicy",
    "PseudoDataLiteral",
    "PseudoDataTree",
    "TreeIterator",
    "_Child",
    "_NoDefault",
    "child",
    "no_default",
    "GeneratorWrapper",
    "GeometricRange",
    "IntegerRange",
    "IterationLiteral",
    "LinearRange",
    "NumericRange",
    "NumpyRNG",
    "RandomIterationLeaf",
    "Seed",
    "Sequence",
    "generate_seeds",
    "Accumulator",
    "CartesianProduct",
    "FunctionalTranform",
    "IterationMethod",
    "Lazify",
    "Transform",
    "Union",
    "generate_seeds",
    "restrict_leaves_sizes",
]

from .base import (
    ChildPath,
    DataLiteral,
    DataTree,
    IterationLeaf,
    IterationTree,
    Key,
    ListIterator,
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
    RandomIterationLeaf,
    Seed,
    Sequence,
)
from .node import (
    Accumulator,
    CartesianProduct,
    FunctionalTranform,
    IterationMethod,
    Lazify,
    Transform,
    Union,
)
from .utility import generate_seeds, restrict_leaves_sizes

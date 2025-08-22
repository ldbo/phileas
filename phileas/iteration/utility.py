"""
This module defines utility functions related to iteration.
"""

from enum import Enum
from types import NoneType
from typing import TYPE_CHECKING, Iterable

from .base import (
    ChildPath,
    DataLiteral,
    DataTree,
    InfiniteLength,
    IterationLeaf,
    IterationTree,
    PseudoDataTree,
)
from .leaf import Sequence

if TYPE_CHECKING:
    import pandas


class RestrictionPolicy(Enum):
    #: Iteration leaves only keep their first and last values.
    FIRST_LAST = "FIRST_LAST"

    #: Iteration leaves only keep their two first values.
    FIRST_SECOND = "FIRST_SECOND"

    #: Iteration leaves keep their first, second and last values.
    COMBINED = "COMBINED"


def restrict_leaves_sizes(
    tree: IterationTree,
    policy: RestrictionPolicy = RestrictionPolicy.FIRST_LAST,
) -> IterationTree:
    """
    Restrict the size of the iteration leaves of the tree, depending on the
    restriction policy. This is useful for troubleshooting, or to verify that
    the full range of the leaves is supported by *something*.
    """

    def _restrict(tree: IterationTree, _: ChildPath) -> IterationTree:
        if not isinstance(tree, IterationLeaf):
            return tree

        if policy == RestrictionPolicy.FIRST_LAST:
            try:
                if len(tree) <= 2:
                    return tree
            except InfiniteLength:
                return tree

            tree_iter = iter(tree)
            first = next(tree_iter)
            tree_iter.reverse()
            tree_iter.reset()
            last = next(tree_iter)
            return Sequence([first, last])
        elif policy == RestrictionPolicy.FIRST_SECOND:
            try:
                if len(tree) <= 2:
                    return tree
            except InfiniteLength:
                return tree

            tree_iter = iter(tree)
            return Sequence([next(tree_iter), next(tree_iter)])
        else:
            try:
                if len(tree) <= 3:
                    return tree
            except InfiniteLength:
                return tree

            tree_iter = iter(tree)
            first, second = next(tree_iter), next(tree_iter)
            tree_iter.reverse()
            tree_iter.reset()
            last = next(tree_iter)
            return Sequence([first, second, last])

    return tree.depth_first_modify(_restrict)


def recursive_union(tree1: DataTree, tree2: DataTree) -> DataTree:
    """
    Return the recursive union of two datatrees. If any of those is not a
    dictionary, it returns the latter. Otherwise, it recursively applies the
    union operator of dictionaries.
    """
    if not isinstance(tree1, dict) or not isinstance(tree2, dict):
        return tree2

    union = tree1.copy()
    keys1 = set(tree1.keys())
    keys2 = set(tree2.keys())

    for key in keys2 - keys1:
        union[key] = tree2[key]

    for key in keys1 & keys2:
        union[key] = recursive_union(tree1[key], tree2[key])

    return union


def is_transformed_iteration_leaf(tree: IterationTree) -> bool:
    """
    A transformed iteration leaf is an iteration tree that only contained
    :py:class:`~phileas.iteration.Transform`
    and :py:class:`~phileas.iteration.IterationLeaf` nodes. That is, it does
    not contain :py:class:`~phileas.iteration.IterationMethod` nodes.

    This function checks if a tree is a transformed iteration leaf.
    """
    from phileas.iteration import Transform

    while isinstance(tree, (Transform, IterationLeaf)):
        if isinstance(tree, IterationLeaf):
            return True

        assert isinstance(tree, Transform)
        tree = tree.child

    return False


def flatten_datatree(
    tree: DataTree | PseudoDataTree, key_prefix: None | str = None, separator: str = "."
) -> dict[str, DataLiteral | IterationLeaf] | DataLiteral | IterationLeaf:
    """
    Transform nested ``dict`` and ``list`` objects to a single-level ``dict``.
    :py:attr:`~phileas.iteration.base.DataLiteral`s
    and :py:attr:`~phileas.iteration.base.IterationLeaf`s are left unchanged,
    so that flattening a :py:attr:`~phileas.iteration.base.DataTree` returns a
    :py:attr:`~phileas.iteration.base.DataTree`, and flattening a
    :py:attr:`~phileas.iteration.base.PseudoDataTree` returns
    a :py:attr:`~phileas.iteration.base.PseudoDataTree`.

    Keys are converted to ``str``, and concatenated using the specified
    ``separator``. ``list`` objects are considered as ``int``-keyed ``dict``.

    >>> tree = {
    ...     "key1": {
    ...         "key1-1": 1
    ...     },
    ...     "key2": [1, 2],
    ...     "key3": "value"
    ... }
    >>> flatten_datatree(tree)
    {'key1.key1-1': 1, 'key2.0': 1, 'key2.1': 2, 'key3': 'value'}
    """
    iterable: Iterable[tuple[DataLiteral, DataTree | PseudoDataTree]]
    if isinstance(tree, dict):
        iterable = tree.items()
    elif isinstance(tree, list):
        iterable = enumerate(tree)
    else:
        return tree

    flat_content: dict[str, DataLiteral | IterationLeaf] = {}
    for key, value in iterable:
        flat_key = str(key)
        if key_prefix is not None:
            flat_key = f"{key_prefix}{separator}{flat_key}"

        flat_value = flatten_datatree(value, key_prefix=flat_key)
        if isinstance(flat_value, dict):
            flat_content |= flat_value
        else:  # flat_value is a DataLiteral or IterationLeaf
            flat_content[flat_key] = flat_value

    return flat_content


def iteration_tree_to_xarray_parameters(
    tree: IterationTree,
) -> tuple[dict[str, list | DataLiteral], list[str], list[int]]:
    """
    Generate the arguments required to build an :py:class:`xarray.DataArray` or
    :py:class:`xarray.DataFrame`. You can then modify them, if needed, and build
    the xarray container used to store the results of your experiment.

    This function can only be applied to trees that only have finite leaves.

    >>> import numpy as np
    >>> import xarray as xr
    >>> coords, dims_name, dims_shape = iteration_tree_to_xarray_parameters(tree)
    >>> # Single data to be recorded for each iteration point
    >>> xr.DataArray(data=np.empty(dims_shape), coords=coords, dims=dims_name)
    >>> # Multiple data for each iteration point
    >>> xr.Dataset(
    >>>     data_vars=dict(
    >>>             measurement_1=(dims_name, np.full(dims_shape, np.nan)),
    >>>             measurement_2=(dims_name, np.full(dims_shape, np.nan)),
    >>>     ),
    >>>     coords=coords,
    >>> )

    .. caution::

       This function is tested against trees whose iteration methods are
       cartesian products only, as they represent the coordinates of full
       gridded datasets. You can use it on other kinds of trees, but then make
       sure to verify that its behaves properly.

    .. seealso::

       :py:func:`~phileas.iteration.utility.data_tree_to_xarray_index` to index
       an xarray container  built with this function.

       :py:func:`~phileas.iteration.utility.iteration_tree_to_multiindex` if you
       want to work with Pandas tabular datasets.
    """
    flattened_tree = flatten_datatree(tree.to_pseudo_data_tree())
    if isinstance(flattened_tree, dict):
        coords: dict[str, list | DataLiteral] = {}
        dims_name: list[str] = []
        dims_shape: list[int] = []
        for name, value in flattened_tree.items():
            if isinstance(value, IterationLeaf):
                coords[name] = list(value)
                dims_name.append(name)
                dims_shape.append(len(value))
            elif isinstance(value, (bool, int, float, str, NoneType)):
                coords[name] = value
            else:
                raise TypeError(f"Leaf with type {type(value)} detected.")

        return coords, dims_name, dims_shape
    elif isinstance(flattened_tree, IterationLeaf):
        return {"dim_0": list(flattened_tree)}, ["dim_0"], [len(flattened_tree)]
    else:
        assert isinstance(flattened_tree, (NoneType, bool, str, int, float))
        return {"dim_0": [flattened_tree]}, ["dim_0"], [1]


def data_tree_to_xarray_index(
    tree: DataTree, dims_name: list[str]
) -> dict[str, DataLiteral]:
    """
    Convert a data tree generated by iterating over an iteration tree to an
    index suitable for indexing an xarray container initialized
    using :py:func:`~phileas.iteration.utility.iteration_tree_to_xarray_parameters`.
    """
    flat_tree = flatten_datatree(tree)
    if isinstance(tree, dict):
        assert isinstance(flat_tree, dict)
        return {
            coord: v  # type: ignore[misc]
            for coord, v in flat_tree.items()
            if coord in dims_name
        }
    else:
        raise ValueError("Cannot generate an Xarray index from a non-dict tree.")


def iteration_tree_to_multiindex(tree: IterationTree) -> "pandas.MultiIndex":
    """
    Create a :py:class:`pandas.MultiIndex` that holds the values stored in an
    iteration tree. This method iterates over the whole tree in order to create
    the index. Thus, it requires the iterated trees to have all the same shape.

    .. seealso::

       :py:func:`~phileas.iteration.utility.iteration_tree_to_xarray_parameters`
       if you want to work with xarray gridded datasets.
    """
    import pandas as pd

    flattened_tree = flatten_datatree(tree.to_pseudo_data_tree())
    index: pd.MultiIndex
    if isinstance(flattened_tree, dict):
        names = sorted(flattened_tree.keys())
        tuples = []
        for datatree in tree:
            fd = flatten_datatree(datatree)

            error_message = "Iteration tree that change shape are not supported."
            if not isinstance(fd, dict):
                raise ValueError(error_message)

            try:
                tuples.append(tuple(fd[key] for key in names))
            except KeyError as e:
                raise ValueError(error_message) from e

        index = pd.MultiIndex.from_tuples(tuples, names=names)
    elif isinstance(flattened_tree, IterationLeaf):
        index = pd.MultiIndex.from_arrays([(dt,) for dt in flattened_tree])
    else:
        assert isinstance(flattened_tree, (NoneType, bool, str, int, float))
        index = pd.MultiIndex.from_arrays([(flattened_tree,)])

    return index.sortlevel()[0]

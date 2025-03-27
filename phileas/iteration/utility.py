"""
This module defines utility functions related to iteration.
"""

import dataclasses
from enum import Enum
from types import NoneType
from typing import Iterable

from .base import (
    ChildPath,
    DataLiteral,
    DataTree,
    IterationLeaf,
    IterationTree,
    PseudoDataTree,
)
from .leaf import RandomIterationLeaf, Seed, Sequence


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
            if len(tree) <= 2:
                return tree

            tree_iter = iter(tree)
            first = next(tree_iter)
            tree_iter.reverse()
            tree_iter.reset()
            last = next(tree_iter)
            return Sequence([first, last])
        elif policy == RestrictionPolicy.FIRST_SECOND:
            if len(tree) <= 2:
                return tree

            tree_iter = iter(tree)
            return Sequence([next(tree_iter), next(tree_iter)])
        else:
            if len(tree) <= 3:
                return tree

            tree_iter = iter(tree)
            first, second = next(tree_iter), next(tree_iter)
            tree_iter.reverse()
            tree_iter.reset()
            last = next(tree_iter)
            return Sequence([first, second, last])

    return tree.depth_first_modify(_restrict)


def generate_seeds(tree: IterationTree, salt: DataTree | None = None) -> IterationTree:
    """
    Populate the seeds of the random leaves of the tree, using the given salt.
    """

    def _generate_seed(tree: IterationTree, path: ChildPath) -> IterationTree:
        if isinstance(tree, RandomIterationLeaf):
            return dataclasses.replace(tree, seed=Seed(path, salt))
        else:
            return tree

    return tree.depth_first_modify(_generate_seed)


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


def flatten_datatree(
    tree: DataTree | PseudoDataTree, key_prefix: None | str = None, separator: str = "."
) -> dict[str, DataLiteral | IterationLeaf] | DataLiteral | IterationLeaf:
    """
    Transform nested `dict`s and `list`s to a single-level dict. `DataLiteral`s
    and `IterationLeaf`s are left unchanged, so that flatting a `DataTree`
    returns a `DataTree`, and flattening a `PseudoDataTree` returns a
    `PseudoDataTree`.

    Keys are converted to `str`, and concatenated using the specified
    `separator`. `list`s are considered as `int`-keyed `dict`s.

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
) -> tuple[dict[str, list], list[str], list[int]]:
    """
    Generate the arguments required to build an `xr.DataArray` or
    `xr.DataFrame`. You can then modify them, if needed, and build the `xarray`
    used to store the results of your experiment.

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
    """
    flattened_tree = flatten_datatree(tree.to_pseudo_data_tree())
    if isinstance(flattened_tree, dict):
        coords: dict[str, list] = {}
        dims_name: list[str] = []
        dims_shape: list[int] = []
        for name, value in flattened_tree.items():
            if isinstance(value, IterationLeaf):
                coords[name] = list(value)
                dims_name.append(name)
                dims_shape.append(len(value))
            elif isinstance(value, (bool, int, float, NoneType)):
                pass
            else:
                raise TypeError(f"Leaf with type {type(value)} detected.")

        return coords, dims_name, dims_shape
    elif isinstance(flattened_tree, IterationLeaf):
        return {"dim_0": list(flattened_tree)}, ["dim_0"], [len(flattened_tree)]
    else:
        assert isinstance(flattened_tree, (NoneType, bool, str, int, float))
        return {"dim_0": [flattened_tree]}, ["dim_0"], [1]

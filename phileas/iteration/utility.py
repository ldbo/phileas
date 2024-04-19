"""
This module defines utility functions related to iteration.
"""

import dataclasses
from enum import Enum

from .base import ChildPath, DataTree, IterationLeaf, IterationTree
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

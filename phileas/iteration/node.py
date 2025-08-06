"""
This module defines abstract and concrete iteration tree nodes, which are
iteration methods and transform nodes, as well as their iterators.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
from typing import Callable, ClassVar, Literal, Sequence, TypeVar

import numpy as np

from phileas import _rust
from phileas.iteration import utility
from phileas.iteration.random import RandomTree
from phileas.logging import logger

from .base import (
    ChildPath,
    DataTree,
    DefaultIndex,
    InfiniteLength,
    IterationTree,
    Key,
    NoDefaultError,
    NoDefaultPolicy,
    OneWayTreeIterator,
    PseudoDataTree,
    TreeIterator,
    _Child,
    _NoDefault,
    child,
    no_default,
)

### Iteration method nodes ###


@dataclass(frozen=True)
class IterationMethod(IterationTree):
    """
    Iteration node having multiple children, supplied either as a list or
    dictionary.

    In order to implement a concrete iteration method, you should sub-class
    :py:class:`IterationMethod` and implement a
    corresponding :py:class:`IterationMethodIterator`, which is returned by
    :py:meth:`_iter`.

    This should remain the only node in an iteration tree that can hold ``dict``
    and ``list`` children. If you are tempted to create another node doing so,
    you should verify that it cannot be done by sub-classing
    :py:class:`IterationMethod` instead.
    """

    #: The children of the node. It must not be empty.
    children: list[IterationTree] | dict[Key, IterationTree]

    #: Order of iteration over the children. How it is used depends on the
    #: concrete iteration method implementation. It must be a permutation of the
    #: set of keys of ``children``.
    order: list[Key] | None = None

    #: Notify the iteration method to be lazy. For now, this feature is only
    #: supported for ``dict`` children. In this case, lazy iteration will just
    #: yield the keys that have changed at each step.
    #:
    #: Note that concrete iteration method classes are not required to actually
    #: implement lazy iteration, and if they don't, they will probably do so
    #: silently. Refer to their documentation or their implementation.
    lazy: bool = field(default=False)

    def __post_init__(self):
        if len(self.children) == 0:
            raise ValueError("Empty children are forbidden.")

        if not isinstance(self.children, (list, dict)):
            raise TypeError("IterationMethod must have list or dict children.")

        self.__verify_order()

        if self.lazy and not isinstance(self.children, dict):
            raise TypeError("Lazy iteration is only supported for dictionary children.")

        object.__setattr__(
            self, "configurations", self.__extract_children_configurations()
        )

    def __verify_order(self):
        if self.order is not None:
            keys: set[Key]
            if isinstance(self.children, list):
                keys = set(range(len(self.children)))
            else:
                keys = set(self.children.keys())

            if len(self.order) != len(set(self.order)):
                raise ValueError("The iteration order must not have repetitions.")

            if set(self.order) != keys:
                msg = "The iteration order must be a permutation of the children keys."
                raise ValueError(msg)

    def __extract_children_configurations(self) -> frozenset[Key]:
        if isinstance(self.children, dict):
            children_configurations = (
                child.configurations for child in self.children.values()
            )
        else:
            assert isinstance(self.children, list)
            children_configurations = (child.configurations for child in self.children)

        return frozenset(chain(*children_configurations))

    def _get_configuration(self, config_name: Key) -> IterationTree:
        if self.configurations == {}:
            return self

        new_children: list[IterationTree] | dict[Key, IterationTree]
        if isinstance(self.children, list):
            new_children = [
                child._get_configuration(config_name) for child in self.children
            ]
        else:
            assert isinstance(self.children, dict)
            new_children = {}
            for child_key, child_value in self.children.items():
                if isinstance(child_value, Configurations):
                    self.__update_children_with_configuration(
                        new_children, config_name, child_key, child_value
                    )
                else:
                    new_children[child_key] = child_value._get_configuration(
                        config_name
                    )

        return dataclasses.replace(self, children=new_children)

    def __update_children_with_configuration(
        self,
        children: dict[Key, IterationTree],
        requested_config_name: Key,
        configs_key: Key,
        configs: Configurations,
    ):
        """
        Given a dict containing the currently processed ``children`` of the tree,
        update it with configuration ``requested_config_name`` from the child
        ``configs``, identified by ``configs_key``, of the current node.
        """
        from .leaf import IterationLiteral

        requested_config = configs._get_configuration(requested_config_name)
        is_transformed_leaf = utility.is_transformed_iteration_leaf(requested_config)
        if configs.move_up:
            if not isinstance(requested_config, IterationMethod):
                raise ValueError(
                    "Configurations with move_up must only have iteration "
                    "methods children."
                )

            if not isinstance(requested_config.children, dict):
                raise ValueError("Configurations with move_up must have dict children.")

            self.__config_update_children(children, requested_config.children)

            if configs.insert_name:
                self.__config_update_children(
                    children, {configs_key: IterationLiteral(requested_config_name)}
                )
        elif is_transformed_leaf:  # and not configs.move_up
            self.__config_update_children(children, {configs_key: requested_config})

            if configs.insert_name:
                name_key = f"_{configs_key}_configuration"
                self.__config_update_children(
                    children,
                    {name_key: IterationLiteral(requested_config_name)},
                )
        else:  # not is_transformed_leaf and not configs.move_up
            if configs.insert_name:
                try:
                    requested_config = requested_config.insert_child(
                        ["_configuration"], IterationLiteral(requested_config_name)
                    )
                except TypeError:
                    assert isinstance(requested_config, IterationMethod)
                    assert isinstance(requested_config.children, list)
                    name_key = f"_{configs_key}_configuration"
                    self.__config_update_children(
                        children,
                        {name_key: IterationLiteral(requested_config_name)},
                    )

            self.__config_update_children(children, {configs_key: requested_config})

    @staticmethod
    def __config_update_children(
        children: dict[Key, IterationTree], new_dict: dict[Key, IterationTree]
    ):
        new_keys = set(new_dict.keys())
        conflicting_keys = set(children.keys()) & new_keys
        children.update(new_dict)

        if len(conflicting_keys) == 0:
            return

        logger.warning(
            f"Overriding keys {conflicting_keys} during configuration access."
        )

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        if isinstance(self.children, list):
            return [child.to_pseudo_data_tree() for child in self.children]
        else:
            assert isinstance(self.children, dict)
            return {
                key: value.to_pseudo_data_tree() for key, value in self.children.items()
            }

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree | _NoDefault:
        # List children
        if isinstance(self.children, list):
            default_list: list[DataTree | _NoDefault] = [None] * len(self.children)
            for position, child_tree in enumerate(self.children):
                error: None | NoDefaultError = None
                try:
                    value = child_tree.default(no_default_policy)

                    if (
                        value == no_default
                        and no_default_policy == NoDefaultPolicy.SKIP
                    ):
                        msg = (
                            "NoDefaultPolicy.SKIP is not supported for "
                            " iteration methods with list children."
                        )
                        error = NoDefaultError(msg, [])

                    default_list[position] = value
                except NoDefaultError as err:
                    error = NoDefaultError(err.args[0], [position] + err.path)

                if error is not None:
                    raise error

            return default_list

        # Dict children
        default_dict: dict[Key, DataTree] = {}
        for key, child_tree in self.children.items():
            error = None
            try:
                default_dict[key] = child_tree.default(no_default_policy)
            except NoDefaultError as err:
                own_path: ChildPath = [key]
                error = NoDefaultError(err.args[0], own_path + err.path)

            if error is not None:
                raise error

        if no_default_policy == NoDefaultPolicy.SKIP:
            return {
                key: child_tree
                for key, child_tree in default_dict.items()
                if not child_tree == no_default
            }
        else:
            return default_dict

    # Path API

    # Note: self.children[child_key] is not properly typed. Indeed, child_key
    # can be a dict or a list key, and there is no guarantee that self.children
    # is of the corresponding type. However, if the key does not exist, a
    # :py:class:`KeyError` will be raised, which is the expected behavior. For
    # those reasons, it is valid to ignore[index] all the self.children
    # [child_key] expressions.

    def _get(self, child_key: Key | _Child) -> IterationTree:
        if isinstance(child_key, _Child):
            raise KeyError("Iteration method does not support Child() index.")

        return self.children[child_key]  # type: ignore[index]

    def _insert_child(
        self, child_key: Key | _Child, child: IterationTree
    ) -> IterationTree:
        if isinstance(child_key, _Child):
            raise KeyError("Iteration method does not support Child() index.")

        new_children = self.children.copy()
        new_children[child_key] = child  # type: ignore[index]
        return dataclasses.replace(self, children=new_children)

    def _remove_child(self, child_key: Key | _Child) -> IterationTree:
        new_children = self.children.copy()
        _ = new_children.pop(child_key)  # type: ignore[arg-type]
        return dataclasses.replace(self, children=new_children)

    def _replace_root(
        self, Node: type[IterationTree], *args, **kwargs
    ) -> IterationTree:
        if not issubclass(Node, IterationMethod):
            raise TypeError(f"Cannot replace an iteration method with a {Node}")

        return Node(self.children.copy(), *args, **kwargs)

    def _depth_first_modify(
        self,
        modifier: Callable[[IterationTree, ChildPath], IterationTree],
        path: ChildPath,
    ) -> IterationTree:
        new_children: dict[Key, IterationTree] | list[IterationTree]
        if isinstance(self.children, list):
            new_children = [
                child._depth_first_modify(modifier, path + [position])
                for position, child in enumerate(self.children)
            ]
        else:
            new_children = {
                position: child._depth_first_modify(modifier, path + [position])
                for position, child in self.children.items()
            }

        return modifier(dataclasses.replace(self, children=new_children), path)


T = TypeVar("T", bound=IterationMethod, covariant=True)


class IterationMethodIterator(TreeIterator[T]):
    """
    Base class used to implement concrete :py:class:`IterationMethod` nodes
    iterators, and providing helper attributes to do so.
    """

    #: Children iterators stored in a list. This, with :py:attr:`keys`, allows
    #: child-class to only implement iteration over lists.
    #:
    #: If the iteration tree does specify an order, use it. Otherwise,
    #: dictionaries are sorted by their key value, and lists keep the same
    #: order.
    iterators: list[TreeIterator]

    #: Size of each of the :py:attr:`iterators`. ``None`` represents infinite
    #: trees.
    sizes: list[int | None]

    #: Last returned positions of the child iterators.
    positions: Sequence[int | DefaultIndex | None]

    #: Access keys for the children iterators, such that
    #: ``iterators[i]`` is an iterator of ``tree.children[keys[i]]``.
    #:
    #: Note that the previous statement is not properly typed, as the current
    #: type hints do not state that ``keys`` contains valid keys for
    #: ``children``. However, the constructor takes care of the validity of the
    #: runtime types. Because of that, ignoring type checks can be required
    #: when implementing concrete iteration methods.
    keys: list[Key | int]

    def __init__(self, tree: T) -> None:
        super().__init__(tree)

        if tree.order is not None:
            self.keys = tree.order
        elif isinstance(tree.children, list):
            self.keys = list(range(len(tree.children)))
        else:
            # An exception will be raised if comparison is not possible
            assert isinstance(tree.children, dict)
            self.keys = sorted(tree.children.keys(), key=str)

        self.iterators = [iter(tree.children[key]) for key in self.keys]  # type: ignore[index]
        self.positions = [it.position for it in self.iterators]

        self.sizes = [tree.children[key].safe_len() for key in self.keys]  # type: ignore[index]

    def reset(self):
        super().reset()
        for iterator in self.iterators:
            iterator.reset()

        self.positions = [it.position for it in self.iterators]

    def reverse(self):
        super().reverse()
        for iterator in self.iterators:
            iterator.reverse()

    @abstractmethod
    def _children_positions(self, position: int) -> Sequence[int | DefaultIndex | None]:
        """
        Only method required to implement an iteration method. It returns the
        index of each of the :py:attr:`iterators` corresponding to a global
        ``position``. It supports the index ``None``, meaning that this
        children value should be missing.
        """
        raise NotImplementedError()

    def _current_value(self) -> DataTree:
        """
        Converts the output list of positions of :py:meth:`_children_positions`
        to the expected list or map of children values.
        """
        new_positions = self._children_positions(self.position)

        if isinstance(self.tree.children, list):
            self.positions = new_positions
            assert all(pos is not None for pos in new_positions)
            return [it[pos] for it, pos in zip(self.iterators, new_positions)]  # type: ignore[index]

        ret = {}
        for i, pos in enumerate(new_positions):
            if self.tree.lazy and self.positions[i] == pos:
                continue

            if pos is not None:
                ret[self.keys[i]] = self.iterators[i][pos]

        self.positions = new_positions
        return ret


@dataclass(frozen=True)
class CartesianProduct(IterationMethod):
    """
    Iteration over the cartesian product of the children. The iteration order is
    the same as :py:func:`itertools.product`. In other words, iteration will
    behave roughly as

    .. code-block:: python

        for v1 in c1:
            for v2 in c2:
                for v3 in c3:
                    yield [v1, v2, v3]

    If an order is specified, its first element will correspond to the outermost
    loop, and its last to the innermost one.
    """

    #: Enable snake iteration, which guarantees that successive yielded elements
    #: differ by only one key at most (a la Gray code).
    snake: bool = False

    def _iter(self) -> TreeIterator:
        return CartesianProductIterator(self)

    def _len(self) -> int:
        children: collections.abc.Sized
        if isinstance(self.children, list):
            children = self.children
        else:
            assert isinstance(self.children, dict)
            children = self.children.values()

        return reduce(int.__mul__, map(len, children), 1)


class CartesianProductIterator(IterationMethodIterator[CartesianProduct]):
    #: Backward cumulated products of :py:attr:`sizes`, with ``size + 1``
    #: elements, and ending with a ``1``. Before, and including, the last
    #: infinite child, it only contains ``None``.
    cumsizes: list[int | None]

    def __init__(self, product: CartesianProduct):
        super().__init__(product)
        n = len(self.tree.children)
        self.cumsizes = [None] * n + [1]

        accumulated_size = 1
        child_size: int | None = -1
        child_position = -2
        for child_position in range(n - 1, -1, -1):
            child_size = self.sizes[child_position]
            if child_size is None:
                break

            accumulated_size *= child_size
            self.cumsizes[child_position] = accumulated_size

        if child_size is None:
            logger.warning(
                "A cartesian product contains an infinite tree at position "
                f"{child_position}. During iteration, its preceding siblings "
                "will always yield their first value."
            )

    def _children_positions(self, position: int) -> Sequence[int | DefaultIndex]:
        positions = [0 for _ in self.iterators]
        for i in range(len(self.iterators) - 1, -1, -1):
            cumsize = self.cumsizes[i]
            previous_cumsize = self.cumsizes[i + 1]
            assert isinstance(previous_cumsize, int)

            if cumsize is None:
                row_pos = position // previous_cumsize
                forward = True
            else:
                row_pos = (position % cumsize) // previous_cumsize
                forward = (position // cumsize) % 2 == 0

            if not forward and self.tree.snake:
                size = self.sizes[i]
                assert isinstance(size, int)
                positions[i] = size - 1 - row_pos
            else:
                positions[i] = row_pos

            if cumsize is None:
                break

        return positions


def _has_no_default(t: IterationTree) -> bool:
    try:
        t.default(no_default_policy=NoDefaultPolicy.ERROR)
        return False
    except NoDefaultError:
        return True


@dataclass(frozen=True)
class Union(IterationMethod):
    """
    Iteration over one child at a time, starting with the first one (or the
    first one of the order, if specified). Children that are not being iterated
    over have

    - their default value if it exists and
    - their first value otherwise.
    """

    #: Defines which value to use before starting the iteration over a child.
    #: "first" uses the child's first value and "default" its default value.
    #: ``None`` does not set the children values before iteration. Thus, it is
    #: only applicable to dict children.
    preset: Literal["first"] | Literal["default"] | None = "first"

    #: You can only set it when ``preset == "first"``. Then, children are all set
    #: to their first value at the first iteration. When iterating over them,
    #: they start directly at their second value.
    common_preset: bool = False

    #: Defines which value to use after ending the iteration over a
    #: child. "first" resets it to its starting value, "last" leaves it
    #: unchanged, and "default" resets it to its default value. ``None`` does not
    #: reset the child after iteration. Thus, it is only applicable to dict
    #: children.
    reset: Literal["first"] | Literal["last"] | Literal["default"] | None = "first"

    def __post_init__(self):
        super().__post_init__()

        if self.preset is None and not isinstance(self.children, dict):
            raise ValueError("Union with preset = None requires dict children.")

        if self.common_preset and self.preset != "first":
            raise ValueError("Union with common_preset requires preset = 'first'.")

        if self.reset is None and not isinstance(self.children, dict):
            raise ValueError("Union with reset = None requires dict children.")

        if self.preset == "default" or self.reset == "default":
            children: list[IterationTree]
            if isinstance(self.children, list):
                children = self.children
            else:
                assert isinstance(self.children, dict)
                children = list(self.children.values())

            if any(_has_no_default(child) for child in children):
                raise ValueError(
                    "Union with preset = 'default' or reset = 'default' "
                    + "requires its children to have default values."
                )

    def _iter(self) -> TreeIterator:
        return UnionIterator(self)

    def _len(self) -> int:
        children: list[IterationTree]
        if isinstance(self.children, list):
            children = self.children
        else:
            assert isinstance(self.children, dict)
            children = list(self.children.values())

        length = sum(map(len, children))
        if self.common_preset:
            length -= len(children) - 1

        return length


class UnionIterator(IterationMethodIterator[Union]):
    #: Cumulated sum of the iterated sizes of the children, containing ``int``
    #: or ``math.inf`` values. After, and including, the first infinite children,
    #: it only contains ``math.inf`` values. Its size is ``len(self.sizes) + 1``,
    #: as it is prefixed with a 0.
    cumsizes: list[int | float]

    #: Value of a child before iteration over it starts
    initial_value: int | DefaultIndex | None

    #: Value of a child after iteration over it ends
    final_value: int | DefaultIndex | None

    def __init__(self, tree: Union) -> None:
        super().__init__(tree)

        n = len(self.tree.children)
        self.cumsizes = [0] + [math.inf] * n
        cumsize = 0

        for i, size in enumerate(self.sizes):
            if size is None:
                self.cumsizes[i + 1] = math.inf
                logger.warning(
                    f"A union contains an infinite tree at position {i}. During "
                    "iteration, its following siblings will always yield the "
                    "same value."
                )
                break

            cumsize += size
            if self.tree.common_preset and i > 0:
                cumsize -= 1

            self.cumsizes[i + 1] = cumsize

        if self.tree.preset == "default":
            self.initial_value = DefaultIndex()
        elif self.tree.preset == "first":
            self.initial_value = 0
        else:
            assert self.tree.preset is None
            self.initial_value = None

        if self.tree.reset == "default":
            self.final_value = DefaultIndex()
        elif self.tree.reset == "first":
            self.final_value = 0
        elif self.tree.reset == "last":
            self.final_value = -1
        else:
            assert self.tree.reset is None
            self.final_value = None

    def _children_positions(self, position: int) -> Sequence[int | DefaultIndex | None]:
        positions: list[int | DefaultIndex | None] = [self.initial_value] * len(
            self.iterators
        )

        for i in range(len(self.iterators)):
            if self.cumsizes[i] <= position < self.cumsizes[i + 1]:
                cumsize = self.cumsizes[i]
                assert isinstance(cumsize, int)

                child_pos = position - cumsize
                if self.tree.common_preset and i > 0:
                    child_pos += 1

                positions[i] = child_pos
            elif self.cumsizes[i + 1] <= position:
                if self.final_value == -1:
                    size = self.sizes[i]
                    # Only finite children can reach cumsizes[i + 1] <= position
                    assert size is not None
                    positions[i] = size - 1
                else:
                    positions[i] = self.final_value

        return positions


@dataclass(frozen=True)
class Zip(IterationMethod):
    """
    Iteration over all of the children of the nodes at the same time, in a way
    similar to :py:func:`zip`.
    """

    #: If ``shortest``, iteration stops whenever the first child is exhausted.
    #: If ``longest``, iteration continues until the last child is exhausted.
    #: Note that ``longest`` is only supported for ``dict`` children.
    stops_at: Literal["longest"] | Literal["shortest"] = "shortest"

    #: Ignore children with a fixed value, *ie* those with length 1. When set,
    #: they do not restrict the node size to 1.
    ignore_fixed: bool = True

    def __post_init__(self):
        super().__post_init__()

        if self.stops_at not in ("longest", "shortest"):
            raise ValueError("Zip only supports longest and shortest stops_at.")

        if self.stops_at == "longest" and isinstance(self.children, list):
            raise ValueError("Zip with stops_at = longest requires dict children.")

    def _iter(self) -> TreeIterator:
        size = self.safe_len()

        if isinstance(self.children, dict):
            return ZipIterator(
                self.with_params(
                    [],
                    children={
                        name: First(child, size)
                        for name, child in self.children.items()
                    },
                )
            )
        else:
            return ZipIterator(
                self.with_params(
                    [], children=[First(child, size) for child in self.children]
                )
            )

    def _len(self) -> int:
        children: list[IterationTree]
        if isinstance(self.children, list):
            children = self.children
        else:
            assert isinstance(self.children, dict)
            children = list(self.children.values())

        if self.stops_at == "longest":
            return max(len(child) for child in children)
        else:
            ignored: list[None | int] = [None]
            if self.ignore_fixed:
                ignored.append(1)

            lengths = [child.safe_len() for child in children]
            valid_lengths: list[int] = [
                length for length in lengths if length not in ignored  # type: ignore[misc]
            ]

            if len(valid_lengths) == 0:
                if lengths[0] is None:
                    raise InfiniteLength from ValueError
                else:
                    assert lengths[0] == 1
                    return 1
            else:
                return min(valid_lengths)


class ZipIterator(IterationMethodIterator[Zip]):
    def _children_positions(self, position: int) -> Sequence[int | DefaultIndex | None]:
        positions: list[int | None] = [position] * len(self.sizes)
        for child_pos, size in enumerate(self.sizes):
            if size is not None and size <= position:
                positions[child_pos] = None

            if self.tree.ignore_fixed and size == 1:
                positions[child_pos] = 0

        return positions


@dataclass(frozen=True)
class Pick(RandomTree, IterationMethod):
    """
    Randomly pick and return one child at a time. For finite trees, it behaves
    in a way similar to composing :py:class:`Shuffle` and ``Union
    (reset=False)`` nodes. It can additionally handle infinite children,
    whereas :py:class:`Shuffle` cannot.
    """

    #: Key of the child to use as a default value.
    default_child: Key | None = None

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.children, list):
            if self.default_child is not None and not isinstance(
                self.default_child, int
            ):
                raise TypeError(
                    "The default_child field of a Pick node with list children "
                    "must be None or an int."
                )
        else:
            if (
                self.default_child is not None
                and self.default_child not in self.children
            ):
                raise KeyError(
                    "The default_child field of a Pick node with dict children "
                    "must be a valid children key."
                )

    def _iter(self) -> TreeIterator:
        return PickIterator(self)

    def _len(self) -> int:
        children: collections.abc.Sized
        if isinstance(self.children, list):
            children = self.children
        else:
            assert isinstance(self.children, dict)
            children = self.children.values()

        return sum(len(child) for child in children)

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree | _NoDefault:
        if self.default_child is not None:
            default_child = self.children[self.default_child]  # type: ignore[index]
            default_value = default_child.default(no_default_policy)
            if isinstance(self.children, list):
                return [default_value]
            else:
                return {self.default_child: default_value}

        raise NoDefaultError("Pick node misses a default_child field.", [])


class PickIterator(OneWayTreeIterator, IterationMethodIterator):
    generator: np.random.Generator
    next_positions: list[int]
    child_factory: Callable[[Key, DataTree], DataTree]

    def __init__(self, tree: Pick) -> None:
        OneWayTreeIterator.__init__(self)
        IterationMethodIterator.__init__(self, tree)

        if tree.seed is None:
            raise ValueError("Cannot iterate over a non-seeded Pick node.")

        seed = list(tree.seed.to_bytes())
        self.generator = np.random.Generator(np.random.PCG64(seed))
        self.next_positions = [0] * len(self.iterators)

        if isinstance(tree.children, dict):

            def child_factory(index: Key, child: DataTree) -> DataTree:
                return {index: child}

            self.child_factory = child_factory
        else:

            def child_factory(index: Key, child: DataTree) -> DataTree:
                return [child]

            self.child_factory = child_factory

    def _next(self) -> DataTree:
        while True:
            picked_child = self.generator.integers(len(self.iterators))
            picked_position = self.next_positions[picked_child]

            try:
                value = self.iterators[picked_child][picked_position]
                self.next_positions[picked_child] += 1
                return self.child_factory(self.keys[picked_child], value)
            except IndexError:
                pass

    def _children_positions(self, position: int) -> Sequence[int | DefaultIndex | None]:
        raise NotImplementedError("_children_positions is not required by PickIterator")


### Unary and transform nodes ###


@dataclass(frozen=True)
class UnaryNode(IterationTree):
    child: IterationTree

    def __post_init__(self):
        object.__setattr__(self, "configurations", self.child.configurations)

    def _len(self) -> int:
        return len(self.child)

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self.child.to_pseudo_data_tree()

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree | _NoDefault:
        error: None | NoDefaultError = None
        try:
            return self.child._default(no_default_policy)
        except NoDefaultError as err:
            own_path: ChildPath = [child]
            error = NoDefaultError(err.args[0], own_path + err.path)

        assert error is not None
        raise error

    # Path API

    def _get(self, child_key: Key | _Child) -> IterationTree:
        if not isinstance(child_key, _Child):
            raise KeyError("Unary node child is only accessible with Child().")

        return self.child

    def _insert_child(
        self, child_key: Key | _Child, child: IterationTree
    ) -> IterationTree:
        if not isinstance(child_key, _Child):
            raise KeyError("Unary node child is only accessible with Child().")

        return dataclasses.replace(self, child=child)

    def _remove_child(self, child_key: Key | _Child) -> IterationTree:
        raise TypeError("Unary node does not support child removal.")

    def _replace_root(
        self, Node: type[IterationTree], *args, **kwargs
    ) -> IterationTree:
        # The signature of ``Node`` is statically unknown
        return Node(self.child, *args, **kwargs)  # type: ignore[call-arg]

    def _depth_first_modify(
        self,
        modifier: Callable[["IterationTree", ChildPath], "IterationTree"],
        path: ChildPath,
    ) -> IterationTree:
        new_child = self.child._depth_first_modify(modifier, path + [child])
        return modifier(dataclasses.replace(self, child=new_child), path)

    # Configurations

    def _get_configuration(self, config_name: Key) -> IterationTree:
        if self.configurations == {}:
            return self

        return self._insert_child(_Child(), self.child._get_configuration(config_name))


@dataclass(frozen=True)
class Shuffle(RandomTree, UnaryNode):
    """
    Unary node that shuffles the order of its child. In other words, it iterates
    over a random permutation of its children values.

    Shuffling has a constant memory cost. The permutation of the order of the
    child tree is obtained through the use of a cypher, which means that it is
    entirely represented by a short key.

    To shuffle a tree with size $n$, a symmetric cypher on $\\{ 0, \\dots ,
    n-1 \\}$ is used. It is built with two components:

    - a block cypher working on $2 \\lceil \\log_2 n / 2 \\rceil$-bit words. It consists
      in a 3-round Feistel network which uses SipHash 1-3 as a round function.
    - Cycle walking is used to restrict the message space to $\\{ 0, \\dots, n -
      1 \\}$. The block cypher is iterated until its output is valid.
    """

    def __post_init__(self):
        super().__post_init__()

        try:
            len(self.child)
        except InfiniteLength as e:
            raise ValueError("Shuffle node requires a finite child.") from e

    def _iter(self) -> TreeIterator:
        return ShuffleIterator(self)


class ShuffleIterator(TreeIterator):
    iterator: TreeIterator

    #: Round keys of the generalized Feistel network. They are used for SipHash
    #: 1-3, so they consist in two 64 bit integers. The size of this list must
    #: be :py:attr:`rounds`.
    keys: list[tuple[np.uint64, np.uint64]]

    #: Number of rounds of the generalized Feistel network.
    rounds: ClassVar[int] = 3

    def __init__(self, tree: Shuffle) -> None:
        super().__init__(tree)

        if tree.seed is None:
            raise ValueError("Cannot iterate over a non-seeded Shuffle node.")

        seed_sequence = np.random.SeedSequence(list(tree.seed.to_bytes()))
        all_keys: Sequence[np.uint64] = seed_sequence.generate_state(
            2 * self.rounds, dtype=np.uint64
        )  # type: ignore[assignment]
        self.keys = [(all_keys[i], all_keys[i + 1]) for i in range(self.rounds)]

        self.iterator = iter(tree.child)

        assert self.size is not None

    def _current_value(self) -> DataTree:
        assert self.size is not None
        return _rust.shuffle(self.position, self.size, self.keys)


@dataclass(frozen=True)
class First(UnaryNode):
    """
    Return only the first elements of its child.
    """

    #: Number of elements to keep. If it is ``None``, or is bigger than the
    #: child size, the same number of elements as the child are iterated over.
    size: int | None

    def __post_init__(self):
        return super().__post_init__()

    def _len(self) -> int:
        if self.size is None:
            return len(self.child)
        else:
            child_length = self.child.safe_len()
            if child_length is None:
                return self.size
            return min(self.size, child_length)

    def _iter(self) -> TreeIterator:
        return FirstIterator(self)


class FirstIterator(TreeIterator[First]):
    iterator: TreeIterator

    def __init__(self, tree: First):
        super().__init__(tree)
        self.iterator = iter(tree.child)

    def _current_value(self) -> DataTree:
        return self.iterator[self.position]


@dataclass(frozen=True)
class Transform(UnaryNode):
    """
    Node that modifies the data trees generated by its child during iteration.

    If you want to transform a ``list`` or ``dict`` of iteration trees, you should
    wrap them in an :py:class:`IterationMethod` object first.
    """

    @abstractmethod
    def transform(self, data_tree: DataTree) -> DataTree:
        """
        Method implemented by concrete sub-classes to modify the data tree
        generated by the :py:attr:`child` tree.
        """
        raise NotImplementedError()

    def _iter(self) -> TreeIterator:
        return TransformIterator(self)

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree | _NoDefault:
        return self.transform(super()._default(no_default_policy))


U = TypeVar("U", bound=Transform, covariant=True)


class TransformIterator(TreeIterator[U]):
    transform_node: U
    child_iterator: TreeIterator[U]

    def __init__(self, tree: U):
        super().__init__(tree)
        self.child_iterator = iter(tree.child)

    def reset(self):
        super().reset()
        self.child_iterator.reset()

    def reverse(self):
        super().reverse()
        self.child_iterator.reverse()

    def _current_value(self) -> DataTree:
        return self.tree.transform(self.child_iterator[self.position])


@dataclass(frozen=True)
class FunctionalTranform(Transform):
    """
    Transform node using its function attribute to modify its child.
    """

    f: Callable[[DataTree], DataTree]

    def transform(self, data_tree: DataTree) -> DataTree:
        return self.f(data_tree)


@dataclass(frozen=True)
class Accumulator(Transform):
    """
    Transform node that accumulates its inputs, as a kind of *unlazifying*
    transform:

    - if its successive inputs are dictionaries, merge them using the union
      operator (recursively or not), and return the results;
    - otherwise, leave its inputs untouched.
    """

    #: Specify if the accumulation must be done recursively or not. For example,
    #: accumulating values ``{"a": 1, "b": {"ba": 1}}`` and ``{"a": 2, "b":
    #: {"bb": 2}}`` recursively will return ``{"a": 2, "b": {ba": 1, "bb": 2}}``,
    #: whereas doing it non-recursively will return ``{"a": 2, "b":
    #: {"bb": 2}}``.
    recursive: bool = False

    #: Start value of the accumulator, which must either be a dictionary, or
    #: ``None``.
    start_value: dict[Key, DataTree] | None = None

    def _iter(self) -> TreeIterator:
        return AccumulatorIterator(self)

    def transform(self, data_tree: DataTree) -> DataTree:
        return data_tree


class AccumulatorIterator(TransformIterator[Accumulator]):
    last_value: dict | None

    def __init__(self, tree: Accumulator):
        super().__init__(tree)
        self.last_value = tree.start_value

    def reset(self):
        super().reset()
        self.last_value = self.tree.start_value

    def _current_value(self) -> DataTree:
        data_tree = super()._current_value()
        if isinstance(data_tree, dict):
            if isinstance(self.last_value, dict):
                if self.tree.recursive:
                    new_tree = utility.recursive_union(self.last_value, data_tree)
                    assert isinstance(new_tree, dict)
                    self.last_value = new_tree
                else:
                    self.last_value |= data_tree
            else:
                self.last_value = data_tree

            return self.last_value.copy()
        else:
            return data_tree


@dataclass(frozen=True)
class Lazify(Transform):
    """
    Transform node that only returns the elements that were updated in its
    children values (for dictionary values), or leaves its inputs untouched
    (for other types).
    """

    def _iter(self) -> TreeIterator:
        return LazifyIterator(self)

    def transform(self, data_tree: DataTree) -> DataTree:
        return data_tree


class LazifyIterator(TransformIterator[Lazify]):
    #: Accumulation of the values yielded by the tree iterator. If it generates
    #: a non-dict value, then stores ``None``.
    accumulated_value: dict[Key, DataTree] | None

    def __init__(self, tree: Lazify):
        super().__init__(tree)
        self.last_value = None
        self.accumulated_value = None

    def reset(self):
        super().reset()
        self.last_value = None
        self.accumulated_value = None

    def _current_value(self) -> DataTree:
        new_value = super()._current_value()

        if isinstance(new_value, dict):
            if not isinstance(self.accumulated_value, dict):
                self.accumulated_value = {}

            new_value_keys = set(new_value.keys())
            accumulated_keys = set(self.accumulated_value.keys())
            new_keys = new_value_keys - accumulated_keys
            common_keys = new_value_keys & accumulated_keys
            updated_keys = {
                key
                for key in common_keys
                if self.accumulated_value[key] != new_value[key]
            }
            updated_values = {key: new_value[key] for key in updated_keys | new_keys}

            self.accumulated_value |= new_value
            return updated_values
        else:
            self.accumulated_value = None
            return new_value


@dataclass(frozen=True)
class MoveUpTransform(Transform):
    insert_name: bool | str = False

    def transform(self, data_tree: DataTree) -> DataTree:
        if not isinstance(data_tree, dict) or len(data_tree) != 1:
            raise ValueError("MoveUpTransform expects a dict with a single child.")

        (key,) = list(data_tree.keys())
        (child,) = list(data_tree.values())

        if self.insert_name is not False:
            if not isinstance(child, dict):
                raise ValueError(
                    "MoveUpTransform with insert_name expects a dict children."
                )

            assert isinstance(child, dict)
            if self.insert_name is True:
                child["_key"] = key
            else:
                assert isinstance(self.insert_name, str)
                child[self.insert_name] = key

        return child


### Configurations ###


@dataclass(frozen=True)
class Configurations(IterationMethod):
    """
    Represents a set of named configurations that can be invoked using
    :py:meth:`IterationTree.get_configuration`. When it is called, with argument
    ``name``, all the :py:class:`Configurations` nodes that have a matching
    configuration will be replaced by it. Other ones will be replaced by their
    default value.

    This allows to escape from the recursive and local nature of trees: a single
    call can modify nodes throughout a whole tree. However, it is often
    convenient to convert configurable trees back to classical,
    non-configurable trees. This is done by
    :py:meth:`IterationTree.unroll_configurations`.

    The :py:class:`Configurations` node holds a set of iteration trees,
    called *configurations*, which are identified by their *name*. Two
    parameters, :py:attr:`move_up` and :py:attr:`insert_name`, control how
    :py:meth:`IterationTree.get_configuration` behaves.

    By default, ``move_up == insert_name == False``. The
    :py:class:`Configuration` node is simply replaced by the content of the
    requested configuration.

    >>> tree = CartesianProduct({
    ...     "instrument": Configurations({
    ...         "config1": CartesianProduct({
    ...             "param1": IterationLiteral(value="1-1"),
    ...             "param2": IterationLiteral(value="1-2"),
    ...         }),
    ...         "config2": CartesianProduct({
    ...             "param1": IterationLiteral(value="2-1"),
    ...             "param3": IterationLiteral(value="2-3"),
    ...         })
    ...     })
    ... })
    >>> tree.get_configuration("config1").to_pseudo_data_tree()
    {'instrument': {'param1': '1-1', 'param2': '1-2'}}

    If ``move_up == True``, the content of the chosen configuration is moved one
    level up, so that it is at the same level as the :py:class:`Configurations`
    node. This requires it to have an :py:class:`IterationMethod` parent with
    ``dict`` children. This can be used to factorize configurations.

    >>> tree = CartesianProduct({
    ...     "_": Configurations({
    ...         "config1": CartesianProduct({
    ...             "param1": IterationLiteral(value="1-1"),
    ...         }),
    ...         "config2": CartesianProduct({
    ...             "param1": IterationLiteral(value="2-1"),
    ...         })
    ...     }, move_up=True),
    ...     "param2": IterationLiteral(value="2")
    ... })
    >>> tree.get_configuration("config1").to_pseudo_data_tree()
    {'param1': '1-1', 'param2': '2'}

    If ``insert_name == True``, the name of the requested configuration is
    inserted in the resulting tree. If the requested configuration allows it
    (_ie_. it is an :py:class:`IterationMethod` with ``dict`` children), the
    name is inserted into itself, with the key ``_configuration``.

    >>> tree = CartesianProduct({
    ...     "instrument": Configurations({
    ...         "config1": CartesianProduct({
    ...             "param1": IterationLiteral(value="1-1"),
    ...             "param2": IterationLiteral(value="1-2"),
    ...         }),
    ...         "config2": CartesianProduct({
    ...             "param1": IterationLiteral(value="2-1"),
    ...             "param3": IterationLiteral(value="2-3"),
    ...         })
    ...     }, insert_name=True)
    ... })
    >>> tree.get_configuration("config1").to_pseudo_data_tree()
    {'instrument': {'_configuration': 'config1', 'param1': '1-1', 'param2': '1-2'}}

    If, additionally, ``move_up == True``, the name of the configuration is
    inserted instead of the :py:class:`Configurations` node.

    >>> tree = CartesianProduct({
    ...     "instrument": Configurations({
    ...         "config1": CartesianProduct({
    ...             "param1": IterationLiteral(value="1-1"),
    ...             "param2": IterationLiteral(value="1-2"),
    ...         }),
    ...         "config2": CartesianProduct({
    ...             "param1": IterationLiteral(value="2-1"),
    ...             "param3": IterationLiteral(value="2-3"),
    ...         })
    ...     }, insert_name=True, move_up=True)
    ... })
    >>> tree.get_configuration("config1").to_pseudo_data_tree()
    {'instrument': 'config1', 'param1': '1-1', 'param2': '2'}

    However, if the requested configuration is not an
    :py:class:`IterationMethod`, its ``name`` is inserted as a sibling, assigned
    to the key ``f"_{name}_configuration"``.

    >>> tree = CartesianProduct({
    ...     "param": Configurations({
    ...         "config1": IterationLiteral(value="1"),
    ...         "config2": IterationLiteral(value="2"),
    ...     }, insert_name=True),
    ... })
    >>> tree.get_configuration("config1").to_pseudo_data_tree()
    {'_param_configuration': 'config1', 'param': '1'}

    This means that the content of the configurations affects how they are
    handled. Although this might change in the future, it is made to support
    most situations. It is recommended to keep all the configurations with the
    same shape.

    :py:attr:`insert_name` is not necessary. It is possible to replace it with
    the following kind of tree:

    >>> tree = CartesianProduct({
    ...     "param": Configurations({
    ...         "config1": IterationLiteral(value="1"),
    ...         "config2": IterationLiteral(value="2"),
    ...     },
    ...     "config": Configurations({
    ...         "config1": IterationLiteral(value="config1"),
    ...         "config2": IterationLiteral(value="config2"),
    ...     })),
    ... })
    >>> tree.get_configuration("config1").to_pseudo_data_tree()
    {'param': '1', 'config': 'config1'}
    """

    children: dict[Key, IterationTree]

    #: Key of the default configuration, which must be in the set of keys of
    #: children.
    default_configuration: Key | None = None

    #: If set, the content of a requested configuration is moved up, at the
    #: parent level of the current node. In other words, it becomes a sibling
    #: of the current :py:class:`Configurations` node.
    #:
    #: Otherwise, the content is inserted at the same level as the
    #: configurations themselves.
    move_up: bool = False

    #: If set, insert the name of the requested configuration when calling
    #: :py:meth:`~phileas.iteration.IterationTree.get_configuration`.
    #: If :py:attr:`move_up`, then the :py:class:`Configurations` node is
    #: replaced by this name. Otherwise, a ``"_configuration"`` sibling node
    #: is inserted.
    insert_name: bool = False

    def __post_init__(self):
        super().__post_init__()

        object.__setattr__(
            self, "configurations", self.configurations.union(self.children.keys())
        )

        if (
            self.default_configuration is not None
            and self.default_configuration not in self.children
        ):
            raise KeyError("Default configuration not in the children configurations.")

    def _get_configuration(self, config_name: Key) -> IterationTree:
        try:
            return self.children[config_name]._get_configuration(config_name)
        except KeyError as e:
            if self.default_configuration is not None:
                return self.children[self.default_configuration]._get_configuration(
                    config_name
                )
            else:
                raise KeyError(
                    f"Missing configuration {config_name}, with no default "
                    "configuration."
                ) from e

    def _iter(self) -> TreeIterator:
        raise AssertionError(
            f"{self.__class__.__name__} iteration is handled by "
            "IterationTree.__iter__."
        )

    def _len(self) -> int:
        raise AssertionError(
            f"{self.__class__.__name__} length is handled by IterationTree.__len__."
        )

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree:
        if self.default_configuration is None:
            raise NoDefaultError.build_from(self)

        return self.children[self.default_configuration].default(no_default_policy)

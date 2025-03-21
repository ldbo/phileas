"""
This module defines abstract and concrete iteration tree nodes, which are
iteration methods and transform nodes, as well as their iterators.
"""

import collections.abc
import dataclasses
import math
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
from itertools import accumulate, chain
from operator import mul
from typing import Callable, Sequence, TypeVar

from phileas.iteration import utility
from phileas.logging import logger

from .base import (
    ChildPath,
    DataTree,
    DefaultIndex,
    IterationTree,
    Key,
    NoDefaultError,
    NoDefaultPolicy,
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
    `IterationMethod` and implement a corresponding `IterationMethodIterator`,
    which is returned by `_iter()`.

    This should remain the only node in an iteration tree that can hold `dict`
    and `list` children. If you are tempted to create another node doing so,
    you should verify that it cannot be done by sub-classing `IterationMethod`
    instead.
    """

    #: The children of the node. It must not be empty.
    children: list[IterationTree] | dict[Key, IterationTree]

    #: Order of iteration over the children. How it is used depends on the
    #: concrete iteration method implementation. It must be a permutation of the
    #: set of keys of `children`.
    order: list[Key] | None = None

    #: Notify the iteration method to be lazy. For now, this feature is only
    #: supported for `dict` children. In this case, lazy iteration will just
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

    def _get_configuration(self, config_name: Key) -> "IterationTree":
        if config_name not in self.configurations:
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
        configs: "Configurations",
    ):
        """
        Given a dict containing the currently processed `children` of the tree,
        update it with configuration `requested_config_name` from the child
        `configs`, identified by `configs_key`, of the current node.
        """
        from .leaf import IterationLiteral

        requested_config = configs._get_configuration(requested_config_name)
        is_transformed_leaf = utility.is_transformed_iteration_leaf(requested_config)
        if configs.move_up:
            if not isinstance(requested_config, IterationMethod):
                raise ValueError(
                    "Configurations with move_up must be iteration methods."
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
        else:  # isinstance(self.children, dict)
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

    def __getitem__(self, Key) -> IterationTree:
        return self.children[Key]

    # Path API

    # Note: self.children[child_key] is not properly typed. Indeed, child_key
    # can be a dict or a list key, and there is no guarantee that self.children
    # is of the corresponding type. However, if the key does not exist, a
    # `KeyError` will be raised, which is the expected behavior. For those
    # reasons, it is valid to ignore[index] all the self.children
    # [child_key] expressions.

    def _get(self, child_key: Key | _Child) -> IterationTree:
        if isinstance(child_key, _Child):
            raise KeyError("Iteration method does not support Child() index.")

        return deepcopy(self.children[child_key])  # type: ignore[index]

    def _insert_child(
        self, child_key: Key | _Child, child: IterationTree
    ) -> IterationTree:
        if isinstance(child_key, _Child):
            raise KeyError("Iteration method does not support Child() index.")

        new_children = deepcopy(self.children)
        new_children[child_key] = child  # type: ignore[index]
        return dataclasses.replace(self, children=new_children)

    def _remove_child(self, child_key: Key | _Child) -> IterationTree:
        new_children = deepcopy(self.children)
        _ = new_children.pop(child_key)  # type: ignore[arg-type]
        return dataclasses.replace(self, children=new_children)

    def _replace_root(
        self, Node: type[IterationTree], *args, **kwargs
    ) -> IterationTree:
        if not issubclass(Node, IterationMethod):
            raise TypeError(f"Cannot replace an iteration method with a {Node}")

        return Node(deepcopy(self.children), *args, **kwargs)

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

        return modifier(
            dataclasses.replace(self, children=deepcopy(new_children)), path
        )


T = TypeVar("T", bound=IterationMethod, covariant=True)


class IterationMethodIterator(TreeIterator[T]):
    """
    Base class used to implement concrete `IterationMethod` nodes iterators, and
    providing helper attributes to do so.
    """

    #: Children iterators stored in a list. This, with `keys`, allows
    #: child-class to only implement iteration over lists.
    #:
    #: If the iteration tree does specify an order, use it. Otherwise,
    #: dictionaries are sorted by their key value, and lists keep the same
    #: order.
    iterators: list[TreeIterator]

    #: Size of each of the `iterators`. `None` represents infinite trees.
    sizes: list[int | None]

    #: Last returned positions of the child iterators.
    positions: Sequence[int | DefaultIndex | None]

    #: Access keys for the children iterators, such that
    #: `iterators[i]` is an iterator of `tree.children[keys[i]]`.
    #:
    #: Note that the previous statement is not properly typed, as the current
    #: type hints do not state that `keys` contains valid keys for `children`.
    #: However, the constructor takes care of the validity of the runtime
    #: types. Because of that, ignoring type checks can be required when
    #: implementing concrete iteration methods.
    keys: list[Key | int]

    def __init__(self, tree: T) -> None:
        super().__init__(tree)

        if tree.order is not None:
            self.keys = tree.order
        elif isinstance(tree.children, list):
            self.keys = list(range(len(tree.children)))
        else:  # isinstance(tree.children, dict)
            # An exception will be raised if comparison is not possible
            assert isinstance(tree.children, dict)
            self.keys = sorted(tree.children.keys())

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
        index of each of the `iterators` corresponding to a global `position`.
        It supports the index `None`, meaning that this children value should
        be missing.
        """
        raise NotImplementedError()

    def _current_value(self) -> DataTree:
        """
        Converts the output list of positions of `_children_positions()` to the
        expected list or map of children values.
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
    the same as `itertools.product`. In other words, iteration will behave
    roughly as

    ```py
    for v1 in c1:
        for v2 in c2:
            for v3 in c3:
                yield [v1, v2, v3]
    ```

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
        else:  # isinstance(self.chidren, dict)
            children = self.children.values()

        return reduce(int.__mul__, map(len, children), 1)


class CartesianProductIterator(IterationMethodIterator[CartesianProduct]):
    #: Backward cumulated products of the `sizes` after the last infinite one.
    #: It has an additional 1 at the end.
    #:
    #: TBR
    #: Backward cumulated products of `sizes`, with `size` + 1 elements, and
    #: ending at 1.
    cumsizes: list[int]

    last_infinite_index: int

    def __init__(self, product: CartesianProduct):
        super().__init__(product)

        self.last_infinite_index = (
            len(self.sizes)
            - next(i for i, s in enumerate(self.sizes[::-1] + [None]) if s is None)
            - 1
        )

        if self.last_infinite_index > 0:
            logger.warning(
                "A cartesian product contains an infinite tree at position "
                f"{self.last_infinite_index}. During iteration, its preceding "
                "siblings will always yield their first value."
            )

        last_finite_sizes = self.sizes[self.last_infinite_index + 1 :]
        self.cumsizes = list(accumulate(reversed(last_finite_sizes), mul, initial=1))
        self.cumsizes.reverse()

    def _children_positions(self, position: int) -> Sequence[int | DefaultIndex]:
        # -2 indicates an invalid value
        positions = [-2 for _ in self.iterators]

        for i in range(len(self.iterators) - 1, self.last_infinite_index, -1):
            j = i - (self.last_infinite_index + 1)
            row_pos = (position % self.cumsizes[j]) // self.cumsizes[j + 1]
            forward = (position // self.cumsizes[j]) % 2 == 0

            if not forward and self.tree.snake:
                current_size = self.sizes[i]
                assert current_size is not None
                positions[i] = current_size - 1 - row_pos
            else:
                positions[i] = row_pos

        if self.last_infinite_index > -1:
            positions[self.last_infinite_index] = position // self.cumsizes[0]
            positions[: self.last_infinite_index] = [0] * self.last_infinite_index

        assert all(pos != -2 for pos in positions)
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

    def _iter(self) -> TreeIterator:
        return UnionIterator(self)

    def _len(self) -> int:
        children: list[IterationTree]
        if isinstance(self.children, list):
            children = self.children
        else:  # isinstance(self.children, dict)
            children = list(self.children.values())

        no_default = [_has_no_default(child) for child in children]
        children_without_default = sum(no_default)

        return (
            sum(map(len, children))
            - children_without_default
            + int(children_without_default > 0)
        )


class UnionIterator(IterationMethodIterator[Union]):
    #: Cumulated sum of the iterated sizes of the children, containing `int`s or
    #: `math.inf` values. After, and including, the first infinite children, it
    #: only contains `math.inf` values. Its size is `len(self.sizes) + 1`.
    cumsizes: list[int | float]

    #: Indicates whether a child has a default value or not.
    no_default: list[bool]

    #: Index of the first children without a default value.
    first_without_default: int

    def __init__(self, tree: Union) -> None:
        super().__init__(tree)

        n = len(self.tree.children)
        self.no_default = [False] * n
        self.first_without_default = -2
        self.cumsizes = [0] + [math.inf] * n
        cumsize = 0

        for i, (key, size) in enumerate(zip(self.keys, self.sizes)):
            no_default = _has_no_default(self.tree.children[key])  # type: ignore[index]
            self.no_default[i] = no_default
            if no_default and self.first_without_default == -2:
                self.first_without_default = i

            if size is None:
                self.cumsizes[i + 1] = math.inf

                logger.warning(
                    f"A union contains an infinite tree at position {i}. During "
                    "iteration, its following siblings will always yield the "
                    "same value."
                )

                break

            cumsize += size - int(no_default) + int(self.first_without_default == i)
            self.cumsizes[i + 1] = cumsize

    def _children_positions(self, position: int) -> Sequence[int | DefaultIndex | None]:
        positions: list[int | DefaultIndex] = [0] * len(self.iterators)
        pos = position
        for i in range(len(self.iterators)):
            if self.cumsizes[i] <= pos < self.cumsizes[i + 1]:
                cumsize = self.cumsizes[i]
                assert isinstance(cumsize, int)
                positions[i] = (
                    pos
                    - cumsize
                    + int(self.no_default[i])
                    - int(self.first_without_default == i)
                )
            else:
                if self.no_default[i]:
                    positions[i] = 0
                else:
                    positions[i] = DefaultIndex()

        return positions


### Transform nodes ###


@dataclass(frozen=True)
class Transform(IterationTree):
    """
    Node that modifies the data trees generated by its child during iteration.

    If you want to transform a `list` or `dict` of iteration trees, you should
    wrap them in an `IterationMethod` object first.
    """

    child: IterationTree

    def __post_init__(self):
        object.__setattr__(self, "_configurations", self.child.configurations)

    @abstractmethod
    def transform(self, data_tree: DataTree) -> DataTree:
        """
        Method implemented by concrete sub-classes to modify the data tree
        generated by the `child` tree.
        """
        raise NotImplementedError()

    def _iter(self) -> TreeIterator:
        return TransformIterator(self)

    def _len(self) -> int:
        return len(self.child)

    def to_pseudo_data_tree(self) -> PseudoDataTree:
        return self.child.to_pseudo_data_tree()

    def _default(self, no_default_policy: NoDefaultPolicy) -> DataTree | _NoDefault:
        error: None | NoDefaultError = None
        try:
            # This is the only statement that can raise an exception.
            child_default = self.child.default(no_default_policy)

            # We keep those lines here so that `child_default` is explicitly
            # bound.
            if child_default == no_default:
                return no_default

            return self.transform(child_default)
        except NoDefaultError as err:
            own_path: ChildPath = [child]
            error = NoDefaultError(err.args[0], own_path + err.path)

        assert error is not None
        raise error

    def __getitem__(self, key: Key) -> IterationTree:
        return self.child[key]

    # Path API

    def _get(self, child_key: Key | _Child) -> IterationTree:
        if not isinstance(child_key, _Child):
            raise KeyError("Transform node child is only accessible with Child().")

        return self.child

    def _insert_child(
        self, child_key: Key | _Child, child: IterationTree
    ) -> IterationTree:
        if not isinstance(child_key, _Child):
            raise KeyError("Transform node child is only accessible with Child().")

        return dataclasses.replace(self, child=child)

    def _remove_child(self, child_key: Key | _Child) -> IterationTree:
        raise TypeError("Transform node does not support child removal.")

    def _replace_root(
        self, Node: type[IterationTree], *args, **kwargs
    ) -> IterationTree:
        # The signature of `Node` is statically unknown
        return Node(self.child, *args, **kwargs)  # type: ignore[call-arg]

    def _depth_first_modify(
        self,
        modifier: Callable[["IterationTree", ChildPath], "IterationTree"],
        path: ChildPath,
    ) -> "IterationTree":
        new_child = self.child._depth_first_modify(modifier, path + [child])
        return modifier(dataclasses.replace(self, child=new_child), path)

    # Configurations

    def _get_configuration(self, config_name: Key) -> IterationTree:
        if config_name not in self.configurations:
            return self

        return self._insert_child(_Child(), self.child._get_configuration(config_name))


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
    #: accumulating values `{"a": 1, "b": {"ba": 1}}` and `{"a": 2, "b":
    #: {"bb": 2}}` recursively will return `{"a": 2, "b": {ba": 1, "bb": 2}}`,
    #: whereas doing it non-recursively will return ``{"a": 2, "b":
    #: {"bb": 2}}`.
    recursive: bool = False

    #: Start value of the accumulator, which must either be a dictionary, or
    #: `None`.
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
    #: a non-dict value, then stores `None`.
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
class Configurations(IterationMethod):
    """
    It represents a set of configurations that can be invoked using
    `IterationTree.get_configuration`. This allows to escape from the recursive
    and local nature of trees: requesting a configuration on the root of the
    tree returns a subset of it, which can change its global topology.

    It holds a set of mapping of iteration trees, called *configurations*, which
    are identified by their *name*. By default, the configurations are expected
    to be thought of as dictionaries - that is, they are either literal
    dictionaries, or iteration methods whose children are stored in a
    dictionary -, and the `Configurations` node is supposed to be the child of
    an iteration method. Calling `get_configuration` inserts the content of
    the dictionary as siblings of the `Configurations` node. Additionally, the
    `Configurations` node is replaced by the name of the requested
    configuration.

    >>> tree = CartesianProduct({
    ...     "instrument": CartesianProduct({
    ...         "config_name": Configurations({
    ...             "config1": CartesianProduct({
    ...                 "param1": IterationLiteral(value="1-1"),
    ...                 "param2": IterationLiteral(value="1-2"),
    ...             }),
    ...             "config2": CartesianProduct({
    ...                 "param1": IterationLiteral(value="2-1"),
    ...                 "param3": IterationLiteral(value="2-3"),
    ...             })
    ...         })
    ...     })
    ... })
    >>> tree.get_configuration("config1").to_pseudo_data_tree()
    {'instrument': {'config_name': 'config1', 'param1': '1-1', 'param2': '1-2'}}

    Setting `move_up` to `False` changes this behaviour, so that the content of
    the requested configuration is inserted in place of the `Configurations`
    node. Additionally, `insert_name` allows to control whether the name of the
    requested configuration is kept. If set, with `move_up == False`, the name
    is inserted in the `_configuration` key.

    >>> tree = CartesianProduct({
    ...     "instrument": CartesianProduct({
    ...         "param_set": Configurations({
    ...             "config1": CartesianProduct({
    ...                 "param1": IterationLiteral(value="1-1"),
    ...                 "param2": IterationLiteral(value="1-2"),
    ...             }),
    ...             "config2": CartesianProduct({
    ...                 "param1": IterationLiteral(value="2-1"),
    ...                 "param3": IterationLiteral(value="2-3"),
    ...             })
    ...         },
    ...         move_up=False)
    ...     })
    ... })
    >>> tree.get_configuration("config1").to_pseudo_data_tree()
    {'instrument':
        {'param_set':
            {'_configuration': 'config1',
             'param1': '1-1',
             'param2': '1-2'
            }
        }
    }

    """

    children: dict[Key, IterationTree]

    #: Key of the default configuration, which must be in the set of keys of
    #: children.
    default_configuration: Key | None = None

    #: If set, the content of a requested configuration is moved up, at the
    #: parent level of the current node. In other words, it becomes a sibling
    #: of the current `Configurations` node.
    #:
    #:  Otherwise, the content is inserted at the same level as the
    #:  configurations themselves.
    move_up: bool = True

    #: If set, insert the name of the requested configuration when calling
    #: `get_configuration`. If `move_up`, then the `Configurations` node is
    #: replaced by this name. Otherwise, a `"_configuration"` sibling node is
    #: inserted.
    insert_name: bool = True

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
            return self.children[config_name]
        except KeyError as e:
            if self.default_configuration is not None:
                return self.children[self.default_configuration]
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

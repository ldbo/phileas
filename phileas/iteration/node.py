"""
This module defines abstract and concrete iteration tree nodes, which are
iteration methods and transform nodes, as well as their iterators.
"""

import collections.abc
import dataclasses
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from itertools import accumulate
from operator import mul
from typing import Callable, Sequence, TypeVar

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


@dataclass(frozen=True)
class IterationMethod(IterationTree):
    """
    Iteration node having multiple children, supplied either as a list or
    dictionary.

    In order to implement a concrete iteration method, you should sub-class
    `IterationMethod` and implement a corresponding `IterationMethodIterator`,
    which is returned by `__iter__()`.

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

        if self.lazy and not isinstance(self.children, dict):
            raise TypeError("Lazy iteration is only supported for dictionary children.")

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

        return self.children[child_key]  # type: ignore[index]

    def _insert_child(
        self, child_key: Key | _Child, child: IterationTree
    ) -> IterationTree:
        if isinstance(child_key, _Child):
            raise KeyError("Iteration method does not support Child() index.")

        self.children[child_key] = child  # type: ignore[index]
        return self

    def _remove_child(self, child_key: Key | _Child) -> IterationTree:
        _ = self.children.pop(child_key)  # type: ignore[arg-type]
        return self

    def _replace_root(
        self, Node: type[IterationTree], *args, **kwargs
    ) -> IterationTree:
        if not issubclass(Node, IterationMethod):
            raise TypeError(f"Cannot replace an iteration method with a {Node}")

        return Node(self.children, *args, **kwargs)

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

    #: Size of each of the `iterators`.
    sizes: list[int]

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

        try:
            self.sizes = [len(tree.children[key]) for key in self.keys]  # type: ignore[index]
        except TypeError as error:
            raise Exception(
                "Iteration over iteration methods with infinite children is not"
                " supported."
            ) from error

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

    def __iter__(self) -> TreeIterator:
        return CartesianProductIterator(self)

    def __len__(self) -> int:
        children: collections.abc.Sized
        if isinstance(self.children, list):
            children = self.children
        else:  # isinstance(self.chidren, dict)
            children = self.children.values()

        return reduce(int.__mul__, map(len, children), 1)


class CartesianProductIterator(IterationMethodIterator[CartesianProduct]):
    #: Reversed cumulated products of `sizes`, with `size` + 1 elements, and
    #: ending at 1.
    cumsizes: list[int]

    def __init__(self, product: CartesianProduct):
        super().__init__(product)

        self.cumsizes = list(accumulate(reversed(self.sizes), mul, initial=1))
        self.cumsizes.reverse()

    def _children_positions(self, position: int) -> Sequence[int | DefaultIndex]:
        positions = [-2 for _ in self.iterators]
        for i in range(len(self.iterators) - 1, -1, -1):
            row_pos = (position % self.cumsizes[i]) // self.cumsizes[i + 1]
            forward = (position // self.cumsizes[i]) % 2 == 0

            if not forward and self.tree.snake:
                positions[i] = self.sizes[i] - 1 - row_pos
            else:
                positions[i] = row_pos

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

    def __iter__(self) -> TreeIterator:
        return UnionIterator(self)

    def __len__(self) -> int:
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
    #: Cumulated sum of the iterated sizes of the children.
    cumsizes: list[int]

    #: Indicates whether a child has a default value or not.
    no_default: list[bool]

    #: Index of the first children without a default value.
    first_without_default: int

    def __init__(self, tree: Union) -> None:
        super().__init__(tree)

        try:
            n = len(self.tree.children)
            self.no_default = [False] * n
            self.first_without_default = -2
            self.cumsizes = [0] * (n + 1)
            cumsize = 0

            for i, (key, size) in enumerate(zip(self.keys, self.sizes)):
                no_default = _has_no_default(self.tree.children[key])  # type: ignore[index]
                self.no_default[i] = no_default
                if no_default and self.first_without_default == -2:
                    self.first_without_default = i

                cumsize += size - int(no_default) + int(self.first_without_default == i)
                self.cumsizes[i + 1] = cumsize
        except TypeError as error:
            raise Exception("Union is not supported with infinite children.") from error

    def _children_positions(self, position: int) -> Sequence[int | DefaultIndex | None]:
        positions: list[int | DefaultIndex] = [0] * len(self.iterators)
        pos = position
        for i in range(len(self.iterators)):
            if self.cumsizes[i] <= pos < self.cumsizes[i + 1]:
                positions[i] = (
                    pos
                    - self.cumsizes[i]
                    + int(self.no_default[i])
                    - int(self.first_without_default == i)
                )
            else:
                if self.no_default[i]:
                    positions[i] = 0
                else:
                    positions[i] = DefaultIndex()

        return positions


@dataclass(frozen=True)
class Transform(IterationTree):
    """
    Node that modifies the data trees generated by its child during iteration.

    If you want to transform a `list` or `dict` of iteration trees, you should
    wrap them in an `IterationMethod` object first.
    """

    child: IterationTree

    @abstractmethod
    def transform(self, data_tree: DataTree) -> DataTree:
        """
        Method implemented by concrete sub-classes to modify the data tree
        generated by the `child` tree.
        """
        raise NotImplementedError()

    def __iter__(self) -> TreeIterator:
        return TransformIterator(self)

    def __len__(self) -> int:
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
        operator, and return the results;
      - else, leave its inputs untouched.
    """

    #: Start value of the accumulator, which must either be a dictionary, or
    #: `None`.
    start_value: dict[Key, DataTree] | None = None

    def __iter__(self) -> TreeIterator:
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

    def __iter__(self) -> TreeIterator:
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
        print(f"{self.accumulated_value=}")

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

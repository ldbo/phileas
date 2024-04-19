"""
This module defines abstract and concrete iteration tree nodes, which are
iteration methods and transform nodes, as well as their iterators.
"""

import collections.abc
import dataclasses
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable

from .base import (
    ChildPath,
    DataTree,
    IterationTree,
    Key,
    ListIterator,
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

    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        # List children
        if isinstance(self.children, list):
            if no_default_policy == NoDefaultPolicy.SKIP:
                msg = "NoDefaultPolicy.SKIP is not supported by iteration "
                msg += "methods with list children."
                raise TypeError(msg)

            default_list: list[DataTree | _NoDefault] = [None] * len(self.children)
            for position, child_tree in enumerate(self.children):
                error: None | NoDefaultError = None
                try:
                    default_list[position] = child_tree.default(no_default_policy)
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


class IterationMethodIterator(TreeIterator):
    """
    Base class used to implement concrete `IterationMethod` nodes iterators, and
    providing helper attributes to do so.
    """

    tree: IterationMethod

    #: Children iterators stored in a list. This, with `keys`, allows
    #: child-class to only implement iteration over lists.
    #:
    #: If the iteration tree does specify an order, use it. Otherwise,
    #: dictionaries are sorted by their key value, and lists keep the same
    #: order.
    iterators: list[TreeIterator]

    #: Access keys for the children iterators, such that
    #: `iterators[i]` is an iterator of `tree.children[keys[i]]`.
    #:
    #: Note that the previous statement is not properly typed, as the current
    #: type hints do not state that `keys` contains valid keys for `children`.
    #: However, the constructor takes care of the validity of the runtime
    #: types. Because of that, ignoring type checks can be required when
    #: implementing concrete iteration methods.
    keys: list[Key | int]

    def __init__(self, tree: IterationMethod) -> None:
        super().__init__()
        self.tree = tree

        if tree.order is not None:
            self.keys = tree.order
        elif isinstance(tree.children, list):
            self.keys = list(range(len(tree.children)))
        else:  # isinstance(tree.children, dict)
            # An exception will be raised if comparison is not possible
            self.keys = sorted(tree.children.keys())  # type: ignore[type-var]

        self.iterators = [iter(tree.children[key]) for key in self.keys]  # type: ignore[index]

    def reset(self):
        for iterator in self.iterators:
            iterator.reset()

    def reverse(self):
        super().reverse()
        for iterator in self.iterators:
            iterator.reverse()


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


class CartesianProductIterator(IterationMethodIterator):
    base: list | dict

    #: Indicates if the iterator is exhausted, in the forward and backward
    #: directions.
    done: tuple[bool, bool]

    #: Contains, at `position`, whether the tree children at the position
    #: `key[position]` is 1 element long.
    trivial_length_child: list[bool]

    def __init__(self, product: CartesianProduct) -> None:
        super().__init__(product)
        self.trivial_length_child = [
            len(self.tree.children[key]) == 1 for key in self.keys  # type: ignore[index]
        ]
        self.reset()

    def __base(self):
        if isinstance(self.tree.children, list):
            return [next(iterator) for iterator in self.iterators]
        else:
            return {
                key: next(iterator) for key, iterator in zip(self.keys, self.iterators)
            }

    def reset(self):
        super().reset()
        self.done = (False, True)

    def reverse(self):
        super().reverse()
        self.done = (self.done[1], self.done[0])

    def __next__(self) -> DataTree:
        """
        Lazy iteration is supported, and works except for the case where a child
        has only one element.
        """
        # Last iteration in the current direction
        if self.done[0]:
            raise StopIteration

        # First iteration in the current direction
        if self.done[1]:
            self.base = self.__base()
            self.done = (False, False)
            return self.base.copy()

        if self.tree.lazy:
            self.base = {}

        # First, find all the children iterators that are exhausted.
        position = len(self.iterators) - 1
        while position >= 0:
            iterator = self.iterators[position]
            try:
                self.base[self.keys[position]] = next(iterator)  # type: ignore[index]
                break
            except StopIteration:
                position -= 1

        # At this point, all the children at positions > `position` are
        # exhausted in their current direction.

        # Then,
        #   - register that the iterator is exhausted if all of them are
        #     exhausted;
        #   - or reset/reverse them, and yield the current value.
        if position == -1:
            self.done = (True, False)
            raise StopIteration
        else:
            position += 1
            while position < len(self.iterators):
                iterator = self.iterators[position]

                assert isinstance(self.tree, CartesianProduct)
                if self.tree.snake:
                    iterator.reverse()
                else:
                    iterator.reset()

                if self.tree.snake and self.tree.lazy:
                    # Just consume the former last, now first, value, as it does
                    # not require to be updated
                    next(iterator)
                elif self.tree.lazy and self.trivial_length_child[position]:
                    # This "new" value is the same as before, as the child has
                    # only one element. Just consume it.
                    next(iterator)
                else:
                    self.base[self.keys[position]] = next(iterator)  # type: ignore[index]

                position += 1

            self.done = (False, False)

            return self.base.copy()


@dataclass(frozen=True)
class Union(IterationMethod):
    """
    Iteration over one child at a time, starting with the first one (or the
    first one of the order, if specified).

    For children that have a default value, they will be reset to it when they
    are not being iterated. However, there will be no complain about children
    not implementing a default value.
    """

    #: Policy used while iterating, to generate the values returned for the
    #: children that are not being iterated over.
    no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR

    def __iter__(self) -> TreeIterator:
        return UnionIterator(self)

    def __len__(self) -> int:
        children: collections.abc.Sized
        if isinstance(self.children, list):
            children = self.children
        else:  # isinstance(self.children, dict)
            children = self.children.values()
        return sum(map(len, children))


@dataclass
class UnionIterator(IterationMethodIterator):
    #: Iterator over the children indices, whose values are stored in
    #: `current_index`, and used to know which child is currently iterated
    #: over.
    indices: ListIterator = field(init=False)

    #: Special value -1 just after reset. Even for lazy iteration, returns all
    #: the values.
    current_index: int = field(init=False)

    #: Indicates that, at the next iteration, a new `current_index` will be
    #: drawn.
    change_index: bool = field(init=False)

    #: Cached children default values.
    default_values: list | dict = field(init=False)

    def __init__(self, union: Union) -> None:
        super().__init__(union)

        self.indices = ListIterator(list(range(len(self.iterators))))
        self.reset()

        self.yield_all_children = True
        self.default_values = self.__default_values()

    def __default_values(self) -> dict | list:
        assert isinstance(self.tree, Union)
        default_values = self.tree.default(self.tree.no_default_policy)
        assert isinstance(default_values, (dict, list))

        return default_values

    def reset(self):
        super().reset()
        self.indices.reset()
        self.current_index = -1
        self.change_index = True

    def reverse(self):
        super().reverse()
        self.indices.reverse()

    def __next__(self) -> DataTree:
        return self.__next()

    def __next(self, children: None | list | dict = None) -> DataTree:
        # First iteration after reset
        if self.current_index == -1:
            assert self.change_index
            children = self.__default_values()

        # First iteration with a new index
        if self.change_index:
            index = next(self.indices)
            assert isinstance(index, int)
            self.current_index = index
            self.change_index = False

        # Children is not None just after a child iterator has been exhausted.
        if children is None:
            if self.tree.lazy:
                children = {}
            else:
                children = self.default_values.copy()

        current_key = self.keys[self.current_index]
        try:
            new_value = next(self.iterators[self.current_index])
        except StopIteration:
            # A child iterator has been exhausted
            self.change_index = True

            try:
                children[current_key] = self.default_values[current_key]  # type: ignore[index]
            except KeyError:
                assert isinstance(self.tree, Union)
                if self.tree.no_default_policy != NoDefaultPolicy.SKIP:
                    raise

            return self.__next(children)

        children[current_key] = new_value  # type: ignore[index]

        return children


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

    def default(
        self, no_default_policy: NoDefaultPolicy = NoDefaultPolicy.ERROR
    ) -> DataTree | _NoDefault:
        error: None | NoDefaultError = None
        try:
            # This is the only statement that can raise an exception.
            child_default = self.child.default(no_default_policy)

            # We keep those lines here so that `child_default` is explicitly
            # bound.
            if isinstance(child_default, _NoDefault):
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


@dataclass
class TransformIterator(TreeIterator):
    transform_node: Transform
    child_iterator: TreeIterator = field(init=False)

    def __post_init__(self):
        super().__init__()
        self.child_iterator = iter(self.transform_node.child)

    def reset(self):
        self.child_iterator.reset()

    def reverse(self):
        super().reverse()
        self.child_iterator.reverse()

    def __next__(self) -> DataTree:
        return self.transform_node.transform(next(self.child_iterator))


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
        return AccumulatorIterator(self, last_value=self.start_value)

    def transform(self, data_tree: DataTree) -> DataTree:
        return data_tree


@dataclass
class AccumulatorIterator(TransformIterator):
    last_value: dict | None

    def __next__(self) -> DataTree:
        data_tree = super().__next__()
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


class LazifyIterator(TreeIterator):
    tree: Lazify
    iterator: TreeIterator

    #: Accumulation of the values yielded by the tree iterator. If it generates
    #: a non-dict value, then stores `None`.
    accumulated_value: dict[Key, DataTree] | None

    def __init__(self, tree: Lazify):
        self.tree = tree
        self.iterator = iter(self.tree.child)
        self.reset()

    def reset(self):
        self.iterator.reset()
        self.last_value = None
        self.accumulated_value = None

    def reverse(self):
        super().reverse()
        self.iterator.reverse()

    def __next__(self) -> DataTree:
        new_value = next(self.iterator)
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

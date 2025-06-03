import dataclasses
import datetime
import itertools
import math
import unittest

import hypothesis
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import phileas
from phileas import iteration
from phileas.iteration import (
    Accumulator,
    CartesianProduct,
    Configurations,
    DataLiteral,
    DataTree,
    FunctionalTranform,
    GeneratorWrapper,
    GeometricRange,
    InfiniteLength,
    IntegerRange,
    IterationLiteral,
    IterationMethod,
    IterationTree,
    Key,
    LinearRange,
    NoDefaultPolicy,
    NumericRange,
    NumpyRNG,
    Seed,
    Sequence,
    Transform,
    Union,
)
from phileas.iteration.base import ChildPath
from phileas.iteration.node import First, Lazify, Pick, Shuffle, UnaryNode
from phileas.iteration.random import generate_seeds
from phileas.iteration.utility import (
    flatten_datatree,
    iteration_tree_to_xarray_parameters,
)

# Some tests are close to the 200 ms limit after which hypothesis classifies
# the test as an error, so increase it.
hypothesis.settings.register_profile("ci", deadline=datetime.timedelta(seconds=5))
hypothesis.settings.load_profile("ci")

# Restrict the iterated size of infinite trees
INFINITE_TREE_ITERATED_SIZE = 500

### Hypothesis strategies ###

## Data tree ##

# As only the shape of the iteration trees matters for iteration, and not the
# shape of data trees, we don't bother generating more complex data trees.


data_literal = st.integers(0, 2)
data_tree = data_literal
key = data_literal


## Iteration leaves ##


@st.composite
def iteration_literal(draw):
    return IterationLiteral(draw(data_literal))


@st.composite
def numeric_range(draw):
    return NumericRange(draw(st.floats()), draw(st.floats()), default_value=1)


@st.composite
def linear_range(draw):
    start = draw(st.floats(min_value=-10, max_value=10, allow_nan=False))
    end = draw(st.floats(min_value=-10, max_value=10, allow_nan=False))
    if start == end:
        return LinearRange(start, end, steps=1, default_value=1)
    else:
        return LinearRange(start, end, steps=draw(st.integers(2, 3)))


@st.composite
def geometric_range(draw):
    sign = draw(st.sampled_from([-1, 1]))
    start = sign * draw(
        st.floats(min_value=1e-10, max_value=10, exclude_min=True, allow_nan=False)
    )
    end = sign * draw(
        st.floats(
            min_value=1e-10, max_value=10, exclude_min=True, allow_nan=False
        ).filter(lambda e: abs(e * start) > 0)
    )
    return GeometricRange(start, end, steps=draw(st.integers(2, 3)), default_value=1.0)


@st.composite
def integer_range(draw):
    start = draw(st.integers(-3, 3))
    end = draw(st.one_of(st.just(math.inf), st.just(-math.inf), st.integers(-3, 3)))
    if start == end:
        return IntegerRange(start, end, step=0, default_value=1)
    else:
        return IntegerRange(start, end, step=draw(st.integers(1, 3)), default_value=1)


@st.composite
def sequence(draw):
    seq = draw(st.lists(data_tree, min_size=1, max_size=3))
    return Sequence(seq, default_value=1)


@st.composite
def random_leaf(draw):
    seed = Seed([], draw(data_tree))
    size = draw(st.one_of(st.none(), st.integers(1, 3)))
    default = draw(st.one_of(st.none(), data_tree))
    return NumpyRNG(
        seed=seed,
        size=size,
        distribution=np.random.Generator.uniform,
        default_value=default,
    )


@st.composite
def generator_wrapper(draw):
    size = draw(st.integers(1, 3))
    return GeneratorWrapper(lambda: iter(range(size)), size=size)


iteration_leaf = st.one_of(
    iteration_literal(),
    numeric_range(),
    linear_range(),
    geometric_range(),
    integer_range(),
    sequence(),
    random_leaf(),
    generator_wrapper(),
)

iterable_iteration_leaf = st.one_of(
    iteration_literal(),
    linear_range(),
    geometric_range(),
    integer_range(),
    sequence(),
    random_leaf(),
    generator_wrapper(),
)

## Iteration nodes ##


@st.composite
def iteration_tree_node(draw, children_st: st.SearchStrategy) -> IterationTree:
    if draw(st.booleans()):
        child = draw(children_st)
        return unary_tree_node(draw, child)
    else:
        children = draw(
            st.lists(children_st, min_size=1, max_size=4)
            | st.dictionaries(data_literal, children_st, min_size=1, max_size=4)
        )
        return iteration_method_node(draw, children)


def unary_tree_node(draw, child: IterationTree) -> IterationTree:
    types: list[type[UnaryNode]] = [IdTransform, First]
    try:
        if child.safe_len() is not None:
            types.append(Shuffle)
    except TypeError:  # Occurs for NumericRange
        pass

    Node = draw(st.sampled_from(types))
    if Node == First:
        return Node(child, draw(st.one_of(st.none(), st.integers(1, 3))))

    return Node(child)


def cartesian_product(
    draw, children: dict[Key, IterationTree] | list[IterationTree], lazy: bool
) -> CartesianProduct:
    return CartesianProduct(children, lazy=lazy)


def union(
    draw, children: dict[Key, IterationTree] | list[IterationTree], lazy: bool
) -> IterationMethod:
    presets: list[str | None] = ["first"]
    if isinstance(children, dict):
        presets.append(None)
    preset = draw(st.sampled_from(["first"]))

    common_presets = [False]
    if preset == "first":
        common_presets.append(True)
    common_preset = draw(st.sampled_from(common_presets))

    children_list = children
    if not isinstance(children, list):
        children_list = list(children.values())
    assert isinstance(children_list, list)

    resets = ["first", None]
    try:
        if all(child.safe_len() is not None for child in children_list):
            resets.append("last")
    except TypeError:  # This happens with NumericRange leaves
        pass
    reset = draw(st.sampled_from(["first"]))

    return Union(
        children, lazy=lazy, preset=preset, common_preset=common_preset, reset=reset
    )


def pick(
    draw, children: dict[Key, IterationTree] | list[IterationTree], lazy: bool
) -> IterationMethod:
    return Pick(children, lazy=lazy)


def configurations(
    draw, children: dict[Key, IterationTree] | list[IterationTree], lazy: bool
) -> IterationMethod:
    assert isinstance(children, dict)

    move_up = False
    insert_name = False
    if all(
        isinstance(c, IterationMethod)
        and not isinstance(c, Configurations)
        and isinstance(c.children, dict)
        for c in children.values()
    ):
        move_up = draw(st.booleans())

        if move_up:
            insert_name = False
        else:
            insert_name = draw(st.booleans())

    default_child = draw(st.sampled_from(list(children.keys())))

    return Configurations(
        children,
        lazy=False,
        move_up=move_up,
        insert_name=insert_name,
        default_configuration=default_child,
    )


def iteration_method_node(
    draw, children: dict[Key, IterationTree] | list[IterationTree]
) -> IterationTree:
    node_factories = [
        cartesian_product,
        union,
        pick,
    ]
    lazy = False
    if isinstance(children, dict):
        node_factories.append(configurations)
        lazy = draw(st.booleans())

    node_factory = draw(st.sampled_from(node_factories))
    return node_factory(draw, children, lazy)


class IdTransform(Transform):
    def transform(self, data_tree: DataTree) -> DataTree:
        return data_tree


def transform(child: st.SearchStrategy):
    return st.builds(IdTransform, child)


iteration_tree_with_likely_config_root = st.recursive(
    iteration_leaf,
    lambda children: iteration_tree_node(children) | transform(children),
    max_leaves=8,
)


@st.composite
def iteration_tree(draw) -> st.SearchStrategy:
    tree = draw(iteration_tree_with_likely_config_root)

    if isinstance(tree, Configurations) and tree.move_up:
        Node = draw(st.sampled_from([CartesianProduct, Union]))
        children = {"config": tree}
        for _ in range(draw(st.integers(min_value=0, max_value=3))):
            children[draw(key)] = draw(iteration_tree())
        return Node(children, lazy=False)

    return draw(st.just(tree))


iterable_iteration_tree_with_likely_config_root = st.recursive(
    iterable_iteration_leaf,
    lambda children: iteration_tree_node(children) | transform(children),
    max_leaves=8,
)


@st.composite
def iterable_iteration_tree(draw):
    tree = draw(
        iterable_iteration_tree_with_likely_config_root.filter(
            lambda tree: not isinstance(tree, Configurations) or not tree.move_up
        )
    )
    return generate_seeds(tree)


@st.composite
def iterable_iteration_tree_and_index(draw):
    tree = draw(iterable_iteration_tree())
    size = tree.safe_len()
    if size is None:
        size = INFINITE_TREE_ITERATED_SIZE

    index = draw(st.integers(min_value=0, max_value=size - 1))
    return tree, index


### Tests ###


def is_lazy(tree: IterationTree) -> bool:
    lazy = False

    def inspect(tree: IterationTree, path: ChildPath) -> IterationTree:
        nonlocal lazy
        if isinstance(tree, IterationMethod) and tree.lazy:
            lazy = True

        if isinstance(tree, Lazify):
            lazy = True

        return tree

    tree.depth_first_modify(inspect)

    return lazy


class TestIteration(unittest.TestCase):
    ## Iteration leaves ##
    @given(linear_range() | geometric_range())
    def test_linear_and_geometric_range_length(self, r: LinearRange | GeometricRange):
        self.assertEqual(len(r), r.steps)

    def test_linear_range_iteration(self):
        r = LinearRange(0.0, 5.0, steps=6)
        self.assertEqual(list(r), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_geometric_range_iteration(self):
        r = GeometricRange(1.0, 8.0, steps=4)
        self.assertEqual(list(r), [1.0, 2.0, 4.0, 8.0])

    @given(st.integers(-10, 10), st.integers(-10, 10), st.integers(10))
    def test_integer_range_iteration(self, start: int, end: int, step: int):
        r = iter(IntegerRange(start, end, step=step))
        iterator = r
        value = next(iterator)
        assert isinstance(value, (int, float))
        self.assertEqual(value, start)
        while True:
            try:
                next_value = next(iterator)
                assert isinstance(next_value, (int, float))
                self.assertEqual(abs(next_value - value), step)
                value = next_value
            except StopIteration:
                break

        if end > start:
            self.assertGreaterEqual(end, value)
        else:
            self.assertGreaterEqual(value, end)

    @given(sequence())
    def test_sequence_iteration(self, sequence: Sequence):
        self.assertEqual(list(sequence), sequence.elements)

    @given(st.lists(data_literal, max_size=10))
    def test_generator_wrapper_finite(self, elements: list[DataLiteral]):
        def generator_factory():
            return (e for e in elements)

        tree = GeneratorWrapper(generator_factory, size=len(elements))
        self.assertEqual(list(tree), elements)

    @given(st.integers(min_value=0, max_value=10))
    def test_generator_wrapper_infinite(self, size: int):
        def generator_factory():
            return (e for e in itertools.count())

        tree = GeneratorWrapper(generator_factory, size=size)
        self.assertEqual(list(tree), list(range(size)))

    @given(
        st.integers(min_value=0, max_value=10),
        st.one_of(st.none(), st.integers(min_value=1, max_value=10)),
    )
    def test_generator_wrapper_too_short(
        self, generator_size: int, tree_additional_size: int | None
    ):
        def generator_factory():
            return (e for e in range(generator_size))

        tree_size = (
            None
            if tree_additional_size is None
            else generator_size + tree_additional_size
        )
        tree = GeneratorWrapper(generator_factory, size=tree_size)

        with self.assertRaises(ValueError):
            list(iter(tree))

    ## Iteration nodes ##

    @given(iteration_tree())
    def test_iteration_tree_generation(self, tree: IterationTree):
        """
        This test is voluntarily left empty, in order to test tree strategies.
        """
        del tree

    @given(iterable_iteration_tree(), st.integers(1, 3))
    def test_reverse_changes_forward(self, tree: IterationTree, reverses: int):
        iterator = iter(tree)
        for _ in range(reverses):
            iterator.reverse()

        self.assertEqual(iterator.is_forward(), reverses % 2 == 0)

    @given(iterable_iteration_tree())
    def test_len_consistent_with_iterate_finite(self, tree: IterationTree):
        try:
            n = len(tree)
            formatted_list = "\n".join(f" - {s}" for s in tree)
            hypothesis.note(f"The iterated list is \n{formatted_list}")
            self.assertEqual(n, len(list(tree)))
        except InfiniteLength:
            return

    @given(iterable_iteration_tree(), st.integers(min_value=0, max_value=1000))
    def test_infinite_tree_length_is_unbound(
        self, tree: IterationTree, tested_length: int
    ):
        if tree.safe_len() is not None:
            return

        it = iter(tree)
        for _ in range(tested_length):
            next(it)

    @given(iterable_iteration_tree())
    def test_exhausted_iterator_is_exhausted(self, tree: IterationTree):
        if tree.safe_len() is None:
            return

        iterator = iter(tree)
        _ = list(iterator)
        self.assertEqual(list(iterator), [])

    @given(iterable_iteration_tree())
    def test_reversible_iterator(self, tree: IterationTree):
        if tree.safe_len() is None:
            return

        if is_lazy(tree):
            return

        iterator = iter(tree)

        forward = list(iterator)
        hypothesis.note(f"Forward: {forward}")

        iterator.reverse()
        iterator.reset()
        backward = list(iterator)
        hypothesis.note(f"Backward: {backward}")
        backward.reverse()

        self.assertEqual(forward, backward)

    @given(iterable_iteration_tree())
    def test_reverse_without_reset_is_empty(self, tree: IterationTree):
        iterator = iter(tree)
        iterator.reverse()

        self.assertEqual(list(iterator), [])

    @given(iterable_iteration_tree_and_index())
    def test_reverse_partial_iteration(self, tree_index: tuple[IterationTree, int]):
        tree, index = tree_index
        iterator = iter(tree)

        if is_lazy(tree):
            return

        if index == 0:
            return

        for _ in range(index - 1):
            _ = next(iterator)

        previous, last = next(iterator), next(iterator)
        hypothesis.note(f"Last value: {last}")
        iterator.reverse()
        previous_after_reverse = next(iterator)

        self.assertEqual(previous, previous_after_reverse)

    @given(iterable_iteration_tree(), st.booleans())
    def test_same_iteration_after_reset(self, tree: IterationTree, reverse: bool):
        if reverse and tree.safe_len() is None:
            return

        if is_lazy(tree):
            return

        iterator = iter(tree)
        if reverse:
            iterator.reverse()
            iterator.reset()

        size = tree.safe_len()
        if size is None:
            size = INFINITE_TREE_ITERATED_SIZE

        l_before_reset = list(itertools.islice(iterator, size))
        iterator.reset()
        l_after_reset = list(itertools.islice(iterator, size))

        self.assertEqual(l_before_reset, l_after_reset)

    ## Iteration methods ##

    @given(st.lists(st.integers(min_value=2, max_value=3), min_size=1, max_size=5))
    def test_cartesian_product_forward_lazy_iteration(self, sizes: list[int]):
        tree = CartesianProduct(
            {i: Sequence(list(range(s))) for i, s in enumerate(sizes)}, lazy=False
        )

        non_lazy_list = list(tree)
        hypothesis.note(f"Non lazy list: {non_lazy_list}")

        lazy_tree = dataclasses.replace(tree, lazy=True)
        lazy_list = list(lazy_tree)
        hypothesis.note(f"Lazy list: {lazy_list}")

        accumulated_lazy_list = []
        accumulated_element: dict = {}
        for element in lazy_list:
            assert isinstance(element, dict)
            for key in set(accumulated_element.keys()).intersection(element.keys()):
                self.assertNotEqual(accumulated_element[key], element[key])

            accumulated_element = accumulated_element | element
            accumulated_lazy_list.append(accumulated_element)

        hypothesis.note(f"Accumulated lazy list: {accumulated_lazy_list}")

        self.assertEqual(non_lazy_list, accumulated_lazy_list)

    @given(st.lists(linear_range(), min_size=1, max_size=5))
    def test_cartesian_product_iteration(self, children: list[IterationTree]):
        c = CartesianProduct(children)

        iterated_list = list(c)
        formatted_list = "\n".join(f" - {s}" for s in iterated_list)
        hypothesis.note(f"Iterated list:\n{formatted_list}")

        expected_list = list(map(list, itertools.product(*children)))
        formatted_list = "\n".join(f" - {s}" for s in expected_list)
        hypothesis.note(f"Expected list:\n{formatted_list}")

        self.assertEqual(iterated_list, expected_list)

    def test_cartesian_product_lazy_iteration_explicit(self):
        u = CartesianProduct(
            {
                0: IntegerRange(1, 2, default_value=10),
                1: IntegerRange(1, 2, default_value=10),
                2: IntegerRange(1, 2),
            },
            lazy=True,
        )
        iterated_list = list(u)
        expected_list = [
            {0: 1, 1: 1, 2: 1},
            {2: 2},
            {1: 2, 2: 1},
            {2: 2},
            {0: 2, 1: 1, 2: 1},
            {2: 2},
            {1: 2, 2: 1},
            {2: 2},
        ]

        self.assertEqual(iterated_list, expected_list)

    @given(st.lists(linear_range(), min_size=1, max_size=5))
    def test_cartesian_product_snake_has_same_elements_as_non_snake(
        self, children: list[IterationTree]
    ):
        cp_snake = CartesianProduct(children, snake=True)
        cp_non_snake = CartesianProduct(children, snake=False)

        snake_list = list(cp_snake)
        formatted_list = "\n".join(f" - {s}" for s in snake_list)
        hypothesis.note(f"Snake list:\n{formatted_list}")

        non_snake_list = list(cp_non_snake)
        formatted_list = "\n".join(f" - {s}" for s in non_snake_list)
        hypothesis.note(f"Non-snake list:\n{formatted_list}")

        self.assertEqual(
            {frozenset(e) for e in snake_list},  # type: ignore[arg-type]
            {frozenset(e) for e in non_snake_list},  # type: ignore[arg-type]
        )

    def test_cartesian_product_snake_iteration_explicit(self):
        tree = CartesianProduct(
            children={1: Sequence([1, 2, 3]), 2: Sequence(["a", "b", "c"])},
            lazy=False,
            snake=True,
        )
        iterated_list = list(tree)
        expected_list = [
            {1: 1, 2: "a"},
            {1: 1, 2: "b"},
            {1: 1, 2: "c"},
            {1: 2, 2: "c"},
            {1: 2, 2: "b"},
            {1: 2, 2: "a"},
            {1: 3, 2: "a"},
            {1: 3, 2: "b"},
            {1: 3, 2: "c"},
        ]

        self.assertEqual(iterated_list, expected_list)

    def test_cartesian_product_infinite_children_explicit(self):
        tree = generate_seeds(
            CartesianProduct(
                [
                    Sequence([1, 2]),
                    IntegerRange(start=0, end=math.inf),
                    Sequence([1, 2]),
                    IntegerRange(start=0, end=math.inf),
                    Sequence(["a", "b"]),
                    Sequence([True, False]),
                ]
            )
        )

        iterated_values = []
        with self.assertLogs(phileas.logger, level="WARNING"):
            iterated_values = list(itertools.islice(tree, 10))

        expected_values = [
            [1, 0, 1, 0, "a", True],
            [1, 0, 1, 0, "a", False],
            [1, 0, 1, 0, "b", True],
            [1, 0, 1, 0, "b", False],
            [1, 0, 1, 1, "a", True],
            [1, 0, 1, 1, "a", False],
            [1, 0, 1, 1, "b", True],
            [1, 0, 1, 1, "b", False],
            [1, 0, 1, 2, "a", True],
            [1, 0, 1, 2, "a", False],
        ]

        self.assertEqual(iterated_values, expected_values)

    @given(
        st.dictionaries(key, sequence(), min_size=1, max_size=5),
        st.sampled_from([CartesianProduct, Union]),
    )
    def test_accumulated_lazy_iteration_identical_to_iteration(
        self, children: dict[Key, IterationTree], Type: type[IterationMethod]
    ):
        tree = Type(children, lazy=False)
        non_lazy_list = list(tree)
        hypothesis.note("Non lazy list: \n" + "\n".join(map(str, non_lazy_list)))

        acc_lazy_list = list(
            tree.replace_node([], type(tree), lazy=True).insert_unary([], Accumulator)
        )
        hypothesis.note(
            "Accumulated lazy list: \n" + "\n".join(map(str, acc_lazy_list))
        )

        self.assertEqual(non_lazy_list, acc_lazy_list)

    @given(
        st.dictionaries(key, sequence(), min_size=1, max_size=5),
    )
    def test_cartesian_product_lazy_snake_changes_elements_once_at_a_time(
        self, children: dict[Key, IterationTree]
    ):
        cp = CartesianProduct(children, lazy=True, snake=True)
        iterated_list = list(cp)
        hypothesis.note("Iterated list: \n" + "\n".join(map(str, iterated_list)))

        sizes = list(map(len, iterated_list))  # type: ignore[arg-type]
        expected_sizes = [len(children)] + [1] * (len(iterated_list) - 1)

        self.assertEqual(sizes, expected_sizes)

    def test_cartesian_product_lazy_snake_iteration_explicit(self):
        tree = CartesianProduct(
            children={1: Sequence([1, 2, 3]), 2: Sequence(["a", "b", "c"])},
            lazy=True,
            snake=True,
        )
        iterated_list = list(tree)
        expected_list = [
            {1: 1, 2: "a"},
            {2: "b"},
            {2: "c"},
            {1: 2},
            {2: "b"},
            {2: "a"},
            {1: 3},
            {2: "b"},
            {2: "c"},
        ]

        self.assertEqual(iterated_list, expected_list)

    def test_union_iteration_finite(self):
        u = Union(
            {
                0: IntegerRange(1, 2, default_value=10),
                1: IntegerRange(1, 2, default_value=10),
                2: IntegerRange(1, 2, default_value=10),
            },
            preset="default",
            reset="default",
        )
        iterated_list = list(u)
        expected_list = [
            {0: 1, 1: 10, 2: 10},
            {0: 2, 1: 10, 2: 10},
            {0: 10, 1: 1, 2: 10},
            {0: 10, 1: 2, 2: 10},
            {0: 10, 1: 10, 2: 1},
            {0: 10, 1: 10, 2: 2},
        ]

        self.assertEqual(iterated_list, expected_list)

    def test_union_iteration_infinite_last(self):
        tree = generate_seeds(
            Union(
                [
                    Sequence([1, 2]),
                    Sequence([1, 2]),
                    IntegerRange(start=0, end=math.inf),
                ]
            )
        )

        iterated_list = []
        with self.assertLogs(phileas.logger, level="WARNING"):
            iterated_list = list(itertools.islice(tree, 10))

        expected_list = [
            [1, 1, 0],
            [2, 1, 0],
            [1, 1, 0],
            [1, 2, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 3],
            [1, 1, 4],
            [1, 1, 5],
        ]

        self.assertEqual(iterated_list, expected_list)

    def test_union_iteration_infinite_inner(self):
        tree = generate_seeds(
            Union(
                [
                    Sequence([1, 2]),
                    Sequence([1, 2], default_value=3),
                    IntegerRange(start=0, end=math.inf),
                    Sequence([1, 2], default_value=3),
                ]
            )
        )

        iterated_list = []
        with self.assertLogs(phileas.logger, level="WARNING"):
            iterated_list = list(itertools.islice(tree, 10))

        expected_list = [
            [1, 1, 0, 1],
            [2, 1, 0, 1],
            [1, 1, 0, 1],
            [1, 2, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 2, 1],
            [1, 1, 3, 1],
            [1, 1, 4, 1],
            [1, 1, 5, 1],
        ]

        self.assertEqual(iterated_list, expected_list)

    def test_union_lazy_iteration(self):
        u = Union(
            {
                0: IntegerRange(1, 2, default_value=10),
                1: IntegerRange(1, 2, default_value=10),
                2: IntegerRange(1, 2, default_value=10),
            },
            lazy=True,
            preset="default",
            reset="default",
        )
        iterated_list = list(u)
        expected_list = [
            {0: 1, 1: 10, 2: 10},
            {0: 2},
            {0: 10, 1: 1},
            {1: 2},
            {1: 10, 2: 1},
            {2: 2},
        ]

        self.assertEqual(iterated_list, expected_list)

    def test_pick(self):
        tree = Pick(
            children={
                "1": IntegerRange(start=1, end=5, step=1),
                "2": Sequence(elements=["a", "b", "c", "d"]),
            },
            seed=Seed(path=[], salt="test"),
        )
        expected_list = [
            {"1": 1},
            {"2": "a"},
            {"2": "b"},
            {"1": 2},
            {"2": "c"},
            {"1": 3},
            {"1": 4},
            {"1": 5},
            {"2": "d"},
        ]
        iterated_list = list(tree)
        self.assertEqual(expected_list, iterated_list)

    @given(st.integers(1, 10))
    def test_shuffle_yields_a_permutation(self, size: int):
        child = IntegerRange(start=0, end=size - 1)
        values = set(child)
        shuffle = generate_seeds(Shuffle(child))
        shuffled_values = list(shuffle)
        hypothesis.note(f"Shuffled values: {shuffled_values}")

        self.assertEqual(len(values), len(shuffled_values))
        self.assertEqual(values, set(shuffled_values))

    def test_first_finite(self):
        tree = First(IntegerRange(start=0, end=math.inf, step=1), 10)
        self.assertEqual(list(tree), list(range(10)))

    def test_first_infinite(self):
        tree = First(IntegerRange(start=0, end=5, step=1), None)
        self.assertEqual(list(tree), list(range(6)))

    @given(iteration_tree())
    def test_requested_configuration_is_not_configurable(self, tree: IterationTree):
        def assert_not_configurable(
            tree: IterationTree, path: ChildPath
        ) -> IterationTree:
            try:
                self.assertEqual(tree.configurations, set())
            except AssertionError:
                hypothesis.note(f"Configurable path: {path}")
                raise

            return tree

        for name in tree.configurations:
            configured_tree = tree.get_configuration(name)
            try:
                configured_tree.depth_first_modify(assert_not_configurable)
            except AssertionError:
                hypothesis.note(f"Configuration: {name}")
                hypothesis.note(f"Configured tree: {configured_tree}")
                hypothesis.note(f"Configurable tree: {tree}")
                raise

    def test_configurations_sample_nomoveup_noinsertname_node(self):
        tree = CartesianProduct(
            children={
                "c": Configurations(
                    children={
                        "config1": CartesianProduct(
                            children={
                                "param1": IterationLiteral(value=11),
                                "param2": IterationLiteral(value=12),
                            },
                        ),
                        "config2": CartesianProduct(
                            children={
                                "param1": IterationLiteral(value=21),
                                "param2": IterationLiteral(value=22),
                            },
                        ),
                    },
                    move_up=False,
                    insert_name=False,
                )
            },
        )
        config = tree.get_configuration("config1")
        expected_config = CartesianProduct(
            children={
                "c": CartesianProduct(
                    children={
                        "param1": IterationLiteral(value=11),
                        "param2": IterationLiteral(value=12),
                    },
                )
            },
        )

        self.assertEqual(config, expected_config)

    def test_configurations_sample_nomoveup_noinsertname_leaf(self):
        tree = CartesianProduct(
            children={
                "c": Configurations(
                    children={
                        "config1": IterationLiteral(1),
                        "config2": IterationLiteral(2),
                    },
                    move_up=False,
                    insert_name=False,
                )
            },
        )
        config = tree.get_configuration("config1")
        expected_config = CartesianProduct(
            children={
                "c": IterationLiteral(1),
            },
        )

        self.assertEqual(config, expected_config)

    def test_configurations_sample_moveup_noinsertname_node(self):
        tree = CartesianProduct(
            children={
                "c": Configurations(
                    children={
                        "config1": CartesianProduct(
                            children={
                                "param1": IterationLiteral(value=11),
                                "param2": IterationLiteral(value=12),
                            },
                        ),
                        "config2": CartesianProduct(
                            children={
                                "param1": IterationLiteral(value=21),
                                "param2": IterationLiteral(value=22),
                            },
                        ),
                    },
                    move_up=True,
                    insert_name=False,
                )
            },
        )
        config = tree.get_configuration("config1")
        expected_config = CartesianProduct(
            children={
                "param1": IterationLiteral(value=11),
                "param2": IterationLiteral(value=12),
            },
        )

        self.assertEqual(config, expected_config)

    def test_configurations_sample_moveup_noinsertname_leaf(self):
        tree = CartesianProduct(
            children={
                "c": Configurations(
                    children={
                        "config1": IterationLiteral(1),
                        "config2": IterationLiteral(2),
                    },
                    move_up=True,
                    insert_name=False,
                )
            },
        )

        with self.assertRaises(ValueError):
            tree.get_configuration("config1")

    def test_configurations_sample_moveup_insertname_node(self):
        tree = CartesianProduct(
            children={
                "c": Configurations(
                    children={
                        "config1": CartesianProduct(
                            children={
                                "param1": IterationLiteral(value=11),
                                "param2": IterationLiteral(value=12),
                            },
                        ),
                        "config2": CartesianProduct(
                            children={
                                "param1": IterationLiteral(value=21),
                                "param2": IterationLiteral(value=22),
                            },
                        ),
                    },
                    move_up=True,
                    insert_name=True,
                )
            },
        )
        config = tree.get_configuration("config1")
        expected_config = CartesianProduct(
            children={
                "c": IterationLiteral(value="config1"),
                "param1": IterationLiteral(value=11),
                "param2": IterationLiteral(value=12),
            },
        )

        self.assertEqual(config, expected_config)

    def test_configurations_sample_moveup_insertname_leaf(self):
        tree = CartesianProduct(
            children={
                "c": Configurations(
                    children={
                        "config1": IterationLiteral(1),
                        "config2": IterationLiteral(2),
                    },
                    move_up=True,
                    insert_name=True,
                )
            },
        )

        with self.assertRaises(ValueError):
            tree.get_configuration("config1")

    def test_configurations_sample_nomoveup_insertname_node(self):
        tree = CartesianProduct(
            children={
                "c": Configurations(
                    children={
                        "config1": CartesianProduct(
                            children={
                                "param1": IterationLiteral(value=11),
                                "param2": IterationLiteral(value=12),
                            },
                        ),
                        "config2": CartesianProduct(
                            children={
                                "param1": IterationLiteral(value=21),
                                "param2": IterationLiteral(value=22),
                            },
                        ),
                    },
                    move_up=False,
                    insert_name=True,
                )
            },
        )
        config = tree.get_configuration("config1")
        expected_config = CartesianProduct(
            children={
                "c": CartesianProduct(
                    children={
                        "_configuration": IterationLiteral(value="config1"),
                        "param1": IterationLiteral(value=11),
                        "param2": IterationLiteral(value=12),
                    },
                )
            },
        )

        self.assertEqual(config, expected_config)

    def test_configurations_sample_nomoveup_insertname_leaf(self):
        tree = CartesianProduct(
            children={
                "c": Configurations(
                    children={
                        "config1": IterationLiteral(1),
                        "config2": IterationLiteral(2),
                    },
                    move_up=False,
                    insert_name=True,
                )
            },
        )
        config = tree.get_configuration("config1")
        expected_config = CartesianProduct(
            children={
                "_c_configuration": IterationLiteral("config1"),
                "c": IterationLiteral(1),
            },
        )

        self.assertEqual(config, expected_config)

    ## Transform ##

    def test_transform(self):
        class Add1Tranform(Transform):
            def transform(self, data_tree: int) -> int:  # type: ignore
                return data_tree + 1

        t = Add1Tranform(IntegerRange(0, 2))
        t_result = list(t)
        self.assertEqual(t_result, [1, 2, 3])

    def test_functional_transform(self):
        def add1(data_tree: DataTree) -> DataTree:
            return data_tree + 1  # type: ignore

        t = FunctionalTranform(IntegerRange(0, 2), add1)
        t_result = list(t)
        self.assertEqual(t_result, [1, 2, 3])

    @given(
        st.lists(
            st.dictionaries(
                st.integers(min_value=0, max_value=4),
                st.text(max_size=2),
                min_size=1,
                max_size=5,
            ),
            min_size=1,
            max_size=5,
        )
    )
    def test_accumulating_transform(self, dicts: list[dict[int, str]]):
        nac_tree = Sequence(dicts)  # type: ignore[arg-type]
        ac_nac_list = []
        ac_data_tree: dict[int, str] = {}
        for data_tree in nac_tree:
            ac_data_tree |= data_tree  # type: ignore[arg-type]
            ac_nac_list.append(ac_data_tree.copy())

        hypothesis.note("Expected list: \n" + "\n".join(map(str, ac_nac_list)))

        ac_tree = Accumulator(Sequence(dicts))  # type: ignore[arg-type]
        ac_list = list(ac_tree)
        hypothesis.note("Obtained list: \n" + "\n".join(map(str, ac_list)))

        self.assertEqual(ac_list, ac_nac_list)

    ## Default ##

    def test_no_default_policy_skip(self):
        t = CartesianProduct({"a": LinearRange(1, 2)})
        self.assertEqual(t.default(no_default_policy=NoDefaultPolicy.SKIP), {})

    def test_no_default_policy_sentinel(self):
        t = CartesianProduct({"a": LinearRange(1, 2)})
        self.assertEqual(
            t.default(no_default_policy=NoDefaultPolicy.SENTINEL),
            {"a": iteration.no_default},
        )

    def test_no_default_policy_error(self):
        t = CartesianProduct({"a": LinearRange(1, 2)})
        with self.assertRaises(iteration.NoDefaultError):
            t.default(no_default_policy=NoDefaultPolicy.ERROR)

    def test_no_default_policy_first_element(self):
        tree = CartesianProduct({"a": LinearRange(1, 2)})
        value = tree.default(NoDefaultPolicy.FIRST_ELEMENT)
        first_value = next(iter(tree))

        self.assertEqual(value, first_value)

    # Utilities
    def test_flatten_datatree(self):
        tree: DataTree = {"key1": {"key1-1": 1}, "key2": [1, 2], "key3": "value"}
        expected_flat_tree = {
            "key1.key1-1": 1,
            "key2.0": 1,
            "key2.1": 2,
            "key3": "value",
        }
        flattened_tree = flatten_datatree(tree)
        self.assertEqual(expected_flat_tree, flattened_tree)

    @given(iterable_iteration_tree())
    def test_iteration_tree_to_xarray_parameters_raises_no_error(
        self, tree: IterationTree
    ):
        import numpy as np
        import xarray as xr

        hypothesis.note(f"The iteration tree is {tree}")
        flat_tree = flatten_datatree(tree.to_pseudo_data_tree())
        leaves: list[IterationTree | DataLiteral]
        if isinstance(flat_tree, dict):
            leaves = list(flat_tree.values())
        else:
            leaves = [flat_tree]

        leaf_lengths = [
            leaf.safe_len() if isinstance(leaf, IterationTree) else 1 for leaf in leaves
        ]
        if None in leaf_lengths:
            with self.assertRaises(InfiniteLength):
                iteration_tree_to_xarray_parameters(tree)
        else:
            coords, dims_name, dims_shape = iteration_tree_to_xarray_parameters(tree)
            xr.DataArray(data=np.empty(dims_shape), coords=coords, dims=dims_name)

import dataclasses
import datetime
import itertools
import unittest

import hypothesis
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from phileas import iteration
from phileas.iteration import (
    Accumulator,
    CartesianProduct,
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
    Sequence,
    Transform,
    Union,
)
from phileas.iteration.leaf import NumpyRNG, Seed
from phileas.iteration.utility import (
    flatten_datatree,
    generate_seeds,
    iteration_tree_to_xarray_parameters,
)

# Some tests are close to the 200 ms limit after which hypothesis classifies
# the test as an error, so increase it.
hypothesis.settings.register_profile("ci", deadline=datetime.timedelta(seconds=5))
hypothesis.settings.load_profile("ci")

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
    start, end = draw(st.integers(-3, 3)), draw(st.integers(-3, 3))
    if start == end:
        return IntegerRange(start, end, default_value=1)
    else:
        return IntegerRange(start, end, step=draw(st.integers(2, 3)), default_value=1)


@st.composite
def sequence(draw):
    seq = draw(st.lists(data_tree, min_size=1, max_size=3))
    return Sequence(seq, default_value=1)


@st.composite
def random_leaf(draw):
    seed = Seed([], draw(data_tree))
    size = draw(st.integers(1, 3))
    default = draw(st.one_of(st.none(), data_tree))
    return NumpyRNG(
        seed=seed,
        size=size,
        distribution=np.random.Generator.uniform,
        default_value=default,
    )


iteration_leaf = st.one_of(
    iteration_literal(),
    numeric_range(),
    linear_range(),
    geometric_range(),
    integer_range(),
    sequence(),
    random_leaf(),
)

iterable_iteration_leaf = st.one_of(
    iteration_literal(),
    linear_range(),
    geometric_range(),
    integer_range(),
    sequence(),
    random_leaf(),
)

## Iteration nodes ##


@st.composite
def iteration_tree_node(draw, children: st.SearchStrategy) -> st.SearchStrategy:
    Node = draw(st.sampled_from([CartesianProduct, Union]))
    children = draw(
        st.lists(children, min_size=1, max_size=4)
        | st.dictionaries(data_literal, children, min_size=1, max_size=4)
    )

    return Node(children, lazy=False)


class IdTransform(Transform):
    def transform(self, data_tree: DataTree) -> DataTree:
        return data_tree


def transform(child: st.SearchStrategy):
    return st.builds(IdTransform, child)


iteration_tree = st.recursive(
    iteration_leaf,
    lambda children: iteration_tree_node(children) | transform(children),
    max_leaves=8,
)

iterable_iteration_tree = st.recursive(
    iterable_iteration_leaf,
    lambda children: iteration_tree_node(children) | transform(children),
    max_leaves=8,
)


@st.composite
def iterable_iteration_tree_and_index(draw):
    tree = draw(iterable_iteration_tree)
    index = draw(st.integers(min_value=0, max_value=len(tree) - 1))
    return tree, index


### Tests ###


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

    @given(iteration_tree)
    def test_iteration_tree_generation(self, tree: IterationTree):
        """
        This test is voluntarily left empty, in order to test tree strategies.
        """
        del tree

    @given(iterable_iteration_tree, st.integers(1, 3))
    def test_reverse_changes_forward(self, tree: IterationTree, reverses: int):
        iterator = iter(tree)
        for _ in range(reverses):
            iterator.reverse()

        self.assertEqual(iterator.is_forward(), reverses % 2 == 0)

    @given(iterable_iteration_tree)
    def test_len_consistent_with_iterate(self, tree: IterationTree):
        try:
            n = len(tree)
            formatted_list = "\n".join(f" - {s}" for s in tree)
            hypothesis.note(f"The iterated list is \n{formatted_list}")
            self.assertEqual(n, len(list(tree)))
        except InfiniteLength:
            return

    @given(iterable_iteration_tree)
    def test_exhausted_iterator_is_exhausted(self, tree: IterationTree):
        iterator = iter(tree)
        _ = list(iterator)
        self.assertEqual(list(iterator), [])

    @given(iterable_iteration_tree)
    def test_reversible_iterator(self, tree: IterationTree):
        iterator = iter(tree)

        forward = list(iterator)
        hypothesis.note(f"Forward: {forward}")

        iterator.reverse()
        iterator.reset()
        backward = list(iterator)
        hypothesis.note(f"Backward: {backward}")
        backward.reverse()

        self.assertEqual(forward, backward)

    @given(iterable_iteration_tree)
    def test_reverse_without_reset_is_empty(self, tree: IterationTree):
        iterator = iter(tree)
        iterator.reverse()

        self.assertEqual(list(iterator), [])

    @given(iterable_iteration_tree_and_index())
    def test_reverse_partial_iteration(self, tree_index: tuple[IterationTree, int]):
        tree, index = tree_index
        iterator = iter(tree)

        if index == 0:
            return

        for _ in range(index - 1):
            _ = next(iterator)

        previous, last = next(iterator), next(iterator)
        hypothesis.note(f"Last value: {last}")
        iterator.reverse()
        previous_after_reverse = next(iterator)

        self.assertEqual(previous, previous_after_reverse)

    @given(iterable_iteration_tree, st.booleans())
    def test_same_iteration_after_reset(self, tree: IterationTree, reverse: bool):
        iterator = iter(tree)
        if reverse:
            iterator.reverse()
            iterator.reset()

        l_before_reset = list(iterator)
        iterator.reset()
        l_after_reset = list(iterator)

        self.assertEqual(l_before_reset, l_after_reset)

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
                    NumpyRNG(),
                    Sequence([1, 2]),
                    NumpyRNG(),
                    Sequence(["a", "b"]),
                    Sequence([True, False]),
                ]
            )
        )

        it = iter(tree)
        iterated_values = list(itertools.islice(it, 10))
        expected_values = [
            [1, 0.9952009928886075, 1, 0.019736141410354624, "a", True],
            [1, 0.9952009928886075, 1, 0.019736141410354624, "a", False],
            [1, 0.9952009928886075, 1, 0.019736141410354624, "b", True],
            [1, 0.9952009928886075, 1, 0.019736141410354624, "b", False],
            [1, 0.9952009928886075, 1, 0.5906680448959815, "a", True],
            [1, 0.9952009928886075, 1, 0.5906680448959815, "a", False],
            [1, 0.9952009928886075, 1, 0.5906680448959815, "b", True],
            [1, 0.9952009928886075, 1, 0.5906680448959815, "b", False],
            [1, 0.9952009928886075, 1, 0.6410922942699461, "a", True],
            [1, 0.9952009928886075, 1, 0.6410922942699461, "a", False],
        ]

        self.assertEqual(iterated_values, expected_values)

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
            tree.replace_node([], type(tree), lazy=True).insert_transform(
                [], Accumulator
            )
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

    def test_union_iteration(self):
        u = Union(
            {
                0: IntegerRange(1, 2, default_value=10),
                1: IntegerRange(1, 2, default_value=10),
                2: IntegerRange(1, 2),
            },
        )
        iterated_list = list(u)
        expected_list = [
            {0: 1, 1: 10, 2: 1},
            {0: 2, 1: 10, 2: 1},
            {0: 10, 1: 1, 2: 1},
            {0: 10, 1: 2, 2: 1},
            {0: 10, 1: 10, 2: 1},
            {0: 10, 1: 10, 2: 2},
        ]

        self.assertEqual(iterated_list, expected_list)

    def test_union_lazy_iteration(self):
        u = Union(
            {
                0: IntegerRange(1, 2, default_value=10),
                1: IntegerRange(1, 2, default_value=10),
                2: IntegerRange(1, 2),
            },
            lazy=True,
        )
        iterated_list = list(u)
        expected_list = [
            {0: 1, 1: 10, 2: 1},
            {0: 2},
            {0: 10, 1: 1},
            {1: 2},
            {1: 10},
            {2: 2},
        ]

        self.assertEqual(iterated_list, expected_list)

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

    @given(iterable_iteration_tree)
    def test_iteration_tree_to_xarray_parameters_raises_no_error(
        self, tree: IterationTree
    ):
        import numpy as np
        import xarray as xr

        coords, dims_name, dims_shape = iteration_tree_to_xarray_parameters(tree)
        xr.DataArray(data=np.empty(dims_shape), coords=coords, dims=dims_name)

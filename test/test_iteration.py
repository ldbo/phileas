import unittest
from itertools import product

import hypothesis
from hypothesis import given
from hypothesis import strategies as st

from phileas import iteration
from phileas.iteration import (
    CartesianProduct,
    DataTree,
    FunctionalTranform,
    GeometricRange,
    IntegerRange,
    IterationLiteral,
    IterationTree,
    LinearRange,
    NoDefaultPolicy,
    NumericRange,
    Sequence,
    Transform,
    Union,
)

### Hypothesis strategies ###

## Data tree ##

# As only the shape of the iteration trees matters for iteration, and not the
# shape of data trees, we don't bother generating more complex data trees.


data_literal = st.integers(0, 2)
data_tree = data_literal


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


iteration_leaf = st.one_of(
    iteration_literal(),
    numeric_range(),
    linear_range(),
    geometric_range(),
    integer_range(),
    sequence(),
)

## Iteration nodes ##


@st.composite
def iteration_tree_node(draw, children: st.SearchStrategy) -> st.SearchStrategy:
    Node = draw(st.sampled_from([CartesianProduct, Union]))
    children = draw(
        st.lists(children, min_size=1, max_size=4)
        | st.dictionaries(data_literal, children, min_size=1, max_size=4)
    )
    return Node(children)


class IdTransform(Transform):
    def transform(self, data_tree: DataTree) -> DataTree:
        return data_tree


def transform(child: st.SearchStrategy):
    return st.builds(IdTransform, child)


iteration_tree = st.recursive(
    iteration_leaf,
    lambda children: iteration_tree_node(children) | transform(children),
    max_leaves=10,
)


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
        self.assertEqual(value, start)
        while True:
            try:
                next_value = next(iterator)
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

    ## Iteration nodes ##

    @given(iteration_tree)
    def test_iteration_tree_generation(self, tree: IterationTree):
        """
        This test is voluntarily left empty, in order to test tree strategies.
        """
        del tree

    @given(iteration_tree)
    def test_len_consistent_with_iterate(self, tree: IterationTree):
        try:
            n = len(tree)
            formatted_list = "\n".join(f" - {s}" for s in tree)
            hypothesis.note(f"The iterated list is \n{formatted_list}")
            self.assertEqual(n, len(list(tree)))
        except TypeError:
            return

    @given(st.lists(linear_range(), min_size=1, max_size=5))
    def test_cartesian_product_iteration(self, children: list[IterationTree]):
        c = CartesianProduct(children)
        hypothesis.note(f"Iteration tree: {c}")
        iterated_list = list(c)
        formatted_list = "\n".join(f" - {s}" for s in iterated_list)
        hypothesis.note(f"Iterated list:\n{formatted_list}")
        expected_list = list(map(list, product(*children)))
        formatted_list = "\n".join(f" - {s}" for s in expected_list)
        hypothesis.note(f"Expected list:\n{formatted_list}")
        self.assertEqual(iterated_list, expected_list)

    def test_cartesian_product_lazy_iteration(self):
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

    def test_union_iteration(self):
        u = Union(
            {
                0: IntegerRange(1, 2, default_value=10),
                1: IntegerRange(1, 2, default_value=10),
                2: IntegerRange(1, 2),
            }
        )
        iterated_list = list(u)
        expected_list = [
            {0: 1, 1: 10},
            {0: 2, 1: 10},
            {0: 10, 1: 1},
            {0: 10, 1: 2},
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
            {0: 1, 1: 10},
            {0: 2},
            {0: 10, 1: 1},
            {1: 2},
            {1: 10, 2: 1},
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

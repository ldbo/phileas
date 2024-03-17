import unittest

import hypothesis
from hypothesis import given
from hypothesis import strategies as st

from phileas.iteration import (
    CartesianProduct,
    GeometricRange,
    IntegerRange,
    IterationLiteral,
    IterationTree,
    LinearRange,
    NumericRange,
    Sequence,
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
    start, end = draw(st.floats()), draw(st.floats())
    if start == end:
        return LinearRange(start, end, steps=1, default_value=1)
    else:
        return LinearRange(start, end, steps=draw(st.integers(2, 3)))


@st.composite
def geometric_range(draw):
    sign = draw(st.sampled_from([-1, 1]))
    start = sign * draw(st.floats(min_value=0, exclude_min=True))
    end = sign * draw(
        st.floats(min_value=0, exclude_min=True).filter(lambda e: abs(e * start) > 0)
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
        st.lists(children, min_size=1, max_size=3)
        | st.dictionaries(data_literal, children, min_size=1, max_size=3)
    )
    return Node(children)


iteration_tree = st.recursive(
    iteration_leaf, lambda children: iteration_tree_node(children), max_leaves=10
)


### Tests ###


class TestIteration(unittest.TestCase):
    ## Iteration leaves ##
    @given(linear_range() | geometric_range())
    def test_linear_and_geometric_range_length(self, r: LinearRange | GeometricRange):
        self.assertEqual(len(r), r.steps)

    def test_linear_range_iteration(self):
        r = LinearRange(0.0, 5.0, steps=6)
        self.assertEqual(list(r.iterate()), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_geometric_range_iteration(self):
        r = GeometricRange(1.0, 8.0, steps=4)
        self.assertEqual(list(r.iterate()), [1.0, 2.0, 4.0, 8.0])

    @given(st.integers(-10, 10), st.integers(-10, 10), st.integers(10))
    def test_integer_range_iteration(self, start: int, end: int, step: int):
        r = IntegerRange(start, end, step=step)
        iterator = r.iterate()
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
        self.assertEqual(list(sequence.iterate()), sequence.elements)

    ## Iteration nodes ##

    @given(iteration_tree)
    def test_iteration_tree_generation(self, tree: IterationTree):
        """
        This test is voluntarily left empty, in order to show when the iteration
        tree strategies are broken.
        """
        del tree

    @given(iteration_tree)
    def test_len_consistent_with_iterate(self, tree: IterationTree):
        try:
            n = len(tree)
            formatted_list = "\n".join(f" - {s}" for s in tree.iterate())
            hypothesis.note(f"The iterated list is \n{formatted_list}")
            self.assertEqual(n, len(list(tree.iterate())))
        except ValueError:
            return


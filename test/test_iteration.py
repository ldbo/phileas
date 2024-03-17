from hypothesis import strategies as st

from phileas.iteration import (
    IterationLiteral,
    NumericRange,
    LinearRange,
    GeometricRange,
    IntegerRange,
    Sequence,
    CartesianProduct,
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
    return NumericRange(draw(st.floats()), draw(st.floats()))


@st.composite
def linear_range(draw):
    start, end = draw(st.floats()), draw(st.floats())
    if start == end:
        return LinearRange(start, end, 1)
    else:
        return LinearRange(start, end, draw(st.integers(2, 3)))


@st.composite
def geometric_range(draw):
    sign = draw(st.sampled_from([-1, 1]))
    start = sign * draw(st.floats(min_value=0, exclude_min=True))
    end = sign * draw(
        st.floats(min_value=0, exclude_min=True).filter(lambda e: abs(e * start) > 0)
    )
    return GeometricRange(start, end, draw(st.integers(2, 3)))


@st.composite
def integer_range(draw):
    start, end = draw(st.integers(-3, 3)), draw(st.integers(-3, 3))
    if start == end:
        return IntegerRange(start, end, 1)
    else:
        return IntegerRange(start, end, draw(st.integers(2, 3)))


sequence = st.builds(Sequence, st.lists(data_tree, min_size=1, max_size=3))

iteration_leaf = st.one_of(
    iteration_literal(),
    numeric_range(),
    linear_range(),
    geometric_range(),
    integer_range(),
    sequence,
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


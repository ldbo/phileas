---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{code-cell} ipython3
:tags: [remove-cell]

import itertools
import math

from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt

from phileas.iteration import *
```

# Iteration trees

*Iteration trees* are used by Phileas to represent complex iterables over *data
trees* in a functional manner. In short, they are a way to represent and work
with iteration as data.

## Data trees

{py:data}`~phileas.iteration.DataTree` are the simplest trees that
Phileas works with. They represent what a simple JSON or YAML file is parsed
to. They are defined as nested Python `dict` and `list` objects.
Eventually, they contain `bool`, `str`, `int`, `float` or `None`
leaves, which are referred to as {py:data}`~phileas.iteration.DataLiteral`.

Actually, {py:data}`~phileas.iteration.DataLiteral` can also be of type
`_NoDefault`, which is a special sentinel value used to represent the default
value of {py:class}`~phileas.iteration.IterationTree` which do not have one.

:::{seealso}

For more information on default values, see [](#default-value).
:::

Non-`None` data literals are used as keys of mapped containers, and are
referred to as {py:data}`~phileas.iteration.Key`.

## Iteration trees: collections of data trees

{py:class}`~phileas.iteration.IterationTree` are iterables
over {py:data}`~phileas.iteration.DataTree` objects, that can be composed in
order to iterate over complex spaces in a diverse manner.

### Iteration

Iteration trees are meant to store collections of data trees and make them
easily accessible. They allow, for example, these statements:

```{code-cell} ipython3
iteration_tree = CartesianProduct({
    "parameter1": Sequence([1, 2, 3]),
    "parameter2": Sequence(["a", "b", "c"]),
})

for data_tree in iteration_tree:
    print(data_tree)

print(list(iteration_tree))

iterator = iter(iteration_tree)
print(f"The third value is {iterator[3]}")
```

Under the hood, iteration trees are Python iterables, and iteration is managed
by {py:class}`~phileas.iteration.TreeIterator`. They are usual Python one-way
iterators, with two additional features:

- **two-way iteration**: {py:meth}`~phileas.iteration.TreeIterator.reverse`
  changes the iteration direction of an iterator. This allows to *go back in
  iteration time*, for example to replay experiment iterations found to be
  invalid. The iteration direction is obtained by
  {py:meth}`~phileas.iteration.TreeIterator.is_forward`.
- **random access**: The iteration position can be modified to any supported
    index by {py:meth}`~phileas.iteration.TreeIterator.update`. Alternatively,
    you can use the `[]` operator. Additionally, the iteration position can be
    {py:meth}`~phileas.iteration.TreeIterator.reset` to its starting value. Be
    careful, the starting value depends on the iteration direction: resetting a
    forward iterator brings it back to position 0; resetting a backward
    iterator brings it to the last supported position. Random access allows to
    replay any iteration without the expensive cost of iterating over all its
    preceding indices.

:::{note}

Developers can use{py:class}`~phileas.iteration.OneWayTreeIterator` to implement
a two-way iterator from a one-way implementation. This is useful in situations where true random access is hard to design.
:::

### A recursive tree structure

Iteration trees have a recursive structure, where the different nodes correspond
to different Python classes.

#### Iteration leaves

The leaves of iteration trees represent simple collections, and are stored in
{py:class}`~phileas.iteration.IterationLeaf` objects.

You can store Python lists in {py:class}`~phileas.iteration.Sequence`, and
simple fixed values are represented by
{py:class}`~phileas.iteration.IterationLiteral`. Other iteration leaves,
including numeric ranges and randomly generated sequences, are available in
module{py:mod}`phileas.iteration.leaf`.

Here are some iteration leaves examples:

```{code-cell} ipython3
literal = IterationLiteral([1, 2, 3])
num_range = LinearRange(start=1, end=2, steps=12)
integer_range = IntegerRange(start=0, end=math.inf)
uniform = NumpyRNG()
bigint_uniform = UniformBigIntegerRng(low=0, high=1 << 32)
prime_number = PrimeRng(low=200, high=300)
```

#### N-ary nodes: iteration methods

Iteration leaves can then be composed in order to obtain complex iteration
behaviors, using *iteration methods*.
{py:class}`~phileas.iteration.IterationMethod` is the class of n-ary iteration
tree nodes: it has multiple children, and represents *how to iterate over all
of them at the same time*. These children are supplied either as a list, or as
a dictionary with {py:data}`~phileas.iteration.Key` keys.

:::{seealso}

Iteration methods are implemented in the
{py:mod}`phileas.iteration.node` module.
:::

##### Cartesian product

The simplest iteration method is
{py:class}`~phileas.iteration.CartesianProduct`, which is roughly equivalent to
nested `for` loops.

```{code-cell} ipython3
tree1 = Sequence(["1.1", "1.2"])
tree2 = Sequence(["2.1", "2.2"])
tree3 = Sequence(["3.1", "3.2"])

print("This is equivalent...")
for value1 in tree1:
    for value2 in tree2:
        for value3 in tree3:
            print(f"{value1=}, {value2=}, {value3=}")

print("...to that")
for value in CartesianProduct(dict(value1=tree1, value2=tree2, value3=tree3)):
    print(value)
```

With the `snake` argument, you can enforce that, from an iteration to the other,
the value of only one children changes.

```{code-cell} ipython3
tree1 = Sequence([1, 2, 3])
tree2 = Sequence(["a", "b", "c"])
for value in CartesianProduct(dict(tree1=tree1, tree2=tree2), snake=True):
    print(value)
```

:::{note}

Notice how only the iteration order is affected, and not the set of iterated
values. Generally speaking, iteration methods always represent the same set of
values.
:::

Since only one children value changes at each iteration, it is oftentimes not
necessary to return all the other fixed values. Iteration methods have a `lazy`
boolean parameter, which addresses this situation. When it is set, only the
children values that have changed since the last iteration are returned.

```{code-cell} ipython3
tree1 = Sequence([1, 2, 3])
tree2 = Sequence(["a", "b", "c"])
for value in CartesianProduct(dict(tree1=tree1, tree2=tree2), snake=True, lazy=True):
    print(value)
```

The `lazy` argument is effective only with `dict` children.

:::{note}

For now, iteration methods are not *required* to implement support for `lazy`.
:::

##### Union

While the cartesian product iteration method is equivalent to nested `for`
loops, {py:class}`~phileas.iteration.Union` behaves like `for` loops in series.
It guarantees that the values of each of its children are generated at least
once, but does so with the minimum number of generated elements.

```{code-cell} ipython3
tree1 = Sequence(["1.1", "1.2", "1.3"])
tree2 = Sequence(["2.1", "2.2", "2.3"])
tree3 = Sequence(["3.1", "3.2", "3.3"])

for value in Union(dict(value1=tree1, value2=tree2, value3=tree3)):
    print(value)
```

Notice how, after iteration over a children is done, it is reset to its initial
value. This is controlled by the `reset` parameter, which can have four
different values:

 - `"first"` (default) sets it to its first value, and "last" to its last value;
 - `"default"` sets it to its default value;
 - `None`, which is only valid with `dict` children, does not set the children
   value after iteration.

Similarly, the `preset` parameter controls the value of children whose iteration
has not started yet. It can have values `"first"` (default), `"default"` and
`None`, which have the same meaning as with `reset`.

To keep the children at their last value, and not presetting them, you can use:

```{code-cell} ipython3
tree1 = Sequence(["1.1", "1.2", "1.3"])
tree2 = Sequence(["2.1", "2.2", "2.3"])
tree3 = Sequence(["3.1", "3.2", "3.3"])

for value in Union(dict(value1=tree1, value2=tree2, value3=tree3), preset=None, reset="last"):
    print(value)
```

Finally, when `preset = "first"`, the first iteration of each children
(except for the first) might be redundant, since the first global iteration
already sets the values correctly. Parameter `common_preset` allows to get rid
of it:

```{code-cell} ipython3
tree1 = Sequence(["1.1", "1.2", "1.3"])
tree2 = Sequence(["2.1", "2.2", "2.3"])
tree3 = Sequence(["3.1", "3.2", "3.3"])

for value in Union(dict(value1=tree1, value2=tree2, value3=tree3), common_preset=True):
    print(value)
```

##### Zip

The {py:class}`~phileas.iteration.Zip` node behaves similarly as the built-in
{py:class}`zip` function: it iterates over all of its children at the same time.

```{code-cell} ipython
tree1 = Sequence(["1.1", "1.2", "1.3"])
tree2 = Sequence(["2.1", "2.2"])

for value in Zip([tree1, tree2]):
    print(value)
```

Here, iteration over `tree1` stops before its end is reached. If you want to
reach the end of all the children, you can specify the argument `stops_at="longest"`:

```{code-cell} ipython
tree1 = Sequence(["1.1", "1.2", "1.3"])
tree2 = Sequence(["2.1", "2.2"])

for value in Zip(dict(value1=tree1, value2=tree2), stops_at="longest"):
    print(value)
```

Note that we switched the type of the children to `dict`. It is required when
`stops_at = "longest"`, as some children elements might go missing.

```{code-cell} ipython
:tags: [raises-exception]

tree1 = Sequence(["1.1", "1.2", "1.3"])
tree2 = Sequence(["2.1", "2.2"])

for value in Zip([tree1, tree2], stops_at="longest"):
    print(value)
```

##### Pick

{py:class}`~phileas.iteration.Pick` behaves similarly as union iteration,
however it randomly picks the children which is being iterated over. You can
think of it as a randomized lazy union, with the notable exception that it can
accept infinite children.

```{code-cell} ipython3
tree1 = IntegerRange(start=1, end=math.inf)
tree2 = IntegerRange(start=1, end=math.inf)
tree3 = IntegerRange(start=1, end=math.inf)
pick = generate_seeds(Pick(dict(value1=tree1, value2=tree2, value3=tree3)))

for value in itertools.islice(pick, 10):
    print(value)
```

:::{note}

This is an example of a random iteration tree. They require to be seeded, which
is done here by {py:func}`~phileas.iteration.generate_seeds`.
[This section](#random-trees) covers random iteration trees.
:::

#### Unary nodes

Unary nodes only have a single child, and can do two things with it:

 - either they change its iteration order, or
 - they change the values it generates, in which case they are
   called *transform* nodes.

##### Shuffle

The {py:class}`~phileas.iteration.Shuffle` node applies a random permutation to
its child, which must have a finite length. It is another example of a
[random tree](#random-trees).

```{code-cell} ipython3

shuffle = generate_seeds(Shuffle(Sequence([1, 2, 3, 4, 5])))
print(list(shuffle))
```

##### First

The {py:class}`~phileas.iteration.First` node only returns the first elements of
its child.

```{code-cell} ipython3
first = First(IntegerRange(0, math.inf, 1), 10)
print(list(first))
```

##### Transform nodes

{py:class}`~phileas.iteration.Transform` nodes are used to apply functions that
modify the *value* of a tree during iteration, leaving its iteration order
unmodified.


Let us consider the an iteration tree which specifies a set of positions
expressed in radial coordinates. The radius `rho_m` is in meters, and the
angle `theta_deg` in degrees. However, we want to use cartesian coordinates,
for, let's say, configuring independent linear motors. The first solution is to
directly perform the coordinate transformation after iteration:

```{code-cell} ipython3
:tags: [hide-output]

tree_radial = CartesianProduct({
    "rho_m": LinearRange(1, 2, steps=3),
    "theta_deg": LinearRange(0, 45, steps=3)
})

for position_radial in tree_radial:
    theta_rad = position_radial["theta_deg"] * math.pi / 180
    position_cartesian = {
        "x": position_radial["rho_m"] * math.cos(theta_rad),
        "y": position_radial["rho_m"] * math.sin(theta_rad),
    }
    print(position_cartesian)
```

This solution can easily become unmaintainable, as parameters generation is
split between the iteration tree definition and its iteration logic.
{py:class}`~phileas.iteration.Transform` are meant to solve this issue, by
incorporating data tree modifications directly in the iteration tree.

We could refactor the former example by using the
{py:class}`~phileas.iteration.FunctionalTransform` node, which simply wraps a
Python function:

```{code-cell} ipython3
:tags: [hide-output]

tree_radial = CartesianProduct({
    "rho_m": LinearRange(1, 2, steps=3),
    "theta_deg": LinearRange(0, 45, steps=3)
})

def radial_to_cartesian(t: DataTree) -> DataTree:
    theta_rad = t["theta_deg"] * math.pi / 180
    return {
        "x": t["rho_m"] * math.cos(theta_rad),
        "y": t["rho_m"] * math.sin(theta_rad),
    }

tree_cartesian = FunctionalTranform(tree_radial, radial_to_cartesian)

for position_cartesian in tree_cartesian:
    print(position_cartesian)
```

Other transform nodes are available in the {py:mod}`phileas.iteration.node`
module. They include {py:class}`~phileas.iteration.Lazify` which behaves
similarly as the `lazy` argument of iteration methods, only keeping changed
key of `dict` children, and its converse
node {py:class}`~phileas.iteration.Accumulator`.

#### Random trees

{py:class}`~phileas.iteration.RandomTree` provides an interface for iteration
trees which have random behavior. As iteration trees are designed to be
deterministic, those trees are not actually random. Instead, they provide
deterministic pseudo-random behavior depending only on their
{py:attr}`~phileas.iteration.RandomTree.seed` attribute. You can set it
manually, yet it is not advised. Instead, it is more convenient to use the
{py:func}`~phileas.iteration.random.generate_seeds` utility function, which
generates all the seeds of a tree, guaranteeing that they will be different.

Iteration on various distributions is provided by
{py:class}`~phileas.iteration.NumpyRNG`, which uses a random number generator
provided by numpy.

```{code-cell} ipython3

rng = generate_seeds(NumpyRNG(distribution=np.random.Generator.normal, kwargs=dict(loc=1, scale=0.2)))

ax = plt.subplot()
_ = ax.hist(list(itertools.islice(rng, 10000)), bins=100)
```

Random iteration is provided by the [](#shuffle) and [](#pick) nodes.

### Breaking the recursive tree structure with configurations

Iteration trees are recursive and local by nature: iterations over siblings of a
node are independent. However, sometimes, global iteration tree behavior are
required. Some can be obtained using transform nodes, yet this is usually hacky
and cost expensive. Using *configurations*, on the other end, is a built-in
feature of iteration tree which is efficient and permits global behavior.

Let us study a simple example. We want to generate data for an instrument in two
phases:

- in phase 1, its parameter varies;
- during the multiple iterations of phase 2, its parameter is fixed.

This could be done with two iteration trees. However, this would not be
compatible with a single experiment configuration file. Alternatively, a
transform node could be used, to pick between two entries, but this would be
hard to modify and understand.

Configurations can solve this the following way:

```{code-cell} ipython3

tree = CartesianProduct({"instrument": Configurations({
    "1": CartesianProduct({
        "parameter": Sequence([10, 20, 30])
    }),
    "2": CartesianProduct({
        "parameter": IterationLiteral(0),
        "iteration": IntegerRange(1, 3)
    }),
})})

for data_tree in tree:
    print(data_tree)
```

Notice the log, which reminds us that the tree is configurable. Under the hood,
{py:meth}`~phileas.iteration.IterationTree.unroll_configurations` is used to
convert a configurable tree to a simple tree without any configuration. This is
done by transforming the tree to a simple union of its configurations
[^unroll-configurations-internals].

[^unroll-configurations-internals]: Actually, the
{py:meth}`~phileas.iteration.MoveUpTransform` is also used to modify the shape
of the tree afterward.

:::{note}

This shows that configurations are not strictly required. However, their use is
convenient, and prevents some common mistakes.
:::

{py:meth}`~phileas.iteration.Configurations` node can be configured with two
parameters:

 - `insert_name` keeps the name of the selected configuration, alongside the
   actual configuration value;
 - `move_up` insert the configuration value one level up.

```{code-cell} ipython3
:tags: [hide-input]

tree_insert_name = CartesianProduct({"instrument": Configurations({
    "1": CartesianProduct({
        "parameter": Sequence([10, 20, 30])
    }),
    "2": CartesianProduct({
        "parameter": IterationLiteral(0),
        "iteration": IntegerRange(1, 3)
    }),
}, insert_name=True)})

tree_move_up = CartesianProduct({"instrument": Configurations({
    "1": CartesianProduct({
        "parameter": Sequence([10, 20, 30])
    }),
    "2": CartesianProduct({
        "parameter": IterationLiteral(0),
        "iteration": IntegerRange(1, 3)
    }),
}, move_up=True)})

print("With insert_name...")
for data_tree in tree_insert_name:
    print(data_tree)

print("\nWith move_up...")
for data_tree in tree_move_up:
    print(data_tree)
```

With `insert_name`, the actual configuration is inserted in the `_configuration`
key. If it is combined with `move_up`, it is inserted where the configuration
used to be, here in `"instrument"`:

```{code-cell} ipython3
:tags: [hide-input]

tree_move_up_insert_name = CartesianProduct({"instrument": Configurations({
    "1": CartesianProduct({
        "parameter": Sequence([10, 20, 30])
    }),
    "2": CartesianProduct({
        "parameter": IterationLiteral(0),
        "iteration": IntegerRange(1, 3)
    }),
}, move_up=True, insert_name=True)})

print("With move_up and insert_name...")
for data_tree in tree_move_up_insert_name:
    print(data_tree)
```

Individual configurations can also be obtained directly, without resorting to
iteration. Each iteration tree has a
{py:attr}`~phileas.iteration.IterationTree.configurations` attribute which
contains the set of its configuration names.
{py:meth}`~phileas.iteration.IterationTree.get_configuration` can be used to
query them individually.

### Access and modifications

#### Access with the `[]` operator

Internal nodes of iteration trees can be accessed as if they where nested `dict`
or `list` objects, using the `[]` operator. This is useful, especially for
shallow trees, as in the following example:

```{code-cell} ipython3

tree = CartesianProduct([Sequence([1, 2, 3]), Sequence(["a", "b", "c"])])
print(f"The first element of the second child is {tree[1][0]!r}.")
```

#### Access and modifications with the path API

However, as iteration trees becomes deeper, accessing internal nodes becomes
cumbersome. Indeed, `[index1]...[indexN]` chains are often poorly readable.
Furthermore, they don't allow tree modifications.

The so-called *path API* solves these issues. A node that would be accessed
through the indexing chain `[index1]...[indexN]` is said to be located at the
{py:data}`~phileas.iteration.ChildPath` `[index1, ..., indexN]`. Then,
different methods can be used to get and modify the structure of the tree,
using path arguments.

The different available methods are:

- {py:meth}`~phileas.iteration.IterationTree.get` which returns the node at a
  given path;
- {py:meth}`~phileas.iteration.IterationTree.with_params` which changes some
  parameters of a node;
- {py:meth}`~phileas.iteration.IterationTree.replace_node` which replaces the
  type of node;
- {py:meth}`~phileas.iteration.IterationTree.insert_child` which adds or removes
  a node;
- {py:meth}`~phileas.iteration.IterationTree.remove_child` which removes a node;
- {py:meth}`~phileas.iteration.IterationTree.insert_unary` which inserts a
  unary node as a parent of another node.

Note that a purely functional style is used: an iteration tree object is a
frozen {py:class}`~dataclasses.dataclass` instance, and its attributes must not
be modified. However, a new, modified, iteration tree can be derived from it.
This prevents side-effects bugs, and allows to chain tree modifications as in
the following example.

```{code-cell} ipython3
tree = CartesianProduct([Sequence([1, 2, 3]), Sequence(["a", "b", "c"])])
new_tree = (
    tree.insert_unary([0], Shuffle)               # Shuffle the first child
    .with_params([], snake=True)                  # Enable snake for the root
    .insert_child([1], Sequence([True, False]))   # Replace the second child
)
pprint(new_tree)
```

:::{attention}

Iteration trees must be considered as immutable objects. However, effectively,
there are ways to modify their state. They should not be used, unless for a
really good reason. In this case, side-effects might occur, unexpectedly
violating assumptions and probably breaking your code. Be cautious !
:::

### Default value

Each iteration tree has a default value, which is accessed with the
{py:meth}`~phileas.iteration.IterationTree.default` method. Some iteration
methods use it for iteration (in particular,
{py:class}`~phileas.iteration.Union`). It is also useful to represent a safe
state of an experiment, where it can get back in case of an unexpected event,
or after measurements are done. In this sense, it is another way of breaking
the local nature of iteration trees, beside configurations.

## Pseudo data trees

Iteration trees represent complex collections alongside the way to iterate over
them. Yet, it can sometimes be inconvenient to use them when only iteration
leaves are of interest. On the other hand, data trees are simple Python objects
that are easy to work with, but they miss the information stored in iteration
nodes and in most iteration leaves.
{py:data}`~phileas.iteration.PseudoDataTree` are an intermediate type that is
useful in these cases.

Similarly to data trees, they consist of nested `dict` and `list`, so they don't
convey any information about how to combine nodes. However, their leaves are
either data literals - `None`, `bool`, `str`, `int` or `float` - or actual
iteration leaves. You can obtain them
using {py:meth}`~phileas.iteration.IterationTree.to_pseudo_data_tree`.

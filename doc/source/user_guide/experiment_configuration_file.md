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
# Experiment configuration file

```{code-cell} ipython3
:tags: [remove-cell]

from pprint import pprint

from phileas.parsing import load_iteration_tree_from_yaml_file
```

The experiment configuration file describes the instruments required by an
experiment, their parameter space and how to iterate through it.

```{code-block} yaml
:caption: Typical `experiment.yaml` content

description: This experiment does this and that

instrument1:
    interface: actuator1
    parameter1: value1
    parameter2: value2

instrument2:
    interface: sensor1
    parameter: value

    connections:
        - instrument3

instrument3:
    parameter: value
```

## Purpose

This file should contain most of the *parameters* required by an experiment, be
it to do it for the first time or to replicate or share it. This includes the
parameters of the instruments used on the bench, how they are connected
together, and any other information useful for the experimenter
(description, expected results, manual configurations, *etc*.).

## Syntax

It is a YAML file, containing a top level mapping. During the initialization of
an {py:class}`~phileas.factory.ExperimentFactory`, it is parsed to an
{py:class}`~phileas.iteration.IterationTree` which is then stored in its
{py:attr}`~phileas.factory.ExperimentFactory.experiment_config` attribute.

In order to ease the representation of the different iteration methods and
leaves, custom YAML types are supported. Whereas YAML only supports lists,
mapping, integer, floats, booleans, strings and none values, this file can also
represent Phileas objects. They are identified by a `!` prefix. For example, a
sequence of values can be represented by `!sequence [1, 2, 3]`.

:::{seealso}

See [](#custom-yaml-types) for more details about the supported types.
:::

## Instrument entries

Each of the mapping entries with an `interface` key represents an experiment
instrument. Its value is matched with the
{py:attr}`~phileas.factory.Loader.interfaces` values of bench instruments'
loaders. Think of it this way:

- in a bench configuration file, the set of available instruments is listed;
- in the experiment configuration file, you list the requirements of an
  instrument (*eg*. it must be an oscilloscope);
- Phileas takes care of providing you with a suitable instrument.

The other non-reserved entries of the instrument represent an
{py:class}`~phileas.iteration.IterationTree` which contains the set of its
configurations. You can use the methods
{py:meth}`~phileas.factory.ExperimentFactory.configure_instrument` and
{py:meth}`~phileas.factory.ExperimentFactory.configure_experiment` of
{py:class}`~phileas.factory.ExperimentFactory` to apply them. Internally, they
use {py:meth}`phileas.factory.Loader.configure`.

## Bench instrument filtering

Multiple bench instruments can have the same interface. For example, there might
be multiple motors or lights on the same bench. In this case, they would all
expose the same interfaces, which would prevent Phileas from knowing which
instrument to provide you. This issue can be solved by the `filter` entry. It
contains a set of attributes and values to be matched against the bench
instruments attributes.

Consider the following bench file:

```{code-block} yaml
:caption: `bench.yaml` with instruments exposing the same interfaces

light-motor:
    loader: brandXX-modelXX
    moves: light

probe-motor:
    loader: brandYY-modelYY
    moves: probe
```

Let us say that only the probe motor is used, while the light remains
stationary. Both loaders `brandXX-modelXX` and `brandYY-modelYY` expose
the same `motor` interface. Using filters, this experiment configuration file
selects only `probe-motor`:

```{code-block} yaml
:caption: `experiment.yaml` using filters to chose a bench instrument

probe-position:
    interface: motor
    filter:
        moves: probe

    x: 0.5
    y: 1.2
    z: 0
```

`probe-position` is matched to `probe-motor`, because it is the only suitable
bench instrument with the attribute `moves = probe`.

## Connections

An experiment is represented by its instruments, how they are configured, but
also how they interact together. An experiment configuration file can represent
those interactions with so-called *connections*, which are the edges in the
global interaction graph of the experiment, where instruments are nodes.

The optional top-level `connections` entry contains the list of the
interaction graph edges. Each edge contains:

- a mandatory origin, stored in the `from` entry,
- a mandatory destination, stored in the `to` entry,
- and, optionally, arbitrary data stored in its `attributes` entry.

The origin and destination of an edge are usually instruments. You can suffix
them with *ports*, separating them with dots. For example, `oscilloscope.chA`
states that the connections involves the A channel of the oscilloscope.

```{code-block} yaml
:caption: Global connections definition

connections:
    - from: probe
      to: oscilloscope.chA
    - from: holder
      to: probe
      attributes: holds
```

In this example, the probe is connected to the channel A of the oscilloscope,
and is hold by `holder`.

However, it is not always convenient to define the experiment graph in a
distinct entry, separating it from the instruments. In these cases, you can
define interactions directly in the entry of an instrument. The instrument
entry `connections` can be used similarly as the top-level entry. However, you
can use a simplified notation: if the origin or destination of an edge is the
instrument itself, you can simply omit the instrument name, starting directly
at its first port.

```{code-block} yaml
:caption: Instrument connections definition

temperature-measurement:
    interface: oscilloscope
    connections:
        - from: chA
          to: temperature-probe
```

This examples states that channel A of the oscilloscope is connected to the
temperature probe.

This experiment graph is built by Phileas, but it does not use it. You can
access it through
{py:meth}`~phileas.factory.ExperimentFactory.get_experiment_graph`.

## Documentation

Phileas ignores all the top-level entries which are not mappings with key
`interface`. It is advised to document the experiment with them. For example,
you can add a `description` entry, which gives a unique `name` to the
experiment, describes the tested `hypothesis`, its `measurements` and the
`expected-results`.

This whole file is parsed and made available in the factory
{py:attr}`~phileas.factory.ExperimentFactory.experiment_config`. You can then
retrieve that information in simple nested Python `dict` and `list` by using
{py:meth}`~phileas.iteration.IterationTree.to_pseudo_data_tree`.

## Custom YAML types

The experiment configuration file is parsed to an
{py:class}`~phileas.iteration.IterationTree`, which is built using the
different Python classes defined in {py:mod}`~phileas.iteration`. Custom YAML
data types can be used to represent these. They are identified by a `!`
prefix.

### Cartesian product

By default, a YAML mapping or sequence is parsed as a cartesian product:

```{code-cell} ipython3
:tags: [hide-input]

content = """
parameter1: !sequence [1, 2, 3]
parameter2: !sequence [a, b, c]
"""
tree = load_iteration_tree_from_yaml_file(content)
print(f"{content}\n\t\t|\n\t\tv\n")
pprint(tree)
```

You can also directly invoke `!product`, which allows to set the node parameters, prefixed by an underscore:

```{code-cell} ipython3
:tags: [hide-input]

content = """
!product
    _lazy: true
    _snake: true
    parameter1: !sequence [1, 2, 3]
    parameter2: !sequence [a, b, c]
"""
tree = load_iteration_tree_from_yaml_file(content)
print(f"{content}\n\t\t|\n\t\tv\n")
pprint(tree)
```

### Other iteration methods

Similarly, you can use `!union`, `!pick` and `!configurations` to build
{py:class}`~phileas.iteration.Union`, {py:class}`~phileas.iteration.Pick` and
{py:class}`~phileas.iteration.Configurations`.

### Unary nodes

`!shuffle` builds {py:class}`~phileas.iteration.Shuffle` node, and requires its
only child to be stored in the `child` key:

```{code-cell} ipython3
:tags: [hide-input]

content = """
!shuffle
    child: !sequence [1, 2, 3]
"""
tree = load_iteration_tree_from_yaml_file(content)
print(f"{content}\n\t\t|\n\t\tv\n")
pprint(tree)
```

For now, transform nodes are not supported. You have to add them to the iteration tree manually, from a Python script.

### Iteration leaves

`!sequence` builds {py:class}`~phileas.iteration.Sequence` nodes. It can be used in the short form `!sequence [a, b, c]`, or its long form

```{code-block} yaml
!sequence
    elements: [a, b, c]
    default: z
```

`!range` is often used, as it represents different kinds of numeric ranges. It
expects a `start` and `end` parameters, and either

 - `steps`, the total number of points in the range, or
 - `resolution`, the maximum distance between two points in the range.

`!random` represent random distribution, and is parsed to
{py:class}`~phileas.iteration.NumpyRNG`. It expects a `distribution` argument,
which is the string name of a numpy distribution. You can find them in
{py:class}`numpy.random.Generator`. Its parameters are given in the
`parameters` field, as a mapping. You can additionally give it a finite `size`, and a `default` value.

## Template generation

A bench template file can be generated automatically by Phileas. See `python -m
phileas generate experiment -h` for the available options.

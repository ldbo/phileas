---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Example: side-channel analysis

This example illustrates how to use Phileas to take measurements for a typical
side-channel analysis experiment.

```{code-cell} ipython3
---
tags: [hide-cell]
mystnb:
  code_prompt_show: Imports and initial setup
  code_prompt_hide: Imports and initial setup
---
from pprint import pprint
import logging
import sys

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

np.random.seed(int.from_bytes("sca.md".encode("utf-8")) % (1 << 32))

from phileas.factory import ExperimentFactory
from phileas.iteration import generate_seeds
import phileas.mock_instruments

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
```

## Experimental setup

A simple current side-channel dataset is acquired, which requires the following
experimental setup:

 - the device under test consists of an electronic component with an embedded
   implementation of the PRESENT cypher.
 - A control computer can communicate with the tested device through a serial
   link. To establish the connection, it requires a baudrate and a peripheral
   path.
 - Its current consumption is monitored by plugging a current probe to its power
   supply.
 - The probe output is sampled by an oscilloscope, which is synchronized with
   the target algorithm using an external trigger.

## Bench description

The first step to prepare the data acquisition is to **describe the
instruments** available on the bench, and **how to establish a connection with
them**. This is done by the *bench configuration file*:

```{code-cell} ipython3
bench = """
simulated_present_dut:
  loader: phileas-mock_present-phileas
  device: /dev/ttyUSB1
  baudrate: 115200
  probe: simulated_current_probe

simulated_current_probe:
  loader: phileas-current_probe-phileas

simulated_oscilloscope:
  loader: phileas-mock_oscilloscope-phileas
  probe: generic
  probe-name: simulated_current_probe
  width: 32
"""
```

Here, we declare that our bench is composed of three *bench instruments*: the
embedded PRESENT implementation `simulated_present_dut` is connected to a
current probe `simulated_current_probe`. An oscilloscope
`simulated_oscilloscope` records the output of the probe.

:::{attention}

Here, we store the content of the bench configuration file in the `bench`
variable, as it is easier to work with it from a notebook. However, you should
usually store this in a YAML file, say `bench.yaml`. Indeed, it is then easier
to share, maintain and modify, and this way it does not pollute the acquisition
script.
:::

## Experiment description

We can then describe **what instruments are required** and **how to configure
them**, which is done by the *experiment configuration file*. If we want to
perform a single encryption, with a unique key and plaintext, we could do it
with:

```{code-cell} ipython3
experiment = """
dut:
  interface: present-encryption
  key: 12
  plaintext: 1

oscilloscope:
  interface: oscilloscope
  amplitude: 10
"""
```

This configuration requires two *experiment instruments*. `dut` must be a device
which supports the `present-encryption` interface. Its configuration consists
of the specified key and plaintext. Similarly, `oscilloscope` must be an
oscilloscope device.

:::{attention}

Here, we store the content of the experiment configuration file in the
`experiment` variable, as it is easier to work with it from a notebook.
However, you should usually store this in a YAML file, say `experiment.yaml`.
Indeed, it is then easier to share, maintain and modify, and this way it does
not pollute the acquisition script.
:::

## Instruments instantiation

We can finally let Phileas handle the instantiation of the instruments, which is
done by {py:class}`~phileas.factory.ExperimentFactory`:

```{code-cell} ipython3
factory = ExperimentFactory(bench, experiment)
factory.initiate_connections()
print("Here are the experiment instruments:")
pprint(factory.experiment_instruments)

oscilloscope = factory.experiment_instruments["oscilloscope"]
dut = factory.experiment_instruments["dut"]
```

The different log records let us follow what is going on under the hood.

  1. First, *loaders* are assigned to the different bench instruments. A loader
  is a sub-class of {py:class}`~phileas.factory.Loader` which handles the
  instantiation and configuration of an instrument. It is obtained by matching
  the `loader` field of the bench instrument entries to the
  {py:attr}`~phileas.factory.Loader.name` attribute of the registered loader
  classes. For more information about loaders, see
  [this page](/user_guide/implementing_loaders).

  2. Then, the instruments required in the experiment configuration file are
  matched to the available bench instruments. The `interface` required by each
  experiment instrument is compared to the
  {py:attr}`~phileas.factory.Loader.interfaces` field of the loaders previously
  matched with the bench instruments. If only a single bench instrument
  correspond to the required interface, it is assigned to the experiment
  instrument. Here, `dut` (from `experiment`) is matched with
  `simulated_aes_dut` (from `bench`).

  3. Finally,
  {py:meth}`~phileas.factory.ExperimentFactory.initiate_connections`
  instantiates all the required drivers, using the connection information
  available in the bench configuration file. You can access these instruments
  in {py:attr}`~phileas.factory.ExperimentFactory.experiment_instruments`.


:::{hint}

Phileas exclusively uses the `phileas` logging handler. However, in Python, logs
are by default not displayed. The easiest way to turn this on is to use
{py:func}`logging.basicConfig`.

```python
import logging

logging.basicConfig(level=logging.INFO)
```
:::

We can now configure the instruments, and get a side-channel trace:

```{code-cell} ipython3
factory.configure_experiment()
cyphertext = dut.encrypt(factory.experiment_config["dut"]["plaintext"].value)
print(f"Ciphertext: {cyphertext:016x}")
trace = oscilloscope.get_measurement()

plt.plot(trace)
plt.show()
```

See how the experiment configuration has been parsed and is stored in
{py:attr}`~phileas.factory.ExperimentFactory.experiment_config`. We can
directly access it in order to retrieve the plaintext. However, the plaintext
is not directly an `int`, but it encapsulates one in its `value` field:

```{code-cell} ipython3
factory.experiment_config["dut"]["plaintext"]
```

## Representing multiple configurations

It is now easy to transform the simple experiment configuration to one which
represents a whole set of parameters. Different custom YAML types, identified
by a `!` prefix, can be used. For example, to use different plaintext values,
we could specify:

```{code-cell} ipython3
experiment = """
dut:
  interface: present-encryption
  key: 12
  plaintext: !range
    start: 1
    end: 3
    resolution: 1

oscilloscope:
  interface: oscilloscope
  amplitude: 10
"""

factory = ExperimentFactory(bench, experiment)
factory.initiate_connections()
pprint(factory.experiment_instruments)

oscilloscope = factory.experiment_instruments["oscilloscope"]
dut = factory.experiment_instruments["dut"]
```

The experiment configuration still requires the same set of instruments.
However, multiple instruments configurations are represented by the experiment
configuration. They are available in the
{py:attr}`~phileas.factory.Factory.experiment_config` attribute of the
experiment factory, which is an iterable:

```{code-cell} ipython3
for config in factory.experiment_config:
  print(config)
  factory.configure_experiment(config)
```

However, the factory instantiated some new instruments, which was not required
here. We can bypass this by directly parsing the experiment configuration:

```{code-cell} ipython3
configurations = phileas.parsing.load_iteration_tree_from_yaml_file(experiment)
pprint(configurations)
```

## Measurements acquisition and storage

+++

Let us now see how to manage the measurements, and store them conveniently.
Phileas provides some interoperability functions to use Pandas and Xarray
containers. Xarray is often more convenient to use, as the parameters spaces
are often simple cartesian products.
{py:func}`~phileas.iteration.utility.iteration_tree_to_xarray_parameters` can
be used to prepare the datasets. It reads the experiment configurations, and
produces the arguments required to initialize a{py:class}`xarray.DataArray` or
{py:class}`xarray.Dataset`:

```{code-cell} ipython3
coords, dims_name, dims_shape = phileas.iteration.utility.iteration_tree_to_xarray_parameters(configurations)
traces = xr.DataArray(coords=coords, dims=dims_name)
traces
```

However, the oscilloscope outputs 1D arrays, so we should add a dimension, say
`oscilloscope.sample`, to `traces`.

```{code-cell} ipython3
coords["oscilloscope.sample"] = list(range(trace.shape[0]))
dims_name.append("oscilloscope.sample")
dims_shape.append(trace.shape[0])
traces = xr.DataArray(coords=coords, dims=dims_name)
traces
```

We can now populate `traces` with measurements. Converting a configuration to a
dictionary that can be used by Xarray for indexing can be done by
{py:func}`~phileas.iteration.utility.data_tree_to_xarray_index`:

```{code-cell} ipython3
for config in configurations:
    index = phileas.iteration.utility.data_tree_to_xarray_index(config, dims_name)
    factory.configure_experiment(config)
    dut.encrypt(config["dut"]["plaintext"])
    trace = oscilloscope.get_measurement()
    traces.loc[index] = trace
```

```{code-cell} ipython3
traces.plot(x="oscilloscope.sample", col="dut.plaintext")
plt.show()
```

We can see that the traces are quite noisy. PRESENT consists of 32 similar
rounds, so we could expect to notice 32 similar patterns. Let us average the
measurements, to get rid of some noise and hopefully expose these patterns.

## Traces de-noising

Before we do it, we should disable outputting the logs to the standard output.
Indeed, now that we have checked that the experiment runs properly, we don't
need to verify them anymore. Since averaging requires a lot more
configurations, they would totally clutter the output. We can disable those
which are less severe than warnings with

```{code-cell} ipython3
logger = logging.getLogger("phileas")
logger.setLevel(logging.WARNING)
```

:::{note}

Please note that in an actual experiment, depending on the surrounding workflow,
it might be better to redirect the logs to an external file instead.
:::

Let us now carry out 1000 measurements for each of the DUT configurations. We
can do this by adding a new parameter, say `iteration`, to the top-level
experiment configuration:

```{code-cell} ipython3
experiment = """
dut:
  interface: present-encryption
  key: 12
  plaintext: !sequence [12345, 67890]

oscilloscope:
  interface: oscilloscope
  amplitude: 10

iteration: !range
  start: 1
  end: 1000
  resolution: 1
"""

configurations = phileas.parsing.load_iteration_tree_from_yaml_file(experiment)

coords, dims_name, dims_shape = phileas.iteration.utility.iteration_tree_to_xarray_parameters(configurations)
coords["oscilloscope.sample"] = list(range(trace.shape[0]))
dims_name.append("oscilloscope.sample")
dims_shape.append(trace.shape[0])
traces = xr.DataArray(coords=coords, dims=dims_name)

for config in configurations:
    index = phileas.iteration.utility.data_tree_to_xarray_index(config, dims_name)
    factory.configure_experiment(config)
    dut.encrypt(config["dut"]["plaintext"])
    trace = oscilloscope.get_measurement()
    traces.loc[index] = trace

traces
```

See how a new coordinate, `iteration`, has been automatically added to the
Xarray container. We can then display the averaged traces with:

```{code-cell} ipython3
traces.mean("iteration").plot(x="oscilloscope.sample", col="dut.plaintext")
plt.show()
```

It's better, the rounds are now visible. However, we are only using two
different plaintexts, which is not enough for most side-channel attacks. Let's
see how to generate a lot more !

## Generating random data

We could generate each of the 64-bit plaintexts supported by PRESENT. However,
this would be too long to be practically feasible. Let us see how to randomly
sample them uniformly in the message space.

The `!random` node represents values generated by the Numpy random number
generators. We can generate random 64-bit integers with this sample:

```{code-cell} ipython3
pt_yaml = """
!random
  distribution: integers
  parameters:
    dtype: uint64
    low: 0
    high: 0x10000000000000000
  size: 10
"""

pt = phileas.parsing.load_iteration_tree_from_yaml_file(pt_yaml)
```

The distribution refers to {py:meth}`numpy.random.Generator.integers`, whose
arguments are taken from `parameters`. `size` specifies the number of generated
values, which will be infinitely many if it is `none` or not given.

Let us see what values this contains:

```{code-cell} ipython3
:tags: [raises-exception]

list(pt)
```

With Phileas, iteration over all configurations must be deterministic, so *true*
randomness cannot be achieved. To circumvent this, deterministic pseudo-random
number generators are used, which generate seemingly random numbers in a
repeatable way. However, they must be seeded before use: you can use
{py:func}`~phileas.iteration.random.generate_seeds`, which populate the random
nodes of a tree with seeds that depend on their location, and a user-supplied
salt. This salt can then be used to generate multiple different experiments
with the same configuration file.

```{code-cell} ipython3
print("With salt='first'")
pprint(list(phileas.iteration.random.generate_seeds(pt, salt="first")))
print("\nWith salt='second'")
pprint(list(phileas.iteration.random.generate_seeds(pt, salt="second")))
```

:::{note}

The RNG from Numpy can only work with at most 64-bit wide integers. If you want
to generate arbitrarily big ones, you can use `!random_uniform_bigint`
instead.
:::

## Randomizing the iteration order

Let's suppose that there is a low-frequency noise source in the experimental
setup. It affects the measurements, and its value slowly changes in time. Thus,
the measurements related to the same plaintext will likely suffer from the same
noise value. This will bias the statistical algorithms used to carry out the
side-channel attack.

A solution against this is to randomize the iteration order. Instead of
recording 1000 traces related with the first plaintext, then 1000 linked to the
second, and so on, we can pick the plaintexts randomly, and ensure that each
one is picked 1000 times.

With Phileas, this is done by the `!shuffle` node:

```{code-cell} ipython3
pt_yaml = """
!shuffle
  child: !product
    plaintext: !sequence [111, 222, 333]
    iteration: !range
      start: 1
      end: 3
      resolution: 1
"""

pt = phileas.iteration.random.generate_seeds(phileas.parsing.load_iteration_tree_from_yaml_file(pt_yaml))
pprint(list(pt))
```

## Wrapping up and getting further

We have seen how to use Phileas to:

 - automatically instantiate and configure the instruments of the experiment
   bench;
 - represent simple experiments with configuration files;
 - create storage containers for the acquired data;
 - generate random data values and
 - shuffle the configurations iteration order.

We can wrap all of this in a single experiment file and script:

```{code-cell} ipython3
from phileas.parsing import load_iteration_tree_from_yaml_file
from phileas.iteration.utility import (
    iteration_tree_to_xarray_parameters,
    data_tree_to_xarray_index
)
from phileas.iteration.random import generate_seeds

experiment = """!shuffle
  child:
    dut:
      interface: present-encryption
      key: !random
        distribution: integers
        parameters:
          dtype: uint64
          low: 0
          high: 0x10000000000000000
        size: 10
      plaintext: !random
        distribution: integers
        parameters:
          dtype: uint64
          low: 0
          high: 0x10000000000000000
        size: 10

    oscilloscope:
      interface: oscilloscope
      amplitude: 10

    iteration: !range
      start: 1
      end: 10
      resolution: 1
"""

configurations = generate_seeds(load_iteration_tree_from_yaml_file(experiment))

coords, dims_name, dims_shape = iteration_tree_to_xarray_parameters(configurations)
dut.key = 12
dut.encrypt(0)
trace = oscilloscope.get_measurement()
coords["oscilloscope.sample"] = list(range(trace.shape[0]))
dims_name.append("oscilloscope.sample")
dims_shape.append(trace.shape[0])
traces = xr.DataArray(coords=coords, dims=dims_name)

for config in configurations:
    factory.configure_experiment(config)

    dut.encrypt(config["dut"]["plaintext"])
    trace = oscilloscope.get_measurement()

    index = data_tree_to_xarray_index(config, dims_name)
    traces.loc[index] = trace

traces
```

You can discover new features of Phileas by browsing through the [user guide]
(/user_guide/index), or by exploring [other examples](/examples/index).

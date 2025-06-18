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

# Example: side-channel analysis

```{code-cell} ipython3
:tags: [remove-cell]

from pprint import pprint
import logging
import sys

from phileas.factory import ExperimentFactory
from phileas.iteration import generate_seeds
import phileas.mock_instruments

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
```

This example illustrates how to use Phileas to take measurements for a typical
side-channel analysis experiment.

## Experimental setup

A simple current side-channel dataset is acquired, which requires the following
experimental setup:

 - the device under test consists in an electronic component with an embedded
   implementation of the AES cypher.
 - A control computer can communicate with the tested device through a serial
   link. To establish the connection, it requires a baudrate and a peripheral
   path.
 - Its current consumption is monitored by plugging a current probe to its power
   supply.
 - The probe output is sampled by an oscilloscope, which is synchronized with
   the target algorithm using an external trigger.

Experimental setup:
    - DUT performing some operations: RSA? AES?
    - Oscilloscope
    - Plugged to a probe
    - Probe measuring the current supplied to the DUT
    - Current model: gaussian noise, and increased consumption for each
      round/exponent bit

## Bench description

The first step to prepare the data acquisition is to **describe the
instruments** available on the bench, and **how to connect to them**. This is
done by the *bench configuration file*:

```{code-cell} ipython3
bench = """
simulated_aes_dut:
  loader: phileas-mock_aes-phileas
  device: /dev/ttyUSB1
  baudrate: 115200
  probe: simulated_current_probe

simulated_current_probe:
  loader: phileas-current_probe-phileas
  gain: 1.0e-2

simulated_oscilloscope:
  loader: phileas-mock_oscilloscope-phileas
  probe: generic
  probe-name: simulated_current_probe
"""
```

Here, we declare that our bench is composed of three *bench instruments*: the
embedded AES implementation `simulated_aes_dut` is connected to a current probe
`simulated_current_probe`. An oscilloscope `simulated_oscilloscope` records the
output of the probe.

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
  interface: aes
  key: 12
  plaintext: 1

oscilloscope:
  interface: oscilloscope
  amplitude: 1
"""
```

This configuration requires two *experiment instruments*. `dut` must be an `aes`
device, configured with the specified key and plaintext. Similarly,
`oscilloscope` must be an oscilloscope device.

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
  available in the bench configuration file. You can access these instruments in {py:attr}`~phileas.factory.ExperimentFactory.experiment_instruments`.


:::{hint}

Phileas exclusively uses the `phileas` logging handler. However, in Python, logs are by default not displayed. The easiest way to turn this on is to use {py:func}`logging.basicConfig`.

```python
import logging

logging.basicConfig(level=logging.INFO)
```
:::

We can now configure the instruments, and get a side-channel trace:

```{code-cell} ipython3
factory.configure_experiment()
dut.encrypt(factory.experiment_config["dut"]["plaintext"])
oscilloscope.get_measurement()
```

See how the experiment configuration has been parsed and is stored in
{py:attr}`~phileas.factory.ExperimentFactory.experiment_config`. We can
directly access it in order to retrieve the plaintext.

## Using multiple parameters

Now, it is easy to transform the simple experiment configuration to one which
represents a whole set of parameters. Different custom YAML types, identified
by a `!` prefix, can be used. For example, to use different plaintext values,
we could specify:

```{code-cell} ipython3
experiment = """
dut:
  interface: aes
  key: 12
  plaintext: !range
    start: 1
    end: 3
    resolution: 1

oscilloscope:
  interface: oscilloscope
  amplitude: 1
"""

factory = ExperimentFactory(bench, experiment)
factory.initiate_connections()
pprint(factory.experiment_instruments)
```

The experiment configuration still requires the same set of instruments.
However, multiple instruments configurations are represented by the experiment
configuration. They are available in the
{py:attr}`~phileas.factory.Factory.experiment_config` attribute of the
experiment factory, which is an iterable:

```{code-cell} ipython
for config in factory.experiment_config:
  print(config)
  factory.configure_experiment(config)
```

## Measurements acquisition and storage

```{code-cell}

```

## Test and train datasets

Now, let's say that we want to work with two dataset, one for training, and the
other for validation. This is easily done with the `!configuration` node:

```{code-cell} ipython3
experiment = """
dut:
  interface: aes
  key: !configurations
    test: !random_uniform_bigint
      low: 0
      high: 256
      size: 2
    train: !random_uniform_bigint
      low: 0
      high: 256
      size: 5
  plaintext: !configurations
    test: test
    train: !range
      start: 0
      end: 5
      resolution: 1

oscilloscope:
  interface: oscilloscope
  amplitude: 1
"""

factory = ExperimentFactory(bench, experiment)
factory.experiment_config = generate_seeds(
  factory.experiment_config, salt="your experiment salt"
)

for config in factory.experiment_config:
  print(config)
```

This example is equivalent to defining two iteration trees, corresponding to the
`test` and `train` configurations. However, it prevents redundancy.

Note the call to {py:class}`~phileas.iteration.generate_seeds`. The
`!random_uniform_bigint` builds a pseudo-random iteration leave. In order to
provide a deterministic behavior, it uses a seed. This seed depends on the
location of the leaf on the iteration tree - so that all similar leaves in the
same tree do not generate the same values -, and on a salt which is shared by
all the tree nodes - so that it is possible to generate a similar experiment
with different values. {py:class}`~phileas.iteration.generate_seeds` is used to
propagate the salt value, and generate the local seed, to each of the tree
node.

## Acquisition randomization

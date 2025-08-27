[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![MIT License](https://img.shields.io/github/license/ldbo/phileas)](https://mit-license.org/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/phileas)](https://pypi.org/project/phileas/)
[![GitHub Actions Tests Workflow Status](https://img.shields.io/github/actions/workflow/status/ldbo/phileas/tests.yaml?label=tests)](https://github.com/ldbo/phileas/actions/workflows/tests.yaml)
[![GitHub Actions Build Workflow Status](https://img.shields.io/github/actions/workflow/status/ldbo/phileas/deployment.yaml?label=build)](https://github.com/ldbo/phileas/actions/workflows/deployment.yaml)
[![Compile the draft JOSS paper](https://github.com/ldbo/phileas/actions/workflows/joss-paper.yaml/badge.svg)](https://github.com/ldbo/phileas/actions/workflows/joss-paper.yaml)
[![Documentation](https://img.shields.io/readthedocs/phileas)](https://phileas.readthedocs.io/en/latest/)
[![Code coverage](https://img.shields.io/coverallsCoverage/github/ldbo/phileas)](https://coveralls.io/github/ldbo/phileas)
[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Phileas - a Python library for hardware security experiments automation

Phileas is a Python library that eases the acquisition of hardware security
datasets. Its goal is to facilitate the setup of fault injection and
side-channel analysis experiments, using simple YAML configuration files to
describe both the experiment bench and the configurations of its instruments.
This provides basic building blocks towards the repeatability of such
experiments. Finally, it provides a transparent workflow that yields datasets
properly annotated to ease the data analysis stage.

## Draft JOSS paper

This branch contains the draft of a JOSS paper. You can compile it with

```sh
uv tool run --from=rust-just just
```

It is automatically compiled by a Github Action workflow. You can see it on the build artifacts that are available [here](https://github.com/ldbo/phileas/actions/workflows/joss-paper.yaml).

## Installation

Phileas supports Python 3.10 up to 3.13, as well as PyPy. You can install it
from:

```sh
pip install phileas
```

## Getting started

Phileas is meant to interact with real-life test and measurement instruments.
Here, we demonstrate some of its features using simulated devices to acquire a
side-channel attack dataset. This is a stripped version of
[this example](https://phileas.readthedocs.io/en/latest/examples/sca.html) from
the documentation. It requires that you install Phileas with the `xarray`
dependency, with

```sh
pip install phileas[xarray]
```

Phileas relies on different files to describe and run an experiment. First,
the *bench configuration file* contains the different instruments that are
available on the bench, as well as connection information. It is a YAML file
that you can store in `bench.yaml`:

```yaml
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

```

Here, we declare that our bench is composed of three *bench instruments*: the
embedded PRESENT implementation `simulated_present_dut` is connected to a
current probe `simulated_current_probe`. An oscilloscope
`simulated_oscilloscope` records the output of the probe.

Then, the parameter space of the experiment is described in the *experiment
configuration file*. It it easy to modify and share, and contains most of the
scientific information of the data acquisition experiment. It is a YAML file
supporting custom types, that you can call `experiment.yaml`:

```yaml
oscilloscope:
  interface: oscilloscope
  amplitude: 10

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

iteration: !range
  start: 1
  end: 5
  resolution: 1
```

Here, we specify how to configure the oscilloscope, and all the parameters that
the DUT will handle. Random keys and plaintexts are generated, and each pair of
value is iterated over 5 times.

We can then write a Python script that lets Phileas prepare the instruments and
configure them, while we focus on data acquisition and storage.

```python
from pathlib import Path

import numpy as np
import xarray as xr

import phileas
from phileas.iteration.utility import (
    iteration_tree_to_xarray_parameters,
    data_tree_to_xarray_index
)
from phileas.iteration.random import generate_seeds

# Prepare the instruments
factory = phileas.ExperimentFactory(Path("bench.yaml"), Path("experiment.yaml"))
factory.initiate_connections()
oscilloscope = factory.experiment_instruments["scope"]
dut = factory.experiment_instruments["dut"]

configurations = generate_seeds(factory.experiment_config)

# Prepare data storage
coords, dims_name, dims_shape = iteration_tree_to_xarray_parameters(configurations)
dut.key = 0
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
```

For more information about this example, see [the SCA example page](https://phileas.readthedocs.io/en/latest/index.html).
It is taken from [the documentation](https://phileas.readthedocs.io/en/latest/index.html)
which contains examples, user guides and the API description.

## Contributing

There are different ways you can contribute to Phileas:

 - if you have *any question about how it works or how to use it*, you can
   [open a discussion](https://github.com/ldbo/phileas/discussions/new);
 - if you have *found a bug* or want to *request a new feature*, you can
   [submit an issue](https://github.com/ldbo/phileas/issues/new);
 - if you want to *add new features*, you can
   [submit pull requests](https://github.com/ldbo/phileas/pulls) targeting the
   `develop` branch.

Have a look at the [contributing guide](https://github.com/ldbo/phileas/blob/develop/CONTRIBUTING.md)
for more information about submitting issues and pull requests! In any case,
please follow the [code of conduct](https://github.com/ldbo/phileas/blob/develop/CODE_OF_CONDUCT.md).

## Acknowledgment

This work has been supported by DGA and ANSSI.

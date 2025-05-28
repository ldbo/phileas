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

# Getting the state of an experiment bench

```{code-cell} ipython3
:tags: [remove-cell]

from pathlib import Path
from pprint import pprint

from phileas import ExperimentFactory
from phileas.factory import BenchState
import phileas.mock_instruments
```

To replicate an experiment, it is useful to compare the state of the original
and the current experiment bench, ensuring they are as close as possible.
Phileas has different tools that allow to do this.


## Example bench

Let us work with the following simple experiment bench:

```{code-cell} ipython3
bench = """
simulated-motors:
    loader: phileas-mock_motors-phileas

simulated-oscilloscope:
    loader: phileas-mock_oscilloscope-phileas
    probe: electric-field-probe
    motors: simulated-motors
"""

experiment = """
oscilloscope:
    interface: oscilloscope
    amplitude: !range
        start: 0.1
        end: 10
        steps: 5

iteration: !sequence [1, 2]
"""

factory = ExperimentFactory(bench, experiment)
factory.initiate_connections()
```

## Verifying the effectiveness of an applied configuration

After having configured an instrument, it is possible that its effective
configuration differs from the required one. You can use
{py:meth}`~phileas.factory.ExperimentFactory.get_effective_instrument_configuration`
and
{py:meth}`~phileas.factory.ExperimentFactory.get_effective_experiment_configuration`
to know the exact state of the instrument.

In our case, the amplitude of the oscilloscope can only be a power of 10.
Configuring it, and getting its effective configuration, can be done with:

```{code-cell} ipython3
for conf in factory.experiment_config:
    factory.configure_experiment(conf)
    effective_conf = factory.get_effective_instrument_configuration("oscilloscope")

    required_amp = conf["oscilloscope"]["amplitude"]
    effective_amp = effective_conf["amplitude"]
    print(f"Configuration: {conf}")
    print(f"Amplitude, required: {required_amp}\tEffective: {effective_amp}\n")
```

Let us now consider the case where we set the `lazy` and `snake` parameters of the configuration:

```{code-cell} ipython3
exp_conf = factory.experiment_config.with_params([], lazy=True, snake=True)
for conf in exp_conf:
    print(conf)
```

The amplitude of the oscilloscope is not always configured. Thus, it is not
efficient to retrieve its configuration at each step of the loop. In this case,
we can give
{py:meth}`~phileas.factory.ExperimentFactory.get_effective_experiment_configuration`
an empty configuration that it then fills, ignoring missing fields:

```{code-cell} ipython3
for conf in exp_conf:
    factory.configure_experiment(conf)
    effective_conf = factory.get_effective_experiment_configuration(conf)

    print(f"Required: {conf}\nEffective: {effective_conf}\n")
```

:::{note}

Not all loaders implement
{py:meth}`~phileas.factory.Loader.get_effective_configuration`. They are
ignored by the methods which retrieve the effective configuration of an
instrument or an entire bench.
:::

## Dumping the complete state of an experiment bench

Retrieving the configuration of an instrument is not sufficient to guarantee
that its state is the same as in another experiment. The experiment factory
provides the
{py:meth}`~phileas.factory.ExperimentFactory.dump_bench_state` method, whose
purpose is to build the most complete representation of the state of an
experiment bench.

```{code-cell} ipython3
pprint(factory.dump_bench_state())
```

You can use the state as is. Alternatively, you can export it to YAML, allowing
to save it or compare it to another saved state:

```{code-cell} ipython3
saved_state = factory.dump_bench_state().to_yaml()
print(saved_state)
```

{py:meth}`~phileas.factory.BenchState.to_yaml` also takes an optional argument,
which is the path of a file where the state can be stored.

```python
factory.dump_bench_state().to_yaml(Path("state.yaml"))
```

If you have a bench and an experiment configuration file, and a script which
registers the required loaders, you can use the CLI to dump the state of the
bench:

```sh
$ python -m phileas --bench bench.yaml --experiment experiment.yaml --script script.py
version: '0.4'
instruments:
  simulated-motors:
    loader: phileas-mock_motors-phileas
    interfaces:
    - 2d-motors
    configuration: {}
    id: mock-motors-driver:1
    state:
  simulated-oscilloscope:
    loader: phileas-mock_oscilloscope-phileas
    interfaces:
    - oscilloscope
    configuration:
      probe: electric-field-probe
      motors: simulated-motors
    id: mock-oscilloscope-driver:1
    state:
```

:::{note}

You can use exported states to compare them, with `diff` for example.
:::

Once you have stored a state, you can use it later to restore the bench:

```{code-cell} ipython3

state = BenchState.from_yaml(saved_state)
factory.restore_bench_state(state)
```

{py:meth}`~phileas.factory.ExperimentFactory.restore_bench_state` has an
optional argument `strict` which, when set, prevents restoring states whose
version, instrument names, loaders, interfaces and ids do not exactly match.

```{code-cell} ipython3
:tags: [raises-exception]

state_with_invalid_id = BenchState.from_yaml("""
version: '0.4'
instruments:
  simulated-motors:
    loader: phileas-mock_motors-phileas
    interfaces:
    - 2d-motors
    configuration: {}
    id: mock-motors-driver:2
    state:
      x: 0.0
      y: 0.0
""")
factory.restore_bench_state(state_with_invalid_id)
```

## Replication of the software environment of an experiment

Given properly implemented loaders, with the same bench and experiment
configuration files, Phileas can guarantee the proper replication of an
experiment. However, the user is in charge of the replication of the software
environment of the experiment.

For this, different tools exist, such as `conda` or `poetry`. It is recommended
that you use them, and store their lock files alongside `state.yaml`.

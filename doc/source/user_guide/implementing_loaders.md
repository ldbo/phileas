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

# Implementing loaders

```{code-cell} ipython3
:tags: [remove-cell]

from phileas import mock_instruments

from phileas import Loader, register_default_loader
```

Phileas relies on loaders to match required instruments with suitable drivers,
instantiate them based on the [](/user_guide/bench_configuration_file)
parameters, and use them to configure the experiment according to the
[](/user_guide/experiment_configuration_file). They are made to be reusable:
once a loader has been developed, it can be used to handle the same instrument
without polluting the experiment scripts.

This pages contains a step-by-step guide to implement a loader for an experiment
instrument. It is based on the simulated oscilloscope available in
{py:class}`phileas.mock_instruments.Oscilloscope`.

## Inside most loaders, a driver

Before implementing a loader, you usually[^usually-a-loader] have to find a
Python driver for the instrument. Otherwise, you have to create it.

[^usually-a-loader]: Sometimes, you might want to define a loader without a
driver. See [](#loaders-without-drivers) for more information.

```{eval-rst}

.. autoclass:: phileas.mock_instruments.Oscilloscope
  :members:
```


In our example, the {py:class}`~phileas.mock_instruments.Oscilloscope` driver is
quite simple, as it is simulated. It exposes a
{py:meth}`phileas.mock_instruments.Oscilloscope.get_measurement` method which
returns the acquisition of the probe that is connected to it. The amplitude and
bit width can be controlled by directly modifying the corresponding
attributes.

```{code-cell} ipython3
probe = mock_instruments.RandomProbe(shape=(10, ))
oscilloscope = mock_instruments.Oscilloscope(probe, amplitude=1)

print(f"Measurements: {oscilloscope.get_measurement()}")
```

Here, we use the {py:class}`~phileas.mock_instruments.RandomProbe` probe, which
returns random values with a given shape.

Developing a loader boils down to creating a class that Phileas can interact
with, in order to automatically using the driver. This is done by sub-classing
{py:class}`~phileas.factory.Loader`.

## Defining the interface of the loader

The first step is to define two attributes that are used to match the loader to
the bench and experiment configuration entries.

```{eval-rst}

.. autoattribute:: phileas.factory.Loader.name
    :no-value:
```

In our case, the loader is called `phileas-mock_oscilloscope-phileas`. It is a
good habit to use consistent naming in your projects. For example, we recommend
`brand-model_number-driver_library`.

```{eval-rst}

.. autoattribute:: phileas.factory.Loader.interfaces
    :no-value:
```

Interfaces are mostly used to provide interoperability between loaders. You
should be able to replace a loader with another, as long as they expose the
same interfaces. In our example, we chose to define `oscilloscope` as the
interface of an instrument which

 - can be configured with an `amplitude` parameter;
 - has a `get_measurement` method, which returns the last acquired values.

## Instantiating the driver

The first method that we cover is
{py:meth}`~phileas.factory.Loader.initiate_connection
Loader.initiate_connection`, which is responsible for instantiating the driver.


```{eval-rst}

.. automethod:: phileas.factory.Loader.initiate_connection
```

It takes a single argument, which is the content of instrument entry of the
bench configuration file. It is a dictionary with
{py:attr}`~phileas.iteration.DataTree` values, and usually string keys. It must
return an instantiated driver which is ready to be used. In our case, we define
it like this[^partial-loader]:

[^partial-loader]: Actually, our example remains simpler than what
{py:mod}`phileas.mock_instruments` provides.

```python
def initiate_connection(self, configuration: dict) -> Oscilloscope:
    """
    Parameters:
     - probe (required): Name of the simulated probe to use.

    Supported probes, and parameters:
     - random-probe:
        shape: tuple with the required probe output shape
    """
    probe = configuration["probe"]
    if probe == "random-probe":
        return Oscilloscope(probe=RandomProbe(shape=configuration["shape"]))
    else:
        raise ValueError(f"Unsupported probe type: {probe}")
```

The bench configuration is expected to have a first parameter indicating which
probe is used. If a random probe is requested, its shape is additionally
retrieved from `configuration`. Finally, the built oscilloscope is returned.

:::{note}

Notice the docstring, enclosed in `"""`, which documents the parameters required
by the method. It is important to define it clearly, so that others can easily
reuse the loader.
:::

With this implementation, a valid bench configuration is:

```yaml
simulated-oscilloscope:
    loader: phileas-mock_oscilloscope-phileas
    probe: random-probe
```

## Configuring the instrument

Now, in order to use the instrument, loaders are expected to implement the
{py:meth}`~phileas.factory.Loader.configure` method.


```{eval-rst}

.. automethod:: phileas.factory.Loader.configure
```

It uses a driver, supplied as the `instrument` argument, to configure the
instrument according to `configuration`, which is derived from an experiment
configuration. We define it to set the amplitude of the oscilloscope:

```python
def configure(self, instrument: Oscilloscope, configuration: dict):
    """
    Parameters:
     - amplitude (optional): Amplitude of the oscilloscope.
    """
    if "amplitude" in configuration:
        amplitude = configuration["amplitude"]
        instrument.amplitude = amplitude
        self.logger.info(f"Amplitude set to {amplitude}.")
```

This construct is often found: the existence of a key in `configuration` is
checked, and then it is used. This is useful to add optional parameters.

Note that this method does not return anything.

A valid experiment configuration is

```yaml
oscilloscope:
    interface: oscilloscope
    amplitude: !range
        start: 1
        end: 5
        resolution: 0.5
```

## Putting it all together

Finally, the resulting loader can be defined as:

```python
class OscilloscopeLoader(Loader):
    """
    Loader of a simulated oscilloscope.
    """

    name = "phileas-mock_oscilloscope-phileas"
    interfaces = {"oscilloscope"}

    def initiate_connection(self, configuration: dict) -> Oscilloscope:
        """
        Parameters:
         - probe (required): Name of the simulated probe to use.

        Supported probes, and parameters:
         - random-probe:
            shape: tuple with the required probe output shape
        """
        probe = configuration["probe"]
        if probe == "random-probe":
            return Oscilloscope(probe=RandomProbe(shape=configuration["shape"]))
        else:
            raise ValueError(f"Unsupported probe type: {probe}")

    def configure(self, instrument: Oscilloscope, configuration: dict):
        """
        Parameters:
         - amplitude (optional): Amplitude of the oscilloscope.
        """
        if "amplitude" in configuration:
            amplitude = configuration["amplitude"]
            instrument.amplitude = amplitude
            self.logger.info(f"Amplitude set to {amplitude}.")
```

However, for Phileas to know how to use it when requested by a bench
configuration, it must be *registered*. This is done by a the
{py:func}`~phileas.factory.register_default_loader` class decorator. Using it
is simple: in our case, we can modify the definition with:

```python
@register_default_loader
class OscilloscopeLoader(Loader):
    ...
```

Alternatively, if you want to register a loader which is defined elsewhere, you
can use it as a usual function:

```python
register_default_loader(OscilloscopeLoader)
```

Now that it is done, Phileas is able to match

 - any bench configuration entry with `loader:
   phileas-mock_oscilloscope-phileas`; and
 - any experiment instrument requiring the `oscilloscope` interface

to the `OscilloscopeLoader` loader.


## Logging

Loaders have a {py:attr}`~phileas.factory.Loader.logger` logging handler that
can be used by the user. This has the advantage of providing situational logs,
which are registered under `phileas.loader-name`.

## Documentation

Loaders wrap drivers in order to let Phileas automatically use them. However,
they should be understood by users in order to write bench and experiment
configurations. For this reason, it is strongly recommended to add docstrings
to the {py:meth}`~phileas.factory.Loader.initiate_connection` and
{py:meth}`~phileas.factory.Loader.configure` methods when implementing them.

The documentation of all the loaders currently registered by Phileas in a script
can be obtained from `python -m phileas list-loaders script.py`.

:::{warning}

`python -m phileas list-loaders script.py` executes the script.
:::

## Bench-only instruments and references

Bench-only instruments only require the
{py:meth}`~phileas.factory.Loader.initiate_connection` of their loader. They
are introduced
[here](/user_guide/bench_configuration_file.md#bench-only-instruments).

These instruments are often referred to when initiating the connection to other
instruments. In this case, you can use
{py:meth}`~phileas.factory.ExperimentFactory.get_bench_instrument` to retrieve
the bench-only instrument:

```python
# ...in the instrument which uses the bench-only instrument
def initiate_connection(self, configuration: dict) -> Any:
    bench_only_name = configuration["bench-only"]
    bench_only = self.instruments_factory.get_bench_instrument(bench_only_name)
    # You can now use the bench-only instrument driver
```

Note that bench instruments initialization is lazy. Thus, only required
instruments are initialized, and it is done only once. In other words, calling
{py:meth}`~phileas.factory.ExperimentFactory.get_bench_instrument` with the
same argument either returns always the same value, or fails.

:::{warning}

Do not introduce {py:meth}`~phileas.factory.Loader.initiate_connection` loops,
as it creates an infinite loop. Indeed, if `instrumentA` calls
`get_bench_instrument("instrumentB")` and `instrumentB` calls
`get_bench_instrument("instrumentA")`, none of these instruments can be
instantiated, and the program loops endlessly.
:::

## Loaders without drivers

Sometimes, it might be convenient not to use a dedicated driver. In this case,
the loader acts as a driver. This is not recommended, however it is allowed.
Since {py:class}`~phileas.factory.Loader` is a Python class, you can add other
methods that specifically drive the instrument.

However, note that `__init__` is not called by you, but by Phileas itself. Thus,
you should not modify its signature, and ensure that
{py:meth}`~phileas.factory.Loader.__init__ Loader.__init__` is called. We
recommend to consider `initiate_connection` as the constructor, and to return
`self`.

:::{note}

You can implement `__init__`, but in this case it won't have any user-supplied
argument.
:::

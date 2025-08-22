# Bench configuration file

The bench configuration file describes the instruments that are available on an
experiment bench. It is a YAML file containing a dictionary. It is parsed
during the initialization of
an {py:class}`~phileas.factory.ExperimentFactory`.

```{code-block} yaml
:caption: Typical `bench.yaml` content

instrument1:
  loader: loader1-name
  parameter1: value1
  parameter2: value2

example_instrument:
  loader: brand-model-loader
  ip: 192.168.0.1
  port: 12345
```

## Purpose

It should contain all the information that is required to instantiate the
drivers of an experiment bench. Thus, you should specify there all the
connection parameters (IP address, port, baudrate, *etc*.). You can also add
here all the information which is not useful for the experiments. This might
include configuration parameters specific to an instrument, credentials, *etc*.

## Bench instrument entries

Each of its mapping entries which contains the `loader` key represents
a *bench instrument*. The `loader` values specifies the name of the loader
that handles the instrument. Prior to being used by
an {py:class}`~phileas.factory.ExperimentFactory`, a suitable loader must have
been registered. The other values of the entry are passed
to {py:func}`~phileas.factory.Loader.initiate_connection`, in order to
initialize the loader.

A bench might have multiple instruments handled by the same loader(*eg*. there
might be multiple motors, which are similarly driven in software but act on
different elements). In order to allow the user to differentiate them, you can
add arbitrary attributes to the entry, which will be ignored by the loader. For
example, you might specify the element that is moved by a motor, the wavelength
of a light, *etc*.

## Arbitrary attributes for filtering

You can also add top-level entries which are not mappings, or miss the
`loader` key. They are simply ignored by phileas. It is a good idea to use
them to document the experiment setup.

## Bench-only instruments

In some situations, it might be useful to have instruments that will never be
required in the experiment configuration file. They are called *bench-only
instruments*.

Let's say that a remote computer provides access to different instruments. You
can access it through an SSH connection. You could then use the following
configuration:

```{code-block} yaml
:caption: Bench-only instruments example

# Bench-only instrument
control-computer:
  loader: ssh-loader
  user: user-name
  password: password-value

instrument:
  loader: remote-instrument
  control: control-computer  # Internal reference
  parameter: value
```

`control-computer` is only used to establish the connection to `instrument`,
which can reference it with `control: control-computer`. The
`remote-instrument` loader internally uses
{py:meth}`~phileas.factory.ExperimentFactory.get_bench_instrument` to get the
`control-computer` instance.

## Template generation

A bench template file can be generated automatically by phileas. See `python -m
phileas generate bench -h` for the available options.

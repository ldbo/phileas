# Configuration files and concepts

## Preliminary remark

- I'm not sure it is ideal to actually separate the configuration in a bench and
  an experiment, but I do it anyway for now. Using yaml inclusion seems more
  flexible, allowing the configuration files organization to be tailored to the
  project.
- All the configuration files use the YAML file format. Besides the API used for
  this tool (*e.g.* a bench instrument should have a `loader` field), any other
  piece of information can be added in any configuration file, as long as it
  remains a valid YAML file. For example, this allows to add a description to
  an experiment, or the name of the maintainer of a bench.

## Bench

- Is in its own configuration file, typically `bench.yaml`
- Describes a bench, and in particular the list of instruments it holds. Those
  are referred to as *bench instruments*.
- Each instrument has a mandatory `loader` field, which specifies a loader name
  that is used to initialize the instrument. Then, it can hold any number of
  configuration fields, usually made to allow to connect to it, by
  specifying *e.g.* its IPs, USB ports, *etc.*
- For now, I think it should not contain information about how instruments are
  interconnected, cf. the Preliminary remark

## Experiment

- Is in its own configuration file, typically `experiment.yaml`
- Contains the description of an actual experiment that takes place, on
  theoretically any bench that meets its requirements
- Specifies a set of instruments to be used. Each so-called *experiment
  instrument* requires an actual bench instrument having a given `interface`
  (*e.g.* be an oscilloscope, a laser, a power supply) and optionally matching
  some filters (*e.g.* have a 980 nm wavelength, have more than 4 channels).
- Each instrument embeds its configuration, and how it is connected to other
  instruments. Essentially, the experiment configuration file should allow
  someone who does not have any information on the actual hardware that is on a
  bench to replicate the experiment.

### Connections

 - The instruments in an experiment can be connected together, describing how
   they are actually wired on the bench.
 - This is either done in the `connections` part of the configuration file, or
   in the `connections` field of an instrument
 - Each of the connections represents a single two-ended "wire". `connections`
   hold a list of dictionaries, with two fields: `from` and `to`
 - The ends of a connection can refer to
   - an already existing instrument, using its name;
   - a port, or sub-port, of an already existing instrument, using the dot as a
     child operator. For example, "power_supply.ch1" refers to the "ch1" port
     of the "power_supply" instrument.
   - Alternatively, it can be anything else, typically for documentation
     purpose.
 - Additionally, when specifying connections directly inside an instrument
   `instrument`, relative naming can be used. Thus, `instrument.port` can be
   replaced by `port`.
 - Each connection can have an `attribute` field, which is a string that is
   not used yet.

### Iteration through multiple configurations

 - An experiment file can contain numeric ranges, which are represented by
   maps with the following structure

 ```yaml
!range
start: <first_element>
end: <last_element>
steps|resolution: ...
[progression: linear|geometric]
 ```

 - Upon parsing the experiment file, the factory replaces numeric ranges in
   `ExperimentFactory.experiment_config` with a `parsing.NumericRange` object,
   which is an iterable and can be converted to a numpy array with `to_array`.
 - When using numeric ranges, the configuration file can be thought of as a
   representation of multiple so-called *literal configurations*, where each
   entry having a numeric range value is replaced by one of the values that the
   range contains. Iterating through those configurations is possible using the
   `[experiment|instrument]_configurations_iterator` and `configured_
   [experiment|instrument]_iterator` methods.

# Python API

## Experiment factory

- The entry point is the `ExperimentFactory` class, which mainly allows to
  instantiate bench and experiment instruments using the bench and experiment
  configuration files.
- It uses some `Loader`s to instantiate and connect to the bench instruments.
  The experiment instrument are matched to bench instruments using their
  `interface`, as well as `filters` restricting the set of suitable bench
  instruments.
- Those experiment instruments are then configured by the loader.
- Finally, the factory allows the user to retrieve the set of bench and
  experiment instruments.

## Loader

- A `Loader` is used to instantiate an object to manipulate an actual hardware
  instrument, using its `initiate_connection`. It can then configure it with
  `configure`.
- It has a `name` used by the `loader` field of bench instruments to specify how
  they should be loaded.
- The set of `interfaces` it supports is used to allow an experiment instrument
  to specify which interface the bench instrument it is linked to should
  `implement`.

# Typical project structure

A project allowing to run an experiment will typically contain the following
files:
 - `experiment.yaml`: experiment configuration file, which is written by the
   experimenter, and is (*ideally*) bench-agnostic.
 - `bench.yaml`: bench configuration file, which is written by the person
   configuring the bench.
 - `project_config.py`: defines a set of loaders, and use
   `register_default_loader` to make them available for the experiment
   factories to be created. It can be shared between different benches. In
   order to help writing configuration files compatible with a set of
   registered loaders, you can use `python -m phileas list-loaders`, whose
   Markdown output can be redirected to a file if needed, with `python -m
   phileas list-loaders > loaders_doc.md`.

# Developer notes

- The repository depends on [Poetry](https://python-poetry.org/) for
  dependencies and virtual environments management, as well as for packaging.
  After installing Poetry, you have to create the development environment and
  install the dependencies using `poetry install`. You can then enter the
  development environment with `poetry shell`. You can leave it later with
  `exit`. Alternatively, from outside of the development environment, you can
  run commands using `poetry run <command>`. They will be run from inside the
  development environment.
- Adding dependencies can be done using `poetry add`. Before committing the
  changes that have been made to `pyproject.toml`, you should rebuild the lock
  file, using `poetry lock`. Then, `poetry.lock` should be committed as well.
- Test files are stored in the `test` module. You can use `unittest` to
  automatically discover and run them, using for example `python -m unittest`
  from the root of the repository.
- Test `TestFunctional1`, and the associated `functional_1_
  {config.py,experiment.yaml,bench.yaml}` configuration files are a good
  example to start using the Phileas.
- Pre-commit hooks are managed by [pre-commmit]
  (https://pre-commit.com/), which is a developer dependency. To use them, call
  `pre-commit install` from the project virtual environment.

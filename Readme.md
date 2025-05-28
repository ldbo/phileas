# Phileas - a Python library for scientific experiment automation

Phileas is a Python library that eases the acquisition of hardware security
datasets. Its goal is to facilitate the setup of fault injection and
side-channel analysis experiments, using simple YAML configuration files to
describe both the experiment bench and the configurations of its instruments.
This provides basic building blocks towards the repeatability of such
experiments. Finally, it provides a transparent workflow that yields datasets
properly annotated to ease the data analysis stage.

## Installation

Phileas supports Python 3.9 up to 3.13 versions.

It is available in PyPI, so you can install it with

```sh
pip install phileas
```

## Documentation

You can build the documentation and serve it with

```sh
poetry run sphinx-build doc/source/ doc/build/
```

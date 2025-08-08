[![MIT License](https://img.shields.io/github/license/ldbo/phileas)](https://mit-license.org/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/phileas)](https://pypi.org/project/phileas/)
[![GitHub Actions Tests Workflow Status](https://img.shields.io/github/actions/workflow/status/ldbo/phileas/tests.yaml?label=tests)](https://github.com/ldbo/phileas/actions/workflows/tests.yaml)
[![GitHub Actions Build Workflow Status](https://img.shields.io/github/actions/workflow/status/ldbo/phileas/deployment.yaml?label=build)](https://github.com/ldbo/phileas/actions/workflows/deployment.yaml)
[![Documentation](https://img.shields.io/readthedocs/phileas)](https://phileas.readthedocs.io/en/latest/)
[![Code coverage](https://img.shields.io/coverallsCoverage/github/ldbo/phileas)](https://coveralls.io/github/ldbo/phileas)
[![Tested with Hypothesis](https://img.shields.io/badge/hypothesis-tested-brightgreen.svg)](https://hypothesis.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Phileas - a Python library for scientific experiment automation

Phileas is a Python library that eases the acquisition of hardware security
datasets. Its goal is to facilitate the setup of fault injection and
side-channel analysis experiments, using simple YAML configuration files to
describe both the experiment bench and the configurations of its instruments.
This provides basic building blocks towards the repeatability of such
experiments. Finally, it provides a transparent workflow that yields datasets
properly annotated to ease the data analysis stage.

## Installation

Phileas supports Python 3.10 up to 3.13, as well as PyPy.

It is available in PyPI, so you can install it with

```sh
pip install phileas
```

## Documentation

You can build the documentation and serve it with

```sh
poetry run sphinx-build doc/source/ doc/build/
```

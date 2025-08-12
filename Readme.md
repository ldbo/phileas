[![MIT License](https://img.shields.io/github/license/ldbo/phileas)](https://mit-license.org/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/phileas)](https://pypi.org/project/phileas/)
[![GitHub Actions Tests Workflow Status](https://img.shields.io/github/actions/workflow/status/ldbo/phileas/tests.yaml?label=tests)](https://github.com/ldbo/phileas/actions/workflows/tests.yaml)
[![GitHub Actions Build Workflow Status](https://img.shields.io/github/actions/workflow/status/ldbo/phileas/deployment.yaml?label=build)](https://github.com/ldbo/phileas/actions/workflows/deployment.yaml)
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

## Installation

Phileas supports Python 3.10 up to 3.13, as well as PyPy. You can install it with `pip`:

```sh
pip install phileas
```

## Documentation

The documentation is available in the
[package documentation](https://phileas.readthedocs.io/en/latest/index.html).
## Contributing

There are different ways you can contribute to Phileas:

 - if you have *any question about how it works or how to use it*, you can [open a discussion](https://github.com/ldbo/phileas/discussions/new);
 - if you have *found a bug* or want to *request a new feature*, you can [submit an issue](https://github.com/ldbo/phileas/issues/new);
 - if you want to *add new features*, you can [submit pull requests](https://github.com/ldbo/phileas/pulls) targeting the `develop` branch.

Have a look at the [contributing guide](https://github.com/ldbo/phileas/blob/develop/CONTRIBUTING.md) for more information about submitting issues and pull requests! In any case, please follow the [code of conduct](https://github.com/ldbo/phileas/blob/develop/CODE_OF_CONDUCT.md).

## Acknowledgment

This work has been supported by DGA and ANSSI.

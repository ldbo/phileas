# Installation

Phileas supports Python 3.9 up to 3.12 versions.

It is available in PyPI, so you can install it with

```sh
pip install phileas
```

## Dependencies

Phileas depends on
 - ``ruamel-yaml`` for parsing YAML files,
 - ``xarray`` and ``pandas`` to support exporting to their dataset formats, if you use the extras ``[xarray]``,
 - ``numpy`` for random numbers generation,
 - ``jinja2`` for templates files generation,
 - ``rich`` for generating the documentation of loaders and
 - ``graphviz`` for experiment graphs generation.

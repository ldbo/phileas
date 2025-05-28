# Developer notes

## Environment setup

You need poetry in order to setup the development environment. Once it is
installed, simply run

```sh
poetry install
poetry run pre-commit install
poetry shell
```

which will enable a virtual environment with the required development
dependencies. It also installs [pre-commit hooks](https://pre-commit.com) that
will run some code quality tools before each commit.

## Tests

Phileas includes different tests, which are located in the `test` package. You
can run the test suite for your current Python version with

```sh
python -m unittest --buffer --verbose
```

You can run the test suite for all the supported Python versions with

```sh
tox
```

## Dependencies

### Supported Python versions

Phileas only supports
[currently maintained](https://devguide.python.org/versions/) Python versions.
For now, it starts at version 3.9, and stops at version 3.13.

### Adding dependencies

To add a new dependency to the project, you must:

 - Use `poetry add` to update the `pyproject.toml` file with your new
   requirement.
 - Update `poetry.lock` with `poetry lock`.
 - This might break the pre-commit type checker. In this case, update
   `.pre-commit.config.yaml` with the required `additional_dependencies`.
 - Update `/doc/source/getting_started/installation.md` accordingly.
 - Finally, commit all these files. Do not forget to include `poetry.lock`.

### Updating dependencies

When you update dependencies, you should also update the pre-commit type checker
`additional_dependencies` accordingly.

## Type checking

Phileas relies on [mypy](https://www.mypy-lang.org) for type checking. It is
called at different steps:

 - you can call it manually whenever you want;
 - the installed pre-commit hooks automatically run it before each commit;
 - `tox` invokes it.

`ruamel.yaml` is used for YAML parsing. Due to
[this unsolved issue](https://github.com/python/mypy/issues/7276) and
[this related one](https://sourceforge.net/p/ruamel-yaml/tickets/328), you are
likely to encounter the error:

```sh

error: Skipping analyzing "ruamel": module is installed, but missing library stubs or py.typed marker  [import-untyped]
```

This can be fixed by using the `--no-incremental` flag. However, this increases
the verification time. This flag is enabled in the `tox` configuration, however
it is not in the pre-commit configuration. Rather, we recommend

 - manually checking the pre-commit output, and ensure that the only error is
   due to the `ruamel.yaml` issue;
 - skip verification with the ` --no-verify` flag of `git commit`.

## CI

`tox` is used to run code quality, coverage and tests for different Python versions. You can simply invoke it with

```
tox
```

The coverage report is then available in `/htmlcov`.

## Documentation

The documentation is generated with sphinx. You can build the HTML pages with

```sh
sphinx-build doc/source/ doc/build/
```

It is then accessible in `/doc/build/index.html`. Alternatively, it is
convenient to use `sphinx-autobuild`, which automatically updates the
documentation, and exposes it on a local server:

```sh
sphinx-autobuild doc/source/ doc/build/
```

The documentation coverage can be obtained with

```sh
sphinx-build -M coverage doc/source/ doc/coverage/
```

It is then accessible in `/doc/coverage/`

# 0.1

- First released version.
- Basic experiment creation, with iteration through parameters using cartesian
  product.
- Experiment graph extraction.

# 0.2

- Improve iterables handling.
- Add the IterationMethod.UNION iteration method, which allows to search
  through iterables one at a time.
- Breaks the iterables API : YAML custom datatypes are now used, and the numeric
  ranges ends are now called start and end.

# 0.3

- Configurations iteration is now handled by iteration trees, whose iteration
  produces data trees.
- Iteration trees have resetable and two-way iterators.
- Implement cartesian product iteration, possibly lazy and using the snake
  method.
- Implement union iteration, possibly lazy.
- Logging now uses a dedicated logger, name `"phileas"`.
- Test values are generated using hypothesis.

# 0.3.1

- Allow iteration nodes to control the iteration order.

# 0.4

- Add support for random iteration.
- Iteration is now random-access based.
- Add template generation to ease starting up a new project.
- Add iteration.restrict_leaves_sizes, which reduces the size of iteration
  leaves, typically used to verify that a configuration is valid.
- Cartesian product iteration is now valid for all iteration leaves sizes.
- Add tox support for testing across different python versions.

# 0.5

## Added

 - New iteration nodes: `Shuffle`, `Pick`, `UniformBigIntegerRng`, `PrimeRng`,
   `Configurations`.
 - Utilities for experiment scripts: `flatten_datatree`,
   `iteration_tree_to_xarray_parameters`, `iteration_tree_to_multiindex`.
 - Improve working with infinite-sized trees, with the addition of
   `IterationTree.safe_len()` and `InfiniteLength`. Concrete iteration methods
   now support infinite children.
 - Tree modification with `with_params`.
 - Tree iterators support negative indexing.
 - Loaders have a logging handler available at `Loader.logger`.
 - `register_default_loader` is a class decorator.
 - Different utilities to dump the state of a bench.
 - Generate a documentation for the API, as well as getting started instructions, developer notes, user guides and examples.
 - Support for Python 3.13 and PyPy.
 - A simple performance benchmark.
 - Rust bindings are setup with maturin.

## Changed

 - Data validation does not use assertions anymore.
 - Iteration trees are now frozen dataclasses.

## Removed

 - Drop support for Python 3.8 and 3.9.

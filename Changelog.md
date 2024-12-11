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

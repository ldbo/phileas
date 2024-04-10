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

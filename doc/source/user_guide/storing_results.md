---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Storing results

```{code-cell} ipython3
:tags: [remove-cell]

from pprint import pprint

import numpy as np
import xarray as xr
import pandas as pd

from phileas.iteration.utility import (
    flatten_datatree,
    iteration_tree_to_multiindex,
    iteration_tree_to_xarray_parameters,
)
from phileas.parsing import load_iteration_tree_from_yaml_file
```

In most experiments, a {py:class}`~phileas.iteration.IterationTree` is used to
represent the configurations of its instruments. For each of them, one or
multiple measurements are carried out. They should then be stored, alongside
the corresponding configuration.

Phileas can be used to configure the instruments. However, it is not responsible
for carrying out the measurements and handling their results, which should be
done by the user. This includes storing the results of the experiment. Yet,
Phileas provides some utility functions that can be used to prepare dataframes
for storing those results.

## Gridded datasets with [xarray](https://docs.xarray.dev/en/stable/index.html)

If the experiment parameter space is a cartesian product of each of the
instruments' parameters' space, then a gridded dataset is appropriate to store
the results of the experiment.
[xarray](https://docs.xarray.dev/en/stable/index.html) datasets are suitable
for this purpose.

The function
{py:func}`~phileas.iteration.utility.iteration_tree_to_xarray_parameters` can
be used to initialize a {py:class}`~xarray.Dataset` or
{py:class}`~xarray.DataArray`. Given an iteration tree, it returns a tuple that
can be used to define the coordinates of the dataset. The user now just needs
to define the number and types of the measurements.

```{code-cell} ipython3
config_file = """
ins_a:
    param1: !sequence [a1-0, a1-1, a1-2]
    param2: !sequence [a2-0, a2-1, a2-2]
ins_b:
    param1: !sequence [b1-0, b1-1, b1-2]
    param2: b2
"""
tree = load_iteration_tree_from_yaml_file(config_file)

coords, dims_name, dims_shape = iteration_tree_to_xarray_parameters(tree)
results = xr.Dataset(
    data_vars=dict(
        field1=(dims_name, np.full(dims_shape, np.nan)),
        field2=(dims_name, np.full(dims_shape, np.nan)),
    ),
    coords=coords
)
results
```

In this simple example, the experiment uses two instruments. The parameter space
contains three dimensions, corresponding to the two parameters of
`ins_a` and the single parameter of `ins_b`. The returned
coordinates are named `"ins_a.param1"`, `"ins_a.param2"`,
`"ins_b.param2"`, each containing the corresponding parameter values.
The user wants to carry out two measurements for each configuration, containing
`float` values.


:::{seealso}

If the measurements dataset is sparse, *ie.* if only a few measurements are
carried out, using numpy arrays is not optimal. Instead, consider using
[sparse](http://sparse.pydata.org) containers.
:::

Modifying `results` requires using another utility function:
{py:func}`~phileas.iteration.utility.flatten_datatree`, which converts a
hierarchical data tree to a flat one that can be used for xarray indexing.
Thus, the acquisition loop usually has this structure:

```{code-cell} ipython3
for config in tree:
    flat_config = {
        coord: v for coord, v in flatten_datatree(config).items()
        if coord in dims_name
    }

    # Acquire measurements ...
    measurements = np.random.rand(2)

    results.field1.loc[flat_config] = measurements[0]
    results.field2.loc[flat_config] = measurements[1]

results
```

Notice how the output of
{py:func}`~phileas.iteration.utility.flatten_datatree` is filtered at line 4:
the goal is to keep indexed coordinates only. Indeed, xarray refuses to use
non-indexed coordinates for indexing.

```{code-cell} ipython3
:tags: [hide-input]

print("config:")
pprint(config)
print("\nflat_config:")
pprint(flat_config)
```

## Tabular datasets with [pandas](http://pandas.pydata.org)

Alternatively, you can chose to store measurements in a tabular dataset, like
the {py:class}`~pandas.DataFrame` provided by pandas. This can be convenient
when the experiment parameter space is not a cartesian product.

The {py:func}`~phileas.iteration.utility.iteration_tree_to_multiindex` function
builds a{py:class}`~pandas.MultiIndex` from an
{py:class}`~phileas.iteration.IterationTree`. It can then be used as a
dataframe index.

```{code-cell} ipython3
config_file = """
ins_a: !union
    _reset: last
    param1: !sequence [a1-0, a1-1, a1-2]
    param2: !sequence [a2-0, a2-1, a2-2]
ins_b:
    param1: !sequence [b1-0, b1-1, b1-2]
    param2: b2
"""
tree = load_iteration_tree_from_yaml_file(config_file)

index = iteration_tree_to_multiindex(tree)

results = pd.DataFrame(
    np.full((len(index), 2), np.nan),
    index=index,
    columns=["field1", "field2"]
)
results
```

This example is a simple variation of the last one. Notice the iteration method
of `ins_a`: it is a {py:class}`~phileas.iteration.Union`, and not a
{py:class}`~phileas.iteration.CartesianProduct` anymore. Thus, a tabular
storage format might be appropriate.

Then, storage of the results is once again done with a modification of the
output of {py:func}`~phileas.iteration.utility.flatten_datatree`:

```{code-cell} ipython3
for config in tree:
    flat_config = flatten_datatree(config)
    current_index = tuple(flat_config[param] for param in results.index.names)

    # Acquire measurements ...
    measurements = np.random.rand(2)

    results.loc[current_index, "field1"] = measurements[0]
    results.loc[current_index, "field2"] = measurements[1]

results
```

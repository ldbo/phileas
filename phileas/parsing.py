from copy import deepcopy
from itertools import product
from math import ceil
from pathlib import Path
from typing import Any, Generator, Iterable

from ruamel.yaml import YAML

import numpy as np


def load_yaml_dict_from_file(file: Path | str) -> dict:
    yaml = YAML(typ="safe")
    data = yaml.load(file)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML file {file} top-level object should be a map")

    return data


def convert_numeric_ranges(range_dict: dict[str, Any] | Any) -> Any:
    """
    Recursively convert numeric ranges in a dictionary to an equivalent numpy array.

    A numeric range is represented by a dictionary which has the following
    keys: `from`, `to`, `steps`|`resolution`, [`progression`].

     - `to` and `from` represent the starting and ending element of the
       represented sequence, and are included in it. Note that they do not
       require to be ordered.
     - `progression` is either `linear` (default) or `geometric`, and controls
       whether the sequence is arithmetic or geometric.
     - `steps` represents the number of elements in the sequence
     - `resolution` is the difference (resp. ratio) between two successive
       elements in an arithmetic (resp. geometric) sequence. It must be
       greater than 0 (resp. 1). As the starting and ending elements of the
       sequence are included, if the resolution is not compatible with them,
       it is rounded down to the first compatible resolution. Thus, the
       actual resolution is guaranteed to be lower or equal to `resolution`

    If `range_dict` is a dictionary which stores numeric ranges, it *will
    be modified*.
    """
    if isinstance(range_dict, list):
        for i in range(len(range_dict)):
            range_dict[i] = convert_numeric_ranges(range_dict[i])
        return range_dict

    if not isinstance(range_dict, dict):
        return range_dict

    keys = set(range_dict.keys())

    required_keys = {"to", "from"}
    one_of_those_keys = {"steps", "resolution"}
    optional_keys = {"progression"}
    allowed_keys = required_keys | one_of_those_keys | optional_keys

    if not keys.issubset(allowed_keys) or not required_keys.issubset(keys):
        excluded_keys = {"connections", "interface"}
        for key in range_dict:
            if key in excluded_keys:
                continue

            range_dict[key] = convert_numeric_ranges(range_dict[key])

        return range_dict

    if len(one_of_those_keys & keys) != 1:
        raise ValueError(f"Exactly one of {one_of_those_keys} must be supplied")

    return _convert_actual_range_to_sequence(range_dict)


def _convert_actual_range_to_sequence(range_dict: dict[str, Any]) -> np.ndarray:
    progression = range_dict.get("progression", "linear")
    start = range_dict["from"]
    end = range_dict["to"]

    if progression == "linear":
        if "steps" in range_dict:
            steps = range_dict["steps"]
        else:  # Then use resolution
            resolution = range_dict["resolution"]
            if resolution <= 0:
                raise ValueError("The resolution of a linear range must be >0")

            steps = ceil(abs(end - start) / resolution) + 1

        return np.linspace(start, end, num=steps)
    else:  # progression == "geometric"
        if "steps" in range_dict:
            steps = range_dict["steps"]
        else:  # "resolution" in r
            resolution = range_dict["resolution"]
            if resolution <= 1:
                raise ValueError("The resolution of a geometric range must be >1")

            global_ratio = end / start
            if global_ratio < 1:
                global_ratio = 1 / global_ratio

            steps = ceil(np.log(global_ratio) / np.log(resolution)) + 1

        return np.geomspace(start, end, num=steps)


def configurations_iterator(config: dict) -> Generator[dict, None, None]:
    """
    Yields the product of the configurations represented by a collection with
    numeric ranges. You can use it to iterate through configurations that don't
    have numeric ranges anymore, but have literal values instead.

    A numeric range is defined as a numpy.ndarray object.

    >>> config = {
    ...     "a": np.linspace(1, 2, 2),
    ...     "b": [np.linspace(0, 1, 2)],
    ...     "c": 3,
    ... }
    >>> for c in configurations_iterator(config):
    ...     print(c)
    {'a': 1.0, 'b': [0.0], 'c': 3}
    {'a': 1.0, 'b': [1.0], 'c': 3}
    {'a': 2.0, 'b': [0.0], 'c': 3}
    {'a': 2.0, 'b': [1.0], 'c': 3}
    """
    range_locations: list[list[str]] = _find_numeric_range_locations(config)
    ranges = [_get_nested_element(config, loc) for loc in range_locations]

    for values in product(*ranges):
        single_config = deepcopy(config)
        for location, value in zip(range_locations, values):
            _set_nested_element(single_config, location, value)

        yield single_config


def _find_numeric_range_locations(
    collection: dict | list | Any,
    current_location: list | None = None,
) -> list[list]:
    """
    Find the locations of the numeric ranges in a nested collection, containing
    of dict, list, literal and numpy.ndarray objects.

    The numeric ranges are defined to be the numpy arrays.

    In a nested collection, the location of an item is defined to be the list
    of indices to recursively access, starting from the top-level collection, in
    order to find the item.
    """
    if current_location is None:
        current_location = []

    locations = []
    next_level_iterator: Iterable[tuple[Any, Any]]
    if isinstance(collection, dict):
        next_level_iterator = collection.items()
    elif isinstance(collection, list):
        next_level_iterator = enumerate(collection)
    else:
        next_level_iterator = []

    for key, value in next_level_iterator:
        new_loc = current_location.copy() + [key]
        if isinstance(value, np.ndarray):
            locations.append(new_loc)
        else:
            locations += _find_numeric_range_locations(value, new_loc)

    return locations


def _get_nested_element(d: dict | list, location: list) -> Any:
    for key in location:
        d = d[key]

    return d


def _set_nested_element(d: dict | list, location: list, v: Any) -> Any:
    for key in location[:-1]:
        d = d[key]

    d[location[-1]] = v

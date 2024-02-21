from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from math import ceil
from pathlib import Path
from typing import Any, ClassVar, Generator, Iterable, Iterator, Literal

import numpy as np
from ruamel.yaml import YAML

_yaml = YAML(typ="safe")


def load_yaml_dict_from_file(file: Path | str) -> dict:
    data = _yaml.load(file)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML file {file} top-level object should be a map")

    return data


@_yaml.register_class
@dataclass
class NumericRange:
    yaml_tag: ClassVar[str] = "!range"
    start: float | int
    end: float | int
    steps: int | None = None
    resolution: float | int | None = None
    progression: Literal["linear"] | Literal["geometric"] = "linear"
    default: float | int | None = None

    def __post_init__(self):
        steps_none = self.steps is None
        resolution_none = self.resolution is None
        if steps_none == resolution_none:
            msg = "!range object should define one of steps and resolution parameters."
            raise ValueError(msg)

        if self.steps is not None and self.steps <= 0:
            raise ValueError("!range object should have a positive number of steps")

        if self.resolution is not None:
            if self.progression == "linear" and self.resolution <= 0:
                raise ValueError("The resolution of a linear !range must be >0")
            elif self.progression == "geometric" and self.resolution <= 1:
                raise ValueError("The resolution of a geometric !range must be >1")

    def to_array(self) -> np.ndarray:
        if self.progression == "linear":
            if self.steps is not None:
                steps = self.steps
            else:  # Then use resolution
                resolution = self.resolution
                assert resolution is not None
                steps = ceil(abs(self.end - self.start) / resolution) + 1

            return np.linspace(self.start, self.end, num=steps)
        else:  # progression == "geometric"
            if self.steps is not None:
                steps = self.steps
            else:  # Then use resolution
                resolution = self.resolution
                assert resolution is not None
                global_ratio = self.end / self.start
                if global_ratio < 1:
                    global_ratio = 1 / global_ratio

                steps = ceil(np.log(global_ratio) / np.log(resolution)) + 1

            return np.geomspace(self.start, self.end, num=steps)

    def __iter__(self) -> Iterator[float]:
        return iter(self.to_array())


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
        if isinstance(value, NumericRange):
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

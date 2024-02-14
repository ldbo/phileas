from math import ceil
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

import numpy as np


def load_yaml_dict_from_file(file: Path | str) -> dict:
    yaml = YAML(typ="safe")
    data = yaml.load(file)

    if data is None:
        data = dict()

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

    return __convert_actual_range_to_sequence(range_dict)


def __convert_actual_range_to_sequence(range_dict: dict[str, Any]) -> np.ndarray:
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

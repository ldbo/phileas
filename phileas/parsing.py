"""
This module allows to parse configuration files (usually using the YAML format)
to data and iteration trees, as defined in the `iteration` module. In
particular, it defines the supported custom YAML types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from types import NoneType
from typing import Any, ClassVar, Generic, Literal, TypeVar

import numpy as np
from ruamel import yaml
from ruamel.yaml import YAML

from phileas import iteration
from phileas.iteration import (
    CartesianProduct,
    DataTree,
    IterationLiteral,
    IterationTree,
    _NoDefault,
)

_data_tree_parser = YAML(typ="safe")
_iteration_tree_parser = YAML(typ="safe")


def load_data_tree_from_yaml_file(file: Path | str) -> DataTree:
    """
    Parses a YAML configuration file (from its path or its content) into a
    data tree.
    """
    return _data_tree_parser.load(file)


def load_iteration_tree_from_yaml_file(file: Path | str) -> IterationTree:
    """
    Parses a YAML configuration file (from its path or its content) into an
    iteration tree. Iteration will be based on cartesian products, and
    iteration leaves can be specified with custom YAML types.
    """
    raw_data = _iteration_tree_parser.load(file)
    return raw_yaml_structure_to_iteration_tree(raw_data)


### Custom YAML types ###


class YamlCustomType(ABC):
    @abstractmethod
    def to_iteration_tree(self) -> IterationTree:
        raise NotImplementedError()


# Numeric type of the Range
RT = TypeVar("RT", bound=int | float)


@_iteration_tree_parser.register_class
@dataclass
class Range(YamlCustomType, Generic[RT]):
    """
    Range of numbers, that can optionally (but usually will) specify an
    iteration method. It can be converted to an iteration leaf using
    `to_iteration_tree`.

    The `start` and `end` attributes are mandatory, and `default` can be
    optionally specified.

    `steps` or `resolution` (but not both) can be specified. If they are, and
    `progression` is not, or equals `"linear"`, the range will represent
      - an `iteration.IntegerRange` if `start` and `end` are integers and
        `resolution` is used and is an integer;
      - an `iteration.LinearRange` otherwise.

    If `progression` is `geometric`, the `Range` will represent an
    `iteration.GeometricRange`.

    If neither `steps` nor `resolution` is specified, the range will represent
    an `iteration.NumericRange`.
    """

    yaml_tag: ClassVar[str] = "!range"
    start: RT
    end: RT
    default: RT | None = None
    steps: int | None = None
    resolution: float | int | None = None
    progression: Literal["linear"] | Literal["geometric"] = "linear"

    def __post_init__(self):
        if self.steps is not None and self.resolution is not None:
            msg = "!range object must have only one of steps and resolution."
            raise ValueError(msg)

        if not isinstance(self.steps, (int, NoneType)):
            raise ValueError("!range steps parameter must be an integer.")

        if self.steps is not None and self.steps <= 0:
            raise ValueError("!range steps must be positive.")

        if self.resolution is not None and self.resolution <= 0:
            raise ValueError("!range resolution parameter must be positive.")

        if self.progression not in ("linear", "geometric"):
            raise ValueError("!range progression must be linear or geometric.")

    def to_iteration_tree(self) -> IterationTree:
        start = self.start
        end = self.end

        default: RT | _NoDefault
        if self.default is None:
            default = iteration.no_default
        else:
            default = self.default

        def is_int(i: object) -> bool:
            return isinstance(i, int)

        if self.steps is None and self.resolution is None:
            return iteration.NumericRange(start, end, default_value=default)
        elif (
            is_int(start)
            and is_int(end)
            and is_int(self.resolution)
            and (is_int(default) or default is iteration.no_default)
        ):
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert isinstance(default, int) or default is iteration.no_default
            assert isinstance(self.resolution, int)
            return iteration.IntegerRange(
                start, end, default_value=default, step=self.resolution  # type: ignore[arg-type]
            )
        elif self.progression in (None, "linear"):
            assert isinstance(end, (int, float))
            assert isinstance(start, (int, float))
            if self.resolution is not None:
                steps = ceil(abs(end - start) / self.resolution) + 1  # type: ignore[operator]
            else:
                assert self.steps is not None
                steps = self.steps

            return iteration.LinearRange(start, end, default_value=default, steps=steps)
        else:
            if self.resolution is not None:
                global_ratio = self.end / self.start  # type: ignore[operator]
                if global_ratio < 1:
                    global_ratio = 1 / global_ratio
                steps = ceil(np.log(global_ratio) / np.log(self.resolution)) + 1
            else:
                assert self.steps is not None
                steps = self.steps

            return iteration.GeometricRange(
                start, end, default_value=default, steps=steps
            )


@_iteration_tree_parser.register_class
@dataclass
class Sequence(YamlCustomType):
    yaml_tag: ClassVar[str] = "!sequence"
    elements: list[iteration.DataTree]
    default: iteration.DataTree | None = None

    @classmethod
    def to_yaml(cls, representer: yaml.Representer, node: "Sequence"):
        if node.default is None:
            return representer.represent_sequence(cls.yaml_tag, node.elements)

        return representer.represent_mapping(
            cls.yaml_tag, {"elements": node.elements, "default": node.default}
        )

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, node: yaml.Node):
        if isinstance(node, yaml.SequenceNode):
            elements = constructor.construct_sequence(node, deep=True)
            default = None
        elif isinstance(node, yaml.MappingNode):
            mapping = constructor.construct_mapping(node, deep=True)
            elements = mapping["elements"]
            if not isinstance(elements, list):
                raise TypeError("!sequence elements field must be a sequence")

            default = mapping.get("default", None)
        else:
            msg = "!sequence must be a scalar sequence of a mapping with keys "
            msg += "elements [and default]"
            raise TypeError(msg)

        return Sequence(elements=elements, default=default)

    def to_iteration_tree(self) -> IterationTree:
        default: DataTree | _NoDefault
        if self.default is None:
            default = iteration.no_default
        else:
            default = self.default

        return iteration.Sequence(self.elements, default)


### Conversion to iteration tree ###


def raw_yaml_structure_to_iteration_tree(structure: Any) -> IterationTree:
    if isinstance(structure, list):
        list_children = list(map(raw_yaml_structure_to_iteration_tree, structure))
        return CartesianProduct(list_children)
    elif isinstance(structure, dict):
        dict_children = {
            key: raw_yaml_structure_to_iteration_tree(value)
            for key, value in structure.items()
        }
        return CartesianProduct(dict_children)
    elif isinstance(structure, YamlCustomType):
        return structure.to_iteration_tree()
    elif isinstance(structure, (NoneType, bool, str, int, float)):
        return IterationLiteral(structure)  # type: ignore[type-var]
    else:
        raise ValueError(f"Unsupported value {structure}.")

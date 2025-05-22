"""
This module allows to parse configuration files (usually using the YAML format)
to data and iteration trees, as defined in the `iteration` module. In
particular, it defines the supported custom YAML types.
"""

from __future__ import annotations

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
from phileas.iteration import Key, NumpyRNG

# Warning: using the safe loader disables calling __post_init__ in dataclasses,
# which effectively skips data verification in some custom YAML types. Using the
# round-trip loader (ie. not specifying `typ`) solve this issue, but changes the
# signature of `construct_mapping`.
_data_tree_parser = YAML(typ="safe")
_iteration_tree_parser = YAML(typ="safe")


def load_data_tree_from_yaml_file(file: Path | str) -> iteration.DataTree:
    """
    Parses a YAML configuration file (from its path or its content) into a
    data tree.
    """
    return _data_tree_parser.load(file)


def load_iteration_tree_from_yaml_file(file: Path | str) -> iteration.IterationTree:
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
    def to_iteration_tree(self) -> iteration.IterationTree:
        raise NotImplementedError()


@_iteration_tree_parser.register_class
@dataclass
class Configurations(YamlCustomType):
    """
    Configurations container. It holds a dictionary of configurations, and has
    three optional arguments: `_default`, `_move_up` and `_insert_name`.

    `_default` specifies the name of the default configuration.

    `_move_up` is a boolean defaulting to `False`. If it is set, the content of
    the chosen configuration will be moved one level up.

    `_insert_name` is a boolean defaulting to `False`. If it is set, the name of
    the chosen configuration will be inserted in the final `DataTree`. If
    `move_up`, then it is assigned to the key that previously hold the
    `!configurations` node. Otherwise, it is inserted under the name
    `"_configuration"`.

    See `iteration.Configurations` for more details on the impact of `_move_up`
    and `_insert_name`.
    """

    yaml_tag: ClassVar[str] = "!configurations"
    configurations: dict[Key, Any]
    default: Key
    move_up: bool
    insert_name: bool

    @classmethod
    def from_yaml(
        cls, constructor: yaml.Constructor, node: yaml.Node
    ) -> Configurations:
        if isinstance(node, yaml.MappingNode):
            mapping = constructor.construct_mapping(node, deep=True)
            default = mapping.pop("_default", None)
            move_up = mapping.pop("_move_up", False)
            insert_name = mapping.pop("_insert_name", False)
            return Configurations(
                configurations=mapping,
                default=default,
                move_up=move_up,
                insert_name=insert_name,
            )
        else:
            raise TypeError("Configurations must be stored in a named map.")

    def to_iteration_tree(self) -> iteration.IterationTree:
        return iteration.Configurations(
            {
                name: raw_yaml_structure_to_iteration_tree(config)
                for name, config in self.configurations.items()
            },
            default_configuration=self.default,
            move_up=self.move_up,
            insert_name=self.insert_name,
        )


@_iteration_tree_parser.register_class
@dataclass
class CartesianProduct(YamlCustomType):
    """
    Cartesian product node, see
    {py:class}`~phileas.iteration.CartesianProduct` for the supported
    arguments.
    """

    yaml_tag: ClassVar[str] = "!product"
    children: dict[Key, Any] | list
    order: list[Key] | None
    lazy: bool
    snake: bool

    @classmethod
    def from_yaml(
        cls, constructor: yaml.Constructor, node: yaml.Node
    ) -> CartesianProduct:
        if isinstance(node, yaml.MappingNode):
            mapping = constructor.construct_mapping(node, deep=True)
            order = mapping.pop("_order", None)
            lazy = mapping.pop("_lazy", False)
            snake = mapping.pop("_snake", False)
            return CartesianProduct(
                children=mapping,
                order=order,
                lazy=lazy,
                snake=snake,
            )
        else:
            children = constructor.construct_sequence(node, deep=True)
            return CartesianProduct(children, order=None, lazy=False, snake=False)

    def to_iteration_tree(self) -> iteration.IterationTree:
        children: dict[Key, iteration.IterationTree] | list[iteration.IterationTree]
        if isinstance(self.children, list):
            children = [
                raw_yaml_structure_to_iteration_tree(child) for child in self.children
            ]
        else:
            assert isinstance(self.children, dict)
            children = {
                name: raw_yaml_structure_to_iteration_tree(child)
                for name, child in self.children.items()
            }
        return iteration.CartesianProduct(
            children,
            order=self.order,
            lazy=self.lazy,
            snake=self.snake,
        )


@_iteration_tree_parser.register_class
@dataclass
class Union(YamlCustomType):
    """
    Union node, see {py:class}`~phileas.iteration.Union` for the supported
    arguments.
    """

    yaml_tag: ClassVar[str] = "!union"
    children: dict[Key, Any] | list
    order: list[Key] | None
    lazy: bool
    preset: Literal["first"] | Literal["default"] | None
    common_preset: bool
    reset: Literal["first"] | Literal["last"] | Literal["default"] | None

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, node: yaml.Node) -> Union:
        if isinstance(node, yaml.MappingNode):
            mapping = constructor.construct_mapping(node, deep=True)
            order = mapping.pop("_order", None)
            lazy = mapping.pop("_lazy", False)
            preset = mapping.pop("_preset", "first")
            common_preset = mapping.pop("_common_preset", False)
            reset = mapping.pop("_reset", "first")
            return Union(
                children=mapping,
                order=order,
                lazy=lazy,
                preset=preset,
                common_preset=common_preset,
                reset=reset,
            )
        else:
            children = constructor.construct_sequence(node, deep=True)
            return Union(
                children,
                order=None,
                lazy=False,
                preset="first",
                common_preset=False,
                reset="first",
            )

    def to_iteration_tree(self) -> iteration.IterationTree:
        children: dict[Key, iteration.IterationTree] | list[iteration.IterationTree]
        if isinstance(self.children, list):
            children = [
                raw_yaml_structure_to_iteration_tree(child) for child in self.children
            ]
        else:
            assert isinstance(self.children, dict)
            children = {
                name: raw_yaml_structure_to_iteration_tree(child)
                for name, child in self.children.items()
            }
        return iteration.Union(
            children,
            order=self.order,
            lazy=self.lazy,
            preset=self.preset,
            common_preset=self.common_preset,
            reset=self.reset,
        )


@_iteration_tree_parser.register_class
@dataclass
class Shuffle(YamlCustomType):
    """
    Shuffle node, whose iteration returns a permutation of its only child. It
    has a single `child` field.
    """

    yaml_tag: ClassVar[str] = "!shuffle"
    child: Any

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, node: yaml.Node) -> Shuffle:
        mapping = constructor.construct_mapping(node, deep=True)
        try:
            child = mapping.pop("child")
        except KeyError as e:
            raise ValueError("!shuffle requires a child field") from e
        return Shuffle(child)

    @classmethod
    def to_yaml(cls, representer: yaml.Representer, node: Shuffle):
        if isinstance(node.child, dict):
            return representer.represent_mapping(cls.yaml_tag, node.child)
        else:
            return representer.represent_sequence(cls.yaml_tag, node.child)

    def to_iteration_tree(self) -> iteration.IterationTree:
        return iteration.Shuffle(raw_yaml_structure_to_iteration_tree(self.child))


@_iteration_tree_parser.register_class
@dataclass
class Pick(YamlCustomType):
    """
    Pick node, whose iteration alternatively returns a single of its children.
    It contains a named mapping, with a `_default_child` field, containing
    the key of its default child.
    """

    yaml_tag: ClassVar[str] = "!pick"
    children: dict[Key, Any]
    default_child: Key

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, node: yaml.Node) -> Pick:
        mapping = constructor.construct_mapping(node, deep=True)
        default_child = mapping.pop("_default_child", None)

        return Pick(children=mapping, default_child=default_child)

    @classmethod
    def to_yaml(cls, representer: yaml.Representer, node: Pick):
        return representer.represent_mapping(
            cls.yaml_tag, node.children | {"_default_child": node.default_child}
        )

    def to_iteration_tree(self) -> iteration.IterationTree:
        return iteration.Pick(
            {
                name: raw_yaml_structure_to_iteration_tree(config)
                for name, config in self.children.items()
            },
            default_child=self.default_child,
        )


#: Numeric type of the Range
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

    def to_iteration_tree(self) -> iteration.IterationTree:
        start = self.start
        end = self.end

        default: RT | iteration._NoDefault
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
    def to_yaml(cls, representer: yaml.Representer, node: Sequence):
        if node.default is None:
            return representer.represent_sequence(cls.yaml_tag, node.elements)

        return representer.represent_mapping(
            cls.yaml_tag, {"elements": node.elements, "default": node.default}
        )

    @classmethod
    def from_yaml(cls, constructor: yaml.Constructor, node: yaml.Node) -> Sequence:
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

    def to_iteration_tree(self) -> iteration.IterationTree:
        default: iteration.DataTree | iteration._NoDefault
        if self.default is None:
            default = iteration.no_default
        else:
            default = self.default

        return iteration.Sequence(self.elements, default)


@_iteration_tree_parser.register_class
@dataclass
class Random(YamlCustomType):
    yaml_tag: ClassVar[str] = "!random"
    distribution: str
    parameters: dict[str, Any]
    size: int | None = None
    default: iteration.DataTree | None = None

    def to_iteration_tree(self) -> iteration.IterationTree:
        return NumpyRNG(
            seed=None,
            size=self.size,
            default_value=self.default,
            distribution=getattr(np.random.Generator, self.distribution),
            kwargs=self.parameters,
        )


### Conversion to iteration tree ###


def raw_yaml_structure_to_iteration_tree(structure: Any) -> iteration.IterationTree:
    if isinstance(structure, list):
        list_children = list(map(raw_yaml_structure_to_iteration_tree, structure))
        return iteration.CartesianProduct(list_children)
    elif isinstance(structure, dict):
        dict_children = {
            key: raw_yaml_structure_to_iteration_tree(value)
            for key, value in structure.items()
        }
        return iteration.CartesianProduct(dict_children)
    elif isinstance(structure, YamlCustomType):
        return structure.to_iteration_tree()
    elif isinstance(structure, (NoneType, bool, str, int, float)):
        return iteration.IterationLiteral(structure)  # type: ignore[type-var]
    else:
        raise ValueError(f"Unsupported value {structure}.")

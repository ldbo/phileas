"""
Defines :py:class:`ExperimentFactory` and :py:class:`Loader` related classes.
"""

from __future__ import annotations

import inspect
import io
import operator
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import reduce
from itertools import chain
from logging import Logger
from pathlib import Path
from types import NoneType
from typing import Any, Callable, ClassVar, cast

import dacite
import graphviz  # type: ignore[import]
from ruamel.yaml import YAML

from . import parsing
from .iteration import DataTree, IterationTree, Key
from .logging import logger

### Loaders ###


class Loader(ABC):
    """
    Class used to initiate a connection to an instrument, and then configure it.
    In order to add the support for a new instrument, you should subclass it
    and register it in an :py:class:`ExperimentFactory`.
    """

    #: Name of the loader, which is usually the name of the loaded instrument.
    #: It is to be matched with the bench configuration ``loader`` field.
    name: ClassVar[str]

    #: Interfaces the loaded instrument supports. Interfaces are arbitrary names
    #: referred to in the experiment configuration file, allowing to choose
    #: which bench instrument is used for which experiment instrument, depending
    #: on its ``interface`` field.
    interfaces: ClassVar[set[str]]

    #: Reference to the instrument factory which has instantiated the loader.
    #: This allows loaders to have access, among other things, to already
    #: instantiated instruments.
    instruments_factory: ExperimentFactory

    #: Logging handler that should be used for logging loader-specific
    #: information.
    logger: Logger

    def __init__(self, name: str, instruments_factory: ExperimentFactory):
        """
        When subclassing `Loader`, the signature of the constructor should not
        be modified, and this constructor should be called.
        """
        self.instruments_factory = instruments_factory
        self.logger = logger.getChild(name)

    @abstractmethod
    def initiate_connection(self, configuration: dict) -> Any:
        """
        Initialize a connection to a given instrument, given its bench
        configuration, and return it.
        """
        raise NotImplementedError()

    @abstractmethod
    def configure(self, instrument: Any, configuration: dict):
        """
        Configure an instrument - whose connection has already been initiated -
        using its experiment configuration. The instrument should be modified,
        and is not returned.
        """
        raise NotImplementedError()

    def get_effective_configuration(
        self, instrument: Any, configuration: None | dict[str, DataTree] = None
    ) -> dict[str, DataTree]:
        """
        Returns the effective configuration of an instrument. The purpose of
        this method is to be called after :py:meth:`configure`, in order to
        verify that the required parameters have been properly applied. It
        might be useful in different situations, for example:

        - the instrument might have a parameter with which the driver uses a
          best effort approach. If the user requests an unsupported value, it
          automatically chooses a supported value. For example, the only
          integer values might be supported, and the driver rounds the
          user-supplied parameter.
        - Setting a parameter might not be deterministic. For example, the
          effective position of a piezoelectric actuator differs from its
          target.
        - The user manually configures an instrument, and wants to get its
          configuration back. In this case, :py:meth:`dump_state` might also be
          interesting.

        If ``configuration`` is not supplied, the entire configuration of the
        instrument is returned. It should have the same structure as the
        argument of :py:meth:`configure`.

        Otherwise, if ``configuration`` is supplied, its values are replaced
        with the effective parameter values. This is typically used to verify
        the value of a configuration that has just been applied
        with :py:meth:`configure`. Note that, in this case, ``configuration``
        might be modified, and it is likely that ``get_effective_configuration
        (instrument, configuration) is configuration``.

        It is not necessary to implement this method.
        """
        raise NotImplementedError()

    def get_id(self, instrument: Any) -> str:
        """
        Returns a unique identifier of an instrument. Two physically different
        instruments are expected to have different identifiers.
        """
        raise NotImplementedError()

    def dump_state(self, instrument: Any) -> DataTree:
        """
        Dump the state of an instrument. It should, as much as possible,
        represent the exact state of the instrument. It should contain the
        value of parameters that can be configured with :py:meth:`configure`,
        but also other ones. These might include, among others, calibration
        values, the exact mode which is used, firmware and driver
        versions, *etc*.

        The purpose of this state is to enable comparing similar instruments, to
        find out if they can be swapped or how to make them interchangeable.

        It is not necessary to implement this method. However, when it is
        implemented, together with :py:meth:`restore_state`, it allows to
        guarantee that the instrument always starts in the same state. It can
        also be used to capture a manual configuration, in order to replicate
        it later.
        """
        raise NotImplementedError()

    def restore_state(self, instrument: Any, state: DataTree):
        """
        Put the instrument in a previously recorded state.
        See :py:meth:`dump_state`.
        """
        raise NotImplementedError()

    @classmethod
    def get_markdown_documentation(cls) -> str:
        """
        Generate a markdown documentation of the loader, which is based on the
        docstrings of the :py:meth:`initiate_connection` and
        :py:meth:`configure` methods.
        """
        doc = f"# {cls.__name__}\n"
        doc += f" - Name: `{cls.name}`\n"

        if len(cls.interfaces) > 0:
            doc += " - Interfaces:\n"
            doc += "".join([f"   - `{i}`\n" for i in sorted(cls.interfaces)])
        else:
            doc += " - This loader is for bench-only instruments\n"

        if cls.__doc__ is not None:
            doc += f"\n{inspect.cleandoc(cls.__doc__)}\n"

        if cls.initiate_connection.__doc__ is not None:
            initiate_doc = inspect.cleandoc(cls.initiate_connection.__doc__)
            doc += f"\n## Initialization\n{initiate_doc}\n"

        if len(cls.interfaces) > 0 and cls.configure.__doc__ is not None:
            configure_doc = inspect.cleandoc(cls.configure.__doc__)
            doc += f"\n## Configuration\n{configure_doc}\n"

        return doc


def build_loader(
    name_: str,
    interfaces_: set[str],
    initiate_connection: Callable[[dict], Any],
    configure: Callable[[Any, dict], None],
) -> type[Loader]:
    """
    Build a loader class from its name, interfaces and different methods. Note
    that using this function does not allow a loader instance to access the
    instrument factory that uses it.
    """

    class BuiltLoader(Loader):
        name = name_
        interfaces = interfaces_

        def initiate_connection(self, configuration: dict) -> Any:
            return initiate_connection(configuration)

        def configure(self, instrument: Any, configuration: dict):
            configure(instrument, configuration)

    name = "".join(w.capitalize() for w in name_.lower().split("_"))
    if not name.endswith("Loader"):
        name += "Loader"

    BuiltLoader.__name__ = name
    BuiltLoader.initiate_connection.__doc__ = initiate_connection.__doc__
    BuiltLoader.configure.__doc__ = configure.__doc__

    return BuiltLoader


def _add_loader(
    loaders: dict[str, type[Loader]],
    loader: (
        type[Loader]
        | tuple[
            str,
            set[str],
            Callable[[DataTree], Any],
            Callable[[Any, DataTree], None],
        ]
    ),
):
    """
    Adds a loader to a map of loader classes. The loader can be supplied either
    with
     - a `Loader` class
     - a 4-tuple containing the name, interfaces and different methods of a
       loader. In this case, it is not possible to reference the instruments
       factory from the loader, so for complex cases it is advised to use a
       `Loader` class.
    """
    if not inspect.isclass(loader):  # type: ignore[misc]
        # Here, loader is guaranteed to be a tuple, as it is not a class object,
        # but this is not understood by mypy. Using isinstance(loader, tuple)
        # could lead to issues if the user were to define a loader inheriting
        # from tuple.
        loader = cast(
            tuple[
                str,
                set[str],
                Callable[[DataTree], Any],
                Callable[[Any, DataTree], None],
            ],
            loader,
        )
        loader = build_loader(*loader)

    name = loader.name
    if name in loaders:
        logger.warning(f"Overwriting the loader {name}")

    loaders[name] = loader


#: Default loaders that every instrument factory has on init. It is made to be
#: modified by :py:func:`register_default_loader` and
#: :py:func:`clear_default_loaders` only.
_DEFAULT_LOADERS: dict[str, type[Loader]] = {}


def register_default_loader(
    loader: (
        type[Loader]
        | tuple[
            str,
            set[str],
            Callable[[dict], Any],
            Callable[[Any, dict], None],
        ]
    ),
) -> (
    type[Loader]
    | tuple[
        str,
        set[str],
        Callable[[dict], Any],
        Callable[[Any, dict], None],
    ]
):
    """
    Register a loader to be added on init by every
    new :py:class:`ExperimentFactory`. See :py:func:`_add_loader` for the
    specifications of the arguments.

    This function can either be used directly, or as a class decorator.
    """
    if inspect.isclass(loader):
        name = loader.name
        interfaces = loader.interfaces
    else:
        loader = cast(
            tuple[
                str, set[str], Callable[[DataTree], Any], Callable[[Any, DataTree], Any]
            ],
            loader,
        )
        name = loader[0]
        interfaces = loader[1]

    logger.debug(f"Register default loader {name} for interfaces {interfaces}")
    _add_loader(_DEFAULT_LOADERS, loader)

    return loader


def clear_default_loaders():
    """
    Clear the default loaders database.
    """
    _DEFAULT_LOADERS.clear()


### Instrument structures ###


@dataclass
class BenchInstrument:
    name: str

    #: Own loader of the instrument, which is not shared with any other
    #: instrument.
    loader: Loader

    #: Bench configuration of the instrument, stripped of the reserved keywords
    #: entries.
    configuration: dict[Key, DataTree]

    #: None before initialization.
    instrument: Any | None = None

    def initiate_connection(self):
        self.loader.logger.info("Initiating connection.")
        self.instrument = self.loader.initiate_connection(self.configuration)


@dataclass
class ExperimentInstrument:
    name: str
    interface: str
    bench_instrument: BenchInstrument

    #: Experiment configuration of the instrument, stripped of the reserved
    #: keywords entries
    configuration: dict[str, Any]


### Bench state structures ###


@dataclass
class InstrumentState:
    """
    Serializable state of a :py:class:`BenchInstrument`.
    """

    #: Name of its loader.
    loader: str

    #: Supported interfaces of the loader.
    interfaces: list[str]

    #: Configuration applied to the instrument.
    configuration: DataTree

    #: Identifier of the instrument, see :py:meth:`Loader.get_id`.
    id: str | None

    #: State of the instrument, see :py:meth:`Loader.dump_state`.
    state: DataTree


#: Parser used to load stored states.
_yaml_load_parser = YAML(typ="safe")

#: Parser used to dump states. It is not ``typ="safe"`` in order to preserve the
#: order of mappings.
_yaml_dump_parser = YAML()


@dataclass
class BenchState:
    """
    Serializable state of a whole bench.
    """

    #: Version of Phileas.
    version: str

    #: Instruments of the bench.
    instruments: dict[str, InstrumentState]

    def to_yaml(self, destination: Path | None = None) -> str | None:
        """
        Serialize a bench state to YAML. If you supply a path in
        ``destination``, directly write the state to the file located there.
        Otherwise, return a string.
        """
        raw = asdict(self)
        ordered = {}
        ordered["version"] = raw["version"]
        ordered["instruments"] = raw["instruments"]

        if destination is None:
            buffer = io.StringIO()
            _yaml_dump_parser.dump(ordered, buffer)
            return buffer.getvalue()
        else:
            with open(destination, "x") as f:
                _yaml_dump_parser.dump(ordered, f)

            return None

    @classmethod
    def from_yaml(cls, source: str | Path) -> BenchState:
        """
        Deserialize a bench state from YAML. If you supply a path in ``source``,
        read the state from the file located there. Otherwise, read the
        string.
        """
        raw: Any
        if isinstance(source, str):
            raw = _yaml_load_parser.load(source)
        else:
            with open(source) as f:
                raw = _yaml_load_parser.load(f)

        config = dacite.Config(forward_references={"_NoDefault": None})
        return dacite.from_dict(data_class=BenchState, data=raw, config=config)


### Filters ###


class Filter(ABC):
    """
    Class used to store an expression tree representing the filtering
    expressions stored in the ``filter`` entry of the experiment configuration.
    """

    @abstractmethod
    def verifies(self, instrument: BenchInstrument) -> bool:
        """
        Checks whether a bench instrument satisfies the filter.
        """
        raise NotImplementedError()

    @staticmethod
    def build_filter(filter_entry: dict[str, Any] | list[dict]) -> "Filter":
        """
        Filter parser, which is given the ``filter`` entry of the experiment
        configuration file, and returns the corresponding expression tree.

        Todo:
            Only parsing single-level dict is supported for now. Parsing
            nested filters should be implemented.
        """
        if isinstance(filter_entry, dict):
            filters: list[Filter] = [
                AttributeFilter(k, v) for k, v in filter_entry.items()
            ]
            t: Filter = ConstantFilter(True)
            return reduce(
                lambda f1, f2: FiltersCombination(f1, f2, operator.and_),
                filters,
                t,
            )
        elif isinstance(filter_entry, list):
            raise NotImplementedError("List filters are not yet supported")


@dataclass(frozen=True)
class AttributeFilter(Filter):
    """
    Filter checking that a bench instrument has the expected entry value in its
    configuration. This is one of the leaves of the filters expression tree.
    """

    field: str
    value: Any

    def verifies(self, instrument: BenchInstrument) -> bool:
        return instrument.configuration[self.field] == self.value


@dataclass(frozen=True)
class ConstantFilter(Filter):
    """
    Literal filter.
    """

    value: bool

    def verifies(self, instrument: BenchInstrument) -> bool:
        return self.value


@dataclass(frozen=True)
class FiltersCombination(Filter):
    """
    Filters combination based on a binary operator.
    """

    filter1: Filter
    filter2: Filter
    operation: Callable[[bool, bool], bool]

    def verifies(self, instrument: BenchInstrument) -> bool:
        return self.operation(
            self.filter1.verifies(instrument), self.filter2.verifies(instrument)
        )


@dataclass(frozen=True)
class Connection:
    src: str
    src_port: list[str]
    dst: str
    dst_port: list[str]
    attr: str


class ExperimentFactory:
    """
    The experiment factory is used to
     - parse configuration files, cleaning them of reserved keywords;
     - match the experiment instruments to their bench instruments and loaders;
     - initiate the connection to the instruments, and configure them.
    """

    #: If the bench configuration is supplied by file, stores its path.
    #: Otherwise, stores ``None``.
    bench_file: Path | None

    #: If the experiment configuration is supplied by file, stores its path.
    #: Otherwise, stores ``None``.
    experiment_file: Path | None

    #: Bench configuration stripped of the reserved keyword entries.
    bench_config: dict[str, DataTree]

    #: Experiment configuration stripped of the reserved keyword entries.
    experiment_config: IterationTree

    #: Instruments whose connection has been initiated, configured or not.
    experiment_instruments: dict[str, Any]

    #: The supported loader classes, stored with their name.
    loaders: dict[str, type[Loader]]

    #: Bench instruments, created by :py:meth:`__preinit_bench_instruments`.
    __bench_instruments: dict[str, BenchInstrument]
    #: Experiment instrument, created
    #: by :py:meth:`__preconfigure_experiment_instruments`.
    __experiment_instruments: dict[str, ExperimentInstrument]
    #: Experiment connections, created by :py:meth:`__build_connection_graph`.
    __connections: list[Connection]

    ### Instruments initialization ###

    def __init__(
        self,
        bench: Path | str | dict[str, DataTree],
        experiment: Path | str | IterationTree,
    ):
        """
        Parse the configuration files (given either by their file path, raw
        content or parsed content), clean them and prepare the instruments.
        """
        # Loaders
        self.loaders = {}
        self.loaders.update(_DEFAULT_LOADERS)

        # Bench configuration handling
        bench_config: DataTree
        if isinstance(bench, Path):
            self.bench_file = bench
            bench_config = parsing.load_data_tree_from_yaml_file(bench)
            logger.info(f"Bench configuration loaded from {bench}.")
        else:
            logger.info("Bench configuration supplied as a data tree.")
            self.bench_file = None
            if isinstance(bench, str):
                bench_config = parsing.load_data_tree_from_yaml_file(bench)
            else:
                bench_config = bench  # type: ignore[assignment]

        if bench_config in (None, ""):
            bench_config = {}
        elif not isinstance(bench_config, dict):
            raise ValueError("The bench configuration must be a dictionary.")

        if any(not isinstance(key, str) for key in bench_config.keys()):
            raise ValueError("The bench configuration must only have string keys.")

        self.bench_config = bench_config  # type: ignore[assignment]
        self.__preinit_bench_instruments()
        self.experiment_instruments = {}

        # Experiment configuration handling
        if isinstance(experiment, Path):
            self.experiment_file = experiment
            experiment_config = parsing.load_iteration_tree_from_yaml_file(experiment)
            logger.info(f"Experiment configuration loaded from {experiment}.")
        else:
            self.experiment_file = None
            if isinstance(experiment, str):
                experiment_config = parsing.load_iteration_tree_from_yaml_file(
                    experiment
                )
            else:
                experiment_config = experiment
            logger.info("Experiment configuration supplied as an iteration tree.")

        p_experiment_config = experiment_config.to_pseudo_data_tree()

        if not isinstance(p_experiment_config, (dict, NoneType)):
            msg = "The experiment configuration must be a empty, or a top-level dictionary"
            raise ValueError(msg)

        self.experiment_config = experiment_config
        self.__preconfigure_experiment_instruments(p_experiment_config)
        self.__build_connection_graph(p_experiment_config)

    def __preinit_bench_instruments(self):
        """
        Prepare for the initialization of the bench instruments:
         - create and assign a loader to each of them,
         - clean their configuration of reserved entries (loader),
        but leave the instruments non-initialized.
        """
        self.__bench_instruments = {}
        for name, config in self.bench_config.items():
            if not isinstance(config, dict):
                continue

            if "loader" not in config:
                continue

            loader_value = config.pop("loader")
            if not isinstance(loader_value, str):
                raise TypeError("Loader field must be a string.")

            ChosenLoader = self.loaders[loader_value]
            instrument = BenchInstrument(name, ChosenLoader(name, self), config, None)
            self.__bench_instruments[name] = instrument

            msg = f"Bench instrument {name} assigned to loader {ChosenLoader}."
            logger.info(msg)

    def __preconfigure_experiment_instruments(self, p_experiment_config: dict | None):
        """
        Prepare for the configuration of the experiment instruments:
         - assign a bench instrument to each of them,
         - clean their configuration of reserved entries (interface, filters),
        but leave them in a non-configured state.
        """
        self.__experiment_instruments = {}

        if p_experiment_config is None:
            return

        for name, config in p_experiment_config.items():
            # Filter out non-instrument entries
            if not isinstance(name, str):
                raise ValueError("Experiment instruments must have a string name.")

            if not isinstance(config, dict):
                continue

            # Get the interface
            if "interface" not in config:
                continue
            interface = config["interface"]

            if not isinstance(interface, str):
                raise ValueError(f"Non-str interface {interface} is not supported.")

            self.experiment_config = self.experiment_config.remove_child(
                [name, "interface"]
            )

            # Get the filters
            try:
                filter_dict = config["filter"]

                if not isinstance(filter_dict, dict):
                    raise ValueError(f"filter field {filter_dict} must be a dict.")

                self.experiment_config = self.experiment_config.remove_child(
                    [name, "filter"]
                )
            except KeyError:
                filter_dict = {}

            filter_ = Filter.build_filter(filter_dict)

            # Select the corresponding bench instrument
            bench_instrument = self.__find_bench_instrument(name, interface, filter_)
            self.__experiment_instruments[name] = ExperimentInstrument(
                name, interface, bench_instrument, config
            )

            msg = f"Matching experiment instrument {name} with bench instrument"
            msg += f" {bench_instrument.name}."
            logger.info(msg)

    def __find_bench_instrument(
        self, exp_name: str, interface: str, filter_: Filter
    ) -> BenchInstrument:
        """
        Find a bench instrument matching and interface and a filter.

        Raises:
            KeyError: if there are 0 or more than 1 matching instruments.
        """
        compatible_instruments = []
        for instrument in self.__bench_instruments.values():
            if interface not in instrument.loader.interfaces:
                continue

            if not filter_.verifies(instrument):
                msg = f"Bench instrument {instrument.name} has the interface "
                msg += f"required by {exp_name}, but does not match its filter."
                logger.info(msg)
                continue

            compatible_instruments.append(instrument)

        if len(compatible_instruments) == 0:
            msg = "Cannot find a suitable bench instrument for experiment "
            msg += f"instrument '{exp_name}'"
            raise KeyError(msg)

        if len(compatible_instruments) > 1:
            compatible_names = [i.name for i in compatible_instruments]
            msg = "More than one suitable bench instrument for experiment "
            msg += f"instrument '{exp_name}': {compatible_names}"
            raise KeyError(msg)

        return compatible_instruments[0]

    def __build_connection_graph(self, p_experiment_config: dict | None):
        """
        Extract the connection graph from the experiment configuration, removing
        ``connections`` entries from it.
        """
        connections: list[tuple[str, list[str], str, list[str], str]] = (
            self.__parse_global_connections(p_experiment_config)
        )

        if isinstance(p_experiment_config, dict):
            for name, config in p_experiment_config.items():
                if not isinstance(name, str):
                    continue

                if not isinstance(config, dict):
                    continue

                if "connections" in config:
                    connections += self.__parse_instrument_connections(
                        name, config["connections"], p_experiment_config
                    )
                    self.experiment_config = self.experiment_config.remove_child(
                        [name, "connections"]
                    )

        self.__connections = []
        for src, src_port, dst, dst_port, attr in connections:
            self.__connections.append(
                Connection(
                    src,
                    src_port,
                    dst,
                    dst_port,
                    attr,
                )
            )

    def __parse_global_connections(
        self, p_experiment_config: dict | None
    ) -> list[tuple[str, list[str], str, list[str], str]]:
        connections: list[tuple[str, list[str], str, list[str], str]] = []

        if not isinstance(p_experiment_config, dict):
            return connections

        if "connections" not in p_experiment_config:
            return connections

        global_connections = p_experiment_config["connections"]
        self.experiment_config = self.experiment_config.remove_child(["connections"])

        if not isinstance(global_connections, list):
            raise TypeError("Experiment configuration connections must be a list")

        for connection in global_connections:
            source_str: str = connection["from"]
            destination_str: str = connection["to"]
            attributes: str = connection.get("attributes", "")

            source_elements = source_str.split(".")
            source_instrument = source_elements[0]
            source_port = source_elements[1:]

            destination_elements = destination_str.split(".")
            destination_instrument = destination_elements[0]
            destination_port = destination_elements[1:]

            connections.append(
                (
                    source_instrument,
                    source_port,
                    destination_instrument,
                    destination_port,
                    attributes,
                )
            )

        return connections

    def __parse_instrument_connections(
        self, name: str, config: dict, p_experiment_config: dict
    ) -> list[tuple[str, list[str], str, list[str], str]]:
        connections = []
        for connection in config:
            source_str: str = connection.pop("from", name)
            destination_str: str = connection.pop("to", name)
            attributes: str = connection.get("attributes", "")

            source_elements = source_str.split(".")
            if source_elements[0] in p_experiment_config:
                source_instrument = source_elements[0]
                source_port = source_elements[1:]
            else:
                source_instrument = name
                source_port = source_elements

            destination_elements = destination_str.split(".")
            if destination_elements[0] in p_experiment_config:
                destination_instrument = destination_elements[0]
                destination_port = destination_elements[1:]
            else:
                destination_instrument = name
                destination_port = destination_elements

            connections.append(
                (
                    source_instrument,
                    source_port,
                    destination_instrument,
                    destination_port,
                    attributes,
                )
            )

        return connections

    ### Working with instruments ###

    def initiate_connections(self):
        """
        Lazily initiate the connections of all the bench instruments, *ie.* only
        those that are matched to an experiment instrument are handled.
        """
        logger.info("Initiating connections to the used instruments.")
        for experiment_instrument in self.__experiment_instruments.values():
            be = experiment_instrument.bench_instrument
            be.initiate_connection()
            name = experiment_instrument.name
            self.experiment_instruments[name] = be.instrument

        logger.info("Connections initiation is done.")

    def get_bench_instrument(self, name: str) -> Any:
        """
        Return the bench instrument whose name is specified, using its loader
        :py:meth:`~Loader.initiate_connection` method to create it. If the
        connection to the instrument has already been initiated, the instrument
        is simply returned.

        This is the only way a bench instrument should be retrieved.
        """
        be = self.__bench_instruments[name]
        if be.instrument is None:
            be.initiate_connection()

            for ei_name, ei_inst in self.__experiment_instruments.items():
                bi = ei_inst.bench_instrument
                if bi.name == name:
                    self.experiment_instruments[ei_name] = bi.instrument

        return be.instrument

    def configure_instrument(self, name: str, configuration: dict | None = None):
        """
        Configure an experiment instrument using the given configuration, and
        the :py:meth:`~Loader.configure` method of its loader. If no configuration is
        given, use the instrument default configuration from the experiment
        configuration.
        """
        if configuration is None:
            d_configuration = self.experiment_config[name].default()

            if d_configuration is None:
                return

            if not isinstance(d_configuration, dict):
                raise TypeError("Instrument configuration must be a dict.")

            configuration = d_configuration

        experiment_instrument = self.__experiment_instruments[name]
        instrument = experiment_instrument.bench_instrument.instrument
        loader = experiment_instrument.bench_instrument.loader
        loader.configure(instrument, configuration)

    def configure_experiment(self, configuration: dict | None = None):
        """
        Configure multiple instruments at once, using
        the :py:meth:`~Loader.configure` method of their respective loaders,
        and the entry of ``configuration`` matching their name. If
        ``configuration`` misses the entry of an instrument, the instrument won't
        be configured.

        If no configuration is given, the default experiment configuration is
        used to configure the instruments. In this case, all the instruments are
        configured.
        """
        if configuration is None:
            d_configuration = self.experiment_config.default()

            if d_configuration is None:
                return

            if not isinstance(d_configuration, dict):
                raise TypeError("Instrument configuration must be a dict.")

            configuration = d_configuration

        for name in self.__experiment_instruments:
            if name in configuration:
                config = configuration[name]

                if not isinstance(config, dict):
                    msg = f"Instrument {name} configuration is not a dict."
                    raise TypeError(msg)

                self.configure_instrument(name, config)

    def get_effective_instrument_configuration(
        self, name: str, configuration: dict | None = None
    ) -> dict[str, DataTree]:
        """
        Return the effective configuration of an instrument.
        See :py:meth:`Loader.get_effective_configuration`.
        """
        experiment_instrument = self.__experiment_instruments[name]
        instrument = experiment_instrument.bench_instrument.instrument
        loader = experiment_instrument.bench_instrument.loader
        return loader.get_effective_configuration(instrument, configuration)

    def get_effective_experiment_configuration(
        self, configuration: dict | None = None
    ) -> dict[str, dict[str, DataTree]]:
        """
        Return the effective configuration of the whole bench. If
        ``configuration`` is not supplied, get the configuration of each
        instrument of the bench. Otherwise, only retrieves the configuration of
        instrument which have an entry. The value of the entry is passed to
        :py:meth:`Loader.get_effective_configuration`. If the entry does not
         correspond to an instrument, or its loader does not
         implement :py:meth:`~Loader.get_effective_configuration`, it is
         ignored.
        """
        if configuration is None:
            configuration = dict.fromkeys(self.__experiment_instruments, None)

        effective_configuration = {}
        for name in configuration.keys():
            if name not in self.__experiment_instruments:
                continue

            experiment_instrument = self.__experiment_instruments[name]
            instrument = experiment_instrument.bench_instrument.instrument
            loader = experiment_instrument.bench_instrument.loader
            try:
                effective_configuration[name] = loader.get_effective_configuration(
                    instrument, configuration[name]
                )
            except NotImplementedError:
                logger.warning(
                    f"Loader {loader.name} used by bench instrument {name} does "
                    "not implement get_effective_configuration."
                )

        return effective_configuration

    ### Using the complete state of a bench ###

    def dump_bench_state(self, full: bool = True) -> BenchState:
        """
        Dumps the state of the whole bench.

        When not ``full``, do not dump the state of the instruments. However,
        retrieve its loader and ID information.
        """
        instruments: dict[str, InstrumentState] = {}
        for bench_instrument in self.__bench_instruments.values():
            driver = self.get_bench_instrument(bench_instrument.name)
            loader = bench_instrument.loader

            instrument_state = None
            if full:
                try:
                    instrument_state = bench_instrument.loader.dump_state(driver)
                except NotImplementedError:
                    logger.warning(
                        f"Loader {loader.name} used by bench instrument "
                        f"{bench_instrument.name} does not implement dump_state."
                    )

            instrument_id = None
            try:
                instrument_id = bench_instrument.loader.get_id(driver)
            except NotImplementedError:
                logger.warning(
                    f"Loader {loader.name} used by bench instrument "
                    f"{bench_instrument.name} does not implement get_id."
                )

            instrument_entry = InstrumentState(
                loader=loader.name,
                interfaces=sorted(loader.interfaces),
                configuration=deepcopy(bench_instrument.configuration),
                id=instrument_id,
                state=deepcopy(instrument_state),
            )
            instruments[bench_instrument.name] = instrument_entry

        import phileas

        return BenchState(version=phileas.__version__, instruments=instruments)

    def restore_bench_state(self, state: BenchState, strict: bool = True):
        """
        Restores the a bench state. If ``strict``, requires the following
        serialized values to match with their bench values:

        - :py:attr:`BenchState.version`,
        - the list of the instruments names,
        - :py:attr:`InstrumentState.loader`,
        - :py:attr:`InstrumentState.interfaces`,
        - :py:attr:`InstrumentState.configuration`,
        - :py:attr:`InstrumentState.id`.

        Additionally, every loader is required to
        implement :py:meth:`Loader.restore_state`.

        Otherwise, if ``not strict``, the state of all the instruments with
        matching names and :py:attr:`InstrumentState.loader` is restored.
        Others are ignored.

        Raises:
            ValueError: if the state cannot be restored due to stored state
                        incompatibilities.
        """
        import phileas

        # Check version
        if state.version != phileas.__version__:
            if strict:
                raise ValueError(
                    "Cannot restore a bench state created with Phileas version"
                    f"{state.version}. Current version is {phileas.__version__}"
                )
            logger.warning(
                "Restoring a bench state created with Phileas version "
                f"{state.version}. Current version is {phileas.__version__}."
            )

        # Check instruments list
        state_instruments = set(state.instruments.keys())
        self.__check_instruments_list(state_instruments, strict)

        # Check instruments loaders, ID, interfaces and configuration
        for bi in self.__bench_instruments.values():
            if bi.name not in state_instruments:
                continue

            self.__check_instrument_validity(state.instruments[bi.name], bi, strict)

        # Restore states
        for bi in self.__bench_instruments.values():
            if bi.name not in state_instruments:
                continue

            try:
                bi.loader.restore_state(bi.instrument, state.instruments[bi.name].state)
            except NotImplementedError as err:
                msg = (
                    f"Loader {bi.loader.name} used by bench instrument {bi.name} "
                    "does not implement restore_state."
                )
                if strict:
                    msg += " The state is partially restored."
                    raise NotImplementedError(msg) from err
                logger.warning(msg)

    def __check_instruments_list(self, state_instruments: set[str], strict: bool):
        own_instruments = set(self.__bench_instruments.keys())
        if own_instruments != state_instruments:
            logger.warning("The bench state has different instruments.")

        if not own_instruments.issubset(state_instruments):
            instruments = own_instruments - state_instruments
            logger.warning(
                f"These instruments are only in the current bench: {instruments}."
            )

        if not state_instruments.issubset(own_instruments):
            instruments = state_instruments - own_instruments
            logger.warning(
                f"These instruments are only in the dumped bench state: {instruments}."
            )

        if own_instruments != state_instruments and strict:
            raise ValueError("Cannot restore a bench state with different instruments.")

    @classmethod
    def __check_instrument_validity(
        cls, si: InstrumentState, bi: BenchInstrument, strict: bool
    ):
        def compare(name: str, state_value: Any, bench_value: Any, required: bool):
            if state_value != bench_value:
                msg = (
                    f"State instrument {bi.name} has {name} {state_value}, but "
                    f"in the current bench it has {name} {bench_value}."
                )
                if required or strict:
                    raise ValueError(msg)
                logger.warning(msg)

        compare("loader", si.loader, bi.loader.name, required=True)
        compare("interfaces", set(si.interfaces), bi.loader.interfaces, required=False)
        compare("configuration", si.configuration, bi.configuration, required=False)
        compare("id", si.id, bi.loader.get_id(bi.instrument), required=False)

    ### Loaders ###

    def register_loader(
        self,
        loader: (
            type[Loader]
            | tuple[
                str,
                set[str],
                Callable[[DataTree], Any],
                Callable[[Any, DataTree], Any],
            ]
        ),
    ):
        """
        Register a new loader for this factory. See :py:func:`_add_loader` for
        the specifications of the arguments.
        """
        _add_loader(self.loaders, loader)

    def get_loaders_markdown_documentation(self) -> str:
        """
        Return the Markdown documentation of all the registered loaders of this
        factory, represented by the concatenation of their documentations.
        """
        return "\n\n".join(
            loader.get_markdown_documentation() for loader in self.loaders.values()
        )

    @staticmethod
    def get_default_loaders_markdown_documentation() -> str:
        """
        Return the Markdown documentation of all the default registered loaders,
        represented by the concatenation of their documentations.
        """
        return "\n\n".join(
            loader.get_markdown_documentation() for loader in _DEFAULT_LOADERS.values()
        )

    def get_experiment_graph(self) -> graphviz.Digraph:
        graph = graphviz.Digraph(
            "Experiment instruments connections", node_attr={"shape": "record"}
        )

        instruments_ports: dict[str, set[str]] = {}
        edges: list[tuple[str, str]] = []
        for connection in self.__connections:
            if len(connection.src_port) > 0:
                src_ports = instruments_ports.get(connection.src, set())
                src_ports.add(".".join(connection.src_port))
                instruments_ports[connection.src] = src_ports
                src_port = len(src_ports)
            else:
                src_port = 0

            if len(connection.dst_port) > 0:
                dst_ports = instruments_ports.get(connection.dst, set())
                dst_ports.add(".".join(connection.dst_port))
                instruments_ports[connection.dst] = dst_ports
                dst_port = len(dst_ports)
            else:
                dst_port = 0

            edge = f"{connection.src}:f{src_port}", f"{connection.dst}:f{dst_port}"
            edges.append(edge)

        for instrument, ports in instruments_ports.items():
            concatenated_ports = ("".join(port) for port in sorted(ports))
            labels_iter = enumerate(chain([instrument], concatenated_ports))
            node_label = " | ".join(f"<f{i}> {l}" for i, l in labels_iter)
            graph.node(instrument, node_label)

        graph.edges(edges)

        return graph

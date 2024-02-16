import inspect
import logging
import operator

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import Any, Callable, ClassVar, Generator

import graphviz

from . import parsing


class Loader(ABC):
    """
    Class used to initiate a connection to an instrument, and then configure it.
    In order to add the support for a new instrument, you should subclass it
    and register it in an `ExperimentFactory`.
    """

    #: Name of the loader, which is usually the name of the loaded instrument.
    #: It is to be matched with the bench configuration `loader` field.
    name: ClassVar[str]

    #: Interfaces the loaded instrument supports. Interfaces are arbitrary names
    #: referred to in the experiment configuration file, allowing to choose
    #: which bench instrument is used for which experiment instrument, depending
    #: on its `interface` field.
    interfaces: ClassVar[set[str]]

    #: Reference to the instrument factory which has instantiated the loader.
    #: This allows loaders to have access, among other things, to already
    #: instantiated instruments.
    instruments_factory: "ExperimentFactory"

    def __init__(self, instruments_factory: "ExperimentFactory"):
        """
        When subclassing `Loader`, the signature of the constructor should not
        be modified, and this constructor should be called.
        """
        self.instruments_factory = instruments_factory

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

    @classmethod
    def get_markdown_documentation(cls) -> str:
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
            Callable[[dict], Any],
            Callable[[Any, dict], Any],
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
    if not inspect.isclass(loader):
        loader = build_loader(*loader)

    name = loader.name
    if name in loaders:
        logging.warning(f"Overwriting the loader {name}")

    loaders[name] = loader


#: Default loaders that every instrument factory has on init. It is made to be
#: modified by `register_default_loader` and `clear_default_loaders` only.
_DEFAULT_LOADERS: dict[str, type[Loader]] = {}


def register_default_loader(
    loader: (
        type[Loader]
        | tuple[
            str,
            set[str],
            Callable[[dict], Any],
            Callable[[Any, dict], Any],
        ]
    ),
):
    """
    Register a loader to be added on init by every new `ExperimentFactory`. See
    `_add_loader` for the specifications of the arguments.
    """
    if inspect.isclass(loader):
        name = loader.name
        interfaces = loader.interfaces
    else:
        name = loader[0]
        interfaces = loader[1]

    logging.debug(f"Register default loader {name} for interfaces {interfaces}")
    _add_loader(_DEFAULT_LOADERS, loader)


def clear_default_loaders():
    """
    Clear the default loaders database.
    """
    _DEFAULT_LOADERS.clear()


@dataclass
class BenchInstrument:
    name: str

    #: Own loader of the instrument, which is not shared with any other
    #: instrument
    loader: Loader

    #: Bench configuration of the instrument, stripped of the reserved keywords
    #: entries
    configuration: dict[str, Any]

    #: None before initialization
    instrument: Any | None = None

    def initiate_connection(self):
        self.instrument = self.loader.initiate_connection(self.configuration)


@dataclass
class ExperimentInstrument:
    name: str
    interface: str
    bench_instrument: BenchInstrument

    #: Experiment configuration of the instrument, stripped of the reserved
    #: keywords entries
    configuration: dict[str, Any]


class Filter(ABC):
    """
    Class used to store an expression tree representing the filtering
    expressions stored in the `filter` entry of the experiment configuration.
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
        Filter parser, which is given the `filter` entry of the experiment
        configuration file, and returns the corresponding expression tree.

        # TBD

        Only parsing single-level dict is supported for now. Parsing nested
        filters should be implemented.
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
    Class used to parse configuration files and create the experiment
    environment they describe, making the instruments available.
    """

    bench_file: Path | str
    experiment_file: Path | str

    #: Parsed content of the bench configuration file, which is stripped of the
    #: reserved keyword entries
    bench_config: dict[str, Any]

    #: Content of the experiment configuration file, which is stripped of the
    #: reserved keyword entries
    experiment_config: dict[str, Any]

    #: Instruments whose connection has been initiated, configured or not.
    experiment_instruments: dict[str, Any]

    #: The supported loader classes, stored with their name
    loaders: dict[str, type[Loader]]

    #: Bench instruments, created by __preinit_bench_instruments
    __bench_instruments: dict[str, BenchInstrument]
    #: Experiment instrument, created by __preconfigure_experiment_instruments
    __experiment_instruments: dict[str, ExperimentInstrument]
    #: Experiment connections, created by __build_connection_graph
    __connections: list[Connection]

    def __init__(self, bench_file: Path | str, experiment_file: Path | str):
        """
        Create an experiment factory, parsing the configuration files, so that
        the instruments just need to initiate their connection to be used.
        """
        self.loaders = {}
        self.loaders.update(_DEFAULT_LOADERS)

        self.bench_file = bench_file
        self.bench_config = parsing.load_yaml_dict_from_file(bench_file)
        self.__preinit_bench_instruments()

        self.experiment_file = experiment_file
        exp = parsing.load_yaml_dict_from_file(experiment_file)
        exp = parsing.convert_numeric_ranges(exp)
        self.experiment_config = exp
        self.experiment_instruments = {}
        self.__preconfigure_experiment_instruments()

        self.__build_connection_graph()

    def initiate_connections(self):
        """
        Lazily initiate the connections of all the bench instruments, *ie.* only
        those that are matched to an experiment instrument are handled.
        """
        for experiment_instrument in self.__experiment_instruments.values():
            be = experiment_instrument.bench_instrument
            be.initiate_connection()
            name = experiment_instrument.name
            self.experiment_instruments[name] = be.instrument

    def get_bench_instrument(self, name: str) -> Any:
        """
        Return the bench instrument whose name is specified, after having
        initiated its connection if required. This is the only way a bench
        instrument should be retrieved.
        """
        be = self.__bench_instruments[name]
        if be.instrument is None:
            be.initiate_connection()

            for ei_name, ei_inst in self.__experiment_instruments.items():
                bi = ei_inst.bench_instrument
                if bi.name == name:
                    self.experiment_instruments[ei_name] = bi.instrument

        return be.instrument

    def configure_instrument(self, name: str, configuration: dict):
        """
        Configure an experiment instrument using the given configuration, and
        the `configure` method of its loader.
        """
        experiment_instrument = self.__experiment_instruments[name]
        instrument = experiment_instrument.bench_instrument.instrument
        loader = experiment_instrument.bench_instrument.loader
        loader.configure(instrument, configuration)

    def configure_experiment(self, configuration: dict):
        """
        Configure all the instruments of an experiment using the `configure`
        method of their respective loaders and the entry of configuration
        matching their name.
        """
        for name in self.__experiment_instruments:
            self.configure_instrument(name, configuration[name])

    def experiment_configurations_iterator(
        self,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Creates a generator yielding the successive literal configurations
        represented by the experiment configuration file. See
        `parsing.configurations_iterator` for more details.
        """
        return parsing.configurations_iterator(self.experiment_config)

    def instrument_configurations_iterator(
        self, instrument: str
    ) -> Generator[dict[str, Any], None, None]:
        """
        Creates a generator yielding the successive literal configurations
        represented by an instrument entry of the experiment configuration
        file. See `parsing.configurations_iterator` for more details.
        """
        return parsing.configurations_iterator(self.experiment_config[instrument])

    def configured_experiment_iterator(self) -> Generator[dict[str, Any], None, None]:
        """
        Creates a generator which configures the experiment according to the
        successive literal configurations represented by the experiment
        configuration file, and yields the dictionary of the configured
        instruments at each step.

        See `parsing.configurations_iterator` for more details.
        """
        for config in self.experiment_configurations_iterator():
            self.configure_experiment(config)
            yield self.experiment_instruments

    def configured_instrument_iterator(
        self, instrument: str, lazy: bool = False
    ) -> Generator[Any, None, None]:
        """
        Creates a generator which configures an instrument according to the
        successive literal configurations represented by the experiment
        configuration file, and yields configured instrument at each step.

        See `parsing.configurations_iterator` for more details.
        """
        for config in self.instrument_configurations_iterator(instrument):
            self.configure_instrument(instrument, config)
            yield self.experiment_instruments[instrument]

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

            loader = self.loaders[config.pop("loader")]
            instrument = BenchInstrument(name, loader(self), config, None)
            self.__bench_instruments[name] = instrument

    def __preconfigure_experiment_instruments(self):
        """
        Prepare for the configuration if the experiment instruments:
         - assign a bench instrument to each of them,
         - clean their configuration of reserved entries (interface),
        but leave them in a non-configured state.
        """
        self.__experiment_instruments = {}
        for name, config in self.experiment_config.items():
            if "interface" not in config:
                continue

            interface = config.pop("interface")
            filter_ = Filter.build_filter(config.pop("filter", {}))
            bench_instrument = self.__find_bench_instrument(name, interface, filter_)

            self.__experiment_instruments[name] = ExperimentInstrument(
                name, interface, bench_instrument, config
            )

    def __build_connection_graph(self):
        """
        Extract the connection graph from the experiment configuration, removing
        `connections` entries from it.
        """
        connections: list[tuple[str, list[str], str, list[str], str]] = (
            self.__parse_global_connections()
        )

        for name, config in self.experiment_config.items():
            connections += self.__parse_instrument_connections(
                name, config.pop("connections", [])
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
        self,
    ) -> list[tuple[str, list[str], str, list[str], str]]:
        connections = []
        for connection in self.experiment_config.pop("connections", []):
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
        self, name: str, config: dict
    ) -> list[tuple[str, list[str], str, list[str], str]]:
        connections = []
        for connection in config:
            source_str: str = connection.pop("from", name)
            destination_str: str = connection.pop("to", name)
            attributes: str = connection.get("attributes", "")

            source_elements = source_str.split(".")
            if source_elements[0] in self.experiment_config:
                source_instrument = source_elements[0]
                source_port = source_elements[1:]
            else:
                source_instrument = name
                source_port = source_elements

            destination_elements = destination_str.split(".")
            if destination_elements[0] in self.experiment_config:
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

    def __find_bench_instrument(
        self, exp_name: str, interface: str, filter_: Filter
    ) -> BenchInstrument:
        """
        Find a bench instrument matching and interface and a filter.

        Raises:
         - KeyError if there are 0 or more than 1 matching instruments.
        """
        compatible_instruments = []
        for instrument in self.__bench_instruments.values():
            if interface not in instrument.loader.interfaces:
                continue

            if not filter_.verifies(instrument):
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

    def register_loader(
        self,
        loader: (
            type[Loader]
            | tuple[
                str,
                set[str],
                Callable[[dict], Any],
                Callable[[Any, dict], Any],
            ]
        ),
    ):
        """
        Register a new loader for this factory. See `_add_loader` for the
        specifications of the arguments.
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

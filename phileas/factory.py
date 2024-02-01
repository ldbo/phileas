import yaml
import inspect
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

import graphviz

import graphviz


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
    #: This allows loaders to access, among other things, to already
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
    def configure(self, instrument: Any, configuration: dict) -> Any:
        """
        Configure an instrument - whose connection has already been initiated -
        using its experiment configuration. The configured instrument is then
        returned,
        """
        raise NotImplementedError()


def build_loader(
    name_: str,
    interfaces_: set[str],
    initiate_connection: Callable[[dict], Any],
    configure: Callable[[Any, dict], Any],
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

        def configure(self, instrument: Any, configuration: dict) -> Any:
            return configure(instrument, configuration)

    return BuiltLoader


def __add_loader(
    loaders: dict[str, type[Loader]],
    loader: type[Loader]
    | tuple[
        str,
        set[str],
        Callable[[dict], Any],
        Callable[[Any, dict], Any],
    ],
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
#: modified.
_DEFAULT_LOADERS: dict[str, type[Loader]] = dict()


def register_default_loader(
    loader: type[Loader]
    | tuple[
        str,
        set[str],
        Callable[[dict], Any],
        Callable[[Any, dict], Any],
    ],
):
    """
    Register a loader to be added on init by every new `ExperimentFactory`. See
    `__add_loader` for the specifications of the arguments.
    """
    try:
        name = loader.name
        interfaces = loader.interfaces
    except AttributeError:
        name = loader[0]
        interfaces = loader[1]

    logging.debug(f"Register default loader {name} for interfaces {interfaces}")
    __add_loader(_DEFAULT_LOADERS, loader)


@dataclass
class ExperimentFactory:
    """
    Class used to parse configuration files and create the experiment
    environment they describe.
    """

    bench_file: Path
    experiment_file: Path

    #: Content of the bench configuration file
    bench_config: dict = field(init=False, repr=False)

    #: Content of the experiment configuration file
    experiment_config: dict = field(init=False, repr=False)

    #: The supported loader classes, stored with their name
    loaders: dict[str, type[Loader]] = field(
        init=False, default_factory=dict, repr=False
    )

    #: Instantiated bench instruments, stored with their name
    bench_instruments: dict[str, Any] = field(
        init=False, default_factory=dict, repr=False
    )
    #: Instantiated and configured experiment instruments, stored with their
    #: name
    experiment_instruments: dict[str, Any] = field(
        init=False, default_factory=dict, repr=False
    )

    #: Loaders of the bench instruments
    bench_instruments_loaders: dict[str, Loader] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self):
        with open(self.bench_file, "r") as f:
            self.bench_config = yaml.safe_load(f)

        with open(self.experiment_file, "r") as f:
            self.experiment_config = yaml.safe_load(f)

        self.loaders.update(_DEFAULT_LOADERS)

    def register_loader(
        self,
        loader: type[Loader]
        | tuple[
            str,
            set[str],
            Callable[[dict], Any],
            Callable[[Any, dict], Any],
        ],
    ):
        """
        Register a new loader for this factory. See `__add_loader` for the
        specifications of the arguments.
        """
        __add_loader(self.loaders, loader)

    def prepare_experiment(self):
        """
        Load and configure the instruments of the bench and experiment files.
        """
        self.__initialize_bench_instruments()
        self.__configure_experiment_instruments()

    def __initialize_bench_instruments(self):
        logging.info(
            f"Initializing connections to bench instruments from {self.bench_file}"
        )
        for name, configuration in self.bench_config.items():
            # Instrument configurations are dictionaries with a `loader` field
            if not isinstance(configuration, dict):
                continue

            if "loader" not in configuration:
                continue
            loader_name = configuration["loader"]

            if loader_name not in self.loaders:
                msg = f"No '{loader_name}' loader found for bench instrument "
                msg += f"'{name}'"
                raise KeyError(msg)
            Loader = self.loaders[loader_name]

            loader = Loader(self)

            logging.info(f"Initializing connection to {name} with loader {loader.name}")
            self.bench_instruments_loaders[name] = loader
            self.bench_instruments[name] = loader.initiate_connection(configuration)
            logging.info(f"{name} connection initialization successful")

    def __configure_experiment_instruments(self):
        logging.info(f"Configuring experiment instruments from {self.experiment_file}")
        for name, config in self.experiment_config.items():
            available_instruments_loaders: list[tuple[str, Any, Loader]] = []
            try:
                required_interface = config["interface"]

                for bench_name, loader in self.bench_instruments_loaders.items():
                    if required_interface not in loader.interfaces:
                        continue

                    logging.debug(
                        f"Experiment instrument {name} supports bench "
                        + f"instrument {bench_name} (loader {loader.name})"
                    )

                    if "filter" in config:
                        all_passed = True
                        for field, value in config["filter"].items():
                            if self.bench_config[bench_name][field] == value:
                                continue

                            all_passed = False
                            msg = f"{bench_name} is discarded for {name} "
                            msg += f"({field} != {value})"
                            logging.debug(msg)
                            break

                        if not all_passed:
                            continue

                    available_instruments_loaders.append(
                        (bench_name, self.bench_instruments[bench_name], loader)
                    )
            except TypeError:
                continue
            except KeyError:
                continue

            if len(available_instruments_loaders) == 0:
                raise KeyError(
                    f"Cannot find a suitable bench instrument for '{name}' experiment instrument"
                )
            elif len(available_instruments_loaders) > 1:
                available_bench_instruments = [
                    il[0] for il in available_instruments_loaders
                ]
                raise KeyError(
                    f"More than one suitable bench instrument for '{name}' "
                    + f"experiment instrument: {available_bench_instruments}"
                )
            else:
                bench_name, instrument, loader = available_instruments_loaders[0]
                logging.info(
                    f"Configuring {name} -> {bench_name} with loader {loader.name}"
                )
                self.experiment_instruments[name] = loader.configure(instrument, config)
                logging.info(f"{name} configuration successful")

    def get_experiment_graph(self) -> graphviz.Digraph:
        """
        Extract the instrument connections from the experiment configuration
        file, building a directed graph representing the interconnections of the
        instruments.
        """
        instruments: set[str] = set()
        connections: list[tuple[str, str, str | None, str | None]] = []

        # Load individual instruments descriptions
        instruments.update(self.experiment_config.keys())
        if "connections" in instruments:
            instruments.remove("connections")

        for instrument, configuration in self.experiment_config.items():
            if "connections" not in configuration:
                continue

            for connection in configuration["connections"]:
                src = connection.get("from", "")
                dst = connection.get("to", "")

                src_instrument, src_port = self.__extract_instrument_child_port(
                    instrument, src, instruments
                )
                dst_instrument, dst_port = self.__extract_instrument_child_port(
                    instrument, dst, instruments
                )

                connections.append((src_instrument, dst_instrument, src_port, dst_port))

        # Load connections listed in "connections"
        for connection in self.experiment_config.get("connections", []):
            src = connection["from"]
            dst = connection["to"]

            src_instrument, src_port = self.__extract_instrument_child_port(
                None, src, instruments
            )
            dst_instrument, dst_port = self.__extract_instrument_child_port(
                None, dst, instruments
            )

            instruments.add(src_instrument)
            instruments.add(dst_instrument)
            connections.append((src_instrument, dst_instrument, src_port, dst_port))

        logging.debug(
            f"Extracted nodes {instruments} from the experiment graph"
            + f", with {len(connections)} connections"
        )

        # Extract the list of instruments ports
        ports: dict[str, dict[str, int]] = dict()
        for src, dst, src_port, dst_port in connections:
            for instrument, port in ((src, src_port), (dst, dst_port)):
                if port is not None:
                    if instrument not in ports:
                        ports[instrument] = dict()

                    ports[instrument][port] = len(ports[instrument]) + 1

        return self.__build_graphviz_graph(connections, ports)

    @staticmethod
    def __extract_instrument_child_port(
        parent_instrument: str | None, port: str, instruments: set[str]
    ) -> tuple[str, str | None]:
        parts = port.split(".")

        if parts == []:
            if parent_instrument is not None:
                return parent_instrument, None
            else:
                raise KeyError("A connections must have a source and a destination")

        instrument = parts[0]
        port_parts = parts[1:]

        if instrument not in instruments and parent_instrument is not None:
            instrument = parent_instrument
            port_parts = parts

        child_port: str | None = ".".join(port_parts)

        if child_port == "":
            child_port = None

        return instrument, child_port

    @staticmethod
    def __build_graphviz_graph(
        connections: list[tuple[str, str, str | None, str | None]],
        ports: dict[str, dict[str, int]],
    ) -> graphviz.Digraph:
        graph = graphviz.Digraph(
            "instruments connections", node_attr={"shape": "record"}
        )

        for instrument, instrument_ports in ports.items():
            labels: list[tuple[int, str]] = [(0, instrument)]
            for port, index in instrument_ports.items():
                labels.append((index, port))

            node_label = " | ".join(f"<f{i}> {l}" for i, l in labels)
            graph.node(instrument, node_label)

        for src, dst, src_port, dst_port in connections:
            src_port_index = 0
            if src_port:
                src_port_index = ports[src][src_port]

            dst_port_index = 0
            if dst_port:
                dst_port_index = ports[dst][dst_port]

            graph.edge(f"{src}:f{src_port_index}", f"{dst}:f{dst_port_index}")

        return graph

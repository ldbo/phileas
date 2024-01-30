import yaml
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import graphviz


class Loader(ABC):
    """
    Class used to initiate a connection to an instrument, and then configure it.
    """

    #: Name of the loader, used in the bench `loader` field
    name: str

    #: Functionalities accomplished by the loaded instrument, used to select a
    #: given bench instrument from the requirements of an experiment. It is
    #: matched with the `interface` experiment field.
    interfaces: set[str]

    @abstractmethod
    def initiate_connection(
        self, factory: "InstrumentsFactory", configuration: dict
    ) -> Any:
        """
        Initialize a connection to a given instrument, given its bench
        configuration, and return it. The instruments factory is referred to
        allow references to other instruments.
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


@dataclass
class FunctionalLoader(Loader):
    """
    Loader class to be used as an alternative to sub-classing `Loader`.
    """

    name: str
    interfaces: set[str]
    initialization: Callable[["InstrumentsFactory", dict], Any] = field(repr=False)
    configuration: Callable[[Any, dict], Any] = field(repr=False)

    def initiate_connection(
        self, factory: "InstrumentsFactory", configuration: dict
    ) -> Any:
        return self.initialization(factory, configuration)

    def configure(self, instrument: Any, configuration: dict) -> Any:
        return self.configuration(instrument, configuration)


def __add_loader(
    loaders: dict[str, Loader],
    loader: Loader
    | tuple[
        str,
        set[str],
        Callable[["InstrumentsFactory", dict], Any],
        Callable[[Any, dict], Any],
    ],
):
    """
    Adds a loader to a set of loaders. The loader can be supplied either with
     - a `Loader` instance
     - a 4-tuple containing the name, interfaces, initiate_connection and
     configure fields of a loader
    """
    if not isinstance(loader, Loader):
        loader = FunctionalLoader(*loader)

    name = loader.name
    if name in loaders:
        logging.warning(f"Overwriting the loader {name}")

    loaders[name] = loader


#: Default loaders that every instrument factory has on init. It can be
#: modified.
_DEFAULT_LOADERS: dict[str, Loader] = dict()


def register_default_loader(
    loader: Loader
    | tuple[
        str,
        set[str],
        Callable[["InstrumentsFactory", dict], Any],
        Callable[[Any, dict], Any],
    ],
):
    """
    Register a loader to be added on init by every new `InstrumentFactory`
    """
    __add_loader(_DEFAULT_LOADERS, loader)


@dataclass
class InstrumentsFactory:
    bench_file: Path
    experiment_file: Path

    #: Content of the bench configuration file
    bench_config: dict = field(init=False, repr=False)
    #: Content of the experiment configuration file
    experiment_config: dict = field(init=False, repr=False)

    #: The supported loaders of the factory, stored with their name
    loaders: dict[str, Loader] = field(init=False, default_factory=dict, repr=False)
    #: Bench instruments, stored with their name
    bench_instruments: dict[str, Any] = field(
        init=False, default_factory=dict, repr=False
    )
    #: Experiment instruments, stored with their name
    experiment_instruments: dict[str, Any] = field(
        init=False, default_factory=dict, repr=False
    )
    #: Loaders of the bench instruments
    bench_instruments_loaders: dict[str, Any] = field(
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
        loader: Loader
        | tuple[
            str,
            set[str],
            Callable[["InstrumentsFactory", dict], Any],
            Callable[[Any, dict], Any],
        ],
    ):
        __add_loader(self.loaders, loader)

    def initialize_instruments(self):
        """
        Load and configure the instruments of the bench and experiment files.
        """
        self.__initialize_bench_instruments()
        self.__configure_experiment_instruments()

    def __initialize_bench_instruments(self):
        for name, configuration in self.bench_config.items():
            loader: Loader
            try:
                loader = self.loaders[configuration["loader"]]
            except TypeError:
                continue
            except KeyError:
                continue

            logging.info(f"Initializing {name} with loader {loader.name}")
            self.bench_instruments_loaders[name] = loader
            self.bench_instruments[name] = loader.initiate_connection(
                self, configuration
            )

    def __configure_experiment_instruments(self):
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
                    + "experiment instrument: {available_bench_instruments}"
                )
            else:
                bench_name, instrument, loader = available_instruments_loaders[0]
                self.experiment_instruments[name] = loader.configure(instrument, config)

    def get_experiment_instruments_graph(self) -> graphviz.Digraph:
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
            f"Extracted nodes {instruments} for the experiment instruments graph"
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

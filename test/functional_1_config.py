from dataclasses import dataclass, field

from phileas import Loader, register_default_loader


@dataclass
class LaserBus:
    device: str


class LaserBusLoader(Loader):
    """
    Bench only instrument loader example

    It does not support any interface, as it is a bench-only instrument, used
    for the initialization of other bench instruments. Thus, you need only
    to implement `initiate_connection`.
    """

    name = "laser_bus"
    interfaces = set()

    def initiate_connection(self, configuration: dict) -> LaserBus:
        """
        Parameters:
         - device: Path of the bus device
        """
        return LaserBus(configuration["device"])

    def configure(self, instrument: LaserBus, configuration: dict):
        """
        Being a bench-only instrument, it must not be configured.
        """
        pass


@dataclass
class LaserSource:
    link: LaserBus
    address: int
    wavelength: int


class LaserSourceLoader(Loader):
    """
    Usual instrument loader example

    To implement it, define its name and supported interfaces, as well as
    `initiate_connection` and `configure`.
    """

    #: Matched against the bench instrument `loader` field, from the bench
    #: configuration file
    name = "laser_source"

    #: Matched against the experiment instrument `interface` field, from the
    #: experiment configuration file
    interfaces = {"laser"}

    def initiate_connection(self, configuration: dict) -> LaserSource:
        """
        Called upon instantiating the bench instrument file.

        Parameters:
         - bus: name of the laser bus used for communication
         - address: address of the laser on the bus
         - wavelength (nm): specified central wavelength of the laser source
        """
        link: LaserBus = self.instruments_factory.get_bench_instrument(
            configuration["bus"]
        )
        wavelength = configuration["wavelength"]
        return LaserSource(link, configuration["address"], wavelength)

    def configure(self, instrument: LaserSource, configuration: dict):
        """
        Called after having linked a bench instrument to an experiment
        instrument, effectively configuring the instrument.
        """
        pass


@dataclass
class PrecisionPowerSupply:
    channels: dict = field(default_factory=dict)

    def configure_channel(self, channel: str, configuration: dict):
        config = {}
        config["tension"] = configuration["tension"]
        config["limit_current"] = configuration["limit_current"]
        self.channels[channel] = config


def configure_precision_power_supply(
    power_supply: PrecisionPowerSupply, configuration: dict
):
    """
    Parameters: dict with channel name keys - strings starting with `ch` - and
    dict values, with the following required keys:
     - tension (V): tension of the channel
     - limit_current (A): limit current of the channel
    """
    for key, parameter_value in configuration.items():
        if key.startswith("ch"):
            power_supply.configure_channel(key, parameter_value)


# You now have to register the implemented loaders
register_default_loader(
    (
        "precision_power_supply",
        {
            "power_supply",
        },
        lambda _: PrecisionPowerSupply(),
        configure_precision_power_supply,
    )
)

register_default_loader(LaserBusLoader)
register_default_loader(LaserSourceLoader)

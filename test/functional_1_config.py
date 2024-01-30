from dataclasses import dataclass

from phileas import Loader, register_default_loader


@dataclass
class AlphanovLink:
    device: str


class AlphanovLoader(Loader):
    name = "alphanov"
    interfaces = set()

    def initiate_connection(self, configuration: dict) -> AlphanovLink:
        return AlphanovLink(configuration["device"])

    def configure(self, instrument: AlphanovLink, configuration: dict) -> AlphanovLink:
        return instrument


@dataclass
class AlphanovLaser:
    link: AlphanovLink
    address: int


class AlphanovLaserLoader(Loader):
    name = "pdm"
    interfaces = {"laser"}

    def initiate_connection(self, configuration: dict) -> AlphanovLaser:
        instruments = self.instruments_factory.bench_instruments
        link: AlphanovLink = instruments[configuration["link"]]
        return AlphanovLaser(link, configuration["address"])

    def configure(
        self, instrument: AlphanovLaser, configuration: dict
    ) -> AlphanovLaser:
        return instrument


@dataclass
class AlphanovTombak:
    link: AlphanovLink
    address: int


class AlphanovTombakLoader(Loader):
    name = "tombak"
    interfaces = {"tombak"}

    def initiate_connection(self, configuration: dict) -> AlphanovTombak:
        instruments = self.instruments_factory.bench_instruments
        link: AlphanovLink = instruments[configuration["link"]]
        return AlphanovTombak(link, configuration["address"])

    def configure(
        self, instrument: AlphanovTombak, configuration: dict
    ) -> AlphanovTombak:
        return instrument


register_default_loader(
    (
        "virtual_power_supply",
        {
            "power_supply",
        },
        lambda _: object(),
        lambda i, _: i,
    )
)

register_default_loader(AlphanovLoader)
register_default_loader(AlphanovLaserLoader)
register_default_loader(AlphanovTombakLoader)

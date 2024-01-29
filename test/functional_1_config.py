from dataclasses import dataclass
from typing import Any

from phileas import Loader, InstrumentsFactory, register_default_loader


@dataclass
class AlphanovLink:
    device: str


class AlphanovLoader(Loader):
    name = "alphanov"
    interfaces = set()

    def initiate_connection(
        self, factory: InstrumentsFactory, configuration: dict
    ) -> Any:
        return AlphanovLink(configuration["device"])

    def configure(self, instrument: Any, configuration: dict) -> Any:
        return instrument


@dataclass
class AlphanovLaser:
    link: AlphanovLink
    address: int


class AlphanovLaserLoader(Loader):
    name = "pdm"
    interfaces = {"laser"}

    def initiate_connection(
        self, factory: InstrumentsFactory, configuration: dict
    ) -> Any:
        link: AlphanovLink = factory.bench_instruments[configuration["link"]]
        return AlphanovLaser(link, configuration["address"])

    def configure(self, instrument: Any, configuration: dict) -> Any:
        return instrument


@dataclass
class AlphanovTombak:
    link: AlphanovLink
    address: int


class AlphanovTombakLoader(Loader):
    name = "tombak"
    interfaces = {"tombak"}

    def initiate_connection(
        self, factory: InstrumentsFactory, configuration: dict
    ) -> Any:
        link: AlphanovLink = factory.bench_instruments[configuration["link"]]
        return AlphanovTombak(link, configuration["address"])

    def configure(self, instrument: Any, configuration: dict) -> Any:
        return instrument


register_default_loader(
    (
        "virtual_power_supply",
        {
            "power_supply",
        },
        lambda _, __: object(),
        lambda i, _: i,
    )
)

register_default_loader(AlphanovLoader())
register_default_loader(AlphanovLaserLoader())
register_default_loader(AlphanovTombakLoader())

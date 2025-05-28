"""
This module defines simulated instruments drivers and loaders. They can be used
for testing, or to demonstrate features, among others.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from math import log10
from typing import Any, ClassVar

import numpy as np

from phileas import Loader
from phileas.factory import register_default_loader
from phileas.iteration.base import DataTree


@dataclass
class Motors:
    """
    Simulated 2D motors driver.
    """

    #: X position of the motor.
    x: float = 0.0

    #: Y position of the motor.
    y: float = 0.0

    #: Unique identifier of the motors.
    id: str = "mock-motors-driver:1"

    def __post_init__(self):
        print("[Motors] Connection initiated.")

    def set_position(self, **kwargs: dict[str, float]):
        """
        Set the position of the motors. Expect arguments x and y, which are both
        optional.
        """
        for axis in "x", "y":
            if axis in kwargs:
                setattr(self, axis, kwargs[axis])


@register_default_loader
class MotorsLoader(Loader):
    """
    Loader of simulated 2D motors.
    """

    name = "phileas-mock_motors-phileas"
    interfaces = {"2d-motors"}

    def initiate_connection(self, configuration: dict) -> Motors:
        return Motors()

    def configure(self, instrument: Motors, configuration: dict):
        """
        Parameters:
         - x, y (optional): Required position of the motors.
        """
        instrument.set_position(**configuration)
        self.logger.info(f"Position set to {configuration}.")

    def get_effective_configuration(
        self, instrument: Motors, configuration: None | dict = None
    ) -> dict[str, DataTree]:
        if configuration is None:
            return self.dump_state(instrument)  # type: ignore[return-value]

        for name in configuration.keys():
            configuration[name] = getattr(instrument, name)

        return configuration

    def get_id(self, instrument: Motors) -> str:
        return instrument.id

    def dump_state(self, instrument: Motors) -> DataTree:
        return {"x": instrument.x, "y": instrument.y}

    def restore_state(self, instrument: Motors, state: dict[str, float]):  # type: ignore[override]
        instrument.x = state["x"]
        instrument.y = state["y"]


class Probe:
    """
    Measurement probe interface.
    """

    @abstractmethod
    def get_amplitude(self) -> np.ndarray:
        """Amplitude of the measured quantity."""
        raise NotImplementedError()


@dataclass
class RandomProbe(Probe):
    """
    Probe which returns random values in [0, 1], using
    {py:func}`numpy.random.rand`.
    """

    #: Shape of the output.
    shape: tuple[int, ...] = (1,)

    def get_amplitude(self) -> np.ndarray:
        return np.random.rand(*self.shape)


@dataclass
class ElectricFieldProbe(Probe):
    """
    Simulation of the electric field radiated by an electric dipole.
    """

    #: Motors used to position the probe.
    motor: Motors

    #: Gain of the probe.
    gain: ClassVar[float] = 1

    def get_amplitude(self) -> np.ndarray:
        p = np.array([1, 1]) / 10
        d = np.array([self.motor.x - 0.3, self.motor.y - 0.5])
        d_norm = np.linalg.norm(d)
        dn = d / d_norm
        electric_field = (3 * np.dot(p, dn) * dn - p) / np.power(d_norm, 3)
        multiplicative_noise = np.random.normal(1, 0.05)

        return self.gain * np.dot(electric_field, electric_field) * multiplicative_noise


@dataclass
class Oscilloscope:
    """
    Simulated 8-bit oscilloscope driver. It uses a
    {py:class}`~phileas.mock_instruments.Probe` and quantifies its output.
    """

    #: Probe which is connected to the oscilloscope
    probe: Probe

    #: Actual location of the :py:attr:`amplitude` field.
    _amplitude: float = field(init=False, repr=False, default=1.0)

    #: Unique identifier of the oscilloscope.
    id: str = "mock-oscilloscope-driver:1"

    #: Bit width of the ADC.
    width: ClassVar[int] = 8

    #: Version of the oscilloscope firmware.
    fw_version: ClassVar[str] = "12.3"

    def __post_init__(self):
        print("[Oscilloscope] Connection initiated.")

    def get_measurement(self) -> np.ndarray:
        """Sampled and quantified value of the probe."""
        value = self.probe.get_amplitude()
        trimmed = np.clip(value, -self.amplitude / 2, self.amplitude / 2)
        quantized = np.round(trimmed * (1 << self.width)) / (1 << self.width)

        return quantized

    @property
    def amplitude(self) -> float:
        #: Amplitude of the measurements, which have a null offset.
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        value = 10 ** round(log10(value))
        self._amplitude = value


@register_default_loader
class OscilloscopeLoader(Loader):
    """
    Loader of a simulated oscilloscope.
    """

    name = "phileas-mock_oscilloscope-phileas"
    interfaces = {"oscilloscope"}

    def initiate_connection(self, configuration: dict) -> Oscilloscope:
        """
        Parameters:
         - probe (required): Name of the simulated probe to use.

        Supported probes, and parameters:
         - electric-field-probe:
            motors: name of the motors that position the probe
         - random-probe:
            shape: tuple with the required probe output shape
        """
        probe = configuration["probe"]
        if probe == "electric-field-probe":
            return Oscilloscope(
                probe=ElectricFieldProbe(
                    self.instruments_factory.get_bench_instrument(
                        configuration["motors"]
                    )
                ),
            )
        elif probe == "random-probe":
            return Oscilloscope(probe=RandomProbe(shape=configuration["shape"]))
        else:
            raise ValueError(f"Unsupported probe type: {probe}")

    def configure(self, instrument: Oscilloscope, configuration: dict):
        """
        Parameters:
         - amplitude (optional): Amplitude of the oscilloscope.
        """
        if "amplitude" in configuration:
            amplitude = configuration["amplitude"]
            instrument.amplitude = amplitude
            self.logger.info(f"Amplitude set to {amplitude}.")

    def get_effective_configuration(
        self, instrument: Oscilloscope, configuration: None | dict = None
    ) -> dict:
        eff_conf = {}

        if configuration is None or "amplitude" in configuration:
            eff_conf["amplitude"] = instrument.amplitude

        return eff_conf

    def get_id(self, instrument: Oscilloscope) -> str:
        return instrument.id

    def dump_state(self, instrument: Oscilloscope) -> DataTree:
        return {
            "probe": f"{type(instrument.probe).__name__}",
            "amplitude": instrument.amplitude,
            "bit_width": instrument.width,
            "fw_version": instrument.fw_version,
        }

    def restore_state(self, instrument: Oscilloscope, state: dict[str, Any]):  # type: ignore[override]
        if instrument.fw_version != state["fw_version"]:
            raise ValueError(
                f"Dumped FW version {state['fw_version']} is not compatible with"
                f" instrument FW version {instrument.fw_version}."
            )

        if instrument.width != state["bit_width"]:
            raise ValueError(
                f"Dumped bit width {state['bit_width']} is not compatible with "
                f"instrument bit width {instrument.width}."
            )
        instrument.amplitude = state["amplitude"]

        if type(instrument.probe).__name__ != state["probe"]:
            self.logger.warning("Cannot change the oscilloscope probe.")

"""
This module defines simulated instruments drivers and loaders. They can be used
for testing, or to demonstrate features, among others.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from phileas import Loader
from phileas.factory import register_default_loader


@dataclass
class Motors:
    """
    Simulated 2D motors driver.
    """

    #: X position of the motor.
    x: float = 0.0

    #: Y position of the motor.
    y: float = 0.0

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

    #: Amplitude of the measurements, which have a null offset.
    amplitude: float = 1.0

    #: Bit width of the ADC.
    width: ClassVar[int] = 8

    def __post_init__(self):
        print("[Oscilloscope] Connection initiated.")

    def get_measurement(self) -> np.ndarray:
        """Sampled and quantified value of the probe."""
        value = self.probe.get_amplitude()
        trimmed = np.clip(value, -self.amplitude / 2, self.amplitude / 2)
        quantized = np.round(trimmed * (1 << self.width)) / (1 << self.width)

        return quantized


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
         - electric-field-probe: No parameter
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
                )
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

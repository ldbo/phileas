"""
This module defines simulated instruments drivers and loaders. They can be used
for testing, or to demonstrate features, among others.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from math import log10
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from phileas import Loader, logger
from phileas.factory import register_default_loader
from phileas.iteration.base import DataTree

logger = logger.getChild(__name__)


@dataclass
class SimulatedPRESENTImplementation:
    """
    Simulation of an embedded `PRESENT
    <https://iacr.org/archive/ches2007/47270450/47270450.pdf>`_ implementation.
    It can be plugged to a probe, in order to simulate SCA.
    """

    #: Path of the device used for the serial connection.
    serial_device: Path

    #: Serial connection baudrate.
    baudrate: int

    #: Current probe used to perform simulated SCA. If you want to use it,
    #: simply set this attribute.
    probe: CurrentProbe | None = None

    #: Encryption key, must be :py:attr:`key_size` bits wide.
    _key: int | None = None

    #: Number of PRESENT rounds.
    rounds: int = 32

    #: Block size, in bits.
    block_size: ClassVar[int] = 64

    #: Key size, in bits.
    key_size: ClassVar[int] = 80

    #: Evaluation of the S-box.
    s_box: ClassVar[list[int]] = [
        0xC,
        0x5,
        0x6,
        0xB,
        0x9,
        0x0,
        0xA,
        0xD,
        0x3,
        0xE,
        0xF,
        0x8,
        0x4,
        0x7,
        0x1,
        0x2,
    ]

    #: SNR of the hamming distance of the S-box, as seen from the measured
    #: side-channel.
    snr: float = 1

    #: Number of sampling points per PRESENT round.
    points_per_round: int = 20

    def __post_init__(self) -> None:
        logger.info(
            f"Connected to the PRESENT DUT on {self.serial_device} "
            f"with baudrate {self.baudrate}."
        )

    @property
    def key(self) -> int | None:
        """
        Encryption key, must be :py:attr:`key_size` bits wide.
        """
        return self._key

    @key.setter
    def key(self, value: int):
        value = int(value)
        if value.bit_length() > self.key_size:
            raise ValueError(f"Key size must be {self.key_size} bits wide.")

        self._key = value
        logger.info(f"PRESENT DUT key set to {value}.")

    def encrypt(self, block: int) -> int:
        """
        Encrypt a 64-bit block. The :py:attr:`key` must be set before encryption.
        """
        if self.key is None:
            raise ValueError("The key must be set before encryption.")

        hamming_distance = np.empty(self.rounds, dtype=np.float64)

        # Encryption
        key = self.key
        for round_counter in range(1, self.rounds):
            round_key = key >> (self.key_size - self.block_size)
            block = self.add_round_key(round_key, block)
            before_s_box = block
            block = self.s_box_layer(block)
            after_s_box = block
            block = self.p_layer(block)
            key = self.key_schedule(round_counter, key)

            hamming_distance[round_counter - 1] = (
                after_s_box ^ before_s_box
            ).bit_count()

        block = self.add_round_key(key >> 16, block)

        # SCA simulations
        time = np.linspace(0, 1, self.points_per_round + 20)[1:]
        leakage_kernel = np.power(time[0] / time, 3)
        leakages = (
            leakage_kernel[np.newaxis, :]
            * hamming_distance[:, np.newaxis]
            / self.block_size
        )

        trace = leakages.flatten()
        noise = np.random.normal(0, 1, trace.shape)

        if self.probe is not None:
            self.probe.last_measurement = trace + noise / self.snr

        return block

    # PRESENT internals

    @classmethod
    def add_round_key(cls, key: int, block: int) -> int:
        return key ^ block

    def s_box_layer(self, block: int) -> int:
        s_block = 0
        for nibble in range((self.block_size + 3) // 4):
            s_block |= self.s_box[(block >> (nibble * 4)) & 0xF] << (nibble * 4)

        return s_block

    def p_layer(self, block: int) -> int:
        p_block = 0
        for position in range(self.block_size):
            destination = (position % 4) * 16 + (position // 4)
            bit_value = (block >> position) & 1
            p_block |= bit_value << destination

        return p_block

    def key_schedule(self, round_counter: int, current_key: int) -> int:
        # Rotate 61 bits left
        key = ((current_key << 61) & ((1 << self.key_size) - 1)) + (
            current_key >> (self.key_size - 61)
        )

        # Pass most significant nibble through the S-box
        ms_pos = self.key_size - 4
        ms_nibble = self.s_box[key >> ms_pos]
        key = key & ((1 << ms_pos) - 1) | (ms_nibble << ms_pos)

        # Add round counter
        key ^= round_counter << 15

        return key


@register_default_loader
class SimulatedPRESENTImplementationLoader(Loader):
    """
    Loader of a simulated PRESENT implementation, used for SCA.
    """

    name = "phileas-mock_present-phileas"
    interfaces = {"present-encryption"}

    def initiate_connection(self, configuration: dict) -> Any:
        """
        Parameters:
          - device (required): Path of the device used for the serial connection.
          - baudrate (required): Serial connection baudrate.
          - probe (optional): Name of the current probe to plug to the device,
            for monitoring its current consumption.
          - snr (optional): SNR of the hamming distance of the S-box, as seen
            from the measured side-channel.
          - points_per_round (optional): Number of sampling points per PRESENT
            round.
        """
        present = SimulatedPRESENTImplementation(
            configuration["device"], configuration["baudrate"]
        )

        if "probe" in configuration:
            probe = self.instruments_factory.get_bench_instrument(
                configuration["probe"]
            )
            if not isinstance(probe, CurrentProbe):
                raise TypeError("The simulated PRESENT only supports current probes.")

            present.probe = probe

        if "snr" in configuration:
            present.snr = configuration["snr"]

        if "points_per_round" in configuration:
            present.points_per_round = configuration["points_per_round"]

        return present

    def configure(self, instrument: Any, configuration: dict):
        """
        Parameters:
          - key: Encryption key, a 80-bit integer.
        """
        if "key" in configuration:
            instrument.key = configuration["key"]


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
        logger.info("[Motors] Connection initiated.")

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
class CurrentProbe(Probe):
    last_measurement: np.ndarray | None = None
    noise_level: float = 1.0
    gain: float = 1.0

    def get_amplitude(self) -> np.ndarray:
        if self.last_measurement is None:
            raise ValueError("The probe has no new measurement ready.")

        shape = self.last_measurement.shape
        noise = np.random.normal(0, self.noise_level, size=shape)
        noisy_measurement = self.gain * self.last_measurement + noise
        self.last_measurement = None

        return noisy_measurement


@register_default_loader
class CurrentProbeLoader(Loader):
    name = "phileas-current_probe-phileas"
    interfaces = set()

    def initiate_connection(self, configuration: dict) -> Any:
        probe = CurrentProbe()
        if "noise_level" in configuration:
            probe.noise_level = configuration["noise_level"]

        if "gain" in configuration:
            probe.gain = configuration["gain"]

        return probe

    def configure(self, instrument: Any, configuration: dict):
        raise NotImplementedError()


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
        logger.info("[Oscilloscope] Connection initiated.")

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
         - generic:
            probe-name: Name of the probe bench instrument
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
        elif probe == "generic":
            probe_name = configuration["probe-name"]
            probe_instrument = self.instruments_factory.get_bench_instrument(probe_name)
            if not isinstance(probe_instrument, Probe):
                raise TypeError(
                    "Oscilloscope generic expects a Probe, whereas "
                    f"{probe_name} is a {type(probe_instrument).__name__}."
                )

            return Oscilloscope(probe=probe_instrument)
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

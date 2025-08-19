import importlib
import unittest

import numpy as np
import xarray as xr

import phileas.mock_instruments  # noqa: F401
from phileas.factory import ExperimentFactory
from phileas.iteration.random import generate_seeds
from phileas.iteration.utility import (
    data_tree_to_xarray_index,
    iteration_tree_to_xarray_parameters,
)
from phileas.parsing import load_iteration_tree_from_yaml_file


class TestScaExample(unittest.TestCase):
    def test_sca_example(self):
        importlib.reload(phileas.mock_instruments)

        np.random.seed(
            int.from_bytes("sca.md".encode("utf-8"), byteorder="little") % (1 << 32)
        )

        bench = """
        simulated_present_dut:
          loader: phileas-mock_present-phileas
          device: /dev/ttyUSB1
          baudrate: 115200
          probe: simulated_current_probe

        simulated_current_probe:
          loader: phileas-current_probe-phileas

        simulated_oscilloscope:
          loader: phileas-mock_oscilloscope-phileas
          probe: generic
          probe-name: simulated_current_probe
          width: 32
        """

        experiment = """
        dut:
          interface: present-encryption
          key: !random
            distribution: integers
            parameters:
              dtype: uint64
              low: 0
              high: 0x10000000000000000
            size: 10
          plaintext: !random
            distribution: integers
            parameters:
              dtype: uint64
              low: 0
              high: 0x10000000000000000
            size: 10

        oscilloscope:
          interface: oscilloscope
          amplitude: 10

        iteration: !range
          start: 1
          end: 10
          resolution: 1
        """

        factory = ExperimentFactory(bench, experiment)
        factory.initiate_connections()

        oscilloscope = factory.experiment_instruments["oscilloscope"]
        dut = factory.experiment_instruments["dut"]

        configurations = generate_seeds(load_iteration_tree_from_yaml_file(experiment))

        coords, dims_name, dims_shape = iteration_tree_to_xarray_parameters(
            configurations
        )
        dut.key = 12
        dut.encrypt(0)
        trace = oscilloscope.get_measurement()
        coords["oscilloscope.sample"] = list(range(trace.shape[0]))
        dims_name.append("oscilloscope.sample")
        dims_shape.append(trace.shape[0])
        traces = xr.DataArray(coords=coords, dims=dims_name)

        for config in configurations:
            assert isinstance(config, dict)
            factory.configure_experiment(config)

            dut.encrypt(config["dut"]["plaintext"])  # type: ignore[index,call-overload]
            trace = oscilloscope.get_measurement()

            index = data_tree_to_xarray_index(config, dims_name)
            traces.loc[index] = trace

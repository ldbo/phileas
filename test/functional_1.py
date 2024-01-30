from phileas import *

from pathlib import Path

import functional_1_config

import logging

logging.basicConfig(level=logging.DEBUG)

test_dir = (Path.cwd() / Path(__file__)).parent
factory = InstrumentsFactory(
    test_dir / "functional_1_bench.yaml", test_dir / "functional_1_experiment.yaml"
)
factory.initialize_instruments()

assert "laser" in factory.experiment_instruments
assert isinstance(
    factory.experiment_instruments["laser"], functional_1_config.AlphanovLaser
)
assert "dut_power_supply" in factory.experiment_instruments

# The experiment connections graph is rendered in functional_1.pdf
graph = factory.get_experiment_instruments_graph()
graph.render(Path(__file__).stem, test_dir, format="pdf", cleanup=True)

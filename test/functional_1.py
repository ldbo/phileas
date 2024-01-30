from utilities import *

from phileas import *
import functional_1_config

from pathlib import Path

test_name = Path(__file__).stem

factory = ExperimentFactory(
    test_dir / f"{test_name}_bench.yaml", test_dir / f"{test_name}_experiment.yaml"
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
graph.render(test_name, test_dir, format="pdf", cleanup=True)

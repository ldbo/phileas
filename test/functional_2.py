from utilities import *

from phileas import *
import functional_1_config as _

from pathlib import Path

test_name = Path(__file__).stem

factory = ExperimentFactory(
    test_dir / f"{test_name}_bench.yaml", test_dir / f"{test_name}_experiment.yaml"
)

try:
    factory.prepare_experiment()
    raise AssertionError("Should raise a KeyError")
except KeyError as e:
    assert e.args[0] == "No 'inquisitor' loader found for bench instrument 'inquisitor'"

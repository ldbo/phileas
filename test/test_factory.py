import unittest
from pathlib import Path
from types import NoneType
from typing import ClassVar

from pyfakefs import fake_filesystem_unittest

import phileas
from phileas import ExperimentFactory, register_default_loader, clear_default_loaders


class BaseTestCase(fake_filesystem_unittest.TestCase):
    """
    Test case class creating a fake filesystem on setup, where file
    `bench_config_file` (resp. `experiment_config_file`) has content
    `bench_config` (resp. `experiment_config`), and creating a factory
    configured to use them. To create tests, subclass it, define `bench_config`
    and `experiment_config`, configure loaders on `setUp` and implement tests as
    methods.
    """

    bench_config_file: ClassVar[Path] = Path("/bench.yaml")
    experiment_config_file: ClassVar[Path] = Path("/experiment.yaml")

    #: Content of the bench configuration file, to be defined for each test case
    bench_config: ClassVar[str]

    #: Content of the experiment configuration file, to be defined for each test
    #: case
    experiment_config: ClassVar[str]

    #: Experiment factory configured to use `bench_config` and
    #: `experiment_config` on setup, and made available for tests
    #: implementation.
    factory: ExperimentFactory

    def setUp(self) -> None:
        self.setUpPyfakefs()
        self.fs.create_file(self.bench_config_file, contents=self.bench_config)
        self.fs.create_file(
            self.experiment_config_file, contents=self.experiment_config
        )
        self.factory = ExperimentFactory(
            self.bench_config_file, self.experiment_config_file
        )

    def tearDown(self) -> None:
        clear_default_loaders()


class TestEmptyConfigurationFile(BaseTestCase):
    bench_config = ""
    experiment_config = ""

    def test_empty_configuration_file(self):
        self.assertEqual(self.factory.bench_config, dict())
        self.assertEqual(self.factory.experiment_config, dict())

        self.factory.prepare_experiment()
        self.assertEqual(self.factory.bench_instruments, dict())
        self.assertEqual(self.factory.experiment_instruments, dict())


class TestExperimentFactory(BaseTestCase):
    bench_config = """
bench_instrument:
  loader: instrument_loader
    """

    experiment_config = """
experiment_instrument:
  interface: instrument_interface
    """

    def setUp(self) -> None:
        register_default_loader(
            (
                "instrument_loader",
                {"instrument_interface"},
                lambda _: object(),
                lambda i, _: i,
            )
        )
        super().setUp()

    def test_experiment_preparation(self):
        self.factory.prepare_experiment()
        self.assertIn("bench_instrument", self.factory.bench_instruments)
        self.assertIn("experiment_instrument", self.factory.experiment_instruments)


class TestMissingBenchInstrument(BaseTestCase):
    bench_config = """
    """

    experiment_config = """
experiment_instrument:
  interface: instrument_interface
    """

    def test_missing_bench_instrument(self):
        msg = f"Cannot find a suitable bench instrument for "
        msg += "'experiment_instrument' experiment instrument"
        with self.assertRaises(KeyError, msg=msg):
            self.factory.prepare_experiment()


class TestTooManyBenchInstrument(BaseTestCase):
    bench_config = """
bench_instrument1:
  loader: instrument_loader
  
bench_instrument2:
  loader: instrument_loader
    """

    experiment_config = """
experiment_instrument:
  interface: instrument_interface
    """

    def setUp(self) -> None:
        register_default_loader(
            (
                "instrument_loader",
                {"instrument_interface"},
                lambda _: None,
                lambda i, _: None,
            )
        )
        super().setUp()

    def test_too_many_bench_instruments(self):
        msg = f"More than one suitable bench instrument for 'experiment_instrument' "
        msg += f"experiment instrument: [bench_instrument1, bench_instrument2]"
        with self.assertRaises(KeyError, msg=msg):
            self.factory.prepare_experiment()


class TestLoaderDocumentation(BaseTestCase):
    bench_config = ""
    experiment_config = ""
    loader: type[phileas.factory.Loader]

    def setUp(self) -> None:
        class ClassLoader(phileas.factory.Loader):
            name = "class_loader"
            interfaces = {"interface1", "interface2"}

            def initiate_connection(self, configuration: dict) -> NoneType:
                """
                Initialization documentation.

                Parameters:
                 - param1: param1 description
                 - param2: param2 description
                """
                return None

            def configure(self, instrument: NoneType, configuration: dict) -> NoneType:
                """
                Configuration documentation.

                Parameters:
                 - param1: param1 description
                 - param2: param2 description
                """
                return instrument

        self.class_loader = ClassLoader

        def initiate(configuration: dict) -> None:
            """Initiate documentation"""
            return None

        def configure(instrument: None, configuration: dict) -> None:
            """Configure documentation"""
            return instrument

        self.func_loader = phileas.factory.build_loader(
            "func_loader", {"interface"}, initiate, configure
        )

        super().setUp()
        self.factory.register_loader(self.class_loader)
        self.factory.register_loader(self.func_loader)
        register_default_loader(self.class_loader)
        register_default_loader(self.func_loader)

    def test_class_loader_markdown_documentation_generation(self):
        expected_doc = (
            "# ClassLoader\n"
            " - Name: `class_loader`\n"
            " - Interfaces:\n"
            "   - `interface1`\n"
            "   - `interface2`\n"
            "\n"
            "## Initialization\n"
            "Initialization documentation.\n"
            "\n"
            "Parameters:\n"
            " - param1: param1 description\n"
            " - param2: param2 description\n"
            "\n"
            "## Configuration\n"
            "Configuration documentation.\n"
            "\n"
            "Parameters:\n"
            " - param1: param1 description\n"
            " - param2: param2 description\n"
        )
        self.assertEqual(self.class_loader.get_markdown_documentation(), expected_doc)

    def test_func_loader_markdown_documentation_generation(self):
        expected_doc = (
            "# FuncLoader\n"
            " - Name: `func_loader`\n"
            " - Interfaces:\n"
            "   - `interface`\n"
            "\n"
            "## Initialization\n"
            "Initiate documentation\n"
            "\n"
            "## Configuration\n"
            "Configure documentation\n"
        )
        self.assertEqual(self.func_loader.get_markdown_documentation(), expected_doc)

    def test_factory_markdown_documentation_generation(self):
        expected_doc = (
            "# ClassLoader\n"
            " - Name: `class_loader`\n"
            " - Interfaces:\n"
            "   - `interface1`\n"
            "   - `interface2`\n"
            "\n"
            "## Initialization\n"
            "Initialization documentation.\n"
            "\n"
            "Parameters:\n"
            " - param1: param1 description\n"
            " - param2: param2 description\n"
            "\n"
            "## Configuration\n"
            "Configuration documentation.\n"
            "\n"
            "Parameters:\n"
            " - param1: param1 description\n"
            " - param2: param2 description\n"
            "\n"
            "\n"
            "# FuncLoader\n"
            " - Name: `func_loader`\n"
            " - Interfaces:\n"
            "   - `interface`\n"
            "\n"
            "## Initialization\n"
            "Initiate documentation\n"
            "\n"
            "## Configuration\n"
            "Configure documentation\n"
        )
        self.assertEqual(
            self.factory.get_loaders_markdown_documentation(), expected_doc
        )
        self.assertEqual(
            ExperimentFactory.get_default_loaders_markdown_documentation(), expected_doc
        )


class TestFunctional1(unittest.TestCase):
    test_dir: Path

    def setUp(self) -> None:
        self.test_dir = (Path.cwd() / Path(__file__)).parent
        super().setUp()

    def test_functional1(self):
        from . import functional_1_config as _

        factory = ExperimentFactory(
            self.test_dir / "functional_1_bench.yaml",
            self.test_dir / "functional_1_experiment.yaml",
        )
        factory.prepare_experiment()

        self.assertEqual(
            factory.bench_instruments["laser_bus_1"].device, "/dev/ttyUSB0"
        )
        self.assertIn("laser_1064", factory.bench_instruments)
        self.assertIn("laser_980", factory.bench_instruments)
        self.assertIn("main_power_supply", factory.bench_instruments)
        self.assertNotIn("dut", factory.bench_instruments)

        self.assertEqual(factory.experiment_instruments["laser"].wavelength, 1064)
        self.assertIn("dut_power_supply", factory.experiment_instruments)
        self.assertNotIn("dut", factory.experiment_instruments)
        self.assertNotIn("ampli", factory.experiment_instruments)

        graph = factory.get_experiment_graph()
        graph.render(
            filename="functional_1", directory=self.test_dir, format="pdf", cleanup=True
        )
        self.graph_file = self.test_dir / "functional_1.pdf"
        self.assertTrue(self.graph_file.exists())

    def tearDown(self) -> None:
        clear_default_loaders()

        try:
            if self.graph_file.exists():
                self.graph_file.unlink()
        except AttributeError:
            pass

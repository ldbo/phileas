import contextlib
import unittest
from pathlib import Path
from typing import ClassVar

import phileas
from phileas import (
    ExperimentFactory,
    clear_default_loaders,
    parsing,
    register_default_loader,
)


class BaseTestCase(unittest.TestCase):
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
        self.factory = ExperimentFactory(self.bench_config, self.experiment_config)

    def tearDown(self) -> None:
        clear_default_loaders()


class TestEmptyConfigurationFile(BaseTestCase):
    bench_config = ""
    experiment_config = ""

    def test_empty_configuration_file(self):
        self.assertEqual(self.factory.bench_config, {})
        self.assertEqual(self.factory.experiment_config, {})

        self.assertDictEqual(self.factory.experiment_instruments, {})
        self.factory.initiate_connections()
        self.assertDictEqual(self.factory.experiment_instruments, {})


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
        self.factory.initiate_connections()
        _ = self.factory.get_bench_instrument("bench_instrument")
        self.assertIn("experiment_instrument", self.factory.experiment_instruments)


class TestMissingBenchInstrument(BaseTestCase):
    bench_config = """
    """

    experiment_config = """
experiment_instrument:
  interface: instrument_interface
    """

    def setUp(self) -> None:
        pass

    def test_missing_bench_instrument(self):
        with self.assertRaises(KeyError):
            super().setUp()


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

    def test_too_many_bench_instruments(self):
        with self.assertRaises(KeyError):
            super().setUp()


class TestLoaderDocumentation(BaseTestCase):
    bench_config = ""
    experiment_config = ""
    loader: type[phileas.factory.Loader]

    def setUp(self) -> None:
        class ClassLoader(phileas.factory.Loader):
            name = "class_loader"
            interfaces = {"interface1", "interface2"}

            def initiate_connection(self, configuration: dict) -> object:
                """
                Initialization documentation.

                Parameters:
                 - param1: param1 description
                 - param2: param2 description
                """
                return object()

            def configure(self, instrument: object, configuration: dict):
                """
                Configuration documentation.

                Parameters:
                 - param1: param1 description
                 - param2: param2 description
                """
                pass

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
        _ = factory.get_bench_instrument("laser_1064")  # noqa: F811
        factory.initiate_connections()

        # Bench-level
        self.assertEqual(
            factory.get_bench_instrument("laser_bus_1").device, "/dev/ttyUSB0"
        )
        _ = factory.get_bench_instrument("laser_980")
        _ = factory.get_bench_instrument("main_power_supply")
        with self.assertRaises(KeyError):
            _ = factory.get_bench_instrument("dut")

        # Experiment-level
        self.assertEqual(factory.experiment_instruments["laser"].wavelength, 1064)
        self.assertIn("dut_power_supply", factory.experiment_instruments)
        with self.assertRaises(KeyError):
            _ = factory.get_bench_instrument("dut_power_supply")
        self.assertNotIn("ampli", factory.experiment_instruments)

        # Configuration access
        self.assertEqual(
            factory.experiment_config["laser"]["power"].to_array().shape[0], 10
        )
        self.assertIn("ampli", factory.experiment_config)
        self.assertNotIn("connections", factory.experiment_config)

        # Configuration generation and iteration
        factory.configure_experiment()
        for exp_configuration in factory.configured_experiment_iterator():
            self.assertIn("laser", exp_configuration)
            self.assertIn("dut_power_supply", exp_configuration)
            self.assertIn("dut", exp_configuration)
            self.assertIn("ampli", exp_configuration)

        for instrument_configuration in factory.configured_instrument_iterator(
            "laser", method=parsing.IterationMethod.UNION
        ):
            self.assertIn("power", instrument_configuration)

        # Graph generation
        graph = factory.get_experiment_graph()
        graph.render(
            filename="functional_1",
            directory=self.test_dir,
            format="pdf",
            cleanup=True,
        )
        self.graph_file = self.test_dir / "functional_1.pdf"
        self.assertTrue(self.graph_file.exists())

    def tearDown(self) -> None:
        clear_default_loaders()

        with contextlib.suppress(AttributeError):
            if self.graph_file.exists():
                self.graph_file.unlink()

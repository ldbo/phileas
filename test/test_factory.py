import unittest
from pathlib import Path
from typing import ClassVar

from pyfakefs import fake_filesystem_unittest

import phileas
from phileas import ExperimentFactory


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
        phileas.factory._DEFAULT_LOADERS = dict()


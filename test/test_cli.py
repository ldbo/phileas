import functools
import importlib.resources
import os
import pathlib
import shutil
import sys
import tempfile
import unittest

import hypothesis.strategies as st
from hypothesis import given

from phileas.__main__ import Template, generate_script, generate_template


def in_tempdir(function):
    functools.wraps(function)

    def wrapper(*args, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            sys.path.append(directory)
            base_directory = os.getcwd()
            try:
                os.chdir(directory)
                result = function(*args, **kwargs)
            finally:
                os.chdir(base_directory)
                sys.path.pop()

            return result

    return wrapper


class TestCli(unittest.TestCase):
    @given(
        template_type=st.one_of(st.just(Template.bench), st.just(Template.experiment)),
        no_explanation=st.booleans(),
        no_example=st.booleans(),
    )
    @in_tempdir
    def test_generate_bench_and_experiment(self, template_type: Template, **kwargs):
        self.assertEqual(
            generate_template(
                template_type=template_type,
                file=pathlib.Path(template_type.to_template_file()),
                dont_write=False,
                **kwargs,
            ),
            0,
        )

    @given(
        template_type=st.just(Template.script),
        no_explanation=st.booleans(),
        no_example=st.booleans(),
        bench=st.one_of(st.just(pathlib.Path("bench.yaml")), st.none()),
        no_cli=st.booleans(),
        no_logging=st.booleans(),
    )
    @in_tempdir
    def test_generate_script(
        self, template_type: Template, bench: pathlib.Path, **kwargs
    ):
        test_module = importlib.resources.files("test")
        shutil.copy(str(test_module / "functional_1_bench.yaml"), "./bench.yaml")
        shutil.copy(
            str(test_module / "functional_1_experiment.yaml"), "./experiment.yaml"
        )
        shutil.copy(
            str(test_module / "functional_1_config.py"), "./experiment_config.py"
        )

        if bench is None:
            experiment = None
            experiment_config = None
        else:
            experiment = pathlib.Path("experiment.yaml")
            experiment_config = "experiment_config"

        self.assertEqual(
            generate_script(
                template_type=template_type,
                file=pathlib.Path(template_type.to_template_file()),
                dont_write=False,
                bench=bench,
                experiment=experiment,
                experiment_config_path=experiment_config,
                **kwargs,
            ),
            0,
        )

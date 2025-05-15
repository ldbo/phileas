import argparse
import enum
import importlib
import importlib.resources
import importlib.util
import pathlib
import sys
import typing

import jinja2
import rich.console
import rich.markdown

import phileas

_CONSOLE = rich.console.Console()
_ERROR_CONSOLE = rich.console.Console(stderr=True, style="bold red")

# List loaders


def list_loaders(script: pathlib.Path) -> int:
    inspected_module_spec = importlib.util.spec_from_file_location(
        "inspected_module", script
    )
    if inspected_module_spec is None:
        _ERROR_CONSOLE.print("Cannot open the script.")
        return 1

    loader = inspected_module_spec.loader
    if loader is None:
        _ERROR_CONSOLE.print("Cannot import the script.")
        return 1

    inspected_module = importlib.util.module_from_spec(inspected_module_spec)
    loader.exec_module(inspected_module)

    doc = phileas.ExperimentFactory.get_default_loaders_markdown_documentation()
    if _CONSOLE.is_terminal:
        doc_md = rich.markdown.Markdown(doc)
        _CONSOLE.print(doc_md)
    else:
        _CONSOLE.print(doc)

    return 0


# Template generation


class Template(enum.Enum):
    bench = "bench"
    experiment = "experiment"
    script = "script"

    def __str__(self):
        return self.value

    def to_template_file(self) -> str:
        return {
            "bench": "bench.yaml.template",
            "experiment": "experiment.yaml.template",
            "script": "script.py.template",
        }[self.value]


def generate_template(
    template_type: Template,
    file: pathlib.Path | None,
    dont_write: bool,
    no_explanation: bool,
    no_example: bool,
    **kwargs
) -> int:
    template_environment = jinja2.Environment(
        loader=jinja2.PackageLoader("phileas", "templates"),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = template_environment.get_template(template_type.to_template_file())

    content = template.render(
        no_explanation=no_explanation, no_example=no_example, **kwargs
    )

    if dont_write:
        print(content)
    else:
        assert file is not None
        with open(file, "w") as generated_file:
            generated_file.write(content)

    return 0


def generate_script(
    bench: pathlib.Path | None,
    experiment: pathlib.Path | None,
    experiment_config: str | None,
    **kwargs
) -> int:
    template_data = {}
    if bench is not None and experiment is not None and experiment_config is not None:
        try:
            importlib.import_module(experiment_config)
        except ImportError:
            import experiment_config  # type: ignore[import-not-found]

        factory = phileas.ExperimentFactory(bench, experiment)
        instruments_and_type = []
        dependencies = set()
        for instrument in factory._ExperimentFactory__experiment_instruments.values():
            name = instrument.name
            hints = typing.get_type_hints(
                instrument.bench_instrument.loader.initiate_connection
            )

            instrument_type = None
            if "return" in hints and hints["return"] != typing.Any:
                instrument_type = ".".join(
                    [hints["return"].__module__, hints["return"].__name__]
                )
                dependencies.add(hints["return"].__module__)

            instruments_and_type.append((name, instrument_type))

        template_data["instruments_and_type"] = instruments_and_type
        template_data["bench"] = bench
        template_data["experiment"] = experiment
        template_data["experiment_config"] = experiment_config
        template_data["dependencies"] = dependencies - {experiment_config}
    elif bench is None and experiment is None and experiment_config is None:
        pass
    else:
        print(
            "The three arguments --bench, --experiment and --experiment-config "
            "must either be specified together, or none at all."
        )
        return 1

    return generate_template(**kwargs, **template_data)


def configure_common_template_parser(
    parser: argparse.ArgumentParser, template_type: Template
):
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "-f",
        "--file",
        type=pathlib.Path,
        help="Path of the file to generate.",
        default=pathlib.Path(template_type.to_template_file()),
    )
    output_mode.add_argument(
        "--dont-write",
        action="store_true",
        help="Don't write the generated file to the file system. Instead, send "
        "it to the standard output.",
    )
    parser.add_argument(
        "--no-explanation",
        action="store_true",
        help="Do not add comments explaining how to fill in the templates.",
    )
    parser.add_argument(
        "--no-example",
        action="store_true",
        help="Do not add an example bench instrument in the file.",
    )
    parser.set_defaults(callback=generate_template)


def main():
    parser = argparse.ArgumentParser(description="Phileas CLI utility.")
    commands_parser = parser.add_subparsers(required=True)

    # List loaders

    list_loaders_parser = commands_parser.add_parser(
        "list-loaders",
        description="List the default loaders registered by a script.",
    )
    list_loaders_parser.add_argument(
        "script",
        type=pathlib.Path,
        help="Path of the analyzed script",
    )
    list_loaders_parser.set_defaults(callback=list_loaders)

    # Template generation
    generate_parser = commands_parser.add_parser(
        "generate",
        description=(
            "Template generator, useful to start a new project. It can create "
            "the initial files that you have to use for every project based on "
            "Phileas: a bench and an experiment configuration files, and a "
            "script loading and using them."
        ),
    )
    generate_commands = generate_parser.add_subparsers(required=True)

    generate_bench_parser = generate_commands.add_parser(
        "bench", description="Generate a bench configuration file template."
    )
    configure_common_template_parser(generate_bench_parser, Template.bench)
    generate_bench_parser.set_defaults(template_type=Template.bench)

    generate_experiment_parser = generate_commands.add_parser(
        "experiment", description="Generate an experiment configuration file template."
    )
    configure_common_template_parser(generate_experiment_parser, Template.experiment)
    generate_experiment_parser.set_defaults(template_type=Template.experiment)

    generate_script_parser = generate_commands.add_parser(
        "script", description="Generate a script template."
    )
    configure_common_template_parser(generate_script_parser, Template.script)
    generate_script_parser.add_argument(
        "--bench", "-b", type=pathlib.Path, help="Path of the bench configuration file."
    )
    generate_script_parser.add_argument(
        "--experiment",
        "-e",
        type=pathlib.Path,
        help="Path of the experiment configuration file.",
    )
    generate_script_parser.add_argument(
        "--experiment-config",
        "-c",
        type=str,
        help="Path of the module which defines all the loaders. This must be a "
        "valid Python module path, ie. calling `import EXPERIMENT_CONFIG` must "
        "succeed.",
    )
    generate_script_parser.add_argument(
        "--no-cli", action="store_true", help="Do not add the CLI parser in the script."
    )
    generate_script_parser.add_argument(
        "--no-logging", action="store_true", help="Disable logging."
    )
    generate_script_parser.set_defaults(callback=generate_script)
    generate_script_parser.set_defaults(template_type=Template.script)

    parsed_args = parser.parse_args()
    callback = parsed_args.callback
    args = vars(parsed_args)
    del args["callback"]
    return callback(**args)


if __name__ == "__main__":
    sys.exit(main())

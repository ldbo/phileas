import argparse
import importlib.util
import pathlib
import sys

import rich.console
import rich.markdown

import phileas

_CONSOLE = rich.console.Console()
_ERROR_CONSOLE = rich.console.Console(stderr=True, style="bold red")


def list_loaders(script: pathlib.Path) -> int:
    inspected_module_spec = importlib.util.spec_from_file_location(
        "inspected_module", script
    )
    if inspected_module_spec is None:
        _ERROR_CONSOLE.print("Cannot open the script")
        return 1

    loader = inspected_module_spec.loader
    if loader is None:
        _ERROR_CONSOLE.print("Cannot import the script")
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


def main():
    parser = argparse.ArgumentParser(description="Phileas CLI utility.")
    commands_parser = parser.add_subparsers(required=True)

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

    parsed_args = parser.parse_args()
    callback = parsed_args.callback
    args = vars(parsed_args)
    del args["callback"]
    sys.exit(callback(**args))


if __name__ == "__main__":
    sys.exit(main())

from pathlib import Path

from ruamel.yaml import YAML


def load_yaml_dict_from_file(file: Path | str) -> dict:
    yaml = YAML(typ="safe")
    data = yaml.load(file)

    if data is None:
        data = dict()

    if not isinstance(data, dict):
        raise ValueError(f"YAML file {file} top-level object should be a map")

    return data

from pathlib import Path

from ruamel.yaml import YAML


def load_yaml_dict_from_file(file_path: Path) -> dict:
    yaml = YAML(typ="safe")
    data = yaml.load(file_path)

    if not isinstance(data, dict):
        data = dict()
    return data

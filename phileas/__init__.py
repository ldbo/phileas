# ruff: noqa: F401

__version__ = "0.5"

from . import factory, iteration, parsing, utility
from .factory import (
    ExperimentFactory,
    Loader,
    clear_default_loaders,
    logger,
    register_default_loader,
)

if __debug__:
    logger.warning(
        "phileas code contains assertions that reduce its performances. "
        "Consider running python with the -O or -OO flags, or setting the "
        "PYTHONOPTIMIZE environment variable in production code."
    )

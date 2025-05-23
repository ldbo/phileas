import dataclasses
from dataclasses import dataclass

from phileas.iteration.base import ChildPath, DataTree, IterationTree


@dataclass(frozen=True)
class Seed:
    """
    Seed of a random iteration node, used by its RNG.
    """

    #: Path of the node in the biggest tree that used it.
    path: ChildPath

    #: Salt value, to customize iteration independently of the shape of the
    #: iteration tree.
    salt: DataTree | None

    def to_bytes(self) -> bytes:
        """
        Convert the seed to bytes, for RNG seeding.
        """
        salt = f":{self.salt}" if self.salt is not None else ""
        return ("/".join(map(str, self.path)) + salt).encode("utf-8")


@dataclass(frozen=True)
class RandomTree:
    """
    Additional base class of random iteration trees.
    """

    #: Seed of the generator used by the random tree. It must be guaranteed that
    #: successive iteration values only depend on the seed value.
    #:
    #: For iteration to be possible, the seed must be set. If you don't want to
    #: manually specify the seed, you can use
    #: :py:func:`~phileas.iteration.utility.generate_seeds`.
    seed: Seed | None = None


def generate_seeds(tree: IterationTree, salt: DataTree | None = None) -> IterationTree:
    """
    Populate the seeds of the random nodes and leaves of a tree, using the given
    salt value.
    """

    def _generate_seed(tree: IterationTree, path: ChildPath) -> IterationTree:
        if isinstance(tree, RandomTree):
            new_seed = Seed(path, salt)
            return dataclasses.replace(tree, seed=new_seed)  # type: ignore[call-arg]
        else:
            return tree

    return tree.depth_first_modify(_generate_seed)

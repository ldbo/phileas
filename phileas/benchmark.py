import time

import numpy as np

from phileas.iteration import (
    CartesianProduct,
    IterationTree,
    Pick,
    Shuffle,
    Union,
    Zip,
)
from phileas.iteration.leaf import (
    GeneratorWrapper,
    GeometricRange,
    IntegerRange,
    IterationLiteral,
    LinearRange,
    NumpyRNG,
    PrimeRng,
    Sequence,
    UniformBigIntegerRng,
)
from phileas.iteration.random import generate_seeds

_samples = 100
_max_duration = 10


def benchmark(tree: IterationTree) -> tuple[float, float]:
    benchmark_end = time.time() + _max_duration

    duration = []
    too_long = False
    for sample in range(_samples):
        it = iter(generate_seeds(tree, salt=sample))
        for position in range(len(tree)):
            start = time.time()
            it[position]
            end = time.time()

            if end > benchmark_end:
                too_long = True
                break

            duration.append((end - start) * 1e9)
            position += 1

        if too_long:
            break

    return float(np.mean(duration)), float(np.std(duration))


tests = {
    "IterationLiteral": IterationLiteral(0),
    "GeneratorWrapper_range100": GeneratorWrapper(lambda: iter(range(10)), size=10),
    "LinearRange_10": LinearRange(0, 1, 10),
    "LinearRange_100": LinearRange(0, 1, 100),
    "GeometricRange_10": GeometricRange(1, 10, 10),
    "GeometricRange_100": GeometricRange(1, 10, 100),
    "IntegerRange_10": IntegerRange(0, 10),
    "IntegerRange_100": IntegerRange(0, 100),
    "Sequence_10": Sequence(list(range(10))),
    "Sequence_100": Sequence(list(range(100))),
    "NumpyRNG_uniform_1000": NumpyRNG(size=1000),
    "UniformBigIntegerRng_0xff_1000": UniformBigIntegerRng(low=0, high=0xFF, size=1000),
    "UniformBigIntegerRng_0x100_1000": UniformBigIntegerRng(
        low=0, high=0x100, size=1000
    ),
    # "UniformBigIntegerRng_0xffffffff_1000": UniformBigIntegerRng(
    #     low=0, high=0xFFFFFFFF, size=1000
    # ),
    "PrimeRng_0xff_1000": PrimeRng(low=0, high=0xFF, size=1000),
    # "PrimeRng_0xffffffff_1000": PrimeRng(low=0, high=0xFFFFFFFF, size=1000),
    "CartesianProduct_5x10": CartesianProduct(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
        },
    ),
    "CartesianProduct_5x100": CartesianProduct(
        {
            0: IntegerRange(1, 100),
            1: IntegerRange(1, 100),
            2: IntegerRange(1, 100),
            3: IntegerRange(1, 100),
            4: IntegerRange(1, 100),
        }
    ),
    "CartesianProduct_10x10": CartesianProduct(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
            5: IntegerRange(1, 10),
            6: IntegerRange(1, 10),
            7: IntegerRange(1, 10),
            8: IntegerRange(1, 10),
            9: IntegerRange(1, 10),
        },
    ),
    "CartesianProduct_lazy_5x10": CartesianProduct(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
        },
        lazy=True,
    ),
    "CartesianProduct_lazy_5x100": CartesianProduct(
        {
            0: IntegerRange(1, 100),
            1: IntegerRange(1, 100),
            2: IntegerRange(1, 100),
            3: IntegerRange(1, 100),
            4: IntegerRange(1, 100),
        },
        lazy=True,
    ),
    "CartesianProduct_lazy_10x10": CartesianProduct(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
            5: IntegerRange(1, 10),
            6: IntegerRange(1, 10),
            7: IntegerRange(1, 10),
            8: IntegerRange(1, 10),
            9: IntegerRange(1, 10),
        },
        lazy=True,
    ),
    "CartesianProduct_snake_5x10": CartesianProduct(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
        },
        snake=True,
    ),
    "CartesianProduct_snake_5x100": CartesianProduct(
        {
            0: IntegerRange(1, 100),
            1: IntegerRange(1, 100),
            2: IntegerRange(1, 100),
            3: IntegerRange(1, 100),
            4: IntegerRange(1, 100),
        },
        snake=True,
    ),
    "CartesianProduct_snake_10x10": CartesianProduct(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
            5: IntegerRange(1, 10),
            6: IntegerRange(1, 10),
            7: IntegerRange(1, 10),
            8: IntegerRange(1, 10),
            9: IntegerRange(1, 10),
        },
        snake=True,
    ),
    "CartesianProduct_lazy_snake_5x10": CartesianProduct(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
        },
        lazy=True,
        snake=True,
    ),
    "CartesianProduct_lazy_snake_5x100": CartesianProduct(
        {
            0: IntegerRange(1, 100),
            1: IntegerRange(1, 100),
            2: IntegerRange(1, 100),
            3: IntegerRange(1, 100),
            4: IntegerRange(1, 100),
        },
        lazy=True,
        snake=True,
    ),
    "CartesianProduct_lazy_snake_10x10": CartesianProduct(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
            5: IntegerRange(1, 10),
            6: IntegerRange(1, 10),
            7: IntegerRange(1, 10),
            8: IntegerRange(1, 10),
            9: IntegerRange(1, 10),
        },
        lazy=True,
        snake=True,
    ),
    "Union_5x10": Union(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
        },
    ),
    "Union_5x100": Union(
        {
            0: IntegerRange(1, 100),
            1: IntegerRange(1, 100),
            2: IntegerRange(1, 100),
            3: IntegerRange(1, 100),
            4: IntegerRange(1, 100),
        }
    ),
    "Union_10x10": Union(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
            5: IntegerRange(1, 10),
            6: IntegerRange(1, 10),
            7: IntegerRange(1, 10),
            8: IntegerRange(1, 10),
            9: IntegerRange(1, 10),
        },
    ),
    "Union_lazy_5x10": Union(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
        },
        lazy=True,
    ),
    "Union_lazy_5x100": Union(
        {
            0: IntegerRange(1, 100),
            1: IntegerRange(1, 100),
            2: IntegerRange(1, 100),
            3: IntegerRange(1, 100),
            4: IntegerRange(1, 100),
        },
        lazy=True,
    ),
    "Union_lazy_10x10": Union(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
            5: IntegerRange(1, 10),
            6: IntegerRange(1, 10),
            7: IntegerRange(1, 10),
            8: IntegerRange(1, 10),
            9: IntegerRange(1, 10),
        },
        lazy=True,
    ),
    "Zip_5x10": Zip(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
        },
    ),
    "Zip_5x100": Zip(
        {
            0: IntegerRange(1, 100),
            1: IntegerRange(1, 100),
            2: IntegerRange(1, 100),
            3: IntegerRange(1, 100),
            4: IntegerRange(1, 100),
        },
    ),
    "Zip_10x10": Zip(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
            5: IntegerRange(1, 10),
            6: IntegerRange(1, 10),
            7: IntegerRange(1, 10),
            8: IntegerRange(1, 10),
            9: IntegerRange(1, 10),
        },
    ),
    "Zip_lazy_5x10": Zip(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
        },
        lazy=True,
    ),
    "Zip_lazy_5x100": Zip(
        {
            0: IntegerRange(1, 100),
            1: IntegerRange(1, 100),
            2: IntegerRange(1, 100),
            3: IntegerRange(1, 100),
            4: IntegerRange(1, 100),
        },
        lazy=True,
    ),
    "Zip_lazy_10x10": Zip(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
            5: IntegerRange(1, 10),
            6: IntegerRange(1, 10),
            7: IntegerRange(1, 10),
            8: IntegerRange(1, 10),
            9: IntegerRange(1, 10),
        },
        lazy=True,
    ),
    "Pick_5x10": Pick(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
        },
    ),
    "Pick_5x100": Pick(
        {
            0: IntegerRange(1, 100),
            1: IntegerRange(1, 100),
            2: IntegerRange(1, 100),
            3: IntegerRange(1, 100),
            4: IntegerRange(1, 100),
        },
    ),
    "Pick_10x10": Pick(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
            5: IntegerRange(1, 10),
            6: IntegerRange(1, 10),
            7: IntegerRange(1, 10),
            8: IntegerRange(1, 10),
            9: IntegerRange(1, 10),
        },
    ),
    "Pick_lazy_5x10": Pick(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
        },
        lazy=True,
    ),
    "Pick_lazy_5x100": Pick(
        {
            0: IntegerRange(1, 100),
            1: IntegerRange(1, 100),
            2: IntegerRange(1, 100),
            3: IntegerRange(1, 100),
            4: IntegerRange(1, 100),
        },
        lazy=True,
    ),
    "Pick_lazy_10x10": Pick(
        {
            0: IntegerRange(1, 10),
            1: IntegerRange(1, 10),
            2: IntegerRange(1, 10),
            3: IntegerRange(1, 10),
            4: IntegerRange(1, 10),
            5: IntegerRange(1, 10),
            6: IntegerRange(1, 10),
            7: IntegerRange(1, 10),
            8: IntegerRange(1, 10),
            9: IntegerRange(1, 10),
        },
        lazy=True,
    ),
    "Shuffle_10": Shuffle(IntegerRange(1, 10)),
    "Shuffle_1000": Shuffle(IntegerRange(1, 1000)),
    "Shuffle_10000": Shuffle(IntegerRange(1, 10000)),
}


def run():
    name_width = max(map(len, tests)) + 7
    print(
        f"{'Test':{name_width}}{'Time per iteration (us)':30}{'Iteration frequency (kHz)'}"
    )

    for name, tree in tests.items():
        mean, std = benchmark(tree)
        duration = f"{mean / 1e3:.1f} ± {std / 1e3:.1f}"
        frequency = f"{1e6 / mean:.1f}"
        print(f"{name:{name_width}}{duration:30}{frequency}")

from __future__ import annotations

import gc
import math
import os
import platform
import sys
import time

import numpy as np
import pyroaring

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_pyroaring import BitMap  # noqa: E402


def timeit(function, repeat: int = 7, number: int = 1) -> float:
    best = math.inf
    for _ in range(repeat):
        gc.disable()
        start = time.perf_counter_ns()
        for _ in range(number):
            function()
        elapsed = (time.perf_counter_ns() - start) / number
        gc.enable()
        best = min(best, elapsed)
    return best / 1e6


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    dense_a = range(0, 5_000_000, 2)
    dense_b = range(1_000_000, 6_000_000, 3)
    ours_a, ours_b = BitMap(dense_a), BitMap(dense_b)
    theirs_a, theirs_b = pyroaring.BitMap(dense_a), pyroaring.BitMap(dense_b)

    rng = np.random.default_rng(42)
    sparse_a = rng.integers(0, 1 << 24, 400_000, dtype=np.uint32)
    sparse_b = rng.integers(0, 1 << 24, 400_000, dtype=np.uint32)
    ours_sparse_a, ours_sparse_b = BitMap(sparse_a), BitMap(sparse_b)
    theirs_sparse_a = pyroaring.BitMap(sparse_a)
    theirs_sparse_b = pyroaring.BitMap(sparse_b)

    cases = [
        ("dense union (2.5M / 1.7M)", lambda: ours_a | ours_b, lambda: theirs_a | theirs_b, 1),
        (
            "dense intersection (2.5M / 1.7M)",
            lambda: ours_a & ours_b,
            lambda: theirs_a & theirs_b,
            1,
        ),
        (
            "dense intersection cardinality",
            lambda: ours_a.intersection_cardinality(ours_b),
            lambda: theirs_a.intersection_cardinality(theirs_b),
            3,
        ),
        (
            "sparse union (400k / 400k)",
            lambda: ours_sparse_a | ours_sparse_b,
            lambda: theirs_sparse_a | theirs_sparse_b,
            1,
        ),
        (
            "sparse difference (400k / 400k)",
            lambda: ours_sparse_a - ours_sparse_b,
            lambda: theirs_sparse_a - theirs_sparse_b,
            1,
        ),
        (
            "rank at 4,000,000",
            lambda: ours_a.rank(4_000_000),
            lambda: theirs_a.rank(4_000_000),
            100,
        ),
        (
            "construct range(5,000,000)",
            lambda: BitMap(range(5_000_000)),
            lambda: pyroaring.BitMap(range(5_000_000)),
            1,
        ),
    ]

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}; Python {platform.python_version()}")
    print()
    print("| operation | mojo-pyroaring | pyroaring 1.0.4 | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs, number in cases:
        ours()
        theirs()
        ours_ms = timeit(ours, number=number)
        theirs_ms = timeit(theirs, number=number)
        ratio = theirs_ms / ours_ms
        label = "Mojo faster" if ratio > 1 else "Mojo slower"
        print(
            f"| {name} | {ours_ms:.4f} ms | {theirs_ms:.4f} ms | "
            f"{ratio:.2f}x ({label}) |"
        )


if __name__ == "__main__":
    main()

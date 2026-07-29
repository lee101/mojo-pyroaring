# mojo-pyroaring

`mojo-pyroaring` is a standalone implementation of 32-bit Roaring bitmaps with
the container kernels written in Mojo and a Python API modeled on
[`pyroaring`](https://github.com/Ezibenroc/PyRoaringBitMap). It is a real
compressed bitmap implementation: values are partitioned by their high 16 bits
and each partition switches between a sorted `uint16` array and an 8 KiB
bitset at the standard 4,096-value threshold.

The Python package is named `mojo_pyroaring` so it can be installed beside the
upstream package for parity tests. For the covered API, switching requires only
changing the import:

```python
from mojo_pyroaring import BitMap, FrozenBitMap
```

## Coverage

The port covers `BitMap`, `FrozenBitMap`, and `AbstractBitMap` for unsigned
32-bit values:

- construction, copying, iteration, reverse iteration, membership, length,
  integer indexing, slicing, `to_array`, equality, and ordering relations;
- union, intersection, difference, and symmetric difference through operators,
  named methods, in-place operators, and cardinality-only methods;
- `rank`, `range_cardinality`, `contains_range`, `next_set_bit`,
  `iter_equal_or_larger`, minimum, maximum, Jaccard index, and intersection
  tests;
- single-value, bulk, and range mutation; flip, shift, overwrite, clear, and
  pop;
- container statistics and immutable, hashable `FrozenBitMap` values.

Not covered yet:

- `BitMap64` and `FrozenBitMap64`;
- CRoaring-compatible `serialize` and `deserialize`;
- run containers. `run_optimize()` consequently returns `False`, and
  `shrink_to_fit()` returns zero;
- actual copy-on-write storage. The constructor flag and property are accepted,
  but logical copies share immutable container arrays and mutations replace
  containers rather than modifying shared storage.

The test suite compares behavior directly with conda-forge `pyroaring 1.0.4`,
including randomized sparse values, dense containers, 32-bit boundaries, and
mutating operations.

## Install and run

The repository pins the Mojo nightly used to build it.

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-pyroaring.so`. The Python loader also
rebuilds a missing or stale library when Mojo is available.

## Usage

```python
from mojo_pyroaring import BitMap

active = BitMap([1, 3, 5, 65_536])
recent = BitMap(range(3, 8))

print(list(active & recent))
print(active.union_cardinality(recent))
print(active.rank(5))
```

Output:

```text
[3, 5]
7
3
```

## How it works

A bitmap is a sorted map from a 16-bit key to one of two container layouts.
Sparse containers are contiguous sorted `uint16` arrays. Containers above
4,096 values are fixed 1,024-word `uint64` bitsets. Results automatically
change representation when they cross the threshold.

Python owns the dictionaries and NumPy allocations. Contiguous array addresses
cross a small C ABI as integer values; the Mojo exports reconstruct
`UnsafePointer[..., AnyOrigin[mut=True]]` pointers and never retain or allocate
Python memory. Container addresses are submitted in batches, and result
containers are views into shared NumPy slabs instead of per-container copies.
The Mojo compilation unit implements sorted array merges, SIMD wordwise bitset
algebra and popcount cardinalities, array/bitset conversion, rank, and select.
Large independent array-container batches use a four-worker host pool above a
200,000-value threshold; smaller batches stay serial. There is no GPU path.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14. These are best-of-seven wall-clock measurements;
the benchmark constructs both bitmap inputs before timing each operation.

| operation | mojo-pyroaring | pyroaring 1.0.4 | upstream / Mojo |
|---|---:|---:|---:|
| dense union (2.5M / 1.7M) | 0.8422 ms | 0.1077 ms | 0.13x (Mojo slower) |
| dense intersection (2.5M / 1.7M) | 0.5614 ms | 0.0881 ms | 0.16x (Mojo slower) |
| dense intersection cardinality | 0.3282 ms | 0.0324 ms | 0.10x (Mojo slower) |
| sparse union (400k / 400k) | 3.2510 ms | 0.9499 ms | 0.29x (Mojo slower) |
| sparse difference (400k / 400k) | 4.8814 ms | 0.4769 ms | 0.10x (Mojo slower) |
| rank at 4,000,000 | 0.0013 ms | 0.0006 ms | 0.43x (Mojo slower) |
| construct range(5,000,000) | 0.2263 ms | 0.0155 ms | 0.07x (Mojo slower) |

Upstream remains faster in every measured case. It is a thin Cython binding to
the heavily optimized CRoaring C library and benefits from run containers,
which this port does not yet implement. Regressions are reported directly
rather than hidden by extrapolation.

Run the locked benchmark again with:

```bash
pixi run bench
```

The Pixi task takes a machine-wide `flock` so concurrent jobs do not distort the
result.

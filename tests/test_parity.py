from __future__ import annotations

from array import array

import numpy as np
import pytest
import pyroaring as upstream

from mojo_pyroaring import BitMap, FrozenBitMap


def same(ours, theirs) -> None:
    assert list(ours) == list(theirs)
    assert len(ours) == len(theirs)
    assert bool(ours) == bool(theirs)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [3, 12, 3, 0],
        range(10_000),
        range(0, 100_000, 3),
        [0, 65_535, 65_536, 70_000, 2**32 - 1],
    ],
)
def test_construction_and_iteration(values):
    same(BitMap(values), upstream.BitMap(values))


def test_copy_membership_reverse_and_slices_are_independent():
    original = BitMap([0, 3, 12, 65_536, 2**32 - 1])
    copied = original.copy()
    assert original == copied
    assert original != BitMap([0, 3, 12])
    copied.add(99)
    assert 99 not in original
    assert 99 in copied
    assert list(reversed(original)) == list(
        reversed(upstream.BitMap([0, 3, 12, 65_536, 2**32 - 1]))
    )
    same(original[1:4], upstream.BitMap([0, 3, 12, 65_536, 2**32 - 1])[1:4])


def test_random_construction_and_container_boundaries():
    rng = np.random.default_rng(2026)
    values = rng.integers(0, 1 << 22, size=150_000, dtype=np.uint32)
    ours = BitMap(values)
    theirs = upstream.BitMap(values)
    same(ours, theirs)
    assert list(reversed(ours)) == list(reversed(theirs))


@pytest.mark.parametrize(
    ("operator_name", "method_name"),
    [
        ("or_", "union"),
        ("and_", "intersection"),
        ("sub", "difference"),
        ("xor", "symmetric_difference"),
    ],
)
def test_dense_set_algebra(operator_name, method_name):
    import operator

    left_values = np.r_[np.arange(0, 180_000, 2), np.arange(250_000, 330_000)]
    right_values = np.r_[np.arange(60_000, 240_000, 3), np.arange(300_000, 410_000)]
    ours_left, ours_right = BitMap(left_values), BitMap(right_values)
    their_left, their_right = upstream.BitMap(left_values), upstream.BitMap(right_values)
    same(
        getattr(operator, operator_name)(ours_left, ours_right),
        getattr(operator, operator_name)(their_left, their_right),
    )
    same(
        getattr(ours_left, method_name)(ours_right),
        getattr(their_left, method_name)(their_right),
    )


def test_sparse_set_algebra():
    rng = np.random.default_rng(7)
    a = rng.integers(0, 2**32, 70_000, dtype=np.uint32)
    b = rng.integers(0, 2**32, 80_000, dtype=np.uint32)
    for operation in ("union", "intersection", "difference", "symmetric_difference"):
        same(
            getattr(BitMap(a), operation)(BitMap(b)),
            getattr(upstream.BitMap(a), operation)(upstream.BitMap(b)),
        )


def test_multi_bitmap_named_operations():
    values = ([1, 2, 3, 10], [2, 3, 4, 20], [3, 4, 5, 30])
    ours = [BitMap(group) for group in values]
    theirs = [upstream.BitMap(group) for group in values]
    same(ours[0].union(*ours[1:]), theirs[0].union(*theirs[1:]))
    same(ours[0].intersection(*ours[1:]), theirs[0].intersection(*theirs[1:]))
    same(ours[0].difference(*ours[1:]), theirs[0].difference(*theirs[1:]))


def test_cardinality_operations_and_relations():
    a_values = range(0, 300_000, 2)
    b_values = range(100_000, 400_000, 3)
    ours_a, ours_b = BitMap(a_values), BitMap(b_values)
    their_a, their_b = upstream.BitMap(a_values), upstream.BitMap(b_values)
    for name in (
        "intersection_cardinality",
        "union_cardinality",
        "difference_cardinality",
        "symmetric_difference_cardinality",
    ):
        assert getattr(ours_a, name)(ours_b) == getattr(their_a, name)(their_b)
    assert ours_a.intersect(ours_b) == their_a.intersect(their_b)
    assert ours_a.isdisjoint(ours_b) == their_a.isdisjoint(their_b)
    assert ours_a.jaccard_index(ours_b) == pytest.approx(
        their_a.jaccard_index(their_b)
    )
    subset = BitMap(range(0, 10_000, 2))
    assert subset < ours_a
    assert subset <= ours_a
    assert ours_a > subset
    assert ours_a >= subset


def test_rank_index_slice_and_extrema():
    values = np.r_[np.arange(0, 10_000, 3), [65_535, 65_536, 2**32 - 1]]
    ours, theirs = BitMap(values), upstream.BitMap(values)
    for value in (0, 1, 9_999, 65_535, 65_536, 1_000_000, 2**32 - 1):
        assert ours.rank(value) == theirs.rank(value)
    for index in (0, 1, 100, -1, -100):
        assert ours[index] == theirs[index]
    same(ours[5:200:7], theirs[5:200:7])
    assert ours.min() == theirs.min()
    assert ours.max() == theirs.max()
    with pytest.raises(IndexError):
        _ = ours[len(ours)]


def test_simd_rank_tail_and_parallel_batch_threshold(monkeypatch):
    import mojo_pyroaring.bitmap as implementation

    dense_values = range(0, 1 << 16, 2)
    ours_dense = BitMap(dense_values)
    theirs_dense = upstream.BitMap(dense_values)
    for value in (0, 63, 64, 127, 191, 192, 255, 319):
        assert (
            implementation.lib().mpr_bitset_rank(
                implementation.address(ours_dense._containers[0].data), value
            )
            == theirs_dense.rank(value)
        )
        assert ours_dense.rank(value) == theirs_dense.rank(value)

    left_values = [
        (key << 16) | low
        for key in range(9)
        for low in range(0, 600, 2)
    ]
    right_values = [
        (key << 16) | low
        for key in range(9)
        for low in range(100, 700, 3)
    ]
    ours_left, ours_right = BitMap(left_values), BitMap(right_values)
    theirs_left = upstream.BitMap(left_values)
    theirs_right = upstream.BitMap(right_values)

    monkeypatch.setattr(implementation, "PARALLEL_ARRAY_VALUES", 10**9)
    same(ours_left | ours_right, theirs_left | theirs_right)
    monkeypatch.setattr(implementation, "PARALLEL_ARRAY_VALUES", 1)
    same(ours_left - ours_right, theirs_left - theirs_right)


def test_ranges_queries_and_next_iteration():
    ours = BitMap(range(100, 100_000, 3))
    theirs = upstream.BitMap(range(100, 100_000, 3))
    for start, end in ((0, 0), (0, 100), (100, 101), (100, 1_000), (65_000, 70_000)):
        assert ours.range_cardinality(start, end) == theirs.range_cardinality(start, end)
        assert ours.contains_range(start, end) == theirs.contains_range(start, end)
    for value in (0, 100, 101, 65_535):
        assert ours.next_set_bit(value) == theirs.next_set_bit(value)
        assert list(ours.iter_equal_or_larger(value)) == list(
            theirs.iter_equal_or_larger(value)
        )
    with pytest.raises(ValueError):
        theirs.next_set_bit(99_999)
    with pytest.raises(ValueError):
        ours.next_set_bit(99_999)


def test_add_discard_remove_and_pop():
    ours, theirs = BitMap([3, 12]), upstream.BitMap([3, 12])
    for value in (42, 42, 65_536, 2**32 - 1):
        ours.add(value)
        theirs.add(value)
    same(ours, theirs)
    ours.discard(99)
    theirs.discard(99)
    ours.remove(12)
    theirs.remove(12)
    same(ours, theirs)
    assert ours.pop() == theirs.pop()
    same(ours, theirs)
    with pytest.raises(KeyError):
        ours.add_checked(42)
    with pytest.raises(KeyError):
        ours.remove(12)


def test_bulk_mutations_and_overwrite():
    ours, theirs = BitMap([1, 2, 3]), upstream.BitMap([1, 2, 3])
    ours.update([3, 4], BitMap([5, 6]))
    theirs.update([3, 4], upstream.BitMap([5, 6]))
    same(ours, theirs)
    ours.intersection_update([2, 4, 6], BitMap([4, 6, 8]))
    theirs.intersection_update([2, 4, 6], upstream.BitMap([4, 6, 8]))
    same(ours, theirs)
    ours.symmetric_difference_update(BitMap([4, 10]))
    theirs.symmetric_difference_update(upstream.BitMap([4, 10]))
    same(ours, theirs)
    ours.difference_update(BitMap([6]), BitMap([20]))
    theirs.difference_update(upstream.BitMap([6]), upstream.BitMap([20]))
    same(ours, theirs)
    ours.overwrite(BitMap([7, 8]))
    theirs.overwrite(upstream.BitMap([7, 8]))
    same(ours, theirs)
    ours.clear()
    theirs.clear()
    same(ours, theirs)


def test_range_mutations_and_flip():
    ours, theirs = BitMap([5, 7, 100_000]), upstream.BitMap([5, 7, 100_000])
    ours.add_range(6, 70_000)
    theirs.add_range(6, 70_000)
    same(ours, theirs)
    ours.remove_range(10_000, 60_000)
    theirs.remove_range(10_000, 60_000)
    same(ours, theirs)
    same(ours.flip(0, 20), theirs.flip(0, 20))
    ours.flip_inplace(65_530, 65_550)
    theirs.flip_inplace(65_530, 65_550)
    same(ours, theirs)


@pytest.mark.parametrize("offset", [-200, -1, 0, 1, 200])
def test_shift(offset):
    values = [0, 100, 65_535, 65_536, 2**32 - 1]
    same(BitMap(values).shift(offset), upstream.BitMap(values).shift(offset))


def test_inplace_operators():
    for symbol in ("__ior__", "__iand__", "__isub__", "__ixor__"):
        ours, theirs = BitMap(range(0, 20, 2)), upstream.BitMap(range(0, 20, 2))
        getattr(ours, symbol)(BitMap(range(5, 25, 3)))
        getattr(theirs, symbol)(upstream.BitMap(range(5, 25, 3)))
        same(ours, theirs)


def test_frozen_bitmap():
    ours = FrozenBitMap([3, 12, 65_536])
    theirs = upstream.FrozenBitMap([3, 12, 65_536])
    same(ours, theirs)
    assert hash(ours) == hash(FrozenBitMap(ours))
    assert isinstance(ours | FrozenBitMap([9]), FrozenBitMap)
    assert not hasattr(ours, "add")


def test_to_array_copy_on_write_and_statistics():
    ours = BitMap(range(0, 20_000, 2), copy_on_write=True)
    theirs = upstream.BitMap(range(0, 20_000, 2), copy_on_write=True)
    assert isinstance(ours.to_array(), array)
    assert ours.to_array() == theirs.to_array()
    assert ours.copy_on_write == theirs.copy_on_write
    stats = ours.get_statistics()
    assert stats["cardinality"] == len(theirs)
    assert stats["min_value"] == theirs.min()
    assert stats["max_value"] == theirs.max()
    assert (
        stats["n_values_array_containers"] + stats["n_values_bitset_containers"]
        == len(ours)
    )
    assert ours.run_optimize() is False
    assert ours.shrink_to_fit() == 0


def test_native_buffer_boundary_rejects_unsafe_layouts():
    from mojo_pyroaring._lib import address

    with pytest.raises(ValueError, match="empty"):
        address(np.empty(0, dtype=np.uint16))
    with pytest.raises(ValueError, match="contiguous"):
        address(np.arange(8, dtype=np.uint16)[::2])
    with pytest.raises(TypeError, match="NumPy"):
        address(array("H", [1]))


@pytest.mark.parametrize("value", [-1, 2**32])
def test_uint32_validation(value):
    with pytest.raises(OverflowError):
        BitMap([value])
    bitmap = BitMap()
    with pytest.raises(OverflowError):
        bitmap.add(value)


def test_numpy_uint32_zero_copy_input_validation():
    values = np.arange(0, 100_000, 7, dtype=np.uint32)
    same(BitMap(values), upstream.BitMap(values))
    with pytest.raises(OverflowError):
        BitMap(np.array([-1, 2], dtype=np.int64))


def test_empty_errors_and_identity():
    import math

    ours, theirs = BitMap(), upstream.BitMap()
    assert math.isnan(ours.jaccard_index(BitMap()))
    assert math.isnan(theirs.jaccard_index(upstream.BitMap()))
    for name in ("min", "max", "pop"):
        with pytest.raises((ValueError, KeyError)):
            getattr(ours, name)()

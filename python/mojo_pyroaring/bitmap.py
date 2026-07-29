from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import operator
from threading import Lock
from typing import Any

import numpy as np

from ._lib import address, lib

MAX_VALUE = (1 << 32) - 1
ARRAY_LIMIT = 4096
WORDS_PER_CONTAINER = 1024
_OPERATIONS = {"union": 0, "intersection": 1, "difference": 2, "xor": 3}
BATCH_MIN_CONTAINERS = 4
PARALLEL_ARRAY_VALUES = 200_000
PARALLEL_WORKERS = 4
_executor: ThreadPoolExecutor | None = None
_executor_lock = Lock()


def _uint32(value: Any) -> int:
    if isinstance(value, float):
        value = int(value)
    else:
        value = operator.index(value)
    if value < 0:
        raise OverflowError("can't convert negative value to uint32_t")
    if value > MAX_VALUE:
        raise OverflowError("value too large to convert to uint32_t")
    return value


def _range_bound(value: Any) -> int:
    value = operator.index(value)
    if value < 0:
        raise OverflowError("can't convert negative value to uint64_t")
    if value > MAX_VALUE + 1:
        raise OverflowError("value too large to convert to uint32_t")
    return value


@dataclass(slots=True)
class _Container:
    data: np.ndarray
    cardinality: int
    rank_prefix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.data.flags.c_contiguous or not self.data.flags.aligned:
            raise ValueError("container storage must be contiguous and aligned")
        if self.data.dtype == np.uint16:
            if not 0 < self.cardinality <= ARRAY_LIMIT:
                raise ValueError("invalid array-container cardinality")
            if self.data.ndim != 1 or self.data.size != self.cardinality:
                raise ValueError("invalid array-container storage")
        elif self.data.dtype == np.uint64:
            if not ARRAY_LIMIT < self.cardinality <= 1 << 16:
                raise ValueError("invalid bitset-container cardinality")
            if self.data.ndim != 1 or self.data.size != WORDS_PER_CONTAINER:
                raise ValueError("invalid bitset-container storage")
        else:
            raise TypeError("container storage must use uint16 or uint64")
        self.data.flags.writeable = False

    @property
    def dense(self) -> bool:
        return self.data.dtype == np.uint64

    def copy(self) -> "_Container":
        return _Container(self.data, self.cardinality, self.rank_prefix)


_FULL_BITSET = np.full(WORDS_PER_CONTAINER, np.uint64(0xFFFFFFFFFFFFFFFF))
_FULL_BITSET.flags.writeable = False


def _parallel_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=PARALLEL_WORKERS,
                    thread_name_prefix="mojo-pyroaring",
                )
    return _executor


def _array_container(values: np.ndarray) -> _Container:
    result = np.ascontiguousarray(values, dtype=np.uint16)
    return _Container(result, len(result))


def _array_to_dense(container: _Container) -> np.ndarray:
    if container.dense:
        return container.data
    result = np.empty(WORDS_PER_CONTAINER, dtype=np.uint64)
    lib().mpr_array_to_bitset(
        address(container.data), container.cardinality, address(result)
    )
    return result


def _dense_to_container(bits: np.ndarray, cardinality: int) -> _Container | None:
    if cardinality == 0:
        return None
    if cardinality > ARRAY_LIMIT:
        return _Container(np.ascontiguousarray(bits, dtype=np.uint64), cardinality)
    values = np.empty(cardinality, dtype=np.uint16)
    written = lib().mpr_bitset_to_array(address(bits), address(values))
    if written != cardinality:
        raise RuntimeError("Mojo container cardinality invariant failed")
    return _Container(values, cardinality)


def _container_binary(
    left: _Container, right: _Container, operation: str
) -> _Container | None:
    kernel = lib()
    if not left.dense and not right.dense:
        if operation in ("union", "xor"):
            capacity = left.cardinality + right.cardinality
        elif operation == "intersection":
            capacity = min(left.cardinality, right.cardinality)
        else:
            capacity = left.cardinality
        result = np.empty(max(capacity, 1), dtype=np.uint16)
        function = getattr(kernel, f"mpr_array_{operation}")
        cardinality = function(
            address(left.data),
            left.cardinality,
            address(right.data),
            right.cardinality,
            address(result),
        )
        if cardinality == 0:
            return None
        result = result[:cardinality]
        if cardinality <= ARRAY_LIMIT:
            return _Container(result, cardinality)
        bits = np.empty(WORDS_PER_CONTAINER, dtype=np.uint64)
        kernel.mpr_array_to_bitset(address(result), cardinality, address(bits))
        return _Container(bits, cardinality)

    left_bits = _array_to_dense(left)
    right_bits = _array_to_dense(right)
    result_bits = np.empty(WORDS_PER_CONTAINER, dtype=np.uint64)
    cardinality = kernel.mpr_bitset_binary(
        address(left_bits),
        address(right_bits),
        address(result_bits),
        _OPERATIONS[operation],
    )
    return _dense_to_container(result_bits, cardinality)


def _batch_array_binary_serial(
    pairs: list[tuple[int, _Container, _Container]], operation: str
) -> list[tuple[int, _Container | None]]:
    count = len(pairs)
    left_addresses = np.fromiter(
        (address(left.data) for _, left, _ in pairs), dtype=np.uint64, count=count
    )
    right_addresses = np.fromiter(
        (address(right.data) for _, _, right in pairs), dtype=np.uint64, count=count
    )
    left_sizes = np.fromiter(
        (left.cardinality for _, left, _ in pairs), dtype=np.int64, count=count
    )
    right_sizes = np.fromiter(
        (right.cardinality for _, _, right in pairs), dtype=np.int64, count=count
    )
    if operation in ("union", "xor"):
        capacities = left_sizes + right_sizes
    elif operation == "intersection":
        capacities = np.minimum(left_sizes, right_sizes)
    else:
        capacities = left_sizes
    offsets = np.empty(count, dtype=np.int64)
    offsets[0] = 0
    if count > 1:
        np.cumsum(capacities[:-1], out=offsets[1:])
    total_capacity = int(capacities.sum())
    values = np.empty(max(total_capacity, 1), dtype=np.uint16)
    destination_addresses = np.asarray(
        address(values) + offsets * values.itemsize, dtype=np.uint64
    )
    cardinalities = np.empty(count, dtype=np.int64)
    lib().mpr_array_binary_batch(
        address(left_addresses),
        address(left_sizes),
        address(right_addresses),
        address(right_sizes),
        address(destination_addresses),
        address(cardinalities),
        count,
        int(left_sizes.sum() + right_sizes.sum()),
        _OPERATIONS[operation],
    )
    results: list[tuple[int, _Container | None]] = []
    for index, (key, _, _) in enumerate(pairs):
        cardinality = int(cardinalities[index])
        if cardinality == 0:
            results.append((key, None))
            continue
        start = int(offsets[index])
        merged = values[start : start + cardinality]
        if cardinality <= ARRAY_LIMIT:
            container = _Container(merged, cardinality)
        else:
            bits = np.empty(WORDS_PER_CONTAINER, dtype=np.uint64)
            lib().mpr_array_to_bitset(address(merged), cardinality, address(bits))
            container = _Container(bits, cardinality)
        results.append((key, container))
    return results


def _batch_array_binary(
    pairs: list[tuple[int, _Container, _Container]], operation: str
) -> list[tuple[int, _Container | None]]:
    total_values = sum(
        left.cardinality + right.cardinality for _, left, right in pairs
    )
    if len(pairs) < PARALLEL_WORKERS * 2 or total_values < PARALLEL_ARRAY_VALUES:
        return _batch_array_binary_serial(pairs, operation)
    chunk_size = (len(pairs) + PARALLEL_WORKERS - 1) // PARALLEL_WORKERS
    chunks = [
        pairs[start : start + chunk_size]
        for start in range(0, len(pairs), chunk_size)
    ]
    results: list[tuple[int, _Container | None]] = []
    for chunk_results in _parallel_executor().map(
        _batch_array_binary_serial,
        chunks,
        [operation] * len(chunks),
    ):
        results.extend(chunk_results)
    return results


def _batch_dense_binary(
    pairs: list[tuple[int, _Container, _Container]], operation: str
) -> list[tuple[int, _Container | None]]:
    count = len(pairs)
    left_addresses = np.fromiter(
        (address(left.data) for _, left, _ in pairs), dtype=np.uint64, count=count
    )
    right_addresses = np.fromiter(
        (address(right.data) for _, _, right in pairs), dtype=np.uint64, count=count
    )
    result_bits = np.empty((count, WORDS_PER_CONTAINER), dtype=np.uint64)
    cardinalities = np.empty(count, dtype=np.int64)
    lib().mpr_bitset_binary_batch(
        address(left_addresses),
        address(right_addresses),
        address(result_bits),
        address(cardinalities),
        count,
        _OPERATIONS[operation],
    )
    return [
        (
            key,
            _dense_to_container(result_bits[index], int(cardinalities[index])),
        )
        for index, (key, _, _) in enumerate(pairs)
    ]


def _container_intersection_cardinality(
    left: _Container, right: _Container
) -> int:
    kernel = lib()
    if not left.dense and not right.dense:
        return int(
            kernel.mpr_array_intersection_cardinality(
                address(left.data),
                left.cardinality,
                address(right.data),
                right.cardinality,
            )
        )
    scratch = np.empty(WORDS_PER_CONTAINER, dtype=np.uint64)
    return int(
        kernel.mpr_bitset_binary(
            address(_array_to_dense(left)),
            address(_array_to_dense(right)),
            address(scratch),
            _OPERATIONS["intersection"],
        )
    )


def _batch_array_intersection_cardinality(
    pairs: list[tuple[_Container, _Container]],
) -> int:
    count = len(pairs)
    left_addresses = np.fromiter(
        (address(left.data) for left, _ in pairs), dtype=np.uint64, count=count
    )
    right_addresses = np.fromiter(
        (address(right.data) for _, right in pairs), dtype=np.uint64, count=count
    )
    left_sizes = np.fromiter(
        (left.cardinality for left, _ in pairs), dtype=np.int64, count=count
    )
    right_sizes = np.fromiter(
        (right.cardinality for _, right in pairs), dtype=np.int64, count=count
    )
    return int(
        lib().mpr_array_intersection_cardinality_batch(
            address(left_addresses),
            address(left_sizes),
            address(right_addresses),
            address(right_sizes),
            count,
            int(left_sizes.sum() + right_sizes.sum()),
        )
    )


def _batch_dense_intersection_cardinality(
    pairs: list[tuple[_Container, _Container]],
) -> int:
    count = len(pairs)
    left_addresses = np.fromiter(
        (address(left.data) for left, _ in pairs), dtype=np.uint64, count=count
    )
    right_addresses = np.fromiter(
        (address(right.data) for _, right in pairs), dtype=np.uint64, count=count
    )
    return int(
        lib().mpr_bitset_intersection_cardinality_batch(
            address(left_addresses),
            address(right_addresses),
            count,
        )
    )


def _container_from_lows(lows: np.ndarray) -> _Container:
    lows = np.ascontiguousarray(lows, dtype=np.uint16)
    cardinality = len(lows)
    if cardinality <= ARRAY_LIMIT:
        return _Container(lows, cardinality)
    bits = np.empty(WORDS_PER_CONTAINER, dtype=np.uint64)
    lib().mpr_array_to_bitset(address(lows), cardinality, address(bits))
    return _Container(bits, cardinality)


def _full_range_container(start: int, end: int) -> _Container:
    cardinality = end - start
    if cardinality <= ARRAY_LIMIT:
        return _array_container(np.arange(start, end, dtype=np.uint16))
    if cardinality == 1 << 16:
        return _Container(_FULL_BITSET, cardinality)
    bits = np.zeros(WORDS_PER_CONTAINER, dtype=np.uint64)
    first_word = start >> 6
    last_word = (end - 1) >> 6
    if first_word == last_word:
        width = end - start
        mask = ((1 << width) - 1) << (start & 63)
        bits[first_word] = np.uint64(mask)
    else:
        bits[first_word] = np.uint64(
            ((1 << (64 - (start & 63))) - 1) << (start & 63)
        )
        if last_word > first_word + 1:
            bits[first_word + 1 : last_word] = np.uint64(MAX_VALUE << 32 | MAX_VALUE)
        last_bits = end & 63
        bits[last_word] = (
            np.uint64(MAX_VALUE << 32 | MAX_VALUE)
            if last_bits == 0
            else np.uint64((1 << last_bits) - 1)
        )
    return _Container(bits, cardinality)


class AbstractBitMap:
    _containers: dict[int, _Container]
    _copy_on_write: bool
    _rank_cache: tuple[list[int], list[int]] | None

    def __init__(
        self, values: Iterable[int] | None = None, copy_on_write: bool = False
    ) -> None:
        self._containers = {}
        self._copy_on_write = bool(copy_on_write)
        self._rank_cache = None
        if values is None:
            return
        if isinstance(values, AbstractBitMap):
            self._containers = {
                key: container.copy() for key, container in values._containers.items()
            }
            return
        if isinstance(values, range) and values.step == 1:
            self._set_range(values.start, values.stop)
            return
        if isinstance(values, np.ndarray) and values.dtype.kind in "iu":
            flattened = values.reshape(-1)
            if flattened.size:
                if values.dtype.kind == "i" and int(flattened.min()) < 0:
                    raise OverflowError("can't convert negative value to uint32_t")
                if int(flattened.max()) > MAX_VALUE:
                    raise OverflowError("value too large to convert to uint32_t")
            converted_array = np.ascontiguousarray(flattened, dtype=np.uint32)
        else:
            converted = [_uint32(value) for value in values]
            if not converted:
                return
            converted_array = np.asarray(converted, dtype=np.uint32)
        if not converted_array.size:
            return
        unique = np.unique(converted_array)
        keys = unique >> np.uint32(16)
        boundaries = np.flatnonzero(np.diff(keys)) + 1
        for group in np.split(unique, boundaries):
            key = int(group[0] >> np.uint32(16))
            lows = (group & np.uint32(0xFFFF)).astype(np.uint16)
            self._containers[key] = _container_from_lows(lows)

    @classmethod
    def _from_containers(
        cls, containers: dict[int, _Container], copy_on_write: bool = False
    ) -> "AbstractBitMap":
        instance = cls.__new__(cls)
        instance._containers = containers
        instance._copy_on_write = copy_on_write
        instance._rank_cache = None
        return instance

    def _set_range(self, start: int, end: int) -> None:
        start = _range_bound(start)
        end = _range_bound(end)
        if end <= start:
            return
        first_key = start >> 16
        last_key = (end - 1) >> 16
        for key in range(first_key, last_key + 1):
            low_start = start & 0xFFFF if key == first_key else 0
            low_end = ((end - 1) & 0xFFFF) + 1 if key == last_key else 1 << 16
            self._containers[key] = _full_range_container(low_start, low_end)

    def _coerce(self, other: Any) -> "AbstractBitMap":
        if not isinstance(other, AbstractBitMap):
            raise TypeError(
                f"Argument 'other' has incorrect type (expected AbstractBitMap, "
                f"got {type(other).__name__})"
            )
        return other

    def _result_type(self) -> type["AbstractBitMap"]:
        return type(self)

    def _binary(self, other: Any, operation: str) -> "AbstractBitMap":
        other = self._coerce(other)
        left_keys = set(self._containers)
        right_keys = set(other._containers)
        result: dict[int, _Container] = {}
        if operation == "intersection":
            keys = left_keys & right_keys
        elif operation == "difference":
            keys = left_keys
        else:
            keys = left_keys | right_keys
        array_pairs: list[tuple[int, _Container, _Container]] = []
        dense_pairs: list[tuple[int, _Container, _Container]] = []
        for key in keys:
            left = self._containers.get(key)
            right = other._containers.get(key)
            if left is None:
                if operation in ("union", "xor"):
                    result[key] = right.copy()  # type: ignore[union-attr]
            elif right is None:
                result[key] = left.copy()
            elif not left.dense and not right.dense:
                array_pairs.append((key, left, right))
            elif left.dense and right.dense:
                dense_pairs.append((key, left, right))
            else:
                container = _container_binary(left, right, operation)
                if container is not None:
                    result[key] = container
        if len(array_pairs) >= BATCH_MIN_CONTAINERS:
            array_results = _batch_array_binary(array_pairs, operation)
        else:
            array_results = [
                (key, _container_binary(left, right, operation))
                for key, left, right in array_pairs
            ]
        if len(dense_pairs) >= BATCH_MIN_CONTAINERS:
            dense_results = _batch_dense_binary(dense_pairs, operation)
        else:
            dense_results = [
                (key, _container_binary(left, right, operation))
                for key, left, right in dense_pairs
            ]
        for key, container in array_results + dense_results:
            if container is not None:
                result[key] = container
        return self._result_type()._from_containers(result, self._copy_on_write)

    @property
    def copy_on_write(self) -> bool:
        return self._copy_on_write

    def copy(self) -> "AbstractBitMap":
        return self._result_type()._from_containers(
            {key: value.copy() for key, value in self._containers.items()},
            self._copy_on_write,
        )

    def __len__(self) -> int:
        return sum(container.cardinality for container in self._containers.values())

    def __bool__(self) -> bool:
        return bool(self._containers)

    def __iter__(self) -> Iterator[int]:
        for key in sorted(self._containers):
            container = self._containers[key]
            high = key << 16
            if container.dense:
                lows = np.empty(container.cardinality, dtype=np.uint16)
                lib().mpr_bitset_to_array(address(container.data), address(lows))
            else:
                lows = container.data
            for low in lows:
                yield high | int(low)

    def __reversed__(self) -> Iterator[int]:
        return reversed(self.to_array())

    def __contains__(self, value: Any) -> bool:
        try:
            value = _uint32(value)
        except (TypeError, OverflowError, ValueError):
            return False
        container = self._containers.get(value >> 16)
        if container is None:
            return False
        low = value & 0xFFFF
        if container.dense:
            return bool(
                int(container.data[low >> 6]) & (1 << (low & 63))
            )
        index = int(np.searchsorted(container.data, low))
        return index < container.cardinality and int(container.data[index]) == low

    def __getitem__(self, index: int | slice) -> int | "AbstractBitMap":
        if isinstance(index, slice):
            return self._result_type()(self.to_array()[index], self._copy_on_write)
        index = operator.index(index)
        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("bitmap index out of range")
        return self.select(index)

    def select(self, index: int) -> int:
        remaining = operator.index(index)
        if remaining < 0:
            remaining += len(self)
        if remaining < 0:
            raise IndexError("bitmap index out of range")
        for key in sorted(self._containers):
            container = self._containers[key]
            if remaining >= container.cardinality:
                remaining -= container.cardinality
                continue
            if container.dense:
                low = int(lib().mpr_bitset_select(address(container.data), remaining))
            else:
                low = int(container.data[remaining])
            return (key << 16) | low
        raise IndexError("bitmap index out of range")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractBitMap):
            return False
        return len(self) == len(other) and self.intersection_cardinality(other) == len(self)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __or__(self, other: Any) -> "AbstractBitMap":
        return self._binary(other, "union")

    def __and__(self, other: Any) -> "AbstractBitMap":
        return self._binary(other, "intersection")

    def __sub__(self, other: Any) -> "AbstractBitMap":
        return self._binary(other, "difference")

    def __xor__(self, other: Any) -> "AbstractBitMap":
        return self._binary(other, "xor")

    def __le__(self, other: Any) -> bool:
        return self.issubset(other)

    def __lt__(self, other: Any) -> bool:
        other = self._coerce(other)
        return len(self) < len(other) and self.issubset(other)

    def __ge__(self, other: Any) -> bool:
        return self.issuperset(other)

    def __gt__(self, other: Any) -> bool:
        other = self._coerce(other)
        return len(self) > len(other) and self.issuperset(other)

    def union(self, *others: Any) -> "AbstractBitMap":
        result = self.copy()
        for other in others:
            result = result | self._coerce(other)
        return result

    def intersection(self, *others: Any) -> "AbstractBitMap":
        result = self.copy()
        for other in others:
            result = result & self._coerce(other)
        return result

    def difference(self, *others: Any) -> "AbstractBitMap":
        result = self.copy()
        for other in others:
            result = result - self._coerce(other)
        return result

    def symmetric_difference(self, other: Any) -> "AbstractBitMap":
        return self ^ other

    def intersection_cardinality(self, other: Any) -> int:
        other = self._coerce(other)
        total = 0
        array_pairs: list[tuple[_Container, _Container]] = []
        dense_pairs: list[tuple[_Container, _Container]] = []
        for key in self._containers.keys() & other._containers.keys():
            left = self._containers[key]
            right = other._containers[key]
            if not left.dense and not right.dense:
                array_pairs.append((left, right))
            elif left.dense and right.dense:
                dense_pairs.append((left, right))
            else:
                total += _container_intersection_cardinality(left, right)
        if len(array_pairs) >= BATCH_MIN_CONTAINERS:
            total += _batch_array_intersection_cardinality(array_pairs)
        else:
            total += sum(
                _container_intersection_cardinality(left, right)
                for left, right in array_pairs
            )
        if len(dense_pairs) >= BATCH_MIN_CONTAINERS:
            total += _batch_dense_intersection_cardinality(dense_pairs)
        else:
            total += sum(
                _container_intersection_cardinality(left, right)
                for left, right in dense_pairs
            )
        return total

    def union_cardinality(self, other: Any) -> int:
        other = self._coerce(other)
        return len(self) + len(other) - self.intersection_cardinality(other)

    def difference_cardinality(self, other: Any) -> int:
        return len(self) - self.intersection_cardinality(other)

    def symmetric_difference_cardinality(self, other: Any) -> int:
        other = self._coerce(other)
        return len(self) + len(other) - 2 * self.intersection_cardinality(other)

    def intersect(self, other: Any) -> bool:
        return self.intersection_cardinality(other) != 0

    def isdisjoint(self, other: Any) -> bool:
        return not self.intersect(other)

    def issubset(self, other: Any) -> bool:
        other = self._coerce(other)
        return len(self) <= len(other) and self.intersection_cardinality(other) == len(self)

    def issuperset(self, other: Any) -> bool:
        return self._coerce(other).issubset(self)

    def jaccard_index(self, other: Any) -> float:
        other = self._coerce(other)
        intersection = self.intersection_cardinality(other)
        union = len(self) + len(other) - intersection
        return intersection / union if union else float("nan")

    def min(self) -> int:
        if not self:
            raise ValueError("Empty roaring bitmap, there is no minimum.")
        return self.select(0)

    def max(self) -> int:
        if not self:
            raise ValueError("Empty roaring bitmap, there is no maximum.")
        return self.select(len(self) - 1)

    def rank(self, value: int) -> int:
        value = _uint32(value)
        key = value >> 16
        if self._rank_cache is None:
            keys = sorted(self._containers)
            prefix = [0]
            for container_key in keys:
                prefix.append(
                    prefix[-1] + self._containers[container_key].cardinality
                )
            self._rank_cache = (keys, prefix)
        keys, prefix = self._rank_cache
        index = bisect_left(keys, key)
        total = prefix[index]
        if index == len(keys) or keys[index] != key:
            return total
        container = self._containers[key]
        low = value & 0xFFFF
        if container.dense:
            if container.rank_prefix is None:
                prefix = np.empty(WORDS_PER_CONTAINER + 1, dtype=np.uint32)
                prefix[0] = 0
                np.cumsum(np.bitwise_count(container.data), out=prefix[1:])
                prefix.flags.writeable = False
                container.rank_prefix = prefix
            word_index = low >> 6
            bit = low & 63
            mask = (
                0xFFFFFFFFFFFFFFFF
                if bit == 63
                else (1 << (bit + 1)) - 1
            )
            return (
                total
                + int(container.rank_prefix[word_index])
                + (int(container.data[word_index]) & mask).bit_count()
            )
        return total + int(np.searchsorted(container.data, low, side="right"))

    def range_cardinality(self, range_start: int, range_end: int) -> int:
        start = _range_bound(range_start)
        end = _range_bound(range_end)
        if end <= start:
            return 0
        before = self.rank(start - 1) if start else 0
        through_end = len(self) if end == MAX_VALUE + 1 else self.rank(end - 1)
        return through_end - before

    def contains_range(self, range_start: int, range_end: int) -> bool:
        start = _range_bound(range_start)
        end = _range_bound(range_end)
        return end <= start or self.range_cardinality(start, end) == end - start

    def next_set_bit(self, value: int) -> int:
        value = _uint32(value)
        index = self.rank(value - 1) if value else 0
        if index == len(self):
            raise ValueError("No value larger or equal to specified value.")
        return self.select(index)

    def iter_equal_or_larger(self, value: int) -> Iterator[int]:
        value = _uint32(value)
        index = self.rank(value - 1) if value else 0
        for item_index in range(index, len(self)):
            yield self.select(item_index)

    def flip(self, start: int, end: int) -> "AbstractBitMap":
        return self ^ self._result_type()(range(_range_bound(start), _range_bound(end)))

    def shift(self, offset: int) -> "AbstractBitMap":
        offset = operator.index(offset)
        shifted = (
            value + offset
            for value in self
            if 0 <= value + offset <= MAX_VALUE
        )
        return self._result_type()(shifted, self._copy_on_write)

    def to_array(self) -> array:
        return array("I", self)

    def get_statistics(self) -> dict[str, int]:
        array_containers = [
            value for value in self._containers.values() if not value.dense
        ]
        bitset_containers = [
            value for value in self._containers.values() if value.dense
        ]
        return {
            "n_containers": len(self._containers),
            "n_array_containers": len(array_containers),
            "n_run_containers": 0,
            "n_bitset_containers": len(bitset_containers),
            "n_values_array_containers": sum(x.cardinality for x in array_containers),
            "n_values_run_containers": 0,
            "n_values_bitset_containers": sum(x.cardinality for x in bitset_containers),
            "n_bytes_array_containers": sum(x.data.nbytes for x in array_containers),
            "n_bytes_run_containers": 0,
            "n_bytes_bitset_containers": sum(x.data.nbytes for x in bitset_containers),
            "max_value": self.max() if self else 0,
            "min_value": self.min() if self else 0,
            "sum_value": 0,
            "cardinality": len(self),
        }

    def run_optimize(self) -> bool:
        return False

    def shrink_to_fit(self) -> int:
        return 0


class BitMap(AbstractBitMap):
    __hash__ = None

    def _replace(self, other: AbstractBitMap) -> None:
        self._containers = other._containers
        self._rank_cache = None

    def __ior__(self, other: Any) -> "BitMap":
        self._replace(self | other)
        return self

    def __iand__(self, other: Any) -> "BitMap":
        self._replace(self & other)
        return self

    def __isub__(self, other: Any) -> "BitMap":
        self._replace(self - other)
        return self

    def __ixor__(self, other: Any) -> "BitMap":
        self._replace(self ^ other)
        return self

    def add(self, value: int) -> None:
        value = _uint32(value)
        if value not in self:
            self |= BitMap([value])

    def add_checked(self, value: int) -> None:
        value = _uint32(value)
        if value in self:
            raise KeyError(value)
        self.add(value)

    def update(self, *all_values: Iterable[int]) -> None:
        for values in all_values:
            self |= values if isinstance(values, AbstractBitMap) else BitMap(values)

    def intersection_update(self, *all_values: Iterable[int]) -> None:
        for values in all_values:
            self &= values if isinstance(values, AbstractBitMap) else BitMap(values)

    def difference_update(self, *others: Iterable[int]) -> None:
        for values in others:
            self -= values if isinstance(values, AbstractBitMap) else BitMap(values)

    def symmetric_difference_update(self, other: Iterable[int]) -> None:
        self ^= other if isinstance(other, AbstractBitMap) else BitMap(other)

    def discard(self, value: int) -> None:
        value = _uint32(value)
        if value in self:
            self -= BitMap([value])

    def remove(self, value: int) -> None:
        value = _uint32(value)
        if value not in self:
            raise KeyError(value)
        self.discard(value)

    def pop(self) -> int:
        if not self:
            raise KeyError("pop from an empty BitMap")
        value = self.min()
        self.discard(value)
        return value

    def clear(self) -> None:
        self._containers = {}
        self._rank_cache = None

    def overwrite(self, other: AbstractBitMap) -> None:
        other = self._coerce(other)
        self._containers = {
            key: value.copy() for key, value in other._containers.items()
        }
        self._rank_cache = None

    def add_range(self, range_start: int, range_end: int) -> None:
        self |= BitMap(range(_range_bound(range_start), _range_bound(range_end)))

    def remove_range(self, range_start: int, range_end: int) -> None:
        self -= BitMap(range(_range_bound(range_start), _range_bound(range_end)))

    def flip_inplace(self, start: int, end: int) -> None:
        self ^= BitMap(range(_range_bound(start), _range_bound(end)))


class FrozenBitMap(AbstractBitMap):
    def __hash__(self) -> int:
        return hash(tuple(self))

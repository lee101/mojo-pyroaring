from std.sys.info import simd_width_of as simdwidthof
from max.algorithm import parallelize
from std.bit import count_trailing_zeros


comptime U16Ptr = UnsafePointer[UInt16, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def popcount(value: UInt64) -> Int:
    var x = value
    x = x - ((x >> 1) & UInt64(0x5555555555555555))
    x = (x & UInt64(0x3333333333333333)) + (
        (x >> 2) & UInt64(0x3333333333333333)
    )
    x = (x + (x >> 4)) & UInt64(0x0F0F0F0F0F0F0F0F)
    return Int((x * UInt64(0x0101010101010101)) >> 56)


def array_union(a: U16Ptr, na: Int, b: U16Ptr, nb: Int, dst: U16Ptr) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    var j = 0
    var n = 0
    while i < na and j < nb:
        if a[i] < b[j]:
            dst[n] = a[i]
            i += 1
        elif b[j] < a[i]:
            dst[n] = b[j]
            j += 1
        else:
            dst[n] = a[i]
            i += 1
            j += 1
        n += 1
    while i + W <= na:
        dst.store(n, a.load[width=W](i))
        i += W
        n += W
    while i < na:
        dst[n] = a[i]
        i += 1
        n += 1
    while j + W <= nb:
        dst.store(n, b.load[width=W](j))
        j += W
        n += W
    while j < nb:
        dst[n] = b[j]
        j += 1
        n += 1
    return n


def array_intersection(
    a: U16Ptr, na: Int, b: U16Ptr, nb: Int, dst: U16Ptr
) -> Int:
    var i = 0
    var j = 0
    var n = 0
    while i < na and j < nb:
        if a[i] < b[j]:
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            dst[n] = a[i]
            i += 1
            j += 1
            n += 1
    return n


def array_difference(
    a: U16Ptr, na: Int, b: U16Ptr, nb: Int, dst: U16Ptr
) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    var j = 0
    var n = 0
    while i < na and j < nb:
        if a[i] < b[j]:
            dst[n] = a[i]
            i += 1
            n += 1
        elif b[j] < a[i]:
            j += 1
        else:
            i += 1
            j += 1
    while i + W <= na:
        dst.store(n, a.load[width=W](i))
        i += W
        n += W
    while i < na:
        dst[n] = a[i]
        i += 1
        n += 1
    return n


def array_xor(a: U16Ptr, na: Int, b: U16Ptr, nb: Int, dst: U16Ptr) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    var j = 0
    var n = 0
    while i < na and j < nb:
        if a[i] < b[j]:
            dst[n] = a[i]
            i += 1
            n += 1
        elif b[j] < a[i]:
            dst[n] = b[j]
            j += 1
            n += 1
        else:
            i += 1
            j += 1
    while i + W <= na:
        dst.store(n, a.load[width=W](i))
        i += W
        n += W
    while i < na:
        dst[n] = a[i]
        i += 1
        n += 1
    while j + W <= nb:
        dst.store(n, b.load[width=W](j))
        j += W
        n += W
    while j < nb:
        dst[n] = b[j]
        j += 1
        n += 1
    return n


def array_intersection_cardinality(a: U16Ptr, na: Int, b: U16Ptr, nb: Int) -> Int:
    var i = 0
    var j = 0
    var n = 0
    while i < na and j < nb:
        if a[i] < b[j]:
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            i += 1
            j += 1
            n += 1
    return n


def array_to_bitset(a: U16Ptr, n: Int, dst: U64Ptr):
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    while i + W <= 1024:
        dst.store(i, SIMD[DType.uint64, W](0))
        i += W
    while i < 1024:
        dst[i] = 0
        i += 1
    for i in range(n):
        var value = Int(a[i])
        dst[value >> 6] |= UInt64(1) << UInt64(value & 63)


def bitset_to_array(a: U64Ptr, dst: U16Ptr) -> Int:
    var n = 0
    for word_index in range(1024):
        var word = a[word_index]
        while word != 0:
            var bit = Int(count_trailing_zeros(word))
            dst[n] = UInt16(word_index * 64 + bit)
            n += 1
            word &= word - 1
    return n


def bitset_binary(a: U64Ptr, b: U64Ptr, dst: U64Ptr, operation: Int) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var cardinality = 0
    var i = 0
    while i + W <= 1024:
        var word: SIMD[DType.uint64, W]
        if operation == 0:
            word = a.load[width=W](i) | b.load[width=W](i)
        elif operation == 1:
            word = a.load[width=W](i) & b.load[width=W](i)
        elif operation == 2:
            word = a.load[width=W](i) & ~b.load[width=W](i)
        else:
            word = a.load[width=W](i) ^ b.load[width=W](i)
        dst.store(i, word)
        cardinality += word.reduce_bit_count()
        i += W
    while i < 1024:
        var word: UInt64
        if operation == 0:
            word = a[i] | b[i]
        elif operation == 1:
            word = a[i] & b[i]
        elif operation == 2:
            word = a[i] & ~b[i]
        else:
            word = a[i] ^ b[i]
        dst[i] = word
        cardinality += popcount(word)
        i += 1
    return cardinality


def bitset_cardinality_n(a: U64Ptr, n: Int) -> Int:
    comptime W = simdwidthof[DType.float64]()
    var cardinality = 0
    var i = 0
    while i + W <= n:
        cardinality += a.load[width=W](i).reduce_bit_count()
        i += W
    while i < n:
        cardinality += popcount(a[i])
        i += 1
    return cardinality


def bitset_cardinality(a: U64Ptr) -> Int:
    return bitset_cardinality_n(a, 1024)


def bitset_rank(a: U64Ptr, value: Int) -> Int:
    var word_index = value >> 6
    var cardinality = bitset_cardinality_n(a, word_index)
    var bit = value & 63
    var mask: UInt64
    if bit == 63:
        mask = UInt64(0xFFFFFFFFFFFFFFFF)
    else:
        mask = (UInt64(1) << UInt64(bit + 1)) - UInt64(1)
    return cardinality + popcount(a[word_index] & mask)


def array_binary_batch(
    a_addrs_addr: Int,
    a_sizes_addr: Int,
    b_addrs_addr: Int,
    b_sizes_addr: Int,
    dst_addrs_addr: Int,
    cardinalities_addr: Int,
    count: Int,
    total_values: Int,
    operation: Int,
):
    def process(index: Int) {
        imm a_addrs_addr,
        imm a_sizes_addr,
        imm b_addrs_addr,
        imm b_sizes_addr,
        imm dst_addrs_addr,
        imm cardinalities_addr,
        imm operation,
    }:
        var a_addrs = U64Ptr(unsafe_from_address=a_addrs_addr)
        var a_sizes = I64Ptr(unsafe_from_address=a_sizes_addr)
        var b_addrs = U64Ptr(unsafe_from_address=b_addrs_addr)
        var b_sizes = I64Ptr(unsafe_from_address=b_sizes_addr)
        var dst_addrs = U64Ptr(unsafe_from_address=dst_addrs_addr)
        var cardinalities = I64Ptr(unsafe_from_address=cardinalities_addr)
        var a = U16Ptr(unsafe_from_address=Int(a_addrs[index]))
        var b = U16Ptr(unsafe_from_address=Int(b_addrs[index]))
        var dst = U16Ptr(unsafe_from_address=Int(dst_addrs[index]))
        var cardinality: Int
        if operation == 0:
            cardinality = array_union(
                a, Int(a_sizes[index]), b, Int(b_sizes[index]), dst
            )
        elif operation == 1:
            cardinality = array_intersection(
                a, Int(a_sizes[index]), b, Int(b_sizes[index]), dst
            )
        elif operation == 2:
            cardinality = array_difference(
                a, Int(a_sizes[index]), b, Int(b_sizes[index]), dst
            )
        else:
            cardinality = array_xor(
                a, Int(a_sizes[index]), b, Int(b_sizes[index]), dst
            )
        cardinalities[index] = Int64(cardinality)

    if count >= 8 and total_values > 0:
        parallelize(process, count, min(count, 4))
    else:
        for index in range(count):
            process(index)


def array_intersection_cardinality_batch(
    a_addrs_addr: Int,
    a_sizes_addr: Int,
    b_addrs_addr: Int,
    b_sizes_addr: Int,
    count: Int,
    total_values: Int,
) -> Int:
    var total = 0
    for index in range(count):
        var a_addrs = U64Ptr(unsafe_from_address=a_addrs_addr)
        var a_sizes = I64Ptr(unsafe_from_address=a_sizes_addr)
        var b_addrs = U64Ptr(unsafe_from_address=b_addrs_addr)
        var b_sizes = I64Ptr(unsafe_from_address=b_sizes_addr)
        total += (
            array_intersection_cardinality(
                U16Ptr(unsafe_from_address=Int(a_addrs[index])),
                Int(a_sizes[index]),
                U16Ptr(unsafe_from_address=Int(b_addrs[index])),
                Int(b_sizes[index]),
            )
        )
    return total


def bitset_binary_batch(
    a_addrs_addr: Int,
    b_addrs_addr: Int,
    dst_addr: Int,
    cardinalities_addr: Int,
    count: Int,
    operation: Int,
):
    @parameter
    def process(index: Int):
        var a_addrs = U64Ptr(unsafe_from_address=a_addrs_addr)
        var b_addrs = U64Ptr(unsafe_from_address=b_addrs_addr)
        var dst = U64Ptr(unsafe_from_address=dst_addr)
        var cardinalities = I64Ptr(unsafe_from_address=cardinalities_addr)
        cardinalities[index] = Int64(
            bitset_binary(
                U64Ptr(unsafe_from_address=Int(a_addrs[index])),
                U64Ptr(unsafe_from_address=Int(b_addrs[index])),
                dst + index * 1024,
                operation,
            )
        )

    for index in range(count):
        process(index)


def bitset_intersection_cardinality_batch(
    a_addrs_addr: Int,
    b_addrs_addr: Int,
    count: Int,
) -> Int:
    var total = 0
    for index in range(count):
        comptime W = simdwidthof[DType.float64]()
        var a_addrs = U64Ptr(unsafe_from_address=a_addrs_addr)
        var b_addrs = U64Ptr(unsafe_from_address=b_addrs_addr)
        var a = U64Ptr(unsafe_from_address=Int(a_addrs[index]))
        var b = U64Ptr(unsafe_from_address=Int(b_addrs[index]))
        var cardinality = 0
        var word_index = 0
        while word_index + W <= 1024:
            cardinality += (
                a.load[width=W](word_index)
                & b.load[width=W](word_index)
            ).reduce_bit_count()
            word_index += W
        while word_index < 1024:
            cardinality += popcount(a[word_index] & b[word_index])
            word_index += 1
        total += cardinality
    return total


def bitset_select(a: U64Ptr, index: Int) -> Int:
    var remaining = index
    for word_index in range(1024):
        var word = a[word_index]
        var count = popcount(word)
        if remaining < count:
            for bit in range(64):
                if (word & (UInt64(1) << UInt64(bit))) != 0:
                    if remaining == 0:
                        return word_index * 64 + bit
                    remaining -= 1
        else:
            remaining -= count
    return -1


@export("mpr_array_union")
def mpr_array_union(
    a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int
) abi("C") -> Int:
    return array_union(
        U16Ptr(unsafe_from_address=a_addr),
        na,
        U16Ptr(unsafe_from_address=b_addr),
        nb,
        U16Ptr(unsafe_from_address=dst_addr),
    )


@export("mpr_array_intersection")
def mpr_array_intersection(
    a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int
) abi("C") -> Int:
    return array_intersection(
        U16Ptr(unsafe_from_address=a_addr),
        na,
        U16Ptr(unsafe_from_address=b_addr),
        nb,
        U16Ptr(unsafe_from_address=dst_addr),
    )


@export("mpr_array_difference")
def mpr_array_difference(
    a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int
) abi("C") -> Int:
    return array_difference(
        U16Ptr(unsafe_from_address=a_addr),
        na,
        U16Ptr(unsafe_from_address=b_addr),
        nb,
        U16Ptr(unsafe_from_address=dst_addr),
    )


@export("mpr_array_xor")
def mpr_array_xor(
    a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int
) abi("C") -> Int:
    return array_xor(
        U16Ptr(unsafe_from_address=a_addr),
        na,
        U16Ptr(unsafe_from_address=b_addr),
        nb,
        U16Ptr(unsafe_from_address=dst_addr),
    )


@export("mpr_array_intersection_cardinality")
def mpr_array_intersection_cardinality(
    a_addr: Int, na: Int, b_addr: Int, nb: Int
) abi("C") -> Int:
    return array_intersection_cardinality(
        U16Ptr(unsafe_from_address=a_addr),
        na,
        U16Ptr(unsafe_from_address=b_addr),
        nb,
    )


@export("mpr_array_binary_batch")
def mpr_array_binary_batch(
    a_addrs_addr: Int,
    a_sizes_addr: Int,
    b_addrs_addr: Int,
    b_sizes_addr: Int,
    dst_addrs_addr: Int,
    cardinalities_addr: Int,
    count: Int,
    total_values: Int,
    operation: Int,
) abi("C"):
    array_binary_batch(
        a_addrs_addr,
        a_sizes_addr,
        b_addrs_addr,
        b_sizes_addr,
        dst_addrs_addr,
        cardinalities_addr,
        count,
        total_values,
        operation,
    )


@export("mpr_array_intersection_cardinality_batch")
def mpr_array_intersection_cardinality_batch(
    a_addrs_addr: Int,
    a_sizes_addr: Int,
    b_addrs_addr: Int,
    b_sizes_addr: Int,
    count: Int,
    total_values: Int,
) abi("C") -> Int:
    return array_intersection_cardinality_batch(
        a_addrs_addr,
        a_sizes_addr,
        b_addrs_addr,
        b_sizes_addr,
        count,
        total_values,
    )


@export("mpr_array_to_bitset")
def mpr_array_to_bitset(a_addr: Int, n: Int, dst_addr: Int) abi("C"):
    array_to_bitset(
        U16Ptr(unsafe_from_address=a_addr),
        n,
        U64Ptr(unsafe_from_address=dst_addr),
    )


@export("mpr_bitset_to_array")
def mpr_bitset_to_array(a_addr: Int, dst_addr: Int) abi("C") -> Int:
    return bitset_to_array(
        U64Ptr(unsafe_from_address=a_addr),
        U16Ptr(unsafe_from_address=dst_addr),
    )


@export("mpr_bitset_binary")
def mpr_bitset_binary(
    a_addr: Int, b_addr: Int, dst_addr: Int, operation: Int
) abi("C") -> Int:
    return bitset_binary(
        U64Ptr(unsafe_from_address=a_addr),
        U64Ptr(unsafe_from_address=b_addr),
        U64Ptr(unsafe_from_address=dst_addr),
        operation,
    )


@export("mpr_bitset_binary_batch")
def mpr_bitset_binary_batch(
    a_addrs_addr: Int,
    b_addrs_addr: Int,
    dst_addr: Int,
    cardinalities_addr: Int,
    count: Int,
    operation: Int,
) abi("C"):
    bitset_binary_batch(
        a_addrs_addr,
        b_addrs_addr,
        dst_addr,
        cardinalities_addr,
        count,
        operation,
    )


@export("mpr_bitset_intersection_cardinality_batch")
def mpr_bitset_intersection_cardinality_batch(
    a_addrs_addr: Int,
    b_addrs_addr: Int,
    count: Int,
) abi("C") -> Int:
    return bitset_intersection_cardinality_batch(
        a_addrs_addr,
        b_addrs_addr,
        count,
    )


@export("mpr_bitset_cardinality")
def mpr_bitset_cardinality(a_addr: Int) abi("C") -> Int:
    return bitset_cardinality(U64Ptr(unsafe_from_address=a_addr))


@export("mpr_bitset_rank")
def mpr_bitset_rank(a_addr: Int, value: Int) abi("C") -> Int:
    return bitset_rank(U64Ptr(unsafe_from_address=a_addr), value)


@export("mpr_bitset_select")
def mpr_bitset_select(a_addr: Int, index: Int) abi("C") -> Int:
    return bitset_select(U64Ptr(unsafe_from_address=a_addr), index)

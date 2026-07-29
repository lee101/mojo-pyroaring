from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "bitmap.mojo")
LIBRARY = os.environ.get("MOJO_PYROARING_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyroaring.so"
)

I = ctypes.c_int64
_SIGNATURES = {
    "mpr_array_union": ([I, I, I, I, I], I),
    "mpr_array_intersection": ([I, I, I, I, I], I),
    "mpr_array_difference": ([I, I, I, I, I], I),
    "mpr_array_xor": ([I, I, I, I, I], I),
    "mpr_array_intersection_cardinality": ([I, I, I, I], I),
    "mpr_array_binary_batch": ([I, I, I, I, I, I, I, I, I], None),
    "mpr_array_intersection_cardinality_batch": (
        [I, I, I, I, I, I],
        I,
    ),
    "mpr_array_to_bitset": ([I, I, I], None),
    "mpr_bitset_to_array": ([I, I], I),
    "mpr_bitset_binary": ([I, I, I, I], I),
    "mpr_bitset_binary_batch": ([I, I, I, I, I, I], None),
    "mpr_bitset_intersection_cardinality_batch": ([I, I, I], I),
    "mpr_bitset_cardinality": ([I], I),
    "mpr_bitset_rank": ([I, I], I),
    "mpr_bitset_select": ([I, I], I),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_PYROARING_MOJO")
    if override:
        return override.split()
    executable = shutil.which("mojo")
    if executable:
        return [executable]
    pixi = shutil.which("pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if pixi and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; run `pixi run build`")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_PYROARING_LIB") and os.path.exists(LIBRARY) and not force:
        return LIBRARY
    if not os.path.exists(SOURCE):
        if os.path.exists(LIBRARY):
            return LIBRARY
        raise BuildError(f"no Mojo source at {SOURCE} and no library at {LIBRARY}")
    if (
        not force
        and os.path.exists(LIBRARY)
        and os.path.getmtime(LIBRARY) >= os.path.getmtime(SOURCE)
    ):
        return LIBRARY
    os.makedirs(os.path.dirname(LIBRARY), exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        SOURCE,
        "-o",
        LIBRARY,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not os.path.exists(LIBRARY):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIBRARY


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


def address(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if array.size == 0:
        raise ValueError("native buffers must not be empty")
    if not array.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    if not array.flags.aligned:
        raise ValueError("native buffers must be naturally aligned")
    pointer = int(array.ctypes.data)
    if pointer == 0:
        raise ValueError("native buffers must have a non-null address")
    return pointer

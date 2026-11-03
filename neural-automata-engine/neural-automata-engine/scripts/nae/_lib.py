from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterator

_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_INT_P = ctypes.POINTER(ctypes.c_int)


class LibraryNotFound(RuntimeError):
    pass


def _candidates() -> Iterator[Path]:
    override = os.environ.get("NAE_LIB")
    if override:
        yield Path(override)
    root = Path(__file__).resolve().parents[2]
    for name in ("libnae.so", "libnae.dylib"):
        yield root / "build" / name


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    u32, i32 = ctypes.c_uint32, ctypes.c_int

    lib.nae_capi_create_life.restype = ctypes.c_void_p
    lib.nae_capi_create_life.argtypes = [u32, u32, i32, i32, u32, u32, u32, _INT_P]

    lib.nae_capi_create_neural.restype = ctypes.c_void_p
    lib.nae_capi_create_neural.argtypes = [
        u32, u32, u32, i32, i32, u32, i32, u32,
        ctypes.c_float, ctypes.c_float,
        _FLOAT_P, _FLOAT_P, _FLOAT_P, _FLOAT_P,
        _INT_P,
    ]

    lib.nae_capi_destroy.restype = None
    lib.nae_capi_destroy.argtypes = [ctypes.c_void_p]

    lib.nae_capi_step.restype = None
    lib.nae_capi_step.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

    lib.nae_capi_generation.restype = ctypes.c_uint64
    lib.nae_capi_generation.argtypes = [ctypes.c_void_p]

    for name in ("width", "height", "channels"):
        fn = getattr(lib, f"nae_capi_{name}")
        fn.restype = u32
        fn.argtypes = [ctypes.c_void_p]

    lib.nae_capi_read.restype = i32
    lib.nae_capi_read.argtypes = [ctypes.c_void_p, _FLOAT_P, ctypes.c_size_t]

    lib.nae_capi_write.restype = i32
    lib.nae_capi_write.argtypes = [ctypes.c_void_p, _FLOAT_P, ctypes.c_size_t]

    lib.nae_capi_seed_random.restype = None
    lib.nae_capi_seed_random.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_float]

    lib.nae_capi_seed_point.restype = i32
    lib.nae_capi_seed_point.argtypes = [ctypes.c_void_p, u32, u32, ctypes.c_float]

    lib.nae_capi_status_name.restype = ctypes.c_char_p
    lib.nae_capi_status_name.argtypes = [i32]
    return lib


@lru_cache(maxsize=1)
def load() -> ctypes.CDLL:
    for path in _candidates():
        if path.is_file():
            return _bind(ctypes.CDLL(str(path)))
    raise LibraryNotFound("libnae not found, run `make lib` or set NAE_LIB")

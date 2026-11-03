from __future__ import annotations

import ctypes
from array import array
from typing import Iterable, Optional, Sequence, Tuple, Union

from . import _lib
from .weights import NeuralWeights

BOUNDARIES = {"wrap": 0, "zero": 1, "clamp": 2}
KERNELS = {"moore": 0, "von_neumann": 1}


class EngineError(RuntimeError):
    pass


def _choice(table: dict, key: str, label: str) -> int:
    try:
        return table[key]
    except KeyError:
        options = ", ".join(sorted(table))
        raise ValueError(f"unknown {label} {key!r}, expected one of: {options}") from None


def _mask(counts: Iterable[int]) -> int:
    mask = 0
    for n in counts:
        if not 0 <= n < 32:
            raise ValueError(f"neighbour count out of range: {n}")
        mask |= 1 << n
    return mask


def _fptr(data: array) -> "ctypes.Array[ctypes.c_float]":
    return (ctypes.c_float * len(data)).from_buffer(data)


def _status_name(lib: ctypes.CDLL, code: int) -> str:
    return lib.nae_capi_status_name(code).decode("ascii")


class Engine:
    def __init__(self, handle: int, width: int, height: int, channels: int) -> None:
        self._lib = _lib.load()
        self._handle: Optional[int] = handle
        self.width = width
        self.height = height
        self.channels = channels

    @classmethod
    def life(
        cls,
        width: int,
        height: int,
        *,
        boundary: str = "wrap",
        kernel: str = "moore",
        radius: int = 1,
        birth: Iterable[int] = (3,),
        survive: Iterable[int] = (2, 3),
    ) -> "Engine":
        lib = _lib.load()
        status = ctypes.c_int(0)
        handle = lib.nae_capi_create_life(
            width,
            height,
            _choice(BOUNDARIES, boundary, "boundary"),
            _choice(KERNELS, kernel, "kernel"),
            radius,
            _mask(birth),
            _mask(survive),
            ctypes.byref(status),
        )
        if not handle:
            raise EngineError(_status_name(lib, status.value))
        return cls(handle, width, height, 1)

    @classmethod
    def neural(
        cls,
        width: int,
        height: int,
        weights: NeuralWeights,
        *,
        boundary: str = "wrap",
        kernel: str = "moore",
        radius: int = 1,
        include_center: bool = False,
    ) -> "Engine":
        lib = _lib.load()
        status = ctypes.c_int(0)
        handle = lib.nae_capi_create_neural(
            width,
            height,
            weights.channels,
            _choice(BOUNDARIES, boundary, "boundary"),
            _choice(KERNELS, kernel, "kernel"),
            radius,
            int(include_center),
            weights.hidden,
            weights.step_size,
            weights.state_clip,
            _fptr(weights.w1),
            _fptr(weights.b1),
            _fptr(weights.w2),
            _fptr(weights.b2),
            ctypes.byref(status),
        )
        if not handle:
            raise EngineError(_status_name(lib, status.value))
        return cls(handle, width, height, weights.channels)

    def _live(self) -> int:
        if self._handle is None:
            raise EngineError("engine is closed")
        return self._handle

    def _check(self, code: int) -> None:
        if code != 0:
            raise EngineError(_status_name(self._lib, code))

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    @property
    def generation(self) -> int:
        return int(self._lib.nae_capi_generation(self._live()))

    def step(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._lib.nae_capi_step(self._live(), count)

    def read(self) -> array:
        out = array("f", [0.0]) * (self.width * self.height * self.channels)
        self._check(self._lib.nae_capi_read(self._live(), _fptr(out), len(out)))
        return out

    def write(self, data: Union[array, Sequence[float]]) -> None:
        buf = data if isinstance(data, array) and data.typecode == "f" else array("f", data)
        self._check(self._lib.nae_capi_write(self._live(), _fptr(buf), len(buf)))

    def seed_random(self, seed: int = 1, density: float = 0.25) -> None:
        self._lib.nae_capi_seed_random(self._live(), seed, density)

    def seed_point(self, x: int, y: int, value: float = 1.0) -> None:
        self._check(self._lib.nae_capi_seed_point(self._live(), x, y, value))

    def close(self) -> None:
        if self._handle is not None:
            self._lib.nae_capi_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

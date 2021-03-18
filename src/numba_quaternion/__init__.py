from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numba import jit

try:
    from coloredlogs import ColoredFormatter as Formatter
except ImportError:
    from logging import Formatter

__version__ = '0.1.0'

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
handler.setFormatter(Formatter('%(name)s %(levelname)s %(message)s'))
try:
    level = os.environ.get('COSCONLOGLEVEL', logging.WARNING)
    logger.setLevel(level=level)
except ValueError:
    logger.setLevel(level=logging.WARNING)
    logger.error(f'Unknown COSCONLOGLEVEL {level}, set to default WARNING.')


@jit(nopython=True, nogil=True)
def lastcol_quat_to_canonical(quat: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    """Convert from real-part-in-last-column to real-part-in-first-column"""
    return np.ascontiguousarray(quat[..., np.array([3, 0, 1, 2])])


@jit(nopython=True, nogil=True)
def canonical_quat_to_lastcol(quat: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    """Convert from real-part-in-first-column to real-part-in-last-column"""
    return np.ascontiguousarray(quat[..., np.array([1, 2, 3, 0])])


@jit(nopython=True, nogil=True)
def mul(p: np.ndarray[np.complex_], q: np.ndarray[np.complex_]) -> np.ndarray[np.complex_]:
    """Perform quarternion multiplication using complex multiplication"""
    A = p[..., 0]
    B = p[..., 1]
    C = q[..., 0]
    D = q[..., 1]
    real_i_part = A * C - B * np.conjugate(D)
    jk_part = B * np.conjugate(C) + A * D
    return np.column_stack((real_i_part, jk_part))


@jit(nopython=True, nogil=True)
def conjugate(p: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    res = np.empty_like(p)
    res[..., 0] = p[..., 0]
    res[..., 1:] = -p[..., 1:]
    return res


@jit(nopython=True, nogil=True)
def norm(p: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    return np.square(p).sum(axis=-1)


@jit(nopython=True, nogil=True)
def abs(p: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    return np.sqrt(norm(p))


@jit(nopython=True, nogil=True)
def inverse(p: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    return conjugate(p) / norm(p).reshape(*p.shape[:-1], 1)


def real_to_complex(array: np.ndarray[np.float_]) -> np.ndarray[np.complex_]:
    dtype = array.dtype
    if dtype == np.float64:
        return array.view(dtype=np.complex128)
    elif dtype == np.float32:
        return array.view(dtype=np.complex64)
    else:
        raise ValueError(f'Unsupported dtype {dtype}.')


def complex_to_real(array_complex: np.ndarray[np.complex_]) -> np.ndarray[np.float_]:
    dtype = array_complex.dtype
    if dtype == np.complex128:
        return array_complex.view(dtype=np.float64)
    elif dtype == np.complex64:
        return array_complex.view(dtype=np.float32)
    else:
        raise ValueError(f'Unsupported dtype {dtype}.')


@dataclass
class Quaternion:
    array: np.ndarray[np.float_]

    def __post_init__(self):
        assert self.array.shape[-1] == 4

    @cached_property
    def array_complex(self) -> np.ndarray[np.complex_]:
        return real_to_complex(self.array)

    def __add__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self.array + other.array)

    def __iadd__(self, other: Quaternion):
        self.array += other.array

    def __mul__(self, other: Quaternion) -> Quaternion:
        return Quaternion.from_array_complex(mul(self.array_complex, other.array_complex))

    def __imul__(self, other: Quaternion):
        self.array = complex_to_real(mul(self.array_complex, other.array_complex))

    @property
    def conjugate(self):
        return Quaternion(conjugate(self.array))

    @property
    def norm(self):
        return Quaternion(norm(self.array))

    @property
    def abs(self):
        return Quaternion(abs(self.array))

    @property
    def inverse(self):
        return Quaternion(inverse(self.array))

    @classmethod
    def from_array_complex(cls, array_complex: np.ndarray[np.complex_]) -> Quaternion:
        return cls(complex_to_real(array_complex))

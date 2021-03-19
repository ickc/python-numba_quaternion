from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numba import jit, generated_jit

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
def float64_to_complex128(array):
    return array.view(np.complex128)


@jit(nopython=True, nogil=True)
def float32_to_complex64(array):
    return array.view(np.complex64)


@jit(nopython=True, nogil=True)
def complex128_to_float64(array):
    return array.view(np.float64)


@jit(nopython=True, nogil=True)
def complex64_to_float32(array):
    return array.view(np.float32)


@generated_jit(nopython=True, nogil=True)
def float_to_complex(array):
    dtype = str(array.dtype)
    if dtype == 'float64':
        return float64_to_complex128
    elif dtype == 'float32':
        return float32_to_complex64


@generated_jit(nopython=True, nogil=True)
def complex_to_float(array):
    dtype = str(array.dtype)
    if dtype == 'complex128':
        return complex128_to_float64
    elif dtype == 'complex64':
        return complex64_to_float32


@jit(nopython=True, nogil=True)
def mul(p: np.ndarray[np.complex_], q: np.ndarray[np.complex_]) -> np.ndarray[np.complex_]:
    """Perform quarternion multiplication using complex multiplication"""
    A = p[..., 0]
    B = p[..., 1]
    C = q[..., 0]
    D = q[..., 1]
    real_i_part = A * C - B * np.conjugate(D)
    jk_part = B * np.conjugate(C) + A * D
    return np.stack((real_i_part, jk_part), -1)


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


@jit(nopython=True, nogil=True)
def rotate(p: np.ndarray[np.complex_], v: np.ndarray[np.complex_]) -> np.ndarray[np.complex_]:
    p_inv = float_to_complex(inverse(complex_to_float(p)))
    return mul(mul(p, v), p_inv)


@jit(nopython=True, nogil=True)
def quat_to_azimuthal_equidistant_projection_with_orientation(quats: np.ndarray[np.complex_]) -> np.ndarray[np.float_]:
    """Convert from detector pointing to Azimuthal equidistant projection with orientation.

    Returned array is in radian,
    has the last dimension with 3 elements,
    1st as the horizontal angular position,
    2nd as the vertical angular position,
    3rd as the orientation in angle.
    """
    x_axis = np.array([1.j, 0.], dtype=quats.dtype)
    z_axis = np.array([0., 1.j], dtype=quats.dtype)
    # rotation from boresight
    r = rotate(quats, z_axis)
    ds = np.arccos(r[:, 1].imag)
    angles = np.arctan2(r[:, 1].real, r[:, 0].imag)

    orients = rotate(quats, x_axis)
    return np.stack(
        (
            ds * np.cos(angles),
            ds * np.sin(angles),
            np.arctan2(orients[:, 1].real, orients[:, 0].imag)
        ),
        -1,
    )


@dataclass
class Quaternion:
    array: np.ndarray[np.float_]

    def __post_init__(self):
        assert self.array.shape[-1] == 4

    @cached_property
    def array_complex(self) -> np.ndarray[np.complex_]:
        return float_to_complex(self.array)

    def __add__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self.array + other.array)

    def __iadd__(self, other: Quaternion):
        self.array += other.array

    def __mul__(self, other: Quaternion) -> Quaternion:
        return Quaternion.from_array_complex(mul(self.array_complex, other.array_complex))

    def __imul__(self, other: Quaternion):
        self.array = complex_to_float(mul(self.array_complex, other.array_complex))

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

    def rotate(self, other: Quaternion) -> Quaternion:
        return Quaternion.from_array_complex(rotate(self.array_complex, other.array_complex))

    @cached_property
    def azimuthal_equidistant_projection_with_orientation(self) -> np.ndarray[np.float_]:
        """Convert from detector pointing to Azimuthal equidistant projection with orientation.

        Returned array is in radian,
        has the last dimension with 3 elements,
        1st as the horizontal angular position,
        2nd as the vertical angular position,
        3rd as the orientation in angle.
        """
        return quat_to_azimuthal_equidistant_projection_with_orientation(self.array_complex)

    @classmethod
    def from_array_complex(cls, array_complex: np.ndarray[np.complex_]) -> Quaternion:
        return cls(complex_to_float(array_complex))

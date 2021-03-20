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


@jit(nopython=True, nogil=True, cache=True)
def lastcol_quat_to_canonical(quat: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    """Convert from real-part-in-last-column to real-part-in-first-column"""
    return np.ascontiguousarray(quat[..., np.array([3, 0, 1, 2])])


@jit(nopython=True, nogil=True, cache=True)
def canonical_quat_to_lastcol(quat: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    """Convert from real-part-in-first-column to real-part-in-last-column"""
    return np.ascontiguousarray(quat[..., np.array([1, 2, 3, 0])])


@jit(nopython=True, nogil=True, cache=True)
def float64_to_complex128(array: np.ndarray[np.float64]) -> np.ndarray[np.complex128]:
    return array.view(np.complex128)


@jit(nopython=True, nogil=True, cache=True)
def float32_to_complex64(array: np.ndarray[np.float32]) -> np.ndarray[np.complex64]:
    return array.view(np.complex64)


@jit(nopython=True, nogil=True, cache=True)
def complex128_to_float64(array: np.ndarray[np.complex128]) -> np.ndarray[np.float64]:
    return array.view(np.float64)


@jit(nopython=True, nogil=True, cache=True)
def complex64_to_float32(array: np.ndarray[np.complex64]) -> np.ndarray[np.float32]:
    return array.view(np.float32)


@generated_jit(nopython=True, nogil=True, cache=True)
def float_to_complex(array: np.ndarray[np.float_]) -> np.ndarray[np.complex_]:
    dtype = str(array.dtype)
    if dtype == 'float64':
        return float64_to_complex128
    elif dtype == 'float32':
        return float32_to_complex64


@generated_jit(nopython=True, nogil=True, cache=True)
def complex_to_float(array: np.ndarray[np.complex_]) -> np.ndarray[np.float_]:
    dtype = str(array.dtype)
    if dtype == 'complex128':
        return complex128_to_float64
    elif dtype == 'complex64':
        return complex64_to_float32


@jit(nopython=True, nogil=True, cache=True)
def mul(p: np.ndarray[np.complex_], q: np.ndarray[np.complex_]) -> np.ndarray[np.complex_]:
    """Perform quarternion multiplication using complex multiplication"""
    A = p[..., 0]
    B = p[..., 1]
    C = q[..., 0]
    D = q[..., 1]
    real_i_part = A * C - B * np.conjugate(D)
    jk_part = B * np.conjugate(C) + A * D
    return np.stack((real_i_part, jk_part), -1)


@jit(nopython=True, nogil=True, cache=True)
def matmul(p: np.ndarray[np.complex_], q: np.ndarray[np.complex_]) -> np.ndarray[np.complex_]:
    """Perform quarternion matrix multiplication using complex matrix multiplication"""
    A = np.ascontiguousarray(p[..., 0])
    B = np.ascontiguousarray(p[..., 1])
    C = np.ascontiguousarray(q[..., 0])
    D = np.ascontiguousarray(q[..., 1])
    real_i_part = A @ C - B @ np.conjugate(D)
    jk_part = B @ np.conjugate(C) + A @ D
    return np.stack((real_i_part, jk_part), -1)


@jit(nopython=True, nogil=True, cache=True)
def quat_to_rotation_matrix(quats: np.ndarray[np.complex_]) -> np.ndarray[np.float_]:
    """Convert quaternion to rotation matrix.
    """
    I = np.array(
        (
            [1.j, 0.],
            [0. , 1.],
            [0., 1.j],
        ),
        dtype=quats.dtype,
    )
    res = rotate_2d(quats, I)
    return np.stack(
        (
            complex_to_float(res[0])[..., 1:],
            complex_to_float(res[1])[..., 1:],
            complex_to_float(res[2])[..., 1:],
        ),
        -1,
    )


@jit(nopython=True, nogil=True, cache=True)
def rotation_matrix_to_quat(m: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    """Convert rotation matrix to quaternion.

    See https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    """
    Q_xx = m[..., 0, 0]
    Q_xy = m[..., 0, 1]
    Q_xz = m[..., 0, 2]
    Q_yx = m[..., 1, 0]
    Q_yy = m[..., 1, 1]
    Q_yz = m[..., 1, 2]
    Q_zx = m[..., 2, 0]
    Q_zy = m[..., 2, 1]
    Q_zz = m[..., 2, 2]
    w =  0.5                         * np.sqrt(1. + Q_xx + Q_yy + Q_zz)
    x = (0.5 * np.sign(Q_zy - Q_yz)) * np.sqrt(1. + Q_xx - Q_yy - Q_zz)
    y = (0.5 * np.sign(Q_xz - Q_zx)) * np.sqrt(1. - Q_xx + Q_yy - Q_zz)
    z = (0.5 * np.sign(Q_yx - Q_xy)) * np.sqrt(1. - Q_xx - Q_yy + Q_zz)
    return np.stack((w, x, y, z), -1)

@jit(nopython=True, nogil=True, cache=True)
def conjugate(p: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    res = np.empty_like(p)
    res[..., 0] = p[..., 0]
    res[..., 1:] = -p[..., 1:]
    return res


@jit(nopython=True, nogil=True, cache=True)
def norm(p: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    return np.square(p).sum(axis=-1)


@jit(nopython=True, nogil=True, cache=True)
def abs(p: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    return np.sqrt(norm(p))


@jit(nopython=True, nogil=True, cache=True)
def inverse(p: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    return conjugate(p) / norm(p).reshape(*p.shape[:-1], 1)


@jit(nopython=True, nogil=True, cache=True)
def rotate(p: np.ndarray[np.complex_], v: np.ndarray[np.complex_]) -> np.ndarray[np.complex_]:
    """Rotate v by p respecting Numpy broadcasting rule."""
    p_inv = float_to_complex(inverse(complex_to_float(p)))
    return mul(mul(p, v), p_inv)


@jit(nopython=True, nogil=True, cache=True)
def rotate_2d(p: np.ndarray[np.complex_], v: np.ndarray[np.complex_]) -> List[np.ndarray[np.complex_]]:
    """Rotate each row of v by p and stack at an axis.

    :param v: 2d-array
    """
    p_inv = float_to_complex(inverse(complex_to_float(p)))
    return [mul(mul(p, row), p_inv) for row in v]


@jit(nopython=True, nogil=True, cache=True)
def quat_to_azimuthal_equidistant_projection_polar_with_orientation(quats: np.ndarray[np.complex_]) -> np.ndarray[np.float_]:
    """Convert from detector pointing to Azimuthal equidistant projection in polar coordinate with orientation.

    Returned array is in radian,
    has the last dimension with 3 elements,
    1st as the angular distance to North pole,
    2nd as the azimuth,
    3rd as the orientation in angle.
    """
    xz_axes = np.array(
        (
            [1.j, 0.],
            [0., 1.j],
        ),
        dtype=quats.dtype,
    )
    # rotation from boresight
    orients, r = rotate_2d(quats, xz_axes)
    ds = np.arccos(r[:, 1].imag)
    angles = np.arctan2(r[:, 1].real, r[:, 0].imag)

    return np.stack(
        (
            ds,
            angles,
            np.arctan2(orients[:, 1].real, orients[:, 0].imag)
        ),
        -1,
    )


@jit(nopython=True, nogil=True, cache=True)
def quat_to_azimuthal_equidistant_projection_with_orientation(quats: np.ndarray[np.complex_]) -> np.ndarray[np.float_]:
    """Convert from detector pointing to Azimuthal equidistant projection in cartesian coordinate with orientation.

    Returned array is in radian,
    has the last dimension with 3 elements,
    1st as the horizontal angular position,
    2nd as the vertical angular position,
    3rd as the orientation in angle.
    """
    xz_axes = np.array(
        (
            [1.j, 0.],
            [0., 1.j],
        ),
        dtype=quats.dtype,
    )
    # rotation from boresight
    orients, r = rotate_2d(quats, xz_axes)
    ds = np.arccos(r[:, 1].imag)
    angles = np.arctan2(r[:, 1].real, r[:, 0].imag)

    return np.stack(
        (
            ds * np.cos(angles),
            ds * np.sin(angles),
            np.arctan2(orients[:, 1].real, orients[:, 0].imag)
        ),
        -1,
    )


@jit(nopython=True, nogil=True, cache=True)
def azimuthal_equidistant_projection_polar_with_orientation_to_rotation_matrix(array: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    """Convert Azimuthal equidistant projection in polar coordinate with orientation to detector pointing.

    Input array is in radian,
    has the last dimension with 3 elements,
    1st as the angular distance to North pole,
    2nd as the azimuth,
    3rd as the orientation in angle.
    """
    theta = array[..., 0]
    phi = array[..., 1]
    alpha = array[..., 2]

    sin_theta = np.sin(theta)
    Rz = np.stack(
        (
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            np.cos(theta),
        ),
        -1,
    )
    # t is cot(beta)
    t = -np.tan(theta) * np.cos(phi - alpha)
    sin_beta = np.power(1. + np.square(t), -0.5)
    Rx = np.stack(
        (
            sin_beta * np.cos(alpha),
            sin_beta * np.sin(alpha),
            t * sin_beta,
        ),
        -1,
    )

    return np.stack((Rx, np.cross(Rz, Rx), Rz), -1)


@dataclass
class Quaternion:
    array: np.ndarray[np.float_]

    def __post_init__(self):
        assert self.array.shape[-1] == 4

    def clear_cache(self):
        try:
            del self.array_complex
        except AttributeError:
            pass
        try:
            del self.azimuthal_equidistant_projection_polar_with_orientation
        except AttributeError:
            pass
        try:
            del self.azimuthal_equidistant_projection_with_orientation
        except AttributeError:
            pass
        try:
            del self.to_rotation_matrix
        except AttributeError:
            pass

    @cached_property
    def array_complex(self) -> np.ndarray[np.complex_]:
        return float_to_complex(self.array)

    def __add__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self.array + other.array)

    def __iadd__(self, other: Quaternion):
        self.clear_cache()
        self.array += other.array

    def __mul__(self, other: Quaternion) -> Quaternion:
        return Quaternion.from_array_complex(mul(self.array_complex, other.array_complex))

    def __imul__(self, other: Quaternion):
        self.clear_cache()
        self.array = complex_to_float(mul(self.array_complex, other.array_complex))

    def __matmul__(self, other: Quaternion) -> Quaternion:
        return Quaternion.from_array_complex(matmul(self.array_complex, other.array_complex))

    def __imatmul__(self, other: Quaternion) -> Quaternion:
        self.clear_cache()
        self.array = complex_to_float(matmul(self.array_complex, other.array_complex))

    @property
    def conjugate(self) -> Quaternion:
        return Quaternion(conjugate(self.array))

    @property
    def norm(self) -> np.ndarray[np.float_]:
        return norm(self.array)

    @property
    def abs(self) -> np.ndarray[np.float_]:
        return abs(self.array)

    @property
    def normalize(self) -> Quaternion:
        return Quaternion(self.array / self.abs[..., np.newaxis])

    @property
    def inverse(self) -> Quaternion:
        return Quaternion(inverse(self.array))

    def rotate(self, other: Quaternion) -> Quaternion:
        return Quaternion.from_array_complex(rotate(self.array_complex, other.array_complex))

    @cached_property
    def azimuthal_equidistant_projection_polar_with_orientation(self) -> np.ndarray[np.float_]:
        """Convert from detector pointing to Azimuthal equidistant projection in polar coordinate with orientation.

        Returned array is in radian,
        has the last dimension with 3 elements,
        1st as the angular distance to North pole,
        2nd as the azimuth,
        3rd as the orientation in angle.
        """
        return quat_to_azimuthal_equidistant_projection_polar_with_orientation(self.array_complex)

    @cached_property
    def azimuthal_equidistant_projection_with_orientation(self) -> np.ndarray[np.float_]:
        """Convert from detector pointing to Azimuthal equidistant projection in cartesian coordinate with orientation.

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

    @classmethod
    def from_rotation_matrix(cls, array: np.ndarray[np.float_]) -> Quaternion:
        return cls(rotation_matrix_to_quat(array))

    @classmethod
    def from_azimuthal_equidistant_projection_polar_with_orientation(cls, array: np.ndarray[np.float_]) -> Quaternion:
        m = azimuthal_equidistant_projection_polar_with_orientation_to_rotation_matrix(array)
        return cls.from_rotation_matrix(m)

    @cached_property
    def to_rotation_matrix(self) -> np.ndarray[np.float_]:
        return quat_to_rotation_matrix(self.array_complex)

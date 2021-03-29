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

__version__ = '0.2.0'

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
def rotation_matrix_to_quat(m: np.ndarray[np.float_]) -> np.ndarray[np.complex_]:
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
    wx = 0.5                          * np.sqrt(1. + Q_xx + Q_yy + Q_zz) + \
        (0.5j * np.sign(Q_zy - Q_yz)) * np.sqrt(1. + Q_xx - Q_yy - Q_zz)
    yz = (0.5 * np.sign(Q_xz - Q_zx)) * np.sqrt(1. - Q_xx + Q_yy - Q_zz) + \
        (0.5j * np.sign(Q_yx - Q_xy)) * np.sqrt(1. - Q_xx - Q_yy + Q_zz)
    return np.stack((wx, yz), -1)

@jit(nopython=True, nogil=True, cache=True)
def conjugate(p: np.ndarray[np.complex_]) -> np.ndarray[np.complex_]:
    res = np.empty_like(p)
    res[..., 0] = np.conjugate(p[..., 0])
    res[..., 1] = -p[..., 1]
    return res


@jit(nopython=True, nogil=True, cache=True)
def norm(p: np.ndarray[np.complex_]) -> np.ndarray[np.float_]:
    return (p * np.conjugate(p)).sum(axis=-1)


@jit(nopython=True, nogil=True, cache=True)
def abs(p: np.ndarray[np.complex_]) -> np.ndarray[np.float_]:
    return np.sqrt(norm(p))


@jit(nopython=True, nogil=True, cache=True)
def inverse(p: np.ndarray[np.complex_]) -> np.ndarray[np.complex_]:
    return conjugate(p) / norm(p).reshape(*p.shape[:-1], 1)


@jit(nopython=True, nogil=True, cache=True)
def rotate(p: np.ndarray[np.complex_], v: np.ndarray[np.complex_]) -> np.ndarray[np.complex_]:
    """Rotate v by p respecting Numpy broadcasting rule."""
    p_inv = inverse(p)
    return mul(mul(p, v), p_inv)


@jit(nopython=True, nogil=True, cache=True)
def rotate_2d(p: np.ndarray[np.complex_], v: np.ndarray[np.complex_]) -> List[np.ndarray[np.complex_]]:
    """Rotate each row of v by p and stack at an axis.

    :param v: 2d-array
    """
    p_inv = inverse(p)
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


@jit(nopython=True, nogil=True, cache=True)
def dist_spherical(p: np.ndarray[np.complex_], q: np.ndarray[np.complex_]) -> float:
    """Great circle distance between 2 detector quaternions."""
    z = np.array([0., 1.j], dtype=p.dtype)
    p_z = complex_to_float(rotate(p.reshape(1, 2), z))
    q_z = complex_to_float(rotate(q.reshape(1, 2), z))
    return np.arccos((p_z[0, 1:] * q_z[0, 1:]).sum())


@jit(nopython=True, nogil=True, cache=True)
def dist_spherical_pairwise(ps: np.ndarray[np.complex_]) -> np.ndarray[np.float_]:
    """Pair-wise great circle distances between detector quaternions.

    Assume input is a 1-dim array of quarternions (2d-array)
    and return pairwise distance in 1d-array,
    ordered in "row-major" and "j>i" directions.
    E.g. for 3 detectors, [(0, 1), (0, 2), (1, 2)] ordering.
    """
    # make sure it is 2d-array
    n, _ = ps.shape
    size = (n * (n - 1)) // 2
    res = np.empty(size)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            res[k] = dist_spherical(ps[i], ps[j])
            k += 1
    return res


@jit(nopython=True, nogil=True, cache=True)
def dist_spherical_pairwise_from_lastcol_array(ps: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
    return dist_spherical_pairwise(float_to_complex(lastcol_quat_to_canonical(ps)))


@dataclass
class Quaternion:
    array_complex: np.ndarray[np.complex_]

    def __post_init__(self):
        assert self.array_complex.shape[-1] == 2

    def clear_cache(self):
        try:
            del self.array
        except AttributeError:
            pass
        try:
            del self.lastcol_array
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
    def array(self) -> np.ndarray[np.float_]:
        return complex_to_float(self.array_complex)

    @cached_property
    def lastcol_array(self) -> np.ndarray[np.float_]:
        return canonical_quat_to_lastcol(complex_to_float(self.array_complex))

    def __add__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self.array + other.array)

    def __iadd__(self, other: Quaternion):
        self.clear_cache()
        self.array += other.array

    def __mul__(self, other: Quaternion) -> Quaternion:
        return Quaternion(mul(self.array_complex, other.array_complex))

    def __imul__(self, other: Quaternion):
        self.clear_cache()
        self.array_complex = mul(self.array_complex, other.array_complex)

    def __matmul__(self, other: Quaternion) -> Quaternion:
        return Quaternion(matmul(self.array_complex, other.array_complex))

    def __imatmul__(self, other: Quaternion) -> Quaternion:
        self.clear_cache()
        self.array_complex = matmul(self.array_complex, other.array_complex)

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
        return Quaternion(self.array_complex / self.abs[..., None])

    @property
    def inverse(self) -> Quaternion:
        return Quaternion(inverse(self.array_complex))

    def rotate(self, other: Quaternion) -> Quaternion:
        return Quaternion(rotate(self.array_complex, other.array_complex))

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
    def from_array(cls, array: np.ndarray[np.float_]) -> Quaternion:
        """Create Quaternion from real array with last axis as w, x, y, z.
        """
        return cls(float_to_complex(array))

    @classmethod
    def from_lastcol_array(cls, array: np.ndarray[np.float_]) -> Quaternion:
        """Create Quaternion from real array with last axis as x, y, z, w.

        Convention used in TOAST.
        """
        return cls(float_to_complex(lastcol_quat_to_canonical(array)))

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

import numpy as np
import pytest

import numba_quaternion

np.random.seed(0)

random_array = np.random.randn(100, 4)
random_array_broadcast = random_array.reshape(5, 5, 2, 2, 4)


@pytest.mark.parametrize(
    "array",
    [
        random_array,
        random_array.astype(np.float32),
        random_array_broadcast,
        random_array_broadcast.astype(np.float32),
    ]
)
def test_mul_inverse_conjugate_norm(array):
    test_array = numba_quaternion.Quaternion.from_array(array)
    res = (test_array * test_array.inverse).array
    decimal = 15 if array.dtype == np.float64 else 6
    np.testing.assert_array_almost_equal(res[..., 0], 1., decimal=decimal)
    np.testing.assert_array_almost_equal(res[..., 1:], 0., decimal=decimal)


m1 = np.random.randn(100, 100) + 1.j * np.random.randn(100, 100)
m2 = np.random.randn(100, 100) + 1.j * np.random.randn(100, 100)
m1_m2 = m1 @ m2
q1 = np.zeros((100, 100, 2), dtype=np.complex128)
q2 = np.zeros((100, 100, 2), dtype=np.complex128)
q_answer = np.zeros((100, 100, 2), dtype=np.complex128)
q1[:, :, 0] = m1
q2[:, :, 0] = m2
q_answer[:, :, 0] = m1_m2


@pytest.mark.parametrize(
    "array",
    [
        random_array,
        random_array.astype(np.float32),
        random_array_broadcast,
        random_array_broadcast.astype(np.float32),
    ]
)
def test_rotation_matrix(array):
    test_array = numba_quaternion.Quaternion.from_array(array).normalize
    m = test_array.to_rotation_matrix
    assert m.shape[-2] == 3
    assert m.shape[-1] == 3
    res = numba_quaternion.Quaternion.from_rotation_matrix(m)
    # sqrt near 0 in rotation_matrix_to_quat amplifies rounding error to ~sqrt(eps)
    atol = 1e-7 if array.dtype == np.float64 else 1e-3
    # q and -q represent the same rotation
    sign = np.sign((res.array * test_array.array).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(sign * res.array, test_array.array, atol=atol)


@pytest.mark.parametrize(
    "array1,array2,answer",
    [
        (q1, q2, q_answer),
        (
            q1.astype(np.complex64),
            q2.astype(np.complex64),
            q_answer.astype(np.complex64),
        ),
    ]
)
def test_mat_mul(array1, array2, answer):
    test_array1 = numba_quaternion.Quaternion(array1)
    test_array2 = numba_quaternion.Quaternion(array2)
    res = (test_array1 @ test_array2).array_complex
    atol = 1e-12 if array1.dtype == np.complex128 else 1e-3
    np.testing.assert_allclose(res, answer, atol=atol)


az = np.stack(
    (
        np.random.uniform(0.5 * np.pi, size=100),
        np.random.uniform(2. * np.pi, size=100),
        np.random.uniform(2. * np.pi, size=100),
    ),
    -1,
)


@pytest.mark.parametrize(
    "az",
    [
        az,
        az.astype(np.float32),
    ]
)
def test_azimuthal(az):
    m = numba_quaternion.azimuthal_equidistant_projection_polar_with_orientation_to_rotation_matrix(az)
    q = numba_quaternion.rotation_matrix_to_quat(m)
    az_round_trip = numba_quaternion.quat_to_azimuthal_equidistant_projection_polar_with_orientation(q)
    atol = 1e-12 if az.dtype == np.float64 else 1e-4
    np.testing.assert_allclose(az[:, 0], az_round_trip[:, 0], atol=atol)
    # angles are only defined modulo 2 pi
    diff = np.angle(np.exp(1.j * (az[:, 1:] - az_round_trip[:, 1:])))
    np.testing.assert_allclose(diff, 0., atol=atol)


def test_operators():
    p = numba_quaternion.Quaternion.from_array(random_array)
    q = numba_quaternion.Quaternion.from_array(random_array[::-1].copy())
    np.testing.assert_allclose((p + q).array, random_array + random_array[::-1])
    np.testing.assert_allclose(p.conjugate.array[..., 0], random_array[..., 0])
    np.testing.assert_allclose(p.conjugate.array[..., 1:], -random_array[..., 1:])
    expected = (p * q).array
    r = numba_quaternion.Quaternion(p.array_complex.copy())
    r *= q
    np.testing.assert_allclose(r.array, expected)
    r = numba_quaternion.Quaternion(p.array_complex.copy())
    r += q
    np.testing.assert_allclose(r.array, (p + q).array)


def test_dist_spherical_pairwise():
    q = numba_quaternion.Quaternion.from_array(random_array[:10]).normalize
    res = numba_quaternion.dist_spherical_pairwise_from_lastcol_array(q.lastcol_array)
    assert res.shape == (45,)
    assert np.all((0. <= res) & (res <= np.pi))

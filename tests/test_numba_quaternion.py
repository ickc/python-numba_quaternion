import numpy as np
import pytest

import numba_quaternion

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
    test_array = numba_quaternion.Quaternion(array)
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
    test_array = numba_quaternion.Quaternion(array).normalize
    m = test_array.to_rotation_matrix
    assert m.shape[-2] == 3
    assert m.shape[-1] == 3
    res = numba_quaternion.Quaternion.from_rotation_matrix(m)
    atol = 1e15 if array.dtype == np.float64 else 1e5
    np.testing.assert_allclose(res.array, test_array.array, atol=atol)


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
    test_array1 = numba_quaternion.Quaternion.from_array_complex(array1)
    test_array2 = numba_quaternion.Quaternion.from_array_complex(array2)
    res = (test_array1 @ test_array2).array_complex
    atol = 1e15 if array1.dtype == np.float64 else 1e5
    np.testing.assert_allclose(res, answer, atol=atol)

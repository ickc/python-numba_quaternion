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
def test_numba_quaternion(array):
    test_array = numba_quaternion.Quaternion(array)
    res = (test_array * test_array.inverse).array
    decimal = 15 if array.dtype == np.float64 else 6
    np.testing.assert_array_almost_equal(res[..., 0], 1., decimal=decimal)
    np.testing.assert_array_almost_equal(res[..., 1:], 0., decimal=decimal)

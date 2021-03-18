import numpy as np

import numba_quaternion


def test_numba_quaternion():
    test_array = numba_quaternion.Quaternion(np.random.randn(100, 4))
    res = (test_array * test_array.inverse).array
    np.testing.assert_allclose(res[..., 0], 1.)
    np.testing.assert_allclose(res[..., 1:], 0., atol=1e-16)

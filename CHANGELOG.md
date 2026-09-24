# Revision history for `numba_quaternion`

- v0.3.0:
    - support numba>=0.59 (`numba.generated_jit` was removed) and numpy 2; require Python>=3.10.
    - declare scipy as a dependency (numba needs it for matrix multiplication in jitted code).
    - fix `Quaternion.conjugate`, `Quaternion.__add__`, and in-place operators returning `None`.
    - remove the `extras` extra (coloredlogs) and the logging handler set up on import.
- v0.2.0: all quaternion array is now complex to avoid excessive conversion. The Quaternion class is a bit different because of this. Use `Quaternion.from_array` to create from real array.
- v0.1.0: first release and proof of concept.

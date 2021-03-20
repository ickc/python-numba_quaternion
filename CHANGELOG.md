# Revision history for `numba_quaternion`

- v0.2.0: all quaternion array is now complex to avoid excessive conversion. The Quaternion class is a bit different because of this. Use `Quaternion.from_array` to create from real array.
- v0.1.0: first release and proof of concept.

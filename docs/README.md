---
fontsize:	11pt
documentclass:	memoir
classoption: article
geometry:	inner=1in, outer=1in, top=1in, bottom=1.25in
title:	numba_quaternion—quaternion operations that can be used within Numba-jit functions
...

``` {.table}
---
header: false
markdown: true
include: badges.csv
...
```

# Introduction

This package contains some numba-jit-compiled functions that perform Quaternion operations and a convenient class `Quaternion` that provide convenient methods wrapping around those functions.

`Quaternion` behaves like a Numpy array containing quaternion, e.g. respect Numpy broadcast operations, but without really imitating a `numpy.ndarray` and implemented a `dtype`.

This design allows you to write any jit-compiled functions involving those provided jit-compiled functions, and then write your own class methods that calls those functions as a convenient interface (by class inheritance.)

If you do not care to use Quaternion in other jit-compiled functions you write, check out packages below instead.

# Other Python quaternion projects

Other Python projects that implements quaternions and I knew and used are:

- [zonca/quaternionarray](https://github.com/zonca/quaternionarray): written in pure Python using Numpy. Note that unusually they put the real part in the last column. `lastcol_quat_to_canonical` and `canonical_quat_to_lastcol` convert between those and the canonical ordering (where real part comes first.)
- [hpc4cmb/toast](https://github.com/hpc4cmb/toast): toast.qarray is a reimplementation of the above quaternionarray package in C++ with the same interface, and following the same convention.
- [moble/quaternion](https://github.com/moble/quaternion): implement Quaternion as a Numpy dtype in C.
- [moble/quaternionic](https://github.com/moble/quaternionic): implement Quaternion as a Numpy dtype using Numba. This package is inspired by my expectation of quaternionic—I expected I could use them in a Numba-jit-compiled function but it doesn't.

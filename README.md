pyVPT2
==============================
[//]: # (Badges)
![CI workflow](https://github.com/philipmnel/pyvpt2/actions/workflows/CI.yaml/badge.svg)
[![codecov](https://codecov.io/gh/philipmnel/pyvpt2/branch/main/graph/badge.svg?token=goQRxdntmS)](https://codecov.io/gh/philipmnel/pyvpt2)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/philipmnel/pyvpt2/blob/main/LICENSE)

### About
pyVPT2 is a package to calculate VPT2 vibrational frequencies using psi4. Cubic/quartic constants can be optionally computed in a distributed parallel fashion using QCFractal.

For usage check the [wiki page](https://github.com/philipmnel/pyvpt2/wiki/pyVPT2-Manual).

Disclaimer: This codebase is still under active development.

### Dependencies
Required:
- psi4 = 1.7+
- python = 3.8+

Installing psi4 will pull all other required dependencies.

Optional:
- qcportal = `next`

### Installation
To install from source, clone this repository and run
```
pip install .
```

### Copyright

Copyright (c) 2021-2023, Philip Nelson


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.

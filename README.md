pyVPT2
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/philipmnel/pyvpt2/workflows/CI/badge.svg)](https://github.com/philipmnel/pyvpt2/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/philipmnel/pyvpt2/branch/main/graph/badge.svg?token=goQRxdntmS)](https://codecov.io/gh/philipmnel/pyvpt2)

### About
pyVPT2 is a package to calculate VPT2 vibrational frequencies using psi4. Cubic/quartic constants can be optionally computed in a distributed parallel fashion using QCFractal.

### Dependencies
Required:
- psi4 = 1.7
- python = 3.8 or 3.9

Installing psi4 will pull all other required dependencies.

Optional:
- qcportal=0.15.8

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

pyVPT2
==============================
[//]: # (Badges)
[![CI](https://img.shields.io/github/actions/workflow/status/philipmnel/pyvpt2/CI.yaml?logo=github)](https://github.com/philipmnel/pyvpt2/actions?query=workflow%3ACI)
[![Docs](https://img.shields.io/github/actions/workflow/status/philipmnel/pyvpt2/CI.yaml?label=docs&logo=readthedocs&logoColor=white)](https://philipmnel.github.io/pyvpt2/)
[![codecov](https://codecov.io/gh/philipmnel/pyvpt2/branch/main/graph/badge.svg?token=goQRxdntmS)](https://codecov.io/gh/philipmnel/pyvpt2)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/philipmnel/pyvpt2/blob/main/LICENSE)


### About
pyVPT2 is a package to calculate VPT2 vibrational frequencies using psi4 or qcengine. Cubic/quartic constants can be optionally computed in a distributed parallel fashion using QCFractal.

For usage check the [documentation](https://philipmnel.github.io/pyvpt2/).

Disclaimer: This codebase is still under active development.

### Dependencies
Required:
- psi4 = 1.8+
- python = 3.8+

Installing psi4 will pull all other required dependencies.

Optional:
- qcportal = `next`

### Installation
pyVPT2 can be installed from conda:
```
conda install pyvpt2 -c conda-forge
```

Or, to install from source, clone this repository and run
```
pip install .
```

### Copyright

Copyright (c) 2021-2024, Philip Nelson


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.

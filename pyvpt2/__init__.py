"""Package to calculate VPT2 anharmonic corrections."""

import logging
from importlib.metadata import PackageNotFoundError, version

from .vpt2 import *

try:
    __version__ = version("pyvpt2")
except PackageNotFoundError:
    pass

vpt2_log = logging.getLogger("psi4.pyvpt2")
vpt2_log.propagate = True

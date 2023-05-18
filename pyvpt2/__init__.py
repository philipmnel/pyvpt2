"""Package to calculate VPT2 anharmonic corrections."""

import logging

from .vpt2 import *

vpt2_log = logging.getLogger("psi4.pyvpt2")
vpt2_log.propagate = True

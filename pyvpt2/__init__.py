"""Package to calculate VPT2 anharmonic corrections."""

# Add imports here
# Import Psi4 logger
from psi4 import logger

from .vpt2 import *

vpt2_log = logger.getChild("pyvpt2")
vpt2_log.propagate = True

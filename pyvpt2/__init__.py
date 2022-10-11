"""Package to calculate VPT2 anharmonic corrections."""

# Add imports here
from .vpt2 import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

# Import Psi4 logger
from psi4 import logger
vpt2_log = logger.getChild("pyvpt2")
vpt2_log.propagate = True
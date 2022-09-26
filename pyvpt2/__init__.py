"""Package to calculate VPT2 anharmonic corrections."""

# Add imports here
from .vpt2 import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

import logging
logger = logging.getLogger(__name__)
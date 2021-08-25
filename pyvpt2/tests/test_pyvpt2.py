"""
Unit and regression test for the pyvpt2 package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import pyvpt2


def test_pyvpt2_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pyvpt2" in sys.modules

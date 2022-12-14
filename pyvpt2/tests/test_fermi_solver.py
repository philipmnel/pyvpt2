import numpy as np
import pyvpt2
from pyvpt2.fermi_solver import fermi_solver
from psi4 import compare_values
import pytest


def test_identify_fermi():

    omega = [1544.3, 1601.8, 2956.6]
    phi_ijk = np.asarray([ 
    [[158.8875601556632, -247.29101465678508, -166.74685943583202],
    [-247.29101465678508, 120.09522082111998, -142.50632482892297],
    [-166.74685943583202, -142.50632482892297, -281.86663885753416]],
    [[-247.29101465678508, 120.09522082111998, -143.13648130744926],
    [120.09522082111998, -426.7578692752896, 52.4164500795531],
    [-143.13648130744926, 52.4164500795531, 151.5566843963656]],
    [[-166.74685943583202, -148.31003819977596, -281.86663885753416],
    [-148.31003819977596, 52.4164500795531, 151.5566843963656],
    [-281.86663885753416, 151.5566843963656, 2406.274767904205]]])
    n_modes = 3
    v_ind = [0,1,2]
    kwargs = pyvpt2.process_options_keywords()
    fermi_list = pyvpt2.identify_fermi(omega, phi_ijk, n_modes, v_ind, **kwargs)
    assert fermi_list.pop() == (2, (0,0))
    assert len(fermi_list) == 0


def test_fermi_solver():

    ref = {(1,2): 3047.2, (0,): 3116.3, (3,6): 2711.5, (5,): 2851.5, (2,6): 3004.3}

    interaction0 = {"left": {"state": (0,), "nu": 3095.5}, "right": {"state": (1,2), "nu": 3068.0}, "phi": -89.7, "type": 2}
    interaction1 = {"left": {"state": (5,), "nu": 2828.0}, "right": {"state": (2,6), "nu": 2987.4}, "phi": -146.7, "type": 2}
    interaction2 = {"left": {"state": (5,), "nu": 2828.0}, "right": {"state": (3,6), "nu": 2751.9}, "phi": 185.8, "type": 2}
    fermi_list = [interaction0, interaction1, interaction2]

    state_list = fermi_solver(fermi_list)

    for state in state_list.keys():
        assert compare_values(ref[state], state_list[state], 0.1)
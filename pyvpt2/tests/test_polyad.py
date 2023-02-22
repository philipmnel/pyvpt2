from pyvpt2.fermi_solver import Polyad, Interaction, State
from psi4 import compare_values
import pytest

def test_polyad_1():

    phi_ijk = 88.1
    nu = [3106.1, 3152.0]
    ref = {(0,): 3097.2, (1,1): 3160.8}

    interaction = Interaction(left=State(state=(0,), nu=nu[0]), right=State(state=(1,1), 
                            nu=nu[1]), phi=phi_ijk, ftype=1)
    test = Polyad(interaction)
    state_list = test.solve()
    for state in state_list.keys():
        assert compare_values(ref[state], state_list[state], 0.1)

def test_polyad_2():

    phi_ijk = -89.7
    nu = [3095.5, 3068.0]
    ref = {(1,2): 3047.2, (0,): 3116.3}

    interaction = Interaction(left=State(state=(0,), nu=nu[0]), right=State(state=(1,2), 
                            nu=nu[1]), phi=phi_ijk, ftype=2)
    test = Polyad(interaction)
    state_list = test.solve()
    for state in state_list.keys():
        assert compare_values(ref[state], state_list[state], 0.1)

def test_polyad_3():

    ref = {(3,6): 2711.5, (5,): 2851.5, (2,6): 3004.3}
    interaction1 = Interaction(left=State(state=(5,), nu=2828.0), right=State(state=(2,6), 
                            nu=2987.4), phi=-146.7, ftype=2)
    interaction2 = Interaction(left=State(state=(5,), nu=2828.0), right=State(state=(3,6), 
                            nu=2751.9), phi=185.8, ftype=2)

    test = Polyad(interaction1)
    test.add(interaction2)
    state_list = test.solve()
    for state in state_list.keys():
        assert compare_values(ref[state], state_list[state], 0.1)
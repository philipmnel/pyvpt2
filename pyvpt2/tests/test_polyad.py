import pytest
from psi4 import compare_values

from pyvpt2.fermi_solver import Interaction, Polyad, State


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


def test_polyad_two_eigenvectors_peaking_on_one_state():
    # three strongly mixed states: two eigenvectors have their largest component on the same basis state, which the
    # per-eigenvector argmax mapped to one state twice and left another without a frequency (KeyError)
    import numpy as np
    i1 = Interaction(left=State(state=(0,), nu=1000.0), right=State(state=(1, 1), nu=1002.0), phi=400.0, ftype=1)
    i2 = Interaction(left=State(state=(0,), nu=1000.0), right=State(state=(2, 3), nu=1001.0), phi=400.0, ftype=2)
    test = Polyad(i1)
    test.add(i2)
    freqs = test.solve()
    assert set(freqs) == {(0,), (1, 1), (2, 3)}
    assert sorted(freqs.values()) == pytest.approx(sorted(np.linalg.eigvalsh(test.H)))


def test_fermi_solver_merges_polyads_an_interaction_bridges():
    import numpy as np

    from pyvpt2.fermi_solver import fermi_solver
    a = Interaction(left=State(state=(0,), nu=1000.0), right=State(state=(4, 4), nu=1010.0), phi=50.0, ftype=1)
    b = Interaction(left=State(state=(1,), nu=1500.0), right=State(state=(5, 6), nu=1490.0), phi=60.0, ftype=2)
    bridge = Interaction(left=State(state=(1,), nu=1500.0), right=State(state=(4, 4), nu=1010.0), phi=40.0, ftype=1)
    freqs = fermi_solver([a, b, bridge])
    whole = Polyad(a)
    whole.add(b)
    whole.add(bridge)
    ref = whole.solve()
    assert set(freqs) == {(0,), (1,), (4, 4), (5, 6)}
    assert sorted(freqs.values()) == pytest.approx(sorted(np.linalg.eigvalsh(whole.H)))
    for state, value in ref.items():
        assert freqs[state] == pytest.approx(value)

import psi4
import pytest

import pyvpt2


def test_hcn():
    mol = psi4.geometry("""
    H            0.000000000000     0.000000000000    -1.614875631638
    C           -0.000000000000     0.000000000000    -0.548236744990
    N            0.000000000000     0.000000000000     0.586039395549
    """)

    qc_kwargs = {'d_convergence': 1e-10,
                'e_convergence': 1e-10,
                'points': 5}

    options = {'FD': "HESSIAN",
        'DISP_SIZE': 0.05}

    ref_omega = [869.1587, 2421.4515, 3645.1338]
    ref_anharmonic = [-19.8940, -23.6620, -125.1330]

    inp = {"molecule": mol.to_schema(dtype=2),
           "input_specification": [{"model":{
                                        "method": "scf",
                                        "basis": "cc-pvdz"},
                                   "keywords": qc_kwargs}],
           "keywords": options}

    results = pyvpt2.vpt2_from_schema(inp)
    omega = results.omega[-3:]
    anharmonic = results.nu[-3:] - omega

    assert psi4.compare_values(ref_omega, omega, 0.5)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.5)

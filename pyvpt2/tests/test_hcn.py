import psi4
import pytest

import pyvpt2


def test_hcn():
    mol = psi4.geometry("""
    nocom
    noreorient

    H            0.000000000000     0.000000000000    -1.614875631638
    C           -0.000000000000     0.000000000000    -0.548236744990
    N            0.000000000000     0.000000000000     0.586039395549
    """)

    psi4.set_options({'d_convergence': 1e-10,
                'e_convergence': 1e-10,
                'points': 5})

    options = {'METHOD': 'scf/cc-pvdz',
            'FD': "HESSIAN",
            'DISP_SIZE': 0.05}

    ref_omega = [869.1587, 2421.4515, 3645.1338]
    ref_anharmonic = [-19.8940, -23.6620, -125.1330]

    results = pyvpt2.vpt2(mol, **options)
    omega = results.omega[-3:]
    anharmonic = results.nu[-3:] - omega

    assert psi4.compare_values(ref_omega, omega, 0.5)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.5)

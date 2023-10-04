import psi4
import pytest

import pyvpt2


@pytest.mark.parametrize("driver", ["ENERGY", "GRADIENT", "HESSIAN"])
def test_hf_vpt2(driver):

    mol = psi4.geometry("""
    H
    F 1 R1

    R1 = 0.920853
    """)

    qc_kwargs = {'d_convergence': 1e-12,
                'e_convergence': 1e-12,
                'scf_type': 'direct',
                'puream': True}

    options = {'FD': driver,
            'DISP_SIZE': 0.02,
            'QC_PROGRAM': "psi4"}

    ref_omega = 4135.3637
    ref_anharmonic = -153.1174
    ref_harm_zpve = 2067.682
    ref_zpve_corr = -13.595

    inp = {"molecule": mol.to_schema(dtype=2),
           "input_specification": [{"model":{
                                        "method": "scf",
                                        "basis": "6-31g"},
                                   "keywords": qc_kwargs}],
           "keywords": options}

    results = pyvpt2.vpt2_from_schema(inp)
    omega = results.omega[-1]
    anharmonic = results.nu[-1] - omega
    harm_zpve  = results.harmonic_zpve
    zpve_corr = results.anharmonic_zpve - harm_zpve

    assert psi4.compare_values(ref_omega, omega, 0.5)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.5)
    assert psi4.compare_values(ref_harm_zpve, harm_zpve, 0.5)
    assert psi4.compare_values(ref_zpve_corr, zpve_corr, 0.5)

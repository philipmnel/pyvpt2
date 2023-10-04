import psi4
import pytest

import pyvpt2


@pytest.mark.parametrize("driver", ["ENERGY", "GRADIENT", "HESSIAN"])
def test_h2o_multi_vpt2(driver):

    mol = psi4.geometry("""
    O
    H 1 R1
    H 1 R2 2 A

    R1    =        0.9406103293
    R2    =        0.9406103293
    A    =      106.0259742413
    """)

    qc_kwargs = {'d_convergence': 1e-12,
                'e_convergence': 1e-12,}

    options = {'FD': driver,
            'DISP_SIZE': 0.05}

    ref_omega = [1752.8008, 4126.8616, 4226.9977]
    ref_anharmonic = [-63.0767, -181.2757, -190.6156]
    ref_harm_zpve = 5053.3300
    ref_zpve_corr = -79.2130

    inp = {"molecule": mol.to_schema(dtype=2),
           "input_specification": [{"model":{
                                        "method": "scf",
                                        "basis": "cc-pvtz"},
                                   "keywords": qc_kwargs},
                                   {"model":{
                                       "method": "scf",
                                       "basis": "cc-pvdz"},
                                   "keywords": qc_kwargs}
                                   ],
           "keywords": options}

    results = pyvpt2.vpt2_from_schema(inp)
    omega = results.omega[-3:]
    anharmonic = results.nu[-3:] - omega
    harm_zpve  = results.harmonic_zpve
    zpve_corr = results.anharmonic_zpve - harm_zpve

    assert psi4.compare_values(ref_omega, omega, 0.5)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.5)
    assert psi4.compare_values(ref_harm_zpve, harm_zpve, 0.5)
    assert psi4.compare_values(ref_zpve_corr, zpve_corr, 0.5)

import psi4
import pytest

import pyvpt2


@pytest.mark.parametrize("driver", ["ENERGY", "GRADIENT", "HESSIAN"])
def test_h2o_cbs_vpt2(driver):

    mol = psi4.geometry("""
    O
    H 1 R1
    H 1 R2 2 A

    A         =  106.3819454243
    R1        =    0.9392155213
    R2        =    0.9392155213
    """)

    qc_kwargs = {'d_convergence': 1e-12,
                'e_convergence': 1e-12}

    options = {'FD': driver,
            'DISP_SIZE': 0.05}

    inp = {
        "molecule": mol.to_schema(dtype=2),
        "keywords": options,
        "input_specification": [{
            "model": {
                "method": "scf/cc-pv[dt]z",
                "basis": "(auto)"},
            "keywords": qc_kwargs}]
        }

    ref_omega = [1747.4491, 4129.8877, 4230.4755]
    ref_anharmonic = [-53.8382, -157.1904, -171.4518]
    ref_harm_zpve = 5053.9062
    ref_zpve_corr = -69.5146

    results = pyvpt2.vpt2_from_schema(inp)
    omega = results.omega[-3:]
    anharmonic = results.nu[-3:] - omega
    harm_zpve  = results.harmonic_zpve
    zpve_corr = results.anharmonic_zpve - harm_zpve

    assert psi4.compare_values(ref_omega, omega, 0.5)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.5)
    assert psi4.compare_values(ref_harm_zpve, harm_zpve, 0.5)
    assert psi4.compare_values(ref_zpve_corr, zpve_corr, 0.5)

import psi4
import pytest
import qcengine as qcng

import pyvpt2

no_cfour = 'cfour' not in qcng.list_available_programs()

@pytest.mark.skipif(no_cfour, reason="CFOUR not installed")
@pytest.mark.parametrize("driver", ["ENERGY", "GRADIENT", "HESSIAN"])
def test_h2o_vpt2(driver):

    mol = psi4.geometry("""
    nocom
    noreorient

    O
    H 1 R1
    H 1 R2 2 A

    R1    =        0.94731025924472878064
    R2    =        0.94731025924472878064
    A    =      105.5028950965639

    symmetry c2v
    """)

    options = {'METHOD': 'scf/6-31G*',
            'FD': driver,
            'DISP_SIZE': 0.05,
            'QC_PROGRAM': 'cfour',
            'QC_KWARGS': {'SCF_CONV': 12}}

    cfour_omega = [1826.8154, 4060.2203, 4177.8273]
    cfour_anharmonic = [-54.0635, -158.2345, -177.9707]
    cfour_harm_zpve = 5032.431
    cfour_zpve_corr = -70.352

    results = pyvpt2.vpt2(mol, **options)
    omega = results.omega[-3:]
    anharmonic = results.nu[-3:] - omega
    harm_zpve  = results.harmonic_zpve
    zpve_corr = results.anharmonic_zpve - harm_zpve

    assert psi4.compare_values(cfour_omega, omega, 0.2), "omega"
    assert psi4.compare_values(cfour_anharmonic, anharmonic, 0.2), "anharmonic correction"
    assert psi4.compare_values(cfour_harm_zpve, harm_zpve, 0.2), "harmonic ZPVE"
    assert psi4.compare_values(cfour_zpve_corr, zpve_corr, 0.2), "ZPVE correction"

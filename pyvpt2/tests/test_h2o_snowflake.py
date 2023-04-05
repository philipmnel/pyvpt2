import time

import psi4
import pytest

import pyvpt2

no_qcfractal = False
try:
    from qcfractal.snowflake import FractalSnowflake
except:
    no_qcfractal = True

@pytest.mark.skipif(no_qcfractal, reason="QCFractal not installed")
@pytest.mark.parametrize("driver", ["ENERGY", "GRADIENT"]) #HESSIAN harmonics currently broken
def test_h2o_snowflake_vpt2(driver):

    snowflake = FractalSnowflake()
    client = snowflake.client()

    mol = psi4.geometry("""
    nocom
    noreorient

    O
    H 1 R1
    H 1 R2 2 A

    A         =  106.3819454243
    R1        =    0.9392155213
    R2        =    0.9392155213
    symmetry c2v
    """)

    mol.move_to_com()
    mol.fix_com(True)
    mol.fix_orientation(True)

    psi4.set_options({'d_convergence': 1e-10,
                'e_convergence': 1e-10,
                'points': 5})

    options = {'METHOD': 'scf/cc-pv[dt]z',
            'FD': driver,
            'DISP_SIZE': 0.05,
            'RETURN_PLAN': True}

    ref_omega = [1747.4491, 4129.8877, 4230.4755]
    ref_anharmonic = [-53.8382, -157.1904, -171.4518]
    ref_harm_zpve = 5053.9062
    ref_zpve_corr = -69.5146

    harmonic_plan = pyvpt2.vpt2(mol, **options)
    harmonic_plan.compute(client=client)
    snowflake.await_results()
    harmonic_ret = harmonic_plan.get_results(client=client)

    plan = pyvpt2.vpt2_from_harmonic(harmonic_ret, **options)
    plan.compute(client=client)
    snowflake.await_results()
    ret = plan.get_results(client=client)
    results = pyvpt2.process_vpt2(ret, **options)

    omega = results["Harmonic Freq"][-3:]
    anharmonic = results["Freq Correction"][-3:]
    harm_zpve  = results["Harmonic ZPVE"]
    zpve_corr = results["ZPVE Correction"]

    assert psi4.compare_values(ref_omega, omega, 0.5)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.5)
    assert psi4.compare_values(ref_harm_zpve, harm_zpve, 0.5)
    assert psi4.compare_values(ref_zpve_corr, zpve_corr, 0.5)

import psi4
import pyvpt2
import pytest
import time

no_qcfractal = False
try:
    from qcfractal import FractalSnowflakeHandler
except:
    no_qcfractal = True

@pytest.mark.skipif(no_qcfractal, reason="QCFractal not installed")
@pytest.mark.parametrize("driver", ["ENERGY", "GRADIENT"]) #HESSIAN harmonics currently broken
def test_h2o_snowflake_vpt2(driver):
    
    snowflake = FractalSnowflakeHandler()
    client = snowflake.client()

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

    mol.move_to_com()
    mol.fix_com(True)
    mol.fix_orientation(True)

    psi4.set_options({'d_convergence': 1e-10,
                'e_convergence': 1e-10,
                'scf_type': 'direct',
                'puream': True,
                'points': 5})

    options = {'METHOD': 'scf/6-31g*',
            'FD': driver,
            'DISP_SIZE': 0.05,
            'RETURN_PLAN': True}

    cfour_omega = [1826.8154, 4060.2203, 4177.8273]
    cfour_anharmonic = [-54.0635, -158.2345, -177.9707] 
    cfour_harm_zpve = 5032.431
    cfour_zpve_corr = -70.352

    harmonic_plan = pyvpt2.vpt2(mol, **options)
    harmonic_plan.compute(client=client)
    while len(client.query_tasks()) != 0:
        time.sleep(1)
    harmonic_ret = harmonic_plan.get_results(client=client)

    plan = pyvpt2.vpt2_from_harmonic(harmonic_ret, **options)
    plan.compute(client=client)
    while len(client.query_tasks()) != 0:
        time.sleep(1)
    ret = plan.get_results(client=client)
    results = pyvpt2.process_vpt2(ret, **options)

    omega = results["Harmonic Freq"][-3:]
    anharmonic = results["Freq Correction"][-3:]
    harm_zpve  = results["Harmonic ZPVE"]
    zpve_corr = results["ZPVE Correction"]

    assert psi4.compare_values(cfour_omega, omega, 0.1)
    assert psi4.compare_values(cfour_anharmonic, anharmonic, 0.1)
    assert psi4.compare_values(cfour_harm_zpve, harm_zpve, 0.1)
    assert psi4.compare_values(cfour_zpve_corr, zpve_corr, 0.1)
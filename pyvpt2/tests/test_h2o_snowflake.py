import psi4
import pytest

import pyvpt2

no_qcfractal = False
try:
    from qcfractal.snowflake import FractalSnowflake
except:
    no_qcfractal = True

@pytest.mark.skipif(no_qcfractal, reason="QCFractal not installed")
@pytest.mark.parametrize("driver", ["HESSIAN"])
def test_h2o_snowflake_vpt2(driver):

    snowflake = FractalSnowflake()
    client = snowflake.client()

    mol = psi4.geometry("""
    O
    H 1 R1
    H 1 R2 2 A

    A         =  106.3819454243
    R1        =    0.9392155213
    R2        =    0.9392155213
    """)

    options = {'FD': driver,
            'DISP_SIZE': 0.05,
            'RETURN_PLAN': True,
            'QC_PROGRAM': 'psi4'}

    ref_omega = [1747.4491, 4129.8877, 4230.4755]
    ref_anharmonic = [-53.8382, -157.1904, -171.4518]
    ref_harm_zpve = 5053.9062
    ref_zpve_corr = -69.5146

    inp = {
        "molecule": mol.to_schema(dtype=2),
        "keywords": options,
        "input_specification": [{
            "model": {
                "method": "scf/cc-pv[dt]z",
                "basis": "(auto)"},
            "keywords": {
                "e_convergence": 10,
                "d_convergence": 10,
            }
        }]
    }

    harmonic_plan = pyvpt2.vpt2_from_schema(inp)
    harmonic_plan.compute(client=client)
    snowflake.await_results(timeout=120)
    harmonic_ret = harmonic_plan.get_results(client=client)

    plan = pyvpt2.vpt2_from_harmonic(harmonic_ret, qc_spec=inp["input_specification"][0], **options)
    plan.compute(client=client)
    snowflake.await_results(timeout=120)
    ret = plan.get_results(client=client)
    results = pyvpt2.process_vpt2(ret, **options)

    omega = results.omega[-3:]
    anharmonic = results.nu[-3:] - omega
    harm_zpve  = results.harmonic_zpve
    zpve_corr = results.anharmonic_zpve - harm_zpve

    assert psi4.compare_values(ref_omega, omega, 0.5)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.5)
    assert psi4.compare_values(ref_harm_zpve, harm_zpve, 0.5)
    assert psi4.compare_values(ref_zpve_corr, zpve_corr, 0.5)

    snowflake.stop()

import psi4
import pytest
import qcengine as qcng

import pyvpt2

no_qcfractal = False
try:
    from qcfractal.snowflake import FractalSnowflake
except:
    no_qcfractal = True

@pytest.mark.skipif(no_qcfractal, reason="QCFractal not installed")
@pytest.mark.parametrize("driver", ["HESSIAN"])
@pytest.mark.parametrize("program", ["psi4", "cfour", "nwchem"])
def test_h2o_snowflake_vpt2(driver, program):

    if program not in qcng.list_available_programs():
        pytest.skip()

    snowflake = FractalSnowflake()
    client = snowflake.client()

    qc_kwargs = {"psi4": {"e_convergence": 12,
                          "puream": True,
                          "scf_type": "direct"},
                 "cfour": {"SCF_CONV": 12},
                 "nwchem": {"scf__thresh": 1e-8,
                            "basis__spherical": True,
                            }}

    mol = psi4.geometry("""
    O
    H 1 R1
    H 1 R2 2 A

    R1    =        0.94731025924472878064
    R2    =        0.94731025924472878064
    A    =      105.5028950965639
    """)

    options = {'FD': driver,
            'DISP_SIZE': 0.05,
            'RETURN_PLAN': True,
            'QC_PROGRAM': program}

    ref_omega = [1826.8154, 4060.2203, 4177.8273]
    ref_anharmonic = [-54.0635, -158.2345, -177.9707]
    ref_harm_zpve = 5032.431
    ref_zpve_corr = -70.352

    inp = {
        "molecule": mol.to_schema(dtype=2),
        "keywords": options,
        "input_specification": [{
            "model": {
                "method": "scf",
                "basis": "6-31g*"},
            "keywords": qc_kwargs[program]
        }]
    }

    harmonic_plan = pyvpt2.vpt2_from_schema(inp)
    harmonic_plan.compute(client=client)
    snowflake.await_results()
    harmonic_ret = harmonic_plan.get_results(client=client)
    print(harmonic_ret)

    plan = pyvpt2.vpt2_from_harmonic(harmonic_ret, qc_spec=inp["input_specification"][0], **options)
    plan.compute(client=client)
    snowflake.await_results()
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

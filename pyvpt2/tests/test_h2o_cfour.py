import psi4
import pytest
import qcengine as qcng

import pyvpt2


@pytest.mark.parametrize("driver", ["ENERGY", "GRADIENT", "HESSIAN"])
@pytest.mark.parametrize("program", ["psi4", "cfour", "nwchem"])
def test_h2o_vpt2(driver, program):

    if program not in qcng.list_available_programs():
        pytest.skip()

    qc_kwargs = {"psi4": {"e_convergence": 12,
                          "puream": True,
                          "scf_type": "direct"},
                 "cfour": {"SCF_CONV": 12},
                 "nwchem": {"scf": "",
                            "   thresh": 8,
                            "end": "",}}

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
            'QC_PROGRAM': program}

    ref_omega = [1826.8154, 4060.2203, 4177.8273]
    ref_anharmonic = [-54.0635, -158.2345, -177.9707]
    ref_harm_zpve = 5032.431
    ref_zpve_corr = -70.352

    inp = {"molecule": mol.to_schema(dtype=2),
           "input_specification":[{
               "model": {"method": "scf",
                         "basis": "6-31g*"},
                "keywords": qc_kwargs[program]
           }],
           "keywords": options}

    results = pyvpt2.vpt2_from_schema(inp)
    omega = results.omega[-3:]
    anharmonic = results.nu[-3:] - omega
    harm_zpve  = results.harmonic_zpve
    zpve_corr = results.anharmonic_zpve - harm_zpve

    assert psi4.compare_values(ref_omega, omega, 0.2), "omega"
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.2), "anharmonic correction"
    assert psi4.compare_values(ref_harm_zpve, harm_zpve, 0.2), "harmonic ZPVE"
    assert psi4.compare_values(ref_zpve_corr, zpve_corr, 0.2), "ZPVE correction"

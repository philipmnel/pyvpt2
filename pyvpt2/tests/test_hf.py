import psi4
import pyvpt2
import pytest

@pytest.mark.parametrize("driver", ["ENERGY", "GRADIENT", "HESSIAN"])
def test_hf_vpt2(driver):

    mol = psi4.geometry("""
    nocom
    noreorient

    H
    F 1 R1
   
    R1 = 0.920853
    """)

    psi4.set_options({'d_convergence': 1e-12,
                'e_convergence': 1e-12,
                'scf_type': 'direct',
                'puream': True})

    
    options = {'METHOD': 'scf/6-31g',
            'FD': driver,
            'DISP_SIZE': 0.02}

    cfour_omega = 4135.3637
    cfour_anharmonic = -153.1174
    cfour_harm_zpve = 2067.682
    cfour_zpve_corr = -13.595

    results = pyvpt2.vpt2(mol, **options)
    omega = results["Harmonic Freq"][-1]
    anharmonic = results["Freq Correction"][-1]
    harm_zpve  = results["Harmonic ZPVE"]
    zpve_corr = results["ZPVE Correction"]

    assert psi4.compare_values(cfour_omega, omega, 0.5)
    assert psi4.compare_values(cfour_anharmonic, anharmonic, 0.5)
    assert psi4.compare_values(cfour_harm_zpve, harm_zpve, 0.5)
    assert psi4.compare_values(cfour_zpve_corr, zpve_corr, 0.5)
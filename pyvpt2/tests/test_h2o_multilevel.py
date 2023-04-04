import psi4
import pytest

import pyvpt2


@pytest.mark.parametrize("driver", ["ENERGY", "GRADIENT", "HESSIAN"])
def test_h2o_multi_vpt2(driver):
    
    mol = psi4.geometry("""
    nocom
    noreorient

    O
    H 1 R1
    H 1 R2 2 A
   
    R1    =        0.9406103293
    R2    =        0.9406103293
    A    =      106.0259742413

    symmetry c2v
    """)

    psi4.set_options({'d_convergence': 1e-12,
                'e_convergence': 1e-12,
                'points': 5})

    options = {'METHOD': 'scf/cc-pvtz',
            'METHOD2': 'scf/cc-pvdz',
            'FD': driver,
            'DISP_SIZE': 0.05}

    ref_omega = [1752.8008, 4126.8616, 4226.9977]
    ref_anharmonic = [-63.0767, -181.2757, -190.6156] 
    ref_harm_zpve = 5053.3300
    ref_zpve_corr = -79.2130

    results = pyvpt2.vpt2(mol, **options)
    #E, wfn = psi4.frequency("scf/cc-pvtz", return_wfn=True)
    #omega = wfn.frequency_analysis["omega"].data[-3:]
    omega = results["Harmonic Freq"][-3:]
    anharmonic = results["Freq Correction"][-3:]
    harm_zpve  = results["Harmonic ZPVE"]
    zpve_corr = results["ZPVE Correction"]

    assert psi4.compare_values(ref_omega, omega, 0.5)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.5)
    assert psi4.compare_values(ref_harm_zpve, harm_zpve, 0.5)
    assert psi4.compare_values(ref_zpve_corr, zpve_corr, 0.5)
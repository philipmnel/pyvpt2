import psi4
import pyvpt2
import pytest

@pytest.mark.parametrize("driver", ["ENERGY", "GRADIENT", "HESSIAN"])
def test_h2o_vpt2(driver):
    
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

    psi4.set_options({'d_convergence': 1e-12,
                'e_convergence': 1e-12,
                'points': 5})

    options = {'METHOD': 'scf/cc-pv[dt]z',
            'FD': driver,
            'DISP_SIZE': 0.05}

    ref_omega = [1747.4491, 4129.8877, 4230.4755]
    ref_anharmonic = [-53.8382, -157.1904, -171.4518] 
    ref_harm_zpve = 5053.9062
    ref_zpve_corr = -69.5146

    results = pyvpt2.vpt2(mol, options)
    omega = results["Harmonic Freq"][-3:]
    anharmonic = results["Freq Correction"][-3:]
    harm_zpve  = results["Harmonic ZPVE"]
    zpve_corr = results["ZPVE Correction"]

    assert psi4.compare_values(ref_omega, omega, 0.5)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.5)
    assert psi4.compare_values(ref_harm_zpve, harm_zpve, 0.5)
    assert psi4.compare_values(ref_zpve_corr, zpve_corr, 0.5)
"""
Unit and regression test for the pyvpt2 package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import pyvpt2
import psi4


def test_pyvpt2_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pyvpt2" in sys.modules


def test_h2o_energy():
    
    mol = psi4.geometry("""
    nocom
    noreorient

    O
    H 1 R1
    H 1 R2 2 A
   
    R1    =        0.94731025924472878064
    R2    =        0.94731025924472878064
    A    =      105.5028950965639
    """)

    psi4.set_options({'d_convergence': 1e-12,
                'e_convergence': 1e-12,
                'scf_type': 'direct',
                'puream': True,
                'points': 5})

    options = {'METHOD': 'scf/6-31g*',
            'FD': 'ENERGY',
            'DISP_SIZE': 0.05}

    cfour_omega = [1826.8154, 4060.2203, 4177.8273]
    cfour_anharmonic = [-54.0635, -158.2345, -177.9707] 
    cfour_harm_zpve = 5032.431
    cfour_zpve_corr = -70.352

    results = pyvpt2.vpt2(mol, options)
    omega = results["Harmonic Freq"][-3:]
    anharmonic = results["Freq Correction"][-3:]
    harm_zpve  = results["Harmonic ZPVE"]
    zpve_corr = results["ZPVE Correction"]

    assert psi4.compare_values(cfour_omega, omega, 0.1)
    assert psi4.compare_values(cfour_anharmonic, anharmonic, 0.1)
    assert psi4.compare_values(cfour_harm_zpve, harm_zpve, 0.1)
    assert psi4.compare_values(cfour_zpve_corr, zpve_corr, 0.1)

def test_h2o_gradient():
    
    psi4.set_memory('32gb')
    psi4.core.set_num_threads(6)

    mol = psi4.geometry("""
    nocom
    noreorient

    O
    H 1 R1
    H 1 R2 2 A
   
    R1    =        0.94731025924472878064
    R2    =        0.94731025924472878064
    A    =      105.5028950965639
    """)

    psi4.set_options({'d_convergence': 1e-12,
                'e_convergence': 1e-12,
                'scf_type': 'direct',
                'puream': True})

    
    options = {'METHOD': 'scf/6-31g*',
            'FD': 'GRADIENT',
            'DISP_SIZE': 0.05}

    cfour_omega = [1826.8154, 4060.2203, 4177.8273]
    cfour_anharmonic = [-54.0635, -158.2345, -177.9707] 
    cfour_harm_zpve = 5032.431
    cfour_zpve_corr = -70.352

    results = pyvpt2.vpt2(mol, options)
    omega = results["Harmonic Freq"][-3:]
    anharmonic = results["Freq Correction"][-3:]
    harm_zpve  = results["Harmonic ZPVE"]
    zpve_corr = results["ZPVE Correction"]

    assert psi4.compare_values(cfour_omega, omega, 0.1)
    assert psi4.compare_values(cfour_anharmonic, anharmonic, 0.1)
    assert psi4.compare_values(cfour_harm_zpve, harm_zpve, 0.1)
    assert psi4.compare_values(cfour_zpve_corr, zpve_corr, 0.1)
    
def test_h2o_hessian():
    
    psi4.set_memory('32gb')
    psi4.core.set_num_threads(6)

    mol = psi4.geometry("""
    nocom
    noreorient

    O
    H 1 R1
    H 1 R2 2 A
   
    R1    =        0.94731025924472878064
    R2    =        0.94731025924472878064
    A    =      105.5028950965639
    """)

    psi4.set_options({'d_convergence': 1e-12,
                'e_convergence': 1e-12,
                'scf_type': 'direct',
                'puream': True})

    
    options = {'METHOD': 'scf/6-31g*',
            'FD': 'HESSIAN',
            'DISP_SIZE': 0.05}

    cfour_omega = [1826.8154, 4060.2203, 4177.8273]
    cfour_anharmonic = [-54.0635, -158.2345, -177.9707] 
    cfour_harm_zpve = 5032.431
    cfour_zpve_corr = -70.352

    results = pyvpt2.vpt2(mol, options)
    omega = results["Harmonic Freq"][-3:]
    anharmonic = results["Freq Correction"][-3:]
    harm_zpve  = results["Harmonic ZPVE"]
    zpve_corr = results["ZPVE Correction"]

    assert psi4.compare_values(cfour_omega, omega, 0.1)
    assert psi4.compare_values(cfour_anharmonic, anharmonic, 0.1)
    assert psi4.compare_values(cfour_harm_zpve, harm_zpve, 0.1)
    assert psi4.compare_values(cfour_zpve_corr, zpve_corr, 0.1)

def test_hf_hessian():
    
    psi4.set_memory('32gb')
    psi4.core.set_num_threads(6)

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
            'FD': 'HESSIAN',
            'DISP_SIZE': 0.05}

    cfour_omega = 4135.3637
    cfour_anharmonic = -153.1174
    cfour_harm_zpve = 2067.682
    cfour_zpve_corr = -13.595

    results = pyvpt2.vpt2(mol, options)
    omega = results["Harmonic Freq"][-1]
    anharmonic = results["Freq Correction"][-1]
    harm_zpve  = results["Harmonic ZPVE"]
    zpve_corr = results["ZPVE Correction"]

    assert psi4.compare_values(cfour_omega, omega, 0.1)
    assert psi4.compare_values(cfour_anharmonic, anharmonic, 0.1)
    assert psi4.compare_values(cfour_harm_zpve, harm_zpve, 0.1)
    assert psi4.compare_values(cfour_zpve_corr, zpve_corr, 0.1)
import pyvpt2
import psi4
import pytest

def test_h2co_vpt2():
    
    mol = psi4.geometry("""
    nocom
    noreorient
    C       
    O 1 R1
    H 1 R2 2 A1
    H 1 R3 2 A2 3 D

    A1 =  122.1401534366
    A2 =  122.1401534366
    D  = -179.9999893363
    R1 =    1.1843393306
    R2 =    1.0915309601
    R3 =    1.0915309601
    """)

    psi4.set_options({'d_convergence': 1e-12,
                'e_convergence': 1e-12,
                'scf_type': 'direct',
                'puream': True,
                'points': 5})

    options = {'METHOD': 'scf/6-31g*',
            'FD': 'HESSIAN',
            'DISP_SIZE': 0.05,
            'FERMI': True}

    ref_omega = [1335.4895, 1382.7834, 1679.7417, 2031.0176, 3157.7172, 3230.2677]
    ref_anharmonic = [-17.4496, -13.0677, -27.0277, -25.5870, -142.8210, -180.0593] 
    ref_harm_zpve = 6408.506
    ref_zpve_corr = -76.325

    results = pyvpt2.vpt2(mol, **options)
    omega = results["Harmonic Freq"][-6:]
    anharmonic = results["Freq Correction"][-6:]
    harm_zpve  = results["Harmonic ZPVE"]
    zpve_corr = results["ZPVE Correction"]
    print(omega)
    print(anharmonic)
    print(harm_zpve)
    print(zpve_corr)

    assert psi4.compare_values(ref_omega, omega, 0.1)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.1)
    assert psi4.compare_values(ref_harm_zpve, harm_zpve, 0.1)
    assert psi4.compare_values(ref_zpve_corr, zpve_corr, 0.1)
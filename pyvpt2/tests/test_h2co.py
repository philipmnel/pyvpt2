import psi4
import pytest

import pyvpt2


def test_h2co_vpt2():

    mol = psi4.geometry("""
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

    qc_kwargs = {'d_convergence': 1e-12,
                'e_convergence': 1e-12,
                'scf_type': 'direct',
                'puream': True,
                'points': 5}

    options = {'FD': 'HESSIAN',
            'DISP_SIZE': 0.05,
            'FERMI': True,
            'GVPT2': True}

    ref_omega = [1335.4895, 1382.7834, 1679.7417, 2031.0176, 3157.7172, 3230.2677]
    ref_deperturbed = [-17.4496, -18.9212, -33.1555, -25.5869, -142.8295, -184.5589]
    ref_anharmonic = [-17.4496, -18.9212, -33.1555, -25.5869, -142.8295, -129.2199]
    ref_harm_zpve = 6408.5086
    ref_zpve_corr = -77.2184

    inp = {
        "molecule": mol.to_schema(dtype=2),
        "keywords": options,
        "input_specification": [{
            "model": {
                "method": "scf",
                "basis": "6-31g*"
            },
            "keywords": qc_kwargs}]
        }

    results = pyvpt2.vpt2_from_schema(inp)
    omega = results.omega[-6:]
    anharmonic = results.nu[-6:] - omega
    harm_zpve  = results.harmonic_zpve
    zpve_corr = results.anharmonic_zpve - harm_zpve
    depertubed = results.extras["Deperturbed Freq"][-6:]
    depertubed = [depertubed[i] - omega[i] for i in range(len(omega))]

    assert psi4.compare_values(ref_omega, omega, 0.1)
    assert psi4.compare_values(ref_anharmonic, anharmonic, 0.1)
    assert psi4.compare_values(ref_deperturbed, depertubed, 0.1)
    assert psi4.compare_values(ref_harm_zpve, harm_zpve, 0.1)
    assert psi4.compare_values(ref_zpve_corr, zpve_corr, 0.1)

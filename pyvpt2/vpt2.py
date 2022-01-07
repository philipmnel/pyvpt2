import psi4
import numpy as np
import qcelemental as qcel
import itertools
import math

from . import quartic

def harmonic(mol, options):
    """
    harmonic: performs harmonic analysis and parses normal modes

    mol: psi4 molecule object
    options: program options dictionary

    harm: harmonic results dictionary
    """

    method = options["METHOD"]

    E0, wfn = psi4.frequency(method, dertype=1, molecule=mol, return_wfn=True)

    omega = wfn.frequency_analysis["omega"].data
    modes = wfn.frequency_analysis["x"].data
    kforce = wfn.frequency_analysis["k"].data
    trv = wfn.frequency_analysis["TRV"].data
    q = wfn.frequency_analysis["q"].data
    n_modes = len(trv)

    wave_to_hartree = qcel.constants.get("inverse meter-hartree relationship") * 100
    meter_to_bohr = qcel.constants.get("Bohr radius")
    joule_to_hartree = qcel.constants.get("hartree-joule relationship")
    mdyneA_to_hartreebohr = 100 * (meter_to_bohr ** 2) / (joule_to_hartree)

    omega = omega.real
    omega_au = omega * wave_to_hartree
    kforce_au = kforce * mdyneA_to_hartreebohr
    modes_unitless = np.copy(modes)
    gamma = omega_au / kforce_au
    v_ind = []

    for i in range(n_modes):
        if trv[i] == "V" and omega[i] != 0.0:
            modes_unitless[:, i] *= np.sqrt(gamma[i])
            v_ind.append(i)
        else:
            modes_unitless[:, i] *= 0.0

    harm = {}
    harm["E0"] = E0
    harm["omega"] = omega
    harm["modes"] = modes
    harm["v_ind"] = v_ind
    harm["n_modes"] = n_modes
    harm["modes_unitless"] = modes_unitless
    harm["gamma"] = gamma
    harm["q"] = q

    return harm


def coriolis(mol, harm):
    """
    coriolis: calculates coriolis coupling constants

    mol: psi4 molecule object
    harm: harmonic results dictionary

    zeta: coriolis coupling constants
    B: equilibrium rotational constants
    """

    q = harm["q"]
    h = qcel.constants.get("Planck constant")
    c = qcel.constants.get("speed of light in vacuum") * 100
    meter_to_bohr = qcel.constants.get("Bohr radius")
    kg_to_amu = qcel.constants.get("atomic mass constant")
    n_atom = mol.natom()

    inertia = mol.inertia_tensor().np
    B = h / (8 * np.pi ** 2 * c * np.diag(inertia))
    B /= kg_to_amu * meter_to_bohr ** 2

    Mxa = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
    Mya = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]])
    Mza = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

    Mx = np.kron(np.eye(mol.natom()), Mxa)
    My = np.kron(np.eye(mol.natom()), Mya)
    Mz = np.kron(np.eye(mol.natom()), Mza)

    zeta = np.zeros((3, 3 * n_atom, 3 * n_atom))
    zeta[0, :, :] = np.matmul(np.transpose(q), np.matmul(Mx, q))
    zeta[1, :, :] = np.matmul(np.transpose(q), np.matmul(My, q))
    zeta[2, :, :] = np.matmul(np.transpose(q), np.matmul(Mz, q))

    return zeta, B


def vpt2(mol, options=None):
    """
    vpt2: performs vibrational pertubration theory calculation

    mol: psi4 molecule object
    options: program options dictionary

    omega: harmonic frequencies
    anharmonic: vpt2 anharmonic corrections
    """

    if options is None:
        options = {}

    options = {k.upper(): v for k, v in sorted(options.items())}

    if "DISP_SIZE" not in options:
        options["DISP_SIZE"] = 0.02
    if "METHOD" not in options:
        options["METHOD"] = "SCF"
    if "FD" not in options:
        options["FD"] = "HESSIAN"

    harm = harmonic(mol, options)
    n_modes = harm["n_modes"]
    omega = harm["omega"]
    v_ind = harm["v_ind"]

    zeta, B = coriolis(mol, harm)

    print("Harmonic analysis successful. Starting quartic force field calculation.")

    if options["FD"] == "ENERGY":
        phi_ijk, phi_iijj = quartic.force_field_E(mol, harm, options)
    elif options["FD"] == "GRADIENT":
        phi_ijk, phi_iijj = quartic.force_field_G(mol, harm, options)
    elif options["FD"] == "HESSIAN":
        phi_ijk, phi_iijj = quartic.force_field_H(mol, harm, options)

    print("\n\nCUBIC:")
    for [i,j,k] in itertools.product(v_ind, repeat=3):
        if abs(phi_ijk[i, j, k]) > 10:
            print(i + 1, j + 1, k + 1, "    ", phi_ijk[i, j, k])

    quartic.check_cubic(phi_ijk, harm)

    print("\nQUARTIC:")
    for [i,j] in itertools.product(v_ind, repeat=2):
        if abs(phi_iijj[i, j]) > 10:
            print(i + 1, i + 1, j + 1, j + 1, "    ", phi_iijj[i, j])

    quartic.check_quartic(phi_iijj, harm)

    print("\nCORIOLIS:")
    for [i,j,k] in itertools.product(range(3), v_ind, v_ind):
        if abs(zeta[i, j, k]) > 1e-5:
            print(i + 1, j + 1, k + 1, "    ", zeta[i, j, k])

    # Identify Fermi resonances:
    print("\nIdentifying Fermi resonances... ")
    fermi1 = np.zeros((n_modes, n_modes), dtype=bool)
    fermi2 = np.zeros((n_modes, n_modes, n_modes), dtype=bool)
    delta_omega_threshold = 100
    delta_K_threshold = 10


    for [i, j] in itertools.combinations(v_ind, 2):
        if abs(2 * omega[i] - omega[j]) <=  delta_omega_threshold:
            if phi_ijk[i,i,j]**4 / (256*(2*omega[i] - omega[j])**3) <= delta_K_threshold:
                fermi1[i,j] = True
                fermi1[j,i] = True
                print("Detected 2(" + str(i+1) + ") = " + str(j+1))

    for [i, j, k] in itertools.combinations(v_ind,3):
        if abs(omega[i] + omega[j] - omega[k]) <= delta_omega_threshold:
            if phi_ijk[i,j,k]**4 / (64* (omega[i] + omega[j] - omega[k])**3) <= delta_K_threshold:
                for [ii,jj,kk] in itertools.permutations([i,j,k]):
                    fermi2[ii,jj,kk] = True
                    print("Detected " + str(i+1) + " + " + str(j+1) + " = " + str(k+1))


    chi = np.zeros((n_modes, n_modes))
    chi0 = 0.0

    for i in v_ind:

        chi0 += phi_iijj[i, i]
        chi0 -= (7 / 9) * phi_ijk[i, i, i] ** 2 / omega[i]

        for j in v_ind:

            if i == j:

                chi[i, i] = phi_iijj[i, i]

                for k in v_ind:

                    if fermi1[i,k]:
                        chi[i,i] -= (phi_ijk[i, i, k] ** 2 ) / 2 * (1 / (2 * omega[i] + omega[k]) + 4 / omega[k])

                    else:
                        chi[i, i] -= ((8 * omega[i] ** 2 - 3 * omega[k] ** 2) * phi_ijk[i, i, k] ** 2) / (omega[k] * (4 * omega[i] ** 2 - omega[k] ** 2))

                chi[i, i] /= 16

            else:

                chi0 += 3 * omega[i]* phi_ijk[i, j, j] ** 2 / (4 * omega[j] ** 2 - omega[i] ** 2)

                chi[i, j] = phi_iijj[i, j]

                rot = 0
                for b_ind in range(0, 3):
                    rot += B[b_ind] * (zeta[b_ind, i, j]) ** 2

                chi[i, j] += (4 * (omega[i] ** 2 + omega[j] ** 2) / (omega[i] * omega[j]) * rot)

                for k in v_ind:

                    chi[i, j] -= (phi_ijk[i, i, k] * phi_ijk[j, j, k]) / omega[k]

                    if fermi2[i, j, k]:
                        temp = 1 / (omega[i] + omega[j] + omega[k])
                        temp += 1 / (-omega[i] + omega[j] + omega[k])
                        temp += 1 / (omega[i] - omega[j] + omega[k])

                        chi[i, j] += (phi_ijk[i, j, k] ** 2) * temp / 2

                    else:
                        delta = omega[i] + omega[j] - omega[k]
                        delta *= omega[i] + omega[j] + omega[k]
                        delta *= omega[i] - omega[j] + omega[k]
                        delta *= omega[i] - omega[j] - omega[k]

                        chi[i, j] += 2 * omega[k] * (omega[i] ** 2 + omega[j] ** 2 - omega[k] ** 2) * phi_ijk[i, j, k] ** 2 / delta

                    if (j > i) and (k > j):
                        chi0 -=  16 * (omega[i] * omega[j] * omega[k] * phi_ijk[i, j, k] ** 2) / delta

                chi[i, j] /= 4

    chi0 /= 64

    zpe = chi0
    for i in v_ind:
        zpe += (1 / 2) * (omega[i] + (1 / 2) * chi[i, i])
        for j in v_ind:
            if j > i:
                zpe += (1 / 4) * chi[i, j]

    print("\n CHI:")
    for i in v_ind:
        for j in v_ind:
            print(i + 1, j + 1, "    ", chi[i, j])

    anharmonic = np.zeros(n_modes)
    overtone = np.zeros(n_modes)
    band = np.zeros((math.comb(len(v_ind), 2), 3))

    for i in v_ind:
        anharmonic[i] = 2 * chi[i, i]
        overtone[i] = 6 * chi[i,i]

        for j in v_ind:
            if j == i: continue
            anharmonic[i] += 0.5 * chi[i, j]
            overtone[i] += chi[i, j]

    for indx, [i, j] in enumerate(itertools.combinations(v_ind, 2)):
        band[indx, 1] = i; band[indx, 2] = j  
        band[indx, 0] = 2 * chi[i, i] + 2 * chi[j, j] + 2 * chi[i, j]
        for k in v_ind:
            if k == i: continue
            elif k == j: continue
            band[indx, 0] += 0.5 * (chi[i,k] + chi[j,k])

    print("\n Fundamentals FREQ (cm-1):")
    for i in v_ind:
        print(i + 1, omega[i], anharmonic[i], (omega[i] + anharmonic[i]), sep="    ")

    print("\n Overtones FREQ (cm-1):")
    for i in v_ind:
        print("2 " + str(i+1), 2*omega[i], overtone[i], (2*omega[i] + overtone[i]), sep="   ") 

    print("\n Combination Bands FREQ (cm-1:")
    for [band_ij, i, j] in band:
        i = int(i); j = int(j)
        print(i+1, j+1, (omega[i] + omega[j]), band_ij, (omega[i] + omega[j] + band_ij), sep="    ")

    print("\n ZPE:")
    print(zpe)

    return omega, anharmonic

# Library imports:
from statistics import harmonic_mean
import psi4
import numpy as np
import itertools
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING, Tuple, Union
from psi4.driver.driver_findif import _findif_schema_to_wfn
from qcelemental.models import AtomicResult
import logging

#Local imports:
from . import quartic
from .constants import *
from . import logconf

logger = logging.getLogger(__name__)

def harmonic(mol: psi4.core.Molecule, options: Dict) -> Dict:
    """
    harmonic: performs harmonic analysis and parses normal modes

    mol: psi4 molecule object
    options: program options dictionary

    harm: harmonic results dictionary
    """

    method = options["METHOD"]
    plan = psi4.hessian(method, dertype=options["FD"], molecule=mol, return_plan = True)
    return plan

def process_harmonic(wfn):

    frequency_analysis = psi4.vibanal_wfn(wfn)
    omega = frequency_analysis["omega"].data
    modes = frequency_analysis["x"].data
    kforce = frequency_analysis["k"].data
    trv = frequency_analysis["TRV"].data
    q = frequency_analysis["q"].data
    n_modes = len(trv)

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

    zpve = np.sum(list(omega[i] for i in v_ind)) / 2

    harm = {}
    harm["E0"] = wfn.energy() # Energy
    harm["G0"] = wfn.gradient().np # Gradient
    harm["H0"] = wfn.hessian().np # Hessian
    harm["omega"] = omega # Frequencies (cm-1)
    harm["modes"] = modes # Un mass weighted normal modes
    harm["v_ind"] = v_ind # Indices of vibrational modes
    harm["n_modes"] = n_modes # Number of vibrational modes
    harm["modes_unitless"] = modes_unitless # Unitless normal modes, used for displacements
    harm["gamma"] = gamma # Unitless scaling factor
    harm["q"] = q # Normalized, mass weighted normal modes, used for coord transformations
    harm["zpve"] = zpve # Zero point vibrational correction 

    return harm


def coriolis(mol: psi4.core.Molecule, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    coriolis: calculates coriolis coupling constants

    mol: psi4 molecule object
    harm: harmonic results dictionary

    zeta: coriolis coupling constants
    B: equilibrium rotational constants
    """

    n_atom = mol.natom()
    # Need to use inertia_tensor() to preserve ordering of axes
    inertiavals, inertiavecs  = np.linalg.eig(mol.inertia_tensor().np)
    # FIXME: ignoring divide error not working; should find a better way to handle linear mols
    with np.errstate(divide = 'ignore'):
        B = np.where(inertiavals == 0.0, 0.0, h / (8 * np.pi ** 2 * c * inertiavals))
    B /= kg_to_amu * meter_to_bohr ** 2

    Mxa = np.matmul(inertiavecs, np.matmul(np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]]), inertiavecs.T))
    Mya = np.matmul(inertiavecs, np.matmul(np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]]), inertiavecs.T))
    Mza = np.matmul(inertiavecs, np.matmul(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), inertiavecs.T))

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
    if "FERMI" not in options:
        options["FERMI"] = True
    if "FERMI_OMEGA_THRESH" not in options:
        options["FERMI_OMEGA_THRESH"] = 200
    if "FERMI_K_THRESH" not in options:
        options["FERMI_K_THRESH"] = 1

    mol.move_to_com()
    mol.fix_com(True)
    mol.fix_orientation(True)
    rotor_type = mol.rotor_type()

    #logconf.init_logging("vpt2_test")

    if (rotor_type in ["RT_LINEAR", "RT_ASYMMETRIC_TOP"]) == False:
        print("Error: pyVPT2 can only be run on linear or asymmetric top molecules.")
        print("Rotor type is " + rotor_type)
        return {}

    plan = harmonic(mol, options)
    if options.get("RETURN_PLAN", False):
        return plan
    else:
        with psi4.p4util.hold_options_state():
            plan.compute()
        harmonic_result = plan.get_results()

    plan = vpt2_from_harmonic(harmonic_result, options)
    plan.compute()
    quartic_result = plan.get_results()
    result_dict = process_vpt2(quartic_result, options)

    return result_dict

def vpt2_from_harmonic(harmonic_result: AtomicResult, options):

    wfn = _findif_schema_to_wfn(harmonic_result)
    harm = process_harmonic(wfn)
    mol = wfn.molecule()

    method = options.pop("METHOD")
    kwargs = {"options": options, "harm": harm}
    plan = quartic.task_planner(method=method, molecule=mol, **kwargs)
    
    return plan

def identify_fermi(omega, phi_ijk, n_modes, v_ind, options):
    # Identify Fermi resonances:
    fermi1 = np.zeros((n_modes, n_modes), dtype=int) # 2*ind1 = ind2
    fermi2 = np.zeros((n_modes, n_modes, n_modes), dtype=int) # ind1 + ind2 = ind3
    fermi_chi_list = np.zeros((n_modes, n_modes), dtype=int) # list of deperturbed chi constants
    delta_omega_threshold = options["FERMI_OMEGA_THRESH"]
    delta_K_threshold = options["FERMI_K_THRESH"]

    if options["FERMI"]:
        print("\nIdentifying Fermi resonances... ")
        for [i, j] in itertools.permutations(v_ind, 2):
            d_omega = abs(2*omega[i] - omega[j])
            if d_omega <=  delta_omega_threshold:
                d_K = phi_ijk[i,i,j]**4 / (256*d_omega**3)
                if d_K >= delta_K_threshold:
                    fermi1[i,j] = True
                    fermi2[i,i,j] = True
                    fermi_chi_list[i,i] = True
                    fermi_chi_list[i,j] = True
                    print("Detected 2(" + str(i+1) + ") = " + str(j+1) + ", d_omega = " + str(d_omega) + ", d_K = " + str(d_K))

        for [i, j, k] in itertools.permutations(v_ind,3):
            d_omega = abs(omega[i] + omega[j] - omega[k])
            if d_omega <= delta_omega_threshold:
                d_K = phi_ijk[i,j,k]**4 / (64* d_omega**3)
                if d_K >= delta_K_threshold:
                    fermi2[i,j,k] = True
                    fermi_chi_list[i,j] = True
                    fermi_chi_list[i,k] = True
                    fermi_chi_list[j,k] = True
                    if fermi2[j,i,k]: continue    # only print once for each resonance
                    print("Detected " + str(i+1) + " + " + str(j+1) + " = " + str(k+1) + ", d_omega = " + str(d_omega) + ", d_K = " + str(d_K))

    if np.sum(fermi_chi_list) == 0:
        print("None detected.")

    return fermi1, fermi2, fermi_chi_list

def process_vpt2(quartic_result: AtomicResult, options):

    mol = psi4.core.Molecule.from_schema(quartic_result.molecule.dict())
    findifrec = quartic_result.extras["findif_record"]
    phi_ijk = findifrec["reference"]["phi_ijk"]
    phi_iijj = findifrec["reference"]["phi_iijj"]

    harm = findifrec["harm"]
    omega = harm["omega"]
    v_ind = harm["v_ind"]
    n_modes = harm["n_modes"]

    zeta, B = coriolis(mol, harm['q'])
    rotor_type = mol.rotor_type()
    fermi1, fermi2, fermi_chi_list = identify_fermi(omega, phi_ijk, n_modes, v_ind, options)

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
                        temp = (phi_ijk[i, i, k] ** 2 ) / 2 
                        temp *= (1 / (2 * omega[i] + omega[k]) + 4 / omega[k])
                        chi[i,i] -= temp
                    else:
                        temp = ((8 * omega[i] ** 2 - 3 * omega[k] ** 2) * phi_ijk[i, i, k] ** 2)
                        temp /= (omega[k] * (4 * omega[i] ** 2 - omega[k] ** 2))
                        chi[i, i] -=  temp 

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
                        delta = 1 / (omega[i] + omega[j] + omega[k])
                        delta += 1 / (-omega[i] + omega[j] + omega[k])
                        delta += 1 / (omega[i] - omega[j] + omega[k])
                        chi[i, j] += (phi_ijk[i, j, k] ** 2) * delta / 2

                    elif fermi2[j, k, i]:
                        delta = 1 / (omega[i] + omega[j] + omega[k])
                        delta += 1 / (omega[i] + omega[j] - omega[k])
                        delta += 1 / (omega[i] - omega[j] + omega[k])
                        chi[i, j] += (phi_ijk[i, j, k] ** 2) * delta / 2

                    elif fermi2[k, i, j]:
                        delta = 1 / (omega[i] + omega[j] + omega[k])
                        delta += 1 / (-omega[i] + omega[j] + omega[k])
                        delta += 1 / (omega[i] + omega[j] - omega[k])
                        chi[i, j] += (phi_ijk[i, j, k] ** 2) * delta / 2

                    else:
                        delta = omega[i] + omega[j] - omega[k]
                        delta *= omega[i] + omega[j] + omega[k]
                        delta *= omega[i] - omega[j] + omega[k]
                        delta *= omega[i] - omega[j] - omega[k]
                        temp = 2 * omega[k] * (omega[i] ** 2 + omega[j] ** 2 - omega[k] ** 2)
                        chi[i, j] +=  temp * phi_ijk[i, j, k] ** 2 / delta

                    if (j > i) and (k > j):
                        chi0 -=  16 * (omega[i] * omega[j] * omega[k] * phi_ijk[i, j, k] ** 2) / delta

                chi[i, j] /= 4

    for b_ind in range(3):
        if rotor_type == "RT_LINEAR": continue
        zeta_sum = 0
        for [i,j] in itertools.combinations(v_ind, 2):
            zeta_sum += (zeta[b_ind, i, j])**2
        chi0 -= 16 * B[b_ind] * (1 + 2*zeta_sum)

    chi0 /= 64

    zpve = chi0
    for i in v_ind:
        zpve += (1 / 2) * (omega[i] + (1 / 2) * chi[i, i])
        for j in v_ind:
            if j > i:
                zpve += (1 / 4) * chi[i, j]

    print("\nAnharmonic Constants (cm-1)")
    rows = [[i+1, j+1, '*' * fermi_chi_list[i,j], chi[i, j]] for [i,j] in itertools.combinations_with_replacement(v_ind,2)]
    for row in rows:
        print("{: >2} {: >2} {: >1} {: >10.4f}".format(*row))

    anharmonic = np.zeros(n_modes)
    overtone = np.zeros(n_modes)
    band = np.zeros((n_modes, n_modes))

    for i in v_ind:
        anharmonic[i] = 2 * chi[i, i]
        overtone[i] = 6 * chi[i,i]

        for j in v_ind:
            if j == i: continue
            anharmonic[i] += 0.5 * chi[i, j]
            overtone[i] += chi[i, j]

    for [i, j] in itertools.combinations(v_ind, 2):
        band[i, j] = 2 * chi[i, i] + 2 * chi[j, j] + 2 * chi[i, j]
        for k in v_ind:
            if k == i: continue
            elif k == j: continue
            band[i, j] += 0.5 * (chi[i,k] + chi[j,k])
        band[j, i] = band[i, j]
    
    result_dict = {}
    result_dict["Harmonic Freq"] = omega.tolist()
    result_dict["Freq Correction"] = anharmonic.tolist()
    result_dict["Anharmonic Freq"] = (omega + anharmonic).tolist()
    result_dict["Harmonic ZPVE"] = harm["zpve"]
    result_dict["ZPVE Correction"] = zpve - harm["zpve"]
    result_dict["Anharmonic ZPVE"] = zpve
    result_dict["Quartic Schema"] = quartic_result.dict()

    print_result(result_dict, v_ind)

    return result_dict

def print_result(result_dict: Dict, v_ind):

    omega = result_dict["Harmonic Freq"]
    anharmonic = result_dict["Anharmonic Freq"]
    harm_zpve = result_dict["Harmonic ZPVE"]
    zpve = result_dict["Anharmonic ZPVE"]
    phi_ijk = result_dict["Quartic Schema"]["extras"]["findif_record"]["reference"]["phi_ijk"]
    phi_iijj = result_dict["Quartic Schema"]["extras"]["findif_record"]["reference"]["phi_iijj"]

    print("\n\nCubic (cm-1):")
    for [i,j,k] in itertools.product(v_ind, repeat=3):
        if abs(phi_ijk[i, j, k]) > 10:
            print(i + 1, j + 1, k + 1, "    ", phi_ijk[i, j, k])

    print("\nQuartic (cm-1):")
    for [i,j] in itertools.product(v_ind, repeat=2):
        if abs(phi_iijj[i, j]) > 10:
            print(i + 1, i + 1, j + 1, j + 1, "    ", phi_iijj[i, j])

    # print("\nB Rotational Constants (cm-1)")
    # print(B[0], B[1], B[2], sep='    ')

    #print("\nCoriolis Constants (cm-1):")
    #for [i,j,k] in itertools.product(range(3), v_ind, v_ind):
        #if abs(zeta[i, j, k]) > 1e-5:
            #print(i + 1, j + 1, k + 1, "    ", zeta[i, j, k])

    #print("\nVPT2 analysis complete...")
    print("\nFundamentals (cm-1):")
    header = ["", "Harmonic", "Anharmonic", "Anharmonic"]
    header2 = ["Mode", "Frequency", "Correction", "Frequency"]
    rows = [[i+1, omega[i], anharmonic[i], omega[i] + anharmonic[i]] for i in v_ind]
    print("{: >8} {: >20} {: >20} {: >20}".format(*header))
    print("{: >8} {: >20} {: >20} {: >20}".format(*header2))
    for row in rows:
        print("{: >8} {: >20.4f} {: >20.4f} {: >20.4f}".format(*row))


    #print("\nOvertones (cm-1):")
    #header = ["", "", "Harmonic", "Anharmonic", "Anharmonic"]
    #header2 = ["", "Mode", "Frequency", "Correction", "Frequency"]
    #rows = [[2, i+1, 2*omega[i], overtone[i], 2*omega[i] + overtone[i]] for i in v_ind]
    #print("{: >3} {: >4} {: >20} {: >20} {: >20}".format(*header))
    #print("{: >3} {: >4} {: >20} {: >20} {: >20}".format(*header2))
    #for row in rows:
        #print("{: >3} {: >4} {: >20.4f} {: >20.4f} {: >20.4f}".format(*row))

    #print("\nCombination Bands (cm-1):")
    #header = ["", "" , "Harmonic", "Anharmonic", "Anharmonic"]
    #header2 = ["", "Mode", "Frequency", "Correction", "Frequency"]
    #rows = [[i+1, j+1, omega[i] + omega[j], band[i,j], omega[i] + omega[j] + band[i,j]] for [i, j] in itertools.combinations(v_ind,2)]
    #print("{: >3} {: >4} {: >20} {: >20} {: >20}".format(*header))
    #print("{: >3} {: >4} {: >20} {: >20} {: >20}".format(*header2))
    #for row in rows:
        #print("{: >3} {: >4} {: >20.4f} {: >20.4f} {: >20.4f}".format(*row))

    print("\nZero-Point Vibrational Energy:")
    header = ["", "Harmonic", "Anharmonic", "Anharmonic"]
    header2 = ["", "ZPVE", "Correction", "ZPVE"]
    unit_list = [["cm-1:", 1], ["kcal/mol:", wave_to_kcal], ["kJ/mol:", wave_to_kj]]
    rows = [[unit_label, factor * harm_zpve, factor * (zpve - harm_zpve), factor * zpve] for [unit_label, factor] in unit_list]
    print("{: >9} {: >20} {: >20} {: >20}".format(*header))
    print("{: >9} {: >20} {: >20} {: >20}".format(*header2))
    for row in rows:
        print("{: >9} {: >20.4f} {: >20.4f} {: >20.4f}".format(*row))
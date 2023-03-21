# Library imports:
import psi4
import numpy as np
import itertools
from typing import Dict, TYPE_CHECKING, Tuple, Union, List
from qcelemental.models import AtomicResult
from psi4.driver.task_base import AtomicComputer
from psi4.driver.driver_cbs import CompositeComputer
from psi4.driver.driver_findif import FiniteDifferenceComputer
import logging

#Local imports:
from . import quartic
from .fermi_solver import fermi_solver, Interaction, State
from .constants import *

logger = logging.getLogger(__name__)

TaskComputers = Union[AtomicComputer, CompositeComputer, FiniteDifferenceComputer]
def harmonic(mol: psi4.core.Molecule, **kwargs) -> TaskComputers:
    """
    Generates plan for harmonic reference calculation

    Parameters
    ----------
    mol : psi4.core.Molecule 
        Input molecule
    
    Returns
    -------
    TaskComputers 
        Computer for reference harmonic calculation 
    """

    method = kwargs["METHOD"]
    dertype = kwargs["FD"]
    plan = psi4.hessian(method, dertype=dertype, molecule=mol, return_plan=True)
    return plan

def process_harmonic(wfn: psi4.core.Wavefunction) -> Dict:
    """
    Parse harmonic reference wavefunction

    Parameters
    ----------
    wfn : psi4.core.Wavefunction
        Wavefunction from Hessian calculation

    Returns
    -------
    Dict
        Dictionary of reference values from harmonic calculation

    """
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
    gamma = [0.0] * n_modes
    v_ind = []

    for i in range(n_modes):
        if trv[i] == "V" and omega[i] != 0.0:
            gamma[i] = omega_au[i] / kforce_au[i]
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
    Calculates coriolis coupling constants

    Parameters
    ----------
    mol : psi4.core.Molecule
        Input molecule
    q : np.ndarray
        Mass-weighted normal modes

    Returns
    -------
    np.ndarray
        Coriolis coupling constants
    np.ndarray
        Equilibrium rotational constants
    """

    n_atom = mol.natom()
    # Need to use inertia_tensor() to preserve ordering of axes
    inertiavals, inertiavecs  = np.linalg.eig(mol.inertia_tensor().np)
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


def process_options_keywords(**kwargs) -> Dict:
    """
    Process input keywords

    Parameters
    ----------
    kwargs: Dict
        keyword arguments

    Returns
    -------
    Dict
        Processed keywords
    """

    if kwargs is None:
        kwargs = {}

    kwargs = {k.upper(): v for k, v in sorted(kwargs.items())}

    kwargs.setdefault("DISP_SIZE", 0.05)
    kwargs.setdefault("METHOD", "SCF")
    kwargs.setdefault("METHOD_2", None)
    kwargs.setdefault("FD", "HESSIAN")
    kwargs.setdefault("FERMI", True)
    kwargs.setdefault("GVPT2", False)
    kwargs.setdefault("FERMI_OMEGA_THRESH", 200)
    kwargs.setdefault("FERMI_K_THRESH", 1)
    kwargs.setdefault("RETURN_PLAN", False)

    return kwargs

def _findif_schema_to_wfn(findif_model: AtomicResult) -> psi4.core.Wavefunction:
    """
    Helper function to produce Wavefunction and Psi4 files from a FiniteDifference-flavored AtomicResult.
    Some changes from psi4 internal to work with QCFractal next branch.
    """

    # new skeleton wavefunction w/mol, highest-SCF basis (just to choose one), & not energy
    mol = psi4.core.Molecule.from_schema(findif_model.molecule.dict(), nonphysical=True)
    sbasis = "def2-svp" if (findif_model.model.basis == "(auto)") else findif_model.model.basis
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", sbasis, quiet=True)
    wfn = psi4.core.Wavefunction(mol, basis)
    if hasattr(findif_model.provenance, "module"):
        wfn.set_module(findif_model.provenance.module)

    # setting CURRENT E/G/H on wfn below catches Wfn.energy_, gradient_, hessian_
    # setting CURRENT E/G/H on core below is authoritative P::e record
    for qcv, val in findif_model.extras["qcvars"].items():
        if qcv in ["CURRENT DIPOLE", "SCF DIPOLE"]:
            val = np.array(val).reshape(-1,1)
        for obj in [psi4.core, wfn]:
            obj.set_variable(qcv, val)

    return wfn


def vpt2(mol: psi4.core.Molecule, **kwargs) -> Dict:
    """
    Performs vibrational pertubration theory calculation

    Parameters
    ----------
    mol: psi4.core.Molecule
        Input molecule    

    Returns
    -------
    Dict
        VPT2 results dictionary
    """

    kwargs = process_options_keywords(**kwargs)

    mol.move_to_com()
    mol.fix_com(True)
    mol.fix_orientation(True)
    rotor_type = mol.rotor_type()

    if not (rotor_type in ["RT_LINEAR", "RT_ASYMMETRIC_TOP"]):
        raise Exception("pyVPT2 can only be run on linear or asymmetric tops. Detected rotor type is {}".format(rotor_type))

    plan = harmonic(mol, **kwargs)
    if kwargs.get("RETURN_PLAN", False):
        return plan
    else:
        with psi4.p4util.hold_options_state():
            plan.compute()
        harmonic_result = plan.get_results()

    plan = vpt2_from_harmonic(harmonic_result, **kwargs)
    plan.compute()
    quartic_result = plan.get_results()
    result_dict = process_vpt2(quartic_result, **kwargs)

    return result_dict

def vpt2_from_harmonic(harmonic_result: AtomicResult, **kwargs) -> quartic.QuarticComputer:
    """
    Peforms VPT2 calculation starting from harmonic results

    Parameters
    ----------
    harmonic_result : AtomicResult
        Result from reference hessian calculation

    Returns
    -------
    QuarticComputer
        Computer for quartic finite difference calculation
    """
    kwargs = process_options_keywords(**kwargs)
    wfn = _findif_schema_to_wfn(harmonic_result)
    harm = process_harmonic(wfn)
    mol = wfn.molecule()

    method = kwargs.get("METHOD2", kwargs.get("METHOD")) # If no method2, then method (default)
    kwargs = {"options": kwargs, "harm": harm}
    plan = quartic.task_planner(method=method, molecule=mol, **kwargs)
    
    return plan

def identify_fermi(omega: np.ndarray, phi_ijk: np.ndarray, n_modes: np.ndarray, v_ind:  np.ndarray, **kwargs) -> List:
    """
    Identify Fermi resonances

    Parameters
    ----------
    omega : np.ndarray
        Harmonic freqs
    phi_ijk : np.ndarray
        Cubic force constants
    n_modes : np.ndarray
        Number of modes
    v_ind : np.ndarray
        Vibrational indices

    Returns
    -------
    List
        List of Fermi resonant interactions
    """
    # Identify Fermi resonances:
    fermi_chi_list = np.zeros((n_modes, n_modes), dtype=int) # list of deperturbed chi constants
    delta_omega_threshold = kwargs.get("FERMI_OMEGA_THRESH")
    delta_K_threshold = kwargs.get("FERMI_K_THRESH")
    fermi_list = []

    if kwargs.get("FERMI"):
        print("\nIdentifying Fermi resonances... ")
        for [i, j] in itertools.permutations(v_ind, 2):
            d_omega = abs(2*omega[i] - omega[j])
            if d_omega <=  delta_omega_threshold:
                d_K = phi_ijk[i,i,j]**4 / (256*d_omega**3)
                if d_K >= delta_K_threshold:
                    fermi_list.append((j, (i,i)))
                    fermi_chi_list[i,i] = True
                    fermi_chi_list[i,j] = True
                    print("Detected 2(" + str(i+1) + ") = " + str(j+1) + ", d_omega = " + str(d_omega) + ", d_K = " + str(d_K))

        for [i, j, k] in itertools.permutations(v_ind,3):
            d_omega = abs(omega[i] + omega[j] - omega[k])
            if d_omega <= delta_omega_threshold:
                d_K = phi_ijk[i,j,k]**4 / (64* d_omega**3)
                if d_K >= delta_K_threshold:
                    fermi_list.append((k, (i,j)))
                    fermi_chi_list[i,j] = True
                    fermi_chi_list[i,k] = True
                    fermi_chi_list[j,k] = True
                    print("Detected " + str(i+1) + " + " + str(j+1) + " = " + str(k+1) + ", d_omega = " + str(d_omega) + ", d_K = " + str(d_K))

    if np.sum(fermi_chi_list) == 0:
        print("None detected.")

    return fermi_list

def process_vpt2(quartic_result: AtomicResult, **kwargs) -> Dict:
    """
    Calculate VPT2 results from quartic forces

    Parameters
    ----------
    quartic_result : AtomicResult
        Results from quartic finite difference calculation
    
    Returns
    -------
    Dict
        Results dictionary for VPT2 caclutation
    """
    kwargs = process_options_keywords(**kwargs)
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
    fermi_list = identify_fermi(omega, phi_ijk, n_modes, v_ind, **kwargs)

    chi = np.zeros((n_modes, n_modes))
    chi0 = 0.0

    for i in v_ind:

        chi0 += phi_iijj[i, i]
        chi0 -= (7 / 9) * phi_ijk[i, i, i] ** 2 / omega[i]

        for j in v_ind:
            if i == j:
                chi[i, i] = phi_iijj[i, i]

                for k in v_ind:
                    if (k,(i,i)) in fermi_list:
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

                    if (k,(i,j)) in fermi_list:
                        # i + j = k
                        delta_ij = 1 / (omega[i] + omega[j] + omega[k])
                        #delta_ij -= 1 / (omega[i] + omega[j] - omega[k]) deperturbed
                        delta_ij += 1 / (-omega[i] + omega[j] + omega[k])
                        delta_ij += 1 / (omega[i] - omega[j] + omega[k])
                        delta_ij /= -2

                        delta_0 = 1 / (omega[i] + omega[j] + omega[k])
                        #delta_0 -= 1 / (omega[i] + omega[j] - omega[k]) deperturbed
                        delta_0 -= 1 / (omega[i] - omega[j] + omega[k])
                        delta_0 -= 1 / (-omega[i] + omega[j] + omega[k])

                    elif (i,(j,k)) in fermi_list:
                        # j + k = i
                        delta_ij = 1 / (omega[i] + omega[j] + omega[k])
                        delta_ij -= 1 / (omega[i] + omega[j] - omega[k])
                        #delta_ij += 1 / (-omega[i] + omega[j] + omega[k]) deperturbed
                        delta_ij += 1 / (omega[i] - omega[j] + omega[k])
                        delta_ij /= -2

                        delta_0 = 1 / (omega[i] + omega[j] + omega[k])
                        delta_0 -= 1 / (omega[i] + omega[j] - omega[k])
                        delta_0 -= 1 / (omega[i] - omega[j] + omega[k])
                        #delta_0 -= 1 / (-omega[i] + omega[j] + omega[k]) deperturbed

                    elif (j,(i,k)) in fermi_list:
                        # k + i = j
                        delta_ij = 1 / (omega[i] + omega[j] + omega[k])
                        delta_ij -= 1 / (omega[i] + omega[j] - omega[k])
                        delta_ij += 1 / (-omega[i] + omega[j] + omega[k])
                        #delta_ij += 1 / (omega[i] - omega[j] + omega[k]) deperturbed
                        delta_ij /= -2

                        delta_0 = 1 / (omega[i] + omega[j] + omega[k])
                        delta_0 -= 1 / (omega[i] + omega[j] - omega[k])
                        #delta_0 -= 1 / (omega[i] - omega[j] + omega[k]) deperturbed
                        delta_0 -= 1 / (-omega[i] + omega[j] + omega[k])

                    else:
                        delta = omega[i] + omega[j] - omega[k]
                        delta *= omega[i] + omega[j] + omega[k]
                        delta *= omega[i] - omega[j] + omega[k]
                        delta *= omega[i] - omega[j] - omega[k]
                        delta_ij = 2 * omega[k] * (omega[i] ** 2 + omega[j] ** 2 - omega[k] ** 2) / delta
                        delta_0 = -8 * (omega[i] * omega[j] * omega[k]) / delta

                    chi[i,j] += phi_ijk[i,j,k]**2 * delta_ij

                    if (j > i) and (k > j):
                        chi0 +=  2 * phi_ijk[i, j, k] ** 2 * delta_0

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
    rows = [[i+1, j+1, chi[i, j]] for [i,j] in itertools.combinations_with_replacement(v_ind,2)]
    for row in rows:
        print("{: >2} {: >2} {: >10.4f}".format(*row))

    anharmonic = np.zeros(n_modes)
    overtone = np.zeros(n_modes)
    band = np.zeros((n_modes, n_modes))

    for i in v_ind:
        anharmonic[i] = omega[i] + 2 * chi[i, i]
        overtone[i] = 2*omega[i] + 6 * chi[i,i]

        for j in v_ind:
            if j == i: continue
            anharmonic[i] += 0.5 * chi[i, j]
            overtone[i] += chi[i, j]

    for [i, j] in itertools.combinations(v_ind, 2):
        band[i, j] = omega[i] + omega[j] + 2 * chi[i, i] + 2 * chi[j, j] + 2 * chi[i, j]
        for k in v_ind:
            if k == i: continue
            elif k == j: continue
            band[i, j] += 0.5 * (chi[i,k] + chi[j,k])
        band[j, i] = band[i, j]

    if kwargs["FERMI"] and kwargs["GVPT2"]:
        deperturbed = anharmonic.copy()
        fermi_nu, fermi_ind = process_fermi_solver(fermi_list, v_ind, anharmonic, overtone, band, phi_ijk)
        #for ind in fermi_ind:
        #    anharmonic[ind] = fermi_nu[ind]
        # this was already done in-place in process_fermi_solver
        
    result_dict = {}
    result_dict["Harmonic Freq"] = omega.tolist()
    result_dict["Freq Correction"] = (anharmonic - omega).tolist()
    result_dict["Anharmonic Freq"] = anharmonic.tolist()
    result_dict["Harmonic ZPVE"] = harm["zpve"]
    result_dict["ZPVE Correction"] = zpve - harm["zpve"]
    result_dict["Anharmonic ZPVE"] = zpve
    result_dict["Quartic Schema"] = quartic_result

    if kwargs["FERMI"] and kwargs["GVPT2"]:
        result_dict["Deperturbed Freq"] = deperturbed.tolist()

    print_result(result_dict, v_ind)
    result_dict["Quartic Schema"] = quartic_result.dict(encoding="json")

    return result_dict

def process_fermi_solver(fermi_list: List, v_ind: List, nu: np.ndarray, overtone:np.ndarray, band:np.ndarray, phi_ijk:np.ndarray) -> Tuple[np.ndarray, List]:
    """
    Process deperturbed results into fermi solver

    Parameters
    ----------
    fermi_list : list
        List of fermi resonances
    v_ind : list    
        List of vibrational indices
    nu : np.ndarray
        Anharmonic deperturbed frequencies
    overtone : np.ndarray
        Overtone frequencies
    band : np.ndarray
        Combination band frequencies
    phi_ijk : np.ndarray
        Cubic force constants

    Returns
    -------
    np.ndarray
        Variationally corrected frequencies
    List
        Indices of variationally corrected frequencies

    """    

    interaction_list = []
    # process fermi1 resonances
    for [i, j] in itertools.permutations(v_ind, 2):
        if (i, (j,j)) in fermi_list:
            interaction = Interaction(left=State(state=(i,), nu=nu[i]), right=State(state=(j,j), 
                            nu=overtone[j]), phi=phi_ijk[i, j, j], ftype=1)
            interaction_list.append(interaction)

    # process fermi2 resonances
    for [i, j, k] in itertools.permutations(v_ind, 3):
        if (i, (j,k)) in fermi_list:
            if j > k: continue # avoid double counting

            interaction = Interaction(left=State(state=(i,), nu=nu[i]), right=State(state=(j,k), 
                            nu=band[j,k]), phi=phi_ijk[i, j, k], ftype=2)
            interaction_list.append(interaction)

    state_list = fermi_solver(interaction_list)

    updated_indices = []
    for key, value in state_list.items():
        if len(key) == 1:
            # update nu vals
            nu[key[0]] = value
            updated_indices.append(key[0])

    return nu, updated_indices

def print_result(result_dict: Dict, v_ind: np.ndarray):
    """
    Prints VPT2 results

    Parameters
    ----------
    result_dict : dict
        Dictionary of VPT2 results
    v_ind : np.ndarray
        List of vibrational indices
    """

    omega = result_dict["Harmonic Freq"]
    anharmonic = result_dict["Freq Correction"]
    harm_zpve = result_dict["Harmonic ZPVE"]
    zpve = result_dict["Anharmonic ZPVE"]
    phi_ijk = result_dict["Quartic Schema"].dict()["extras"]["phi_ijk"]
    phi_iijj = result_dict["Quartic Schema"].dict()["extras"]["phi_iijj"]

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

# Library imports:
import itertools
import logging
import math
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import numpy as np
import psi4
from psi4.driver.driver_cbs import CompositeComputer
from psi4.driver.driver_findif import FiniteDifferenceComputer
from qcelemental.models import AtomicResult
from qcelemental.models.procedures import QCInputSpecification

#Local imports:
from . import quartic
from .constants import *
from .fermi_solver import Interaction, State, fermi_solver
from .schema import VPTInput, VPTResult, provenance_stamp
from .task_base import AtomicComputer
from .task_planner import hessian_planner, quartic_planner

logger = logging.getLogger(f"psi4.{__name__}")

TaskComputers = Union[AtomicComputer, CompositeComputer, FiniteDifferenceComputer]

def _findif_schema_to_wfn(findif_model: AtomicResult) -> psi4.core.Wavefunction:
    """Helper function to produce Wavefunction and Psi4 files from a FiniteDifference-flavored AtomicResult."""

    # new skeleton wavefunction w/mol, highest-SCF basis (just to choose one), & not energy
    mol = psi4.core.Molecule.from_schema(findif_model.molecule.dict(), nonphysical=True)
    sbasis = "def2-svp" if (findif_model.model.basis == "(auto)") else findif_model.model.basis
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", sbasis, quiet=True)
    wfn = psi4.core.Wavefunction(mol, basis)
    if hasattr(findif_model.provenance, "module"):
        wfn.set_module(findif_model.provenance.module)

    # setting CURRENT E/G/H on wfn below catches Wfn.energy_, gradient_, hessian_
    # setting CURRENT E/G/H on core below is authoritative P::e record
    for obj in [psi4.core, wfn]:

        ret_e = findif_model.properties.return_energy
        ret_g = findif_model.properties.return_gradient
        ret_h = findif_model.properties.return_hessian
        if ret_e is not None:
            obj.set_variable("CURRENT ENERGY", ret_e)
        if ret_g is not None:
            obj.set_variable("CURRENT GRADIENT", ret_g)
        if ret_h is not None:
            obj.set_variable("CURRENT HESSIAN", ret_h)
        dipder = findif_model.extras["qcvars"].get("CURRENT DIPOLE GRADIENT", None)
        if dipder is not None:
            obj.set_variable("CURRENT DIPOLE GRADIENT", dipder)

    return wfn

def process_harmonic(wfn: psi4.core.Wavefunction, **kwargs) -> Dict:
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
    if intensities := frequency_analysis.get("IR_intensities", None):
        intensities = intensities.data
    n_modes = len(trv)
    omega_thresh = kwargs.get("VPT2_OMEGA_THRESH")

    omega = omega.real
    omega_au = omega * wave_to_hartree
    kforce_au = kforce * mdyneA_to_hartreebohr
    modes_unitless = np.copy(modes)
    gamma = [0.0] * n_modes
    v_ind = []
    v_ind_omit = []

    for i in range(n_modes):
        if trv[i] == "V":
            if omega[i] < omega_thresh:
                v_ind_omit.append(i)
            else:
                gamma[i] = omega_au[i] / kforce_au[i]
                modes_unitless[:, i] *= np.sqrt(gamma[i])
                v_ind.append(i)
        else:
            modes_unitless[:, i] *= 0.0

    zpve = np.sum(list(omega[i] for i in (v_ind + v_ind_omit))) / 2

    harm = {}
    harm["E0"] = wfn.energy() # Energy
    G0 = wfn.gradient()
    if G0 is not None:
        harm["G0"] = G0.np # Gradient
    harm["H0"] = wfn.hessian().np # Hessian
    harm["omega"] = omega # Frequencies (cm-1)
    harm["modes"] = modes # Un mass weighted normal modes
    harm["v_ind"] = v_ind # Indices of vibrational modes
    harm["v_ind_omitted"] = v_ind_omit # Indices of vibrational modes omitted from VPT2 treatment
    harm["n_modes"] = n_modes # Number of vibrational modes
    harm["modes_unitless"] = modes_unitless # Unitless normal modes, used for displacements
    harm["gamma"] = gamma # Unitless scaling factor
    harm["q"] = q # Normalized, mass weighted normal modes, used for coord transformations
    harm["zpve"] = zpve # Zero point vibrational correction
    harm["intensities"] = intensities # Harmonic IR intensities (km mol-1)

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
        B = np.where(inertiavals < 1e-4, 0.0, h / (8 * np.pi ** 2 * c * inertiavals))
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

    keyword_defaults = {
        "DISP_SIZE": 0.05,
        "FD": "HESSIAN",
        "FERMI": True,
        "GVPT2": False,
        "FERMI_OMEGA_THRESH": 200,
        "FERMI_K_THRESH": 1,
        "RETURN_PLAN": False,
        "VPT2_OMEGA_THRESH": 1,
        "QC_PROGRAM": "psi4",
        "MULTILEVEL": False,
    }

    for k in kwargs.keys():
        if k not in keyword_defaults.keys():
            raise Warning(f"Ignoring unknown keyword {k}")

    for k, v in keyword_defaults.items():
        kwargs.setdefault(k, v)


    return kwargs

def check_rotor(mol: psi4.core.Molecule):
    """
    Check if rotor type is linear or asymmetric top.
    Can't use the built-in psi4.core.Molecule function since its degeneracy
    tolerance is way too tight. If I switched to qcdb molecule objects in
    the future, this can probably be removed.

    Parameters
    ----------
    mol: psi4.core.Molecule
        Input molecule

    Returns
    -------
    str
        rotor type
    """

    tol = 1e-6
    rot_const = mol.rotational_constants().np
    #inertia_tensor = mol.inertia_tensor().np
    #inertiavals, inertiavecs  = np.linalg.eig(inertia_tensor)
    #with np.errstate(divide = 'ignore'):
        #B = np.where(inertiavals == 0.0, 0.0, h / (8 * np.pi ** 2 * c * inertiavals))
    #rot_const /= kg_to_amu * meter_to_bohr ** 2

    for i in range(3):
        if rot_const[i] is None:
            rot_const[i] = 0.0

    # Determine degeneracy of rotational constants.
    degen = 0
    for i in range(2):
        for j in range(i + 1, 3):
            if degen >= 2:
                continue
            rabs = math.fabs(rot_const[i] - rot_const[j])
            tmp = rot_const[i] if rot_const[i] > rot_const[j] else rot_const[j]
            if rabs > 1e-14:
                rel = rabs / tmp
            else:
                rel = 0.0
            if rel < tol:
                degen += 1

    # Determine rotor type
    if mol.natom() == 1:
        rotor_type = 'RT_ATOM'
    elif rot_const[0] == 0.0:
        rotor_type = 'RT_LINEAR'          # 0  <  IB == IC      inf > B == C
    elif degen == 2:
        rotor_type = 'RT_SPHERICAL_TOP'   # IA == IB == IC       A == B == C
    elif degen == 1:
        rotor_type = 'RT_SYMMETRIC_TOP'   # IA <  IB == IC       A >  B == C --or--
                                            # IA == IB <  IC       A == B >  C
    else:
        rotor_type = 'RT_ASYMMETRIC_TOP'  # IA <  IB <  IC       A >  B >  C
    return rotor_type

def vpt2_from_schema(inp: VPTInput) -> VPTResult:

    from qcelemental.models.molecule import Molecule

    if isinstance(inp, dict):
        inp = VPTInput(**inp)
    elif isinstance(inp, VPTInput):
        inp = inp.copy()
    else:
        raise AssertionError("Input type not recognized.")


    mol = inp.molecule
    kwargs = process_options_keywords(**inp.keywords)
    qc_specification = inp.input_specification[0]
    if len(inp.input_specification) == 1:
        qc_specification2 = inp.input_specification[0]
    else:
        qc_specification2 = inp.input_specification[1]
        kwargs.update({"MULTILEVEL": True})

    mol = mol.orient_molecule()
    mol = mol.dict()
    mol.update({"fix_com": True, "fix_orientation": True})
    mol = Molecule(**mol)
    #rotor_type = check_rotor(mol)

    plan = hessian_planner(mol, qc_specification, **kwargs)

    if kwargs.get("RETURN_PLAN", False):
        return plan
    else:
        with psi4.p4util.hold_options_state():
            plan.compute()
        harmonic_result = plan.get_results()

    plan = vpt2_from_harmonic(harmonic_result, qc_specification2, **kwargs)
    plan.compute()
    quartic_result = plan.get_results()
    result_dict = process_vpt2(quartic_result, **kwargs)

    return result_dict


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
    rotor_type = check_rotor(mol)

    #if not (rotor_type in ["RT_LINEAR", "RT_ASYMMETRIC_TOP"]):
    #    raise Exception("pyVPT2 can only be run on linear or asymmetric tops. Detected rotor type is {}".format(rotor_type))

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

def vpt2_from_harmonic(harmonic_result: AtomicResult, qc_spec, **kwargs) -> quartic.QuarticComputer:
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
    if isinstance(qc_spec, dict):
        qc_spec = QCInputSpecification(**qc_spec)
    elif isinstance(qc_spec, QCInputSpecification):
        qc_spec = qc_spec.copy()
    else:
        raise AssertionError("Input type not recognized.")

    kwargs = process_options_keywords(**kwargs)
    wfn = _findif_schema_to_wfn(harmonic_result)
    harm = process_harmonic(wfn, **kwargs)
    mol = wfn.molecule()

    kwargs = {"options": kwargs, "harm": harm}
    plan = quartic_planner(molecule=mol, qc_spec=qc_spec, **kwargs)

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

    degeneracy = np.ones(n_modes)
    v_ind_nondegen = v_ind.copy() #
    v_ind_degen = [] # degenerate modes
    degen_mode_map = {}
    # find degenerate modes:
    if check_rotor(mol) == "RT_LINEAR":
        # Only do degeneracies on linear mols for now
        for i,j in itertools.combinations(v_ind, 2):
            if abs(omega[i] - omega[j]) < 0.2: #TODO: tolerance value probably not ideal
                degeneracy[i] += 1
                v_ind_degen.append(i)
                v_ind_nondegen.remove(i)
                v_ind_nondegen.remove(j)
                # This assumes only linear mols (degeneracy of 2)
                degen_mode_map[i] = j

    v_ind_all = v_ind_nondegen.copy()
    v_ind_all.extend(v_ind_degen)
    zeta, B = coriolis(mol, harm['q'])
    rotor_type = mol.rotor_type()
    fermi_list = identify_fermi(omega, phi_ijk, n_modes, v_ind, **kwargs)

    chi = np.zeros((n_modes, n_modes))
    g = np.zeros((n_modes, n_modes))
    chi0 = 0.0

    # loop to solve non-degenerate anharmonicities
    for i in v_ind_nondegen:

        # TODO: Fix linear ZPVEs
        chi0 += phi_iijj[i, i]
        chi0 -= (7 / 9) * phi_ijk[i, i, i] ** 2 / omega[i]

        for j in v_ind_nondegen:
            if i == j:
                chi[i, i] = phi_iijj[i, i]

                for k in v_ind_nondegen:
                    if (k, (i,i)) in fermi_list:
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

                for k in v_ind_nondegen:
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

        for j in v_ind_degen:
            chi[i,j] = phi_iijj[i,j]

            #TODO: handle multiple degeneracies
            if (i,(j,j)) in fermi_list:
                temp = 2 * omega[j] * phi_ijk[i,j,j]**2
                temp *= 1 / (2*omega[j] + omega[i])
                temp /= 4 * omega[j]
                chi[i,j] -= temp

            else:
                temp = 2 * omega[j] * phi_ijk[i,j,j]**2
                temp /= (4*omega[j]**2 - omega[i]**2)
                chi[i,j] -= temp

            for k in v_ind_nondegen:
                chi[i,j] -= phi_ijk[i,i,k] * phi_ijk[j,j,k] / omega[k]

            rot = 0
            for b_ind in range(0, 3):
                rot += B[b_ind] * (zeta[b_ind, i, j]) ** 2
            chi[i, j] += 4 * rot * (omega[i] ** 2 + omega[j] ** 2) / (omega[i] * omega[j])
            chi[i, j] /= 4
            chi[j, i] = chi[i, j]

    for i in v_ind_degen:
        for j in v_ind_degen:
            if i == j:
                chi[i, i] = phi_iijj[i, i]
                g[i, i] = -1/3 * phi_iijj[i, i]
                g[i, i] += 7/3 * phi_ijk[i, i, i]**2 / omega[i]

                for k in v_ind_nondegen:
                    if (k,(i,i)) in fermi_list:
                        temp = omega[k] * phi_ijk[k, i, i]**2
                        temp *= 1 / (2*omega[i] + omega[k])
                        temp /= 4 * omega[i]
                        g[i, i] -= temp

                    else:
                        temp = omega[k] * phi_ijk[k, i, i]**2
                        temp /= (4*omega[i]**2 - omega[k]**2)
                        g[i, i] -= temp


                for k in v_ind_all:
                    if (k,(i,i)) in fermi_list:
                        temp = (phi_ijk[i, i, k] ** 2 ) / 2
                        temp *= (1 / (2 * omega[i] + omega[k]) + 4 / omega[k])
                        chi[i,i] -= temp
                    else:
                        temp = ((8 * omega[i] ** 2 - 3 * omega[k] ** 2) * phi_ijk[i, i, k] ** 2)
                        temp /= (omega[k] * (4 * omega[i] ** 2 - omega[k] ** 2))
                        chi[i, i] -=  temp

                chi[i,i] /= 16
                g[i, i] /= 16

            else:
                # TODO: multiple degeneracies
                pass

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

    for i in v_ind_all:

        anharmonic[i] = omega[i] + (1 + degeneracy[i]) * chi[i, i] + g[i,i]
        overtone[i] = 2*omega[i] + 2 * (2 + degeneracy[i]) * chi[i,i]

        for j in v_ind_all:
            if j == i: continue
            anharmonic[i] += 0.5 * chi[i, j] * degeneracy[j]
            overtone[i] += chi[i, j] * degeneracy[j]

        if i in v_ind_degen:
            j = degen_mode_map[i]
            anharmonic[j] = anharmonic[i]

    for [i, j] in itertools.combinations(v_ind, 2):
        band[i, j] = omega[i] + omega[j] + 2 * chi[i, i] + 2 * chi[j, j] + 2 * chi[i, j]
        for k in v_ind:
            if k == i: continue
            elif k == j: continue
            band[i, j] += 0.5 * (chi[i,k] + chi[j,k])
        band[j, i] = band[i, j]

    if (v_ind_omit := harm["v_ind_omitted"]):
        for i in v_ind_omit:
            zpve += 1/2 * omega[i]
            anharmonic[i] = omega[i]

    extras = {}
    if kwargs["FERMI"] and kwargs["GVPT2"]:
        deperturbed = anharmonic.copy()
        extras.update({"Deperturbed Freq": deperturbed})
        fermi_nu, fermi_ind = process_fermi_solver(fermi_list, v_ind, anharmonic, overtone, band, phi_ijk)

    ret = VPTResult(
        molecule = quartic_result.molecule,
        model = quartic_result.model,
        keywords = kwargs,
        omega = omega,
        nu = anharmonic,
        harmonic_zpve = harm["zpve"],
        anharmonic_zpve = zpve,
        harmonic_intensity=harm["intensities"],
        chi = chi,
        phi_ijk = phi_ijk,
        phi_iijj = phi_iijj,
        rotational_constants = B,
        zeta = zeta,
        extras = extras,
        provenance = provenance_stamp()
        )

    v_ind_print = harm["v_ind_omitted"]
    v_ind_print.extend(v_ind)
    print_result(ret, v_ind_print)
    return ret

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

def print_result(results: VPTResult, v_ind: np.ndarray):
    """
    Prints VPT2 results

    Parameters
    ----------
    results : VPT2
        Dataclass of VPT2 results
    v_ind : np.ndarray
        List of vibrational indices
    """

    omega = results.omega
    anharmonic = results.nu
    harm_zpve = results.harmonic_zpve
    zpve = results.anharmonic_zpve
    phi_ijk = results.phi_ijk
    phi_iijj = results.phi_iijj
    B = results.rotational_constants
    zeta = results.zeta

    print("\n\nCubic (cm-1):")
    for [i,j,k] in itertools.product(v_ind, repeat=3):
        if abs(phi_ijk[i, j, k]) > 10:
            print(i + 1, j + 1, k + 1, "    ", phi_ijk[i, j, k])

    print("\nQuartic (cm-1):")
    for [i,j] in itertools.product(v_ind, repeat=2):
        if abs(phi_iijj[i, j]) > 10:
            print(i + 1, i + 1, j + 1, j + 1, "    ", phi_iijj[i, j])

    print("\nB Rotational Constants (cm-1)")
    print(B[0], B[1], B[2], sep='    ')

    print("\nCoriolis Constants (cm-1):")
    for [i,j,k] in itertools.product(range(3), v_ind, v_ind):
        if abs(zeta[i, j, k]) > 1e-5:
            print(i + 1, j + 1, k + 1, "    ", zeta[i, j, k])

    #print("\nVPT2 analysis complete...")
    print("\nFundamentals (cm-1):")
    header = ["", "Harmonic", "Anharmonic", "Anharmonic"]
    header2 = ["Mode", "Frequency", "Correction", "Frequency"]
    rows = [[i+1, omega[i], anharmonic[i] - omega[i], anharmonic[i]] for i in v_ind]
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

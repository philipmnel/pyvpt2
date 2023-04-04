# Library imports:
import copy
import itertools
import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import psi4
from psi4.driver.driver_cbs import CompositeComputer, composite_procedures
from psi4.driver.task_base import AtomicComputer, BaseComputer
from psi4.driver.task_planner import expand_cbs_methods
from pydantic import validator
from qcelemental.models import AtomicResult, DriverEnum

# Local imports:
from .constants import wave_to_hartree

if TYPE_CHECKING:
    import qcportal

logger = logging.getLogger(__name__)

def check_cubic(phi_ijk: np.ndarray, v_ind: List[int]) -> np.ndarray:
    """
    Checks cubic force constants for any numerical inconsistencies, symmetrizes results, 
    prints results to output file

    Parameters
    ----------
    phi_ijk : np.ndarray
        Array of cubic force constants
    v_ind : List[int]
        List of vibrational indices

    Returns
    -------
    np.ndarray
        Symmetrized array of cubic force constants
    """

    no_inconsistency = True

    print("Checking for numerical inconsistencies in cubic terms...")
    for unique_ijk in itertools.combinations_with_replacement(v_ind, 3):
        for [ind1, ind2] in itertools.combinations(set(itertools.permutations(unique_ijk, 3)), 2):
            diff = abs(phi_ijk[ind1] - phi_ijk[ind2])
            if diff >= 1.0:
                print(ind1, ind2, diff)
                no_inconsistency = False

    if no_inconsistency:
        print("No inconsistencies found")

    sym_phi_ijk = np.zeros_like(phi_ijk)
    for ijk_permutes in itertools.permutations(range(3)):
        sym_phi_ijk += phi_ijk.transpose(ijk_permutes)

    return sym_phi_ijk / 6

def check_quartic(phi_iijj: np.ndarray, v_ind: List) -> np.ndarray:
    """
    Checks quartic force constants for any numerical inconsistencies, symmetrizes results,
    prints results to output file

    Parameters
    ----------
    phi_iijj : np.ndarray
        Array of quartic force constants
    v_ind : List[int]
        List of vibrational indices

    Returns
    -------
    np.ndarray
        Symmetrized array of quartic force constants
    """

    no_inconsistency = True

    print("Checking for numerical inconsistencies in quartic terms...")
    for unique_ij in itertools.combinations_with_replacement(v_ind, 2):
        for [ind1, ind2] in itertools.combinations(set(itertools.permutations(unique_ij, 2)), 2):
            diff = abs(phi_iijj[ind1] - phi_iijj[ind2])
            if diff >= 1.0:
                print(ind1, ind2, diff)
                no_inconsistency = False

    if no_inconsistency:
        print("No inconsistencies found")

    phi_iijj = 0.5 * (phi_iijj + phi_iijj.T)
    return phi_iijj

def _displace_cart(geom: np.ndarray, modes: np.ndarray, i_m: Iterator[Tuple], disp_size: float) -> Tuple[np.ndarray, str]:
    """
    Generates a displaced geometry along specified normal mode coordinates

    Parameters
    ----------
    geom : np.ndarray
        Molecular geometry 
    modes : np.ndarray
        Cartesian normal modes (unitless)
    i_m : Iterator[Tuple]
        (displacement index, displacement steps)
    disp_size : float 
        Displacement size (reduced unitless coordinates)
    
    Returns
    -------
    np.ndarray
        Displaced geometry
    """
    label = []
    disp_geom = np.copy(geom)
    # This for loop and tuple unpacking is why the function can handle
    # an arbitrary number of modes.
    for index, disp_steps in i_m:
        disp_geom += modes[:, index].reshape(-1, 3) * disp_size * disp_steps
        label.append(f"{index}: {disp_steps}")

    label = ', '.join(label)
    return disp_geom, label

def _geom_generator(mol: psi4.core.Molecule, data: Dict, mode: int) -> Dict:
    """
    Generates list displaced geometries for a given finite difference job.

    Parameters
    ----------
    mol : psi4.core.Molecule
        Input molecule
    data : Dict
        Dictionary with finite difference job info
    mode : int
        0: energies; 1: gradients; 2: Hessians

    Returns
    -------
    Dict
        Dictionary with list of displacement geometries
    """
    ref_geom = np.array(mol.geometry())
    v_ind = data["harm"]["v_ind"]
    findifrec = {
        "displacement_space": "NormalModes",
        "disp_size": data["options"]["DISP_SIZE"],
        "molecule": mol.to_schema(dtype=2, units='Bohr'),
        "displacements": {},
        "reference": {},
        "harm": data["harm"]
    }

    disp_list = {
        0: {
            "disp1": ((-3, ), (-2, ), (-1, ), (1, ), (2, ), (3, )),
            "disp2": ((-1, -1), (1, -1), (-1, 1), (1, 1)),
            "disp3": ((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (1, 1, -1), (1, -1, 1),
                    (-1, 1, 1), (1, 1, 1))
            },    
        1: {
            "disp1": ((-3, ), (-1, ), (1, ), (3, )),
            "disp2": ((-1, -1), (1, -1), (-1, 1), (1, 1))
            },
        2: {
            "disp1": ((-1, ), (1, ))
            }
        }

    disps = disp_list[mode]

    def append_geoms(indices, steps):
        """Given a list of indices and a list of steps to displace each, append the corresponding geometry to the list."""

        # Next, to make this salc/magnitude composite.
        disp_geom, label = _displace_cart(ref_geom, data["harm"]["modes_unitless"],
                                          zip(indices, steps), findifrec["disp_size"])
        findifrec["displacements"][label] = {"geometry": disp_geom}

    for index in v_ind:
            for val in disps.get("disp1"):
                append_geoms((index, ), val)

    if disps.get("disp2", False):
        for [index1, index2] in itertools.combinations(v_ind, 2):
            for val in disps.get("disp2"):
                append_geoms((index1, index2), val)

    if disps.get("disp3", False):
        for [index1, index2, index3] in itertools.combinations(v_ind, 3):
            for val in disps.get("disp3"):
                append_geoms((index1, index2, index3), val)

    #  reference geometry only if we're doing multi-level calc
    if data["options"].get("METHOD2", False):
        findifrec["reference"]["geometry"] = ref_geom
        findifrec["reference"]["do_reference"] = True

    return findifrec

quartic_from_energies_geometries = partial(_geom_generator, mode = 0)
quartic_from_gradients_geometries = partial(_geom_generator, mode = 1)
quartic_from_hessians_geometries = partial(_geom_generator, mode = 2)

def assemble_quartic_from_energies(findifrec: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cubic and quartic constants by finite difference of energies.

    Parameters
    ----------
    findifrec: Dict
        Dictionary of displaced calculations

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (cubic force constants, quartic force constants)
    """
    n_modes = findifrec["harm"]["n_modes"]
    E0 = findifrec["reference"].get("energy", findifrec["harm"]["E0"])
    v_ind = findifrec["harm"]["v_ind"]
    disp_size = findifrec["disp_size"]
    displacements = findifrec["displacements"]

    phi_ijk = np.zeros((n_modes, n_modes, n_modes))
    phi_iijj = np.zeros((n_modes, n_modes))

    energies = {}
    for label in displacements:
        split_label = label.split(', ')
        for this_label in itertools.permutations(split_label):
            this_label = ', '.join(this_label)
            energies[this_label] = displacements[label]["energy"]

    for i in v_ind:

        E3p = energies[f"{i}: 3"] 
        Ep = energies[f"{i}: 1"]
        En = energies[f"{i}: -1"]
        E3n = energies[f"{i}: -3"]

        phi_ijk[i, i, i] = (E3p - 3 * Ep + 3 * En - E3n) / (8 * disp_size ** 3)

        E2p = energies[f"{i}: 2"]
        E2n = energies[f"{i}: -2"]

        phi_iijj[i, i] = (E2p - 4 * Ep + 6 * E0 - 4 * En + E2n) / (disp_size ** 4)

    
    for [i, j] in itertools.permutations(v_ind, 2):

        Epp = energies[f"{i}: 1, {j}: 1"]
        Epn = energies[f"{i}: 1, {j}: -1"]
        Enp = energies[f"{i}: -1, {j}: 1"]
        Enn = energies[f"{i}: -1, {j}: -1"]

        Eip = energies[f"{i}: 1"]
        Ejp = energies[f"{j}: 1"]
        Ein = energies[f"{i}: -1"]
        Ejn = energies[f"{j}: -1"]

        phi_ijk[i, i, j] = (Epp + Enp - 2 * Ejp - Epn - Enn + 2 * Ejn)
        phi_ijk[i, i, j] /= 2 * disp_size ** 3
        phi_ijk[i, j, i] = phi_ijk[i, i, j]
        phi_ijk[j, i, i] = phi_ijk[i, i, j]

        phi_iijj[i, j] = Epp + Enp + Epn + Enn - 2 * (Eip + Ejp + Ein + Ejn) + 4 * E0
        phi_iijj[i, j] /= disp_size ** 4

    for [i, j, k] in itertools.permutations(v_ind, 3):
        Eppp = energies[f"{i}: 1, {j}: 1, {k}: 1"]
        Enpp = energies[f"{i}: -1, {j}: 1, {k}: 1"]
        Epnp = energies[f"{i}: 1, {j}: -1, {k}: 1"]
        Eppn = energies[f"{i}: 1, {j}: 1, {k}: -1"]
        Epnn = energies[f"{i}: 1, {j}: -1, {k}: -1"]
        Ennp = energies[f"{i}: -1, {j}: -1, {k}: 1"]
        Enpn = energies[f"{i}: -1, {j}: 1, {k}: -1"]
        Ennn = energies[f"{i}: -1, {j}: -1, {k}: -1"]

        phi_ijk[i, j, k] = Eppp - Enpp - Epnp - Eppn + Epnn + Ennp + Enpn - Ennn
        phi_ijk[i, j, k] /= 8 * disp_size ** 3

    phi_ijk /= wave_to_hartree
    phi_ijk = check_cubic(phi_ijk, v_ind)
    phi_iijj /= wave_to_hartree
    phi_iijj = check_quartic(phi_iijj, v_ind)

    return phi_ijk, phi_iijj

def assemble_quartic_from_gradients(findifrec: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cubic and quartic constants by finite difference of gradients.

    Parameters
    ----------
    findifrec: Dict
        Dictionary of displaced calculations

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (cubic force constants, quartic force constants)
    """

    n_modes = findifrec["harm"]["n_modes"]
    grad0 = findifrec["reference"].get("gradient", findifrec["harm"]["G0"])
    q = findifrec["harm"]["modes"]
    gamma = findifrec["harm"]["gamma"]
    v_ind = findifrec["harm"]["v_ind"]
    disp_size = findifrec["disp_size"]
    displacements = findifrec["displacements"]

    phi_ijk = np.zeros((n_modes, n_modes, n_modes))
    phi_iijj = np.zeros((n_modes, n_modes))
    grad = {}

    def transform_grad(grad):
        gradQ = np.matmul(q.transpose(), grad.reshape(-1))
        gradQ = np.einsum("i,i->i", gradQ, np.sqrt(gamma), optimize=True)
        return gradQ

    grad0 = transform_grad(grad0)
    for label in displacements:
        split_label = label.split(', ')
        for this_label in itertools.permutations(split_label):
            this_label = ', '.join(this_label)
            grad[this_label] = transform_grad(displacements[label]["gradient"])

    for i in v_ind:

        grad_p = grad[f"{i}: 1"]
        grad_n = grad[f"{i}: -1"]
        grad_3p = grad[f"{i}: 3"]
        grad_3n = grad[f"{i}: -3"]

        phi_iijj[i, i] = grad_3p[i] - 3 * grad_p[i] + 3 * grad_n[i] - grad_3n[i]
        phi_iijj[i, i] /= 8 * disp_size ** 3

        phi_ijk[i, i, i] = grad_p[i] + grad_n[i] - 2 * grad0[i]
        phi_ijk[i, i, i] /= disp_size ** 2

    for [i, j] in itertools.permutations(v_ind, 2):

        grad_p = grad[f"{j}: 1"]
        grad_n = grad[f"{j}: -1"]

        phi_ijk[i, j, j] = grad_p[i] + grad_n[i] - 2 * grad0[i]
        phi_ijk[i, j, j] /= disp_size ** 2
        phi_ijk[j, j, i] = phi_ijk[i, j, j]
        phi_ijk[j, i, j] = phi_ijk[i, j, j]

        grad_p = grad[f"{i}: 1"]
        grad_n = grad[f"{i}: -1"]
        grad_pp = grad[f"{i}: 1, {j}: 1"]
        grad_nn = grad[f"{i}: -1, {j}: -1"]
        grad_np = grad[f"{i}: -1, {j}: 1"]
        grad_pn = grad[f"{i}: 1, {j}: -1"]

        phi_iijj[i, j] = grad_pp[i] + grad_pn[i] - 2 * grad_p[i] 
        phi_iijj[i, j] -= grad_np[i] + grad_nn[i] - 2 * grad_n[i]
        phi_iijj[i, j] /= 2 * disp_size ** 3

    for [i, j, k] in itertools.permutations(v_ind, 3):

        grad_pj = grad[f"{j}: 1"]
        grad_nj = grad[f"{j}: -1"]
        grad_pk = grad[f"{k}: 1"]
        grad_nk = grad[f"{k}: -1"]
        grad_pp = grad[f"{j}: 1, {k}: 1"]
        grad_nn = grad[f"{j}: -1, {k}: -1"]

        phi_ijk[i, j, k] = grad_pp[i] + grad_nn[i] + 2 * grad0[i]
        phi_ijk[i, j, k] -= grad_pj[i] + grad_pk[i] + grad_nj[i] + grad_nk[i]
        phi_ijk[i, j, k] /= 2 * disp_size ** 2

    phi_ijk = phi_ijk / wave_to_hartree
    phi_ijk = check_cubic(phi_ijk, v_ind)
    phi_iijj = phi_iijj / wave_to_hartree
    phi_iijj = check_quartic(phi_iijj, v_ind)

    return phi_ijk, phi_iijj

def assemble_quartic_from_hessians(findifrec: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cubic and quartic constants by finite difference of Hessians.

    Parameters
    ----------
    findifrec: Dict
        Dictionary of displaced calculations

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (cubic force constants, quartic force constants)
    """

    n_modes = findifrec["harm"]["n_modes"]
    hess0 = findifrec["reference"].get("hessian", findifrec["harm"]["H0"])
    q = findifrec["harm"]["modes"]
    gamma = findifrec["harm"]["gamma"]
    v_ind = findifrec["harm"]["v_ind"]
    disp_size = findifrec["disp_size"]

    phi_ijk = np.zeros((n_modes, n_modes, n_modes))
    phi_iijj = np.zeros((n_modes, n_modes))

    def transform_hessian(hess):
        hessQ = np.matmul(q.transpose(), np.matmul(hess, q))
        hessQ = np.einsum("ij,i,j->ij", hessQ, np.sqrt(gamma), np.sqrt(gamma), optimize=True)
        return hessQ

    hess0 = transform_hessian(hess0)
    displacements = findifrec["displacements"]
    hess_p = {}
    hess_n = {}

    for i in v_ind:
        hess_p[i] = transform_hessian(displacements[f"{i}: 1"]["hessian"])
        hess_n[i] = transform_hessian(displacements[f"{i}: -1"]["hessian"])

    for [i,j,k] in itertools.product(v_ind, repeat=3):
        if (i == j and j == k):
            phi_ijk[i, i, i] = (hess_p[i][i, i] - hess_n[i][i, i]) / (2 * disp_size)

        else:
            phi_ijk[i, j, k] = hess_p[i][j, k] - hess_n[i][j, k] + hess_p[j][k, i]
            phi_ijk[i, j, k] += - hess_n[j][k, i] + hess_p[k][i, j] - hess_n[k][i, j]
            phi_ijk[i, j, k] /= 6 * disp_size

    for [i,j] in itertools.product(v_ind, repeat=2):
        if i == j:
            phi_iijj[i, i] = hess_p[i][i, i] + hess_n[i][i, i] - 2 * hess0[i, i]
            phi_iijj[i, i] /= disp_size ** 2

        else:
            phi_iijj[i, j] = hess_p[j][i, i] + hess_n[j][i, i] + hess_p[i][j, j]
            phi_iijj[i, j] += hess_n[i][j, j] - 2 * hess0[i, i] - 2 * hess0[j, j]
            phi_iijj[i, j] /= 2 * disp_size ** 2


    phi_iijj = phi_iijj / wave_to_hartree
    phi_iijj = check_quartic(phi_iijj, v_ind)
    phi_ijk = phi_ijk / wave_to_hartree
    phi_ijk = check_cubic(phi_ijk, v_ind)

    return phi_ijk, phi_iijj

class QuarticComputer(BaseComputer):

    molecule: Any
    driver: DriverEnum
    metameta: Dict[str, Any] = {}
    task_list: Dict[str, BaseComputer] = {}
    findifrec: Dict[str, Any] = {}
    computer: BaseComputer = AtomicComputer
    method: str

    @validator('driver')
    def set_driver(cls, driver):
        egh = ['energy', 'gradient', 'hessian']
        if driver not in egh:
            raise Exception(f"""Wrapper is unhappy to be calling function ({driver}) not among {egh}.""")

        return driver

    @validator('molecule')
    def set_molecule(cls, mol):
        mol.update_geometry()
        mol.fix_com(True)
        mol.fix_orientation(True)
        return mol

    def __init__(self, **data):
        """Initialize FiniteDifference class.

        data keywords include
        * general AtomicInput keys like molecule, driver, method, basis, and keywords.
        * specialized findif keys like findif_mode, findif_irrep, and those converted from keywords to kwargs:
          findif_stencil_size, findif_step_size, and findif_verbose.

        """

        BaseComputer.__init__(self, **data)

        data['keywords']['PARENT_SYMMETRY'] = self.molecule.point_group().full_name()

        self.method = data['method']

        mode_dict = {"ENERGY": 0, "GRADIENT": 1, "HESSIAN": 2}
        self.metameta['mode'] = mode_dict[data['options']['FD']] 

        if self.metameta['mode'] == 0:
            self.metameta['proxy_driver'] = 'energy'
            self.findifrec = quartic_from_energies_geometries(self.molecule,
                                                               data)

        elif self.metameta['mode'] == 1:
            self.metameta['proxy_driver'] = 'gradient'
            self.findifrec = quartic_from_gradients_geometries(self.molecule,
                                                               data)

        elif self.metameta['mode'] == 2:
            self.metameta['proxy_driver'] = 'hessian'
            self.findifrec = quartic_from_hessians_geometries(self.molecule,
                                                              data)

        ndisp = len(self.findifrec["displacements"])
        info = f""" {ndisp} displacements needed ...\n"""
        logger.debug(info)

        if self.findifrec["reference"].get("do_reference", False):
            packet = {
                "molecule": self.molecule,
                "driver": self.metameta['proxy_driver'],
                "method": self.method,
                "basis": data["basis"],
                "keywords": data["keywords"] or {},
            }
            if 'cbs_metadata' in data:
                packet['cbs_metadata'] = data['cbs_metadata']

            self.task_list["reference"] = self.computer(**packet)

        parent_group = self.molecule.point_group()
        for label, displacement in self.findifrec["displacements"].items():
            clone = self.molecule.clone()
            clone.reinterpret_coordentry(False)
            #clone.fix_orientation(True)

            # Load in displacement into the active molecule
            clone.set_geometry(psi4.core.Matrix.from_array(displacement["geometry"]))

            # If the user insists on symmetry, weaken it if some is lost when displacing.
            # or 'fix_symmetry' in self.findifrec.molecule
            logger.debug(f'SYMM {clone.schoenflies_symbol()}')
            if self.molecule.symmetry_from_input():
                disp_group = clone.find_highest_point_group()
                new_bits = parent_group.bits() & disp_group.bits()
                new_symm_string = psi4.qcdb.PointGroup.bits_to_full_name(new_bits)
                clone.reset_point_group(new_symm_string)

            packet = {
                "molecule": clone,
                "driver": self.metameta['proxy_driver'],
                "method": self.method,
                "basis": data["basis"],
                "keywords": data["keywords"] or {},
            }
            if 'cbs_metadata' in data:
                packet['cbs_metadata'] = data['cbs_metadata']

            self.task_list[label] = self.computer(**packet)

#        for n, displacement in enumerate(findif_meta_dict["displacements"].values(), start=2):
#            _process_displacement(energy, lowername, molecule, displacement, n, ndisp, write_orbitals=False, **kwargs)

    def build_tasks(self, obj, **kwargs):
        # permanently a dummy function
        pass

    def plan(self):
        # uncalled function
        return [t.plan() for t in self.task_list.values()]

    def compute(self, client: Optional["qcportal.FractalClient"] = None):
        """Run each job in task list."""
        instructions = "\n" + psi4.p4util.banner(f" FiniteDifference Computations", strNotOutfile=True) + "\n"
        logger.debug(instructions)

        with psi4.p4util.hold_options_state():
            for t in self.task_list.values():
                t.compute(client=client)

    def get_results(self, client: Optional["qcportal.FractalClient"] = None) -> AtomicResult:
        """Return results as FiniteDifference-flavored QCSchema."""

        results_list = {k: v.get_results(client=client) for k, v in self.task_list.items()}

        if self.findifrec["reference"].get("do_reference", False):
            reference = self.findifrec["reference"]
            task = results_list["reference"]
            response = task.return_result
            reference["module"] = getattr(task.provenance, "module", None)

            if task.driver == 'energy':
                reference['energy'] = response

            elif task.driver == 'gradient':
                reference['gradient'] = response
                reference['energy'] = task.extras['qcvars']['CURRENT ENERGY']

            elif task.driver == 'hessian':
                reference['hessian'] = response
                reference['energy'] = task.extras['qcvars']['CURRENT ENERGY']
                if 'CURRENT GRADIENT' in task.extras['qcvars']:
                    reference['gradient'] = task.extras['qcvars']['CURRENT GRADIENT']

        # load AtomicComputer results into findifrec[displacements]
        for label, displacement in self.findifrec["displacements"].items():
            task = results_list[label]
            response = task.return_result

            if task.driver == 'energy':
                displacement['energy'] = response

            elif task.driver == 'gradient':
                displacement['gradient'] = response
                displacement['energy'] = task.extras['qcvars']['CURRENT ENERGY']

            elif task.driver == 'hessian':
                displacement['hessian'] = response
                displacement['energy'] = task.extras['qcvars']['CURRENT ENERGY']
                if 'CURRENT GRADIENT' in task.extras['qcvars']:
                    displacement['gradient'] = task.extras['qcvars']['CURRENT GRADIENT']

            displacement['provenance'] = task.provenance

        # apply finite difference formulas and load derivatives into findifrec[reference]
        if self.metameta['mode'] == 0:
            phi_ijk, phi_iijj = assemble_quartic_from_energies(self.findifrec)
        if self.metameta['mode'] == 1:
            phi_ijk, phi_iijj = assemble_quartic_from_gradients(self.findifrec)
        if self.metameta['mode'] == 2:
            phi_ijk, phi_iijj = assemble_quartic_from_hessians(self.findifrec)

        self.findifrec["reference"]["phi_ijk"] = phi_ijk
        self.findifrec["reference"]["phi_iijj"] = phi_iijj
        ref = list(results_list.values())[0] # No reference calculation so lets grab a random one
        self.findifrec["reference"]["module"] = getattr(ref.provenance, "module", None)

        properties = {
            "calcinfo_natom": self.molecule.natom(),
            "nuclear_repulsion_energy": self.molecule.nuclear_repulsion_energy(),
            "return_energy": self.findifrec["harm"]["E0"],
        }

        quartic_result = AtomicResult(
              **{
                'driver': self.driver,
                'model': {
                    "basis": self.basis,
                    'method': self.method,
                },
                'molecule': self.molecule.to_schema(dtype=2),
                'properties': properties,
                'provenance': psi4.p4util.provenance_stamp(__name__, module=self.findifrec["reference"]["module"]),
                'extras': {
                    'findif_record': copy.deepcopy(self.findifrec),
                    'phi_ijk': phi_ijk,
                    'phi_iijj': phi_iijj
                },
                'return_result': self.findifrec["harm"]["H0"],
                'success': True,
              })

        return quartic_result

def task_planner(method: str, molecule: psi4.core.Molecule, **kwargs) -> QuarticComputer:
    """
    Generates computer for finite difference calcutations

    Parameters
    ----------
    method : str
        Quantum chemistry method
    molecule: psi4.core.Molecule
        Input molecule
    
    Returns
    -------
    QuarticComputer
        Computer for finite difference calculations
    """
    # keywords are the psi4 option keywords
    keywords = psi4.p4util.prepare_options_for_set_options()
    driver = "hessian"
    dermode = kwargs["options"]["FD"]
    method = method.lower()
    basis = keywords.pop("BASIS", "(auto)")

    # Expand CBS methods
    method, basis, cbsmeta = expand_cbs_methods(method=method, basis=basis, driver=driver)
    if method in composite_procedures:
        kwargs.update(cbsmeta)
        kwargs.update({'cbs_metadata': composite_procedures[method](**kwargs)})
        method = 'cbs'

    packet = {"molecule": molecule, "driver": driver, "method": method, "basis": basis, "keywords": keywords}

    if method == "cbs":
        kwargs.update(cbsmeta)
        logger.info(
            f'PLANNING FD(CBS):  dermode={dermode} packet={packet} kw={kwargs}')
        return QuarticComputer(**packet, computer=CompositeComputer, **kwargs)

    else:
        logger.info(
            f'PLANNING FD:  dermode={dermode} keywords={keywords} kw={kwargs}')
        return QuarticComputer(**packet, **kwargs)

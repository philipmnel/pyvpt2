import psi4
import numpy as np
import qcelemental as qcel
import itertools

wave_to_hartree = qcel.constants.get("inverse meter-hartree relationship") * 100


def gen_disp_geom(mol, disp, harm, options):
    """
    gen_disp_geom: displaces a molecule along a set of normal modes

    mol: psi4 molecule object
    disp: key of normal modes
    harm: harmonic results dictionary
    options: program options dictionary

    disp_mol: psi4 molecule object at displaced geometry
    """

    modes = harm["modes_unitless"]
    disp_size = options["DISP_SIZE"]

    disp_geom = mol.geometry().np
    for i in disp:
        disp_geom += modes[:, i].reshape(-1, 3) * disp_size * disp[i]

    disp_mol = mol.clone()
    disp_mol.set_geometry(psi4.core.Matrix.from_array(disp_geom))
    disp_mol.reinterpret_coordentry(False)
    disp_mol.update_geometry()

    return disp_mol


def disp_energy(mol, disp, harm, options):
    """
    disp_energy: displaces a molecule along a set of normal mode and returns the energy

    mol: psi4 molecule object
    disp: key of normal modes
    harm: harmonic results dictionary
    options: program options dictionary

    E: energy at displaced geometry
    """

    method = options["METHOD"]

    disp_mol = gen_disp_geom(mol, disp, harm, options)

    E = psi4.energy(method, molecule=disp_mol)
    return E


def disp_grad(mol, disp, harm, options):
    """
    disp_grad: displaces a molecule along a set of normal mode and returns the gradient

    mol: psi4 molecule object
    disp: key of normal modes
    harm: harmonic results dictionary
    options: program options dictionary

    gradQ: gradient at displaced geometry in normal mode coordinates; np array of size 3*n_atom
    """

    method = options["METHOD"]
    gamma = harm["gamma"]
    q = harm["modes"]
    n_atom = mol.natom()

    disp_mol = gen_disp_geom(mol, disp, harm, options)

    grad = psi4.gradient(method, molecule=disp_mol).np

    grad = grad.reshape(3 * n_atom, 1)
    gradQ = np.matmul(np.transpose(q), grad)
    gradQ = gradQ.reshape(3 * n_atom)
    gradQ = np.einsum("i,i->i", gradQ, np.sqrt(gamma), optimize=True)

    return gradQ


def disp_hess(mol, disp, harm, options):
    """
    disp_hess: displaces a molecule along a set of normal mode and returns the Hessian

    mol: psi4 molecule object
    disp: key of normal modes
    harm: harmonic results dictionary
    options: program options dictionary

    hessQ: Hessian at displaced geometry, np array of size (3*n_atom, 3*n_atom)
    """

    method = options["METHOD"]
    gamma = harm["gamma"]
    q = harm["modes"]

    disp_mol = gen_disp_geom(mol, disp, harm, options)

    hess = psi4.hessian(method, molecule=disp_mol).np

    hessQ = np.matmul(np.transpose(q), np.matmul(hess, q))
    hessQ = np.einsum("ij,i,j->ij", hessQ, np.sqrt(gamma), np.sqrt(gamma), optimize=True)

    return hessQ


def force_field_E(mol, harm, options):

    n_modes = harm["n_modes"]
    v_ind = harm["v_ind"]
    disp_size = options["DISP_SIZE"]
    E0 = harm["E0"]

    phi_ijk = np.zeros((n_modes, n_modes, n_modes))
    phi_iijj = np.zeros((n_modes, n_modes))

    Ep_matrix = np.zeros((n_modes))
    En_matrix = np.zeros((n_modes))

    Epp_matrix = np.zeros((n_modes, n_modes))
    Epn_matrix = np.zeros((n_modes, n_modes))
    Enn_matrix = np.zeros((n_modes, n_modes))

    Eppp_matrix = np.zeros((n_modes, n_modes, n_modes))
    Ennn_matrix = np.zeros((n_modes, n_modes, n_modes))
    Enpp_matrix = np.zeros((n_modes, n_modes, n_modes))
    Epnn_matrix = np.zeros((n_modes, n_modes, n_modes))

    for i in v_ind:

        Eppp_matrix[i] = disp_energy(mol, {i: 3}, harm, options)
        Ennn_matrix[i] = disp_energy(mol, {i: -3}, harm, options)
        Epp_matrix[i, i] = disp_energy(mol, {i: 2}, harm, options)
        Enn_matrix[i, i] = disp_energy(mol, {i: -2}, harm, options)
        Ep_matrix[i] = disp_energy(mol, {i: 1}, harm, options)
        En_matrix[i] = disp_energy(mol, {i: -1}, harm, options)

        for j in v_ind:

            if j == i:
                continue

            Epn_matrix[i, j] = disp_energy(mol, {i: 1, j: -1}, harm, options)

            if j > i:
                Epp_matrix[i, j] = disp_energy(mol, {i: 1, j: 1}, harm, options)
                Enn_matrix[i, j] = disp_energy(mol, {i: -1, j: -1}, harm, options)

                Epp_matrix[j, i] = Epp_matrix[i, j]
                Enn_matrix[j, i] = Enn_matrix[i, j]

    for i in v_ind:
        for j in v_ind:

            if j == i:
                continue

            for k in v_ind:
                if k == j or k == i:
                    continue

                if k > j:
                    Enpp_matrix[i, j, k] = disp_energy(mol, {i: -1, j: 1, k: 1}, harm, options)
                    Epnn_matrix[i, j, k] = disp_energy(mol, {i: 1, j: -1, k: -1}, harm, options)

                    Enpp_matrix[i, k, j] = Enpp_matrix[i, j, k]
                    Epnn_matrix[i, k, j] = Epnn_matrix[i, j, k]

                if k > j and j > i:

                    Ennn_matrix[i, j, k] = disp_energy(mol, {i: -1, j: -1, k: -1}, harm, options)
                    Eppp_matrix[i, j, k] = disp_energy(mol, {i: 1, j: 1, k: 1}, harm, options)

                    Eppp_matrix[i, k, j] = Eppp_matrix[i, j, k]
                    Eppp_matrix[j, k, i] = Eppp_matrix[i, j, k]
                    Eppp_matrix[j, i, k] = Eppp_matrix[i, j, k]
                    Eppp_matrix[k, i, j] = Eppp_matrix[i, j, k]
                    Eppp_matrix[k, j, i] = Eppp_matrix[i, j, k]

                    Ennn_matrix[i, k, j] = Ennn_matrix[i, j, k]
                    Ennn_matrix[j, k, i] = Ennn_matrix[i, j, k]
                    Ennn_matrix[j, i, k] = Ennn_matrix[i, j, k]
                    Ennn_matrix[k, i, j] = Ennn_matrix[i, j, k]
                    Ennn_matrix[k, j, i] = Ennn_matrix[i, j, k]

    for i in v_ind:

        E3p = Eppp_matrix[i, i, i]
        Ep = Ep_matrix[i]
        En = En_matrix[i]
        E3n = Ennn_matrix[i, i, i]

        phi_ijk[i, i, i] = (E3p - 3 * Ep + 3 * En - E3n) / (8 * disp_size ** 3)

        E2p = Epp_matrix[i, i]
        E2n = Enn_matrix[i, i]

        phi_iijj[i, i] = (E2p - 4 * Ep + 6 * E0 - 4 * En + E2n) / (disp_size ** 4)

    for i in v_ind:
        for j in v_ind:

            if i == j:
                continue

            Epp = Epp_matrix[i, j]
            Epn = Epn_matrix[i, j]
            Enp = Epn_matrix[j, i]
            Enn = Enn_matrix[i, j]

            Eip = Ep_matrix[i]
            Ejp = Ep_matrix[j]
            Ein = En_matrix[i]
            Ejn = En_matrix[j]

            phi_ijk[i, i, j] = (Epp + Enp - 2 * Ejp - Epn - Enn + 2 * Ejn) / (
                2 * disp_size ** 3)
            phi_ijk[i, j, i] = phi_ijk[i, i, j]
            phi_ijk[j, i, i] = phi_ijk[i, i, j]

            phi_iijj[i, j] = (Epp + Enp + Epn + Enn - 2 * (Eip + Ejp + Ein + Ejn) + 4 * E0
            ) / (disp_size ** 4)

    for i in v_ind:
        for j in v_ind:
            for k in v_ind:

                if i == j:
                    continue
                elif i == k:
                    continue
                elif j == k:
                    continue

                Eppp = Eppp_matrix[i, j, k]
                Enpp = Enpp_matrix[i, j, k]
                Epnp = Enpp_matrix[j, i, k]
                Eppn = Enpp_matrix[k, i, j]
                Epnn = Epnn_matrix[i, j, k]
                Ennp = Epnn_matrix[k, i, j]
                Enpn = Epnn_matrix[j, i, k]
                Ennn = Ennn_matrix[i, j, k]

                phi_ijk[i, j, k] = (Eppp - Enpp - Epnp - Eppn + Epnn + Ennp + Enpn - Ennn
                ) / (8 * disp_size ** 3)

    phi_ijk /= wave_to_hartree
    phi_iijj /= wave_to_hartree

    return phi_ijk, phi_iijj


def force_field_G(mol, harm, options):
    """
    force_field_G: generates cubic and quartic force constants using gradients

    mol: psi4 molecule object
    harm: harmonic results dictionary
    options: program options dictionary

    phi_ijk = cubic force constants (cm-1), np array of size (3*n_atom, 3*n_atom, 3*n_atom)
    phi_iijj = quartic force constants (cm-1), np array of size (3*n_atom, 3*n_atom)
    """

    n_modes = harm["n_modes"]
    v_ind = harm["v_ind"]
    disp_size = options["DISP_SIZE"]

    phi_ijk = np.zeros((n_modes, n_modes, n_modes))
    phi_iijj = np.zeros((n_modes, n_modes))

    grad0 = disp_grad(mol, {0: 0}, harm, options)

    grad_p = np.zeros((n_modes, n_modes))
    grad_n = np.zeros((n_modes, n_modes))
    grad_pp = np.zeros((n_modes, n_modes, n_modes))
    grad_nn = np.zeros((n_modes, n_modes, n_modes))
    grad_3p = np.zeros((n_modes, n_modes))
    grad_3n = np.zeros((n_modes, n_modes))
    grad_np = np.zeros((n_modes, n_modes, n_modes))

    for i in v_ind:

        grad_p[:, i] = disp_grad(mol, {i: +1}, harm, options)
        grad_n[:, i] = disp_grad(mol, {i: -1}, harm, options)
        grad_3p[:, i] = disp_grad(mol, {i: +3}, harm, options)
        grad_3n[:, i] = disp_grad(mol, {i: -3}, harm, options)

        for j in v_ind:
            if i == j:
                continue

            if j > i:
                grad_pp[:, i, j] = disp_grad(mol, {i: +1, j: +1}, harm, options)
                grad_nn[:, i, j] = disp_grad(mol, {i: -1, j: -1}, harm, options)

            if j < i:
                grad_pp[:, i, j] = grad_pp[:, j, i]
                grad_nn[:, i, j] = grad_nn[:, j, i]

            grad_np[:, i, j] = disp_grad(mol, {i: -1, j: +1}, harm, options)

    for i in v_ind:

        phi_iijj[i, i] = (grad_3p[i, i] - 3 * grad_p[i, i] + 3 * grad_n[i, i] - grad_3n[i, i]
        ) / (8 * disp_size ** 3)

        for j in v_ind:
            phi_ijk[i, j, j] = (grad_p[i, j] + grad_n[i, j] - 2 * grad0[i]
            ) / (disp_size ** 2)
            phi_ijk[j, j, i] = phi_ijk[i, j, j]
            phi_ijk[j, i, j] = phi_ijk[i, j, j]

            if i == j:
                continue

            phi_iijj[i, j] = (grad_pp[i, i, j] + grad_np[i, j, i] - 2 * grad_p[i, i] 
            - (grad_np[i, i, j] + grad_nn[i, i, j] - 2 * grad_n[i, i])
            ) / (2 * disp_size ** 3)

            for k in v_ind:
                if k == j or k == i:
                    continue

                phi_ijk[i, j, k] = (grad_pp[i, j, k] + grad_nn[i, j, k] + 2 * grad0[i]
                    - (grad_p[i, j] + grad_p[i, k] + grad_n[i, j] + grad_n[i, k])
                ) / (2 * disp_size ** 2)

    phi_ijk = phi_ijk / wave_to_hartree
    phi_iijj = phi_iijj / wave_to_hartree

    return phi_ijk, phi_iijj


def force_field_H(mol, harm, options):
    """
    force_field_H: generates cubic and quartic force constants using Hessians

    mol: psi4 molecule object
    harm: harmonic results dictionary
    options: program options dictionary

    phi_ijk = cubic force constants (cm-1), np array of size (3*n_atom, 3*n_atom, 3*n_atom)
    phi_iijj = quartic force constants (cm-1), np array of size (3*n_atom, 3*n_atom)
    """
    n_modes = harm["n_modes"]
    v_ind = harm["v_ind"]
    disp_size = options["DISP_SIZE"]

    phi_ijk = np.zeros((n_modes, n_modes, n_modes))
    phi_iijj = np.zeros((n_modes, n_modes))

    disp_num = 1 + 2 * len(v_ind)
    disp_counter = 1
    print("{0} displacements needed: ".format(disp_num), end='')

    # REDUNDANT CALCULATION - REWRITE THIS
    hess0 = disp_hess(mol, {0: 0}, harm, options)
    print(disp_counter, end=' '); disp_counter += 1
    hess_p = np.zeros((n_modes, n_modes, n_modes))
    hess_n = np.zeros((n_modes, n_modes, n_modes))

    for i in v_ind:
        hess_p[i, :, :] = disp_hess(mol, {i: +1}, harm, options)
        print(disp_counter, end=' '); disp_counter += 1
        hess_n[i, :, :] = disp_hess(mol, {i: -1}, harm, options)
        print(disp_counter, end=' '); disp_counter += 1

    for i in v_ind:
        phi_iijj[i, i] = (hess_p[i, i, i] + hess_n[i, i, i] - 2 * hess0[i, i]) / (
            disp_size ** 2)
        phi_ijk[i, i, i] = (hess_p[i, i, i] - hess_n[i, i, i]) / (2 * disp_size)

        for j in v_ind:
            for k in v_ind:
                if (i == j) and (j == k):
                    continue
                phi_ijk[i, j, k] = (
                      hess_p[i, j, k]
                    - hess_n[i, j, k]
                    + hess_p[j, k, i]
                    - hess_n[j, k, i]
                    + hess_p[k, i, j]
                    - hess_n[k, i, j]
                ) / (6 * disp_size)

            if i == j:
                continue
            phi_iijj[i, j] = (
                  hess_p[j, i, i]
                + hess_n[j, i, i]
                + hess_p[i, j, j]
                + hess_n[i, j, j]
                - 2 * hess0[i, i]
                - 2 * hess0[j, j]
            ) / (2 * disp_size ** 2)

    phi_iijj = phi_iijj / wave_to_hartree
    phi_ijk = phi_ijk / wave_to_hartree

    return phi_ijk, phi_iijj

def check_cubic(phi_ijk, harm):

    v_ind = harm["v_ind"]

    print("Checking for numerical inconsistencies in cubic terms...")
    for unique_ijk in itertools.combinations_with_replacement(v_ind, 3):
        for [ind1, ind2] in itertools.combinations(set(itertools.permutations(unique_ijk, 3)), 2):
            diff = abs(phi_ijk[ind1] - phi_ijk[ind2])
            if diff >= 1.0:
                print(ind1, ind2, diff)

def check_quartic(phi_iijj, harm):

    v_ind = harm["v_ind"]
    
    print("Checking for numerical inconsistencies in quartic terms...")
    for unique_ij in itertools.combinations_with_replacement(v_ind, 2):
        for [ind1, ind2] in itertools.combinations(set(itertools.permutations(unique_ij, 2)), 2):
            diff = abs(phi_iijj[ind1] - phi_iijj[ind2])
            if diff >= 1.0:
                print(ind1, ind2, diff)
import psi4
import numpy as np
import qcelemental as qcel


def disp_energy(mol, disp, harm, options):

    modes = harm["modes_unitless"]
    disp_size = options["DISP_SIZE"]
    method = options["METHOD"]

    disp_geom = mol.geometry().np
    for i in disp:
        disp_geom += modes[:,i].reshape(-1,3) * disp_size * disp[i]
    
    disp_mol = mol.clone()
    disp_mol.set_geometry(psi4.core.Matrix.from_array(disp_geom))
    disp_mol.update_geometry()

    E = psi4.energy(method, molecule = disp_mol)
    return E

def force_field_E(mol, harm, options):

    n_modes = harm["n_modes"]
    v_ind = harm["v_ind"]
    disp_size = options["DISP_SIZE"]
    E0 = harm["E0"]
    wave_to_hartree = qcel.constants.get("inverse meter-hartree relationship") * 100
   
    phi_ijk = np.zeros((n_modes,n_modes,n_modes))
    phi_iijj = np.zeros((n_modes,n_modes))

    for i in v_ind:

        E3p = disp_energy(mol, {i:3}, harm, options)
        Ep = disp_energy(mol, {i:1}, harm, options)
        En = disp_energy(mol, {i:-1}, harm, options)
        E3n = disp_energy(mol, {i:-3}, harm, options)

        phi_ijk[i,i,i] = ( E3p - 3*Ep + 3*En - E3n ) / (8 * disp_size**3)

        E2p = disp_energy(mol, {i:2}, harm, options)
        E2n = disp_energy(mol, {i:-2}, harm, options)

        phi_iijj[i,i] = ( E2p - 4*Ep + 6*E0 - 4*En + E2n) / (disp_size**4)
        

    for i in v_ind:
        for j in v_ind:

            if i==j: continue

            Epp = disp_energy(mol, {i:1, j:1}, harm, options)
            Epn = disp_energy(mol, {i:1, j:-1}, harm, options)
            Enp = disp_energy(mol, {i:-1, j:1}, harm, options)
            Enn = disp_energy(mol, {i:-1, j:-1}, harm, options)
            
            Eip = disp_energy(mol, {i:1}, harm, options)
            Ejp = disp_energy(mol, {j:1}, harm, options)
            Ein = disp_energy(mol, {i:-1}, harm, options)
            Ejn = disp_energy(mol, {j:-1}, harm, options)

            phi_ijk[i,i,j] = ( Epp + Enp - 2*Ejp - Epn - Enn + 2*Ejn) / (2 * disp_size**3)
            phi_ijk[i,j,i] = phi_ijk[i,i,j]
            phi_ijk[j,i,i] = phi_ijk[i,i,j]

            phi_iijj[i,j] =  ( Epp + Enp + Epn + Enn - 2*( Eip + Ejp + Ein + Ejn ) + 4*E0) / (disp_size**4)

    for i in v_ind:
        for j in v_ind:
            for k in v_ind:

                if i==j: continue
                elif i==k: continue
                elif j==k: continue

                Eppp = disp_energy(mol, {i:1, j:1, k:1}, harm, options) 
                Enpp = disp_energy(mol, {i:-1, j:1, k:1}, harm, options) 
                Epnp = disp_energy(mol, {i:1, j:-1, k:1}, harm, options) 
                Eppn = disp_energy(mol, {i:1, j:1, k:-1}, harm, options) 
                Epnn = disp_energy(mol, {i:1, j:-1, k:-1}, harm, options) 
                Ennp = disp_energy(mol, {i:-1, j:-1, k:1}, harm, options) 
                Enpn = disp_energy(mol, {i:-1, j:1, k:-1}, harm, options) 
                Ennn = disp_energy(mol, {i:-1, j:-1, k:-1}, harm, options) 

                phi_ijk[i,j,k] = ( Eppp - Enpp - Epnp - Eppn + Epnn + Ennp + Enpn - Ennn ) / (8 * disp_size**3)
                phi_ijk[i,k,j] = phi_ijk[i,j,k]
                phi_ijk[j,k,i] = phi_ijk[i,j,k]
                phi_ijk[j,k,j] = phi_ijk[i,j,k]
                phi_ijk[k,i,j] = phi_ijk[i,j,k]
                phi_ijk[k,j,i] = phi_ijk[i,j,k]

    phi_ijk /= wave_to_hartree
    phi_iijj /= wave_to_hartree 

    return phi_ijk, phi_iijj
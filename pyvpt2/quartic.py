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

def disp_grad(mol, disp, harm, options):
    disp_geom = mol.geometry().np
    disp_size = options["DISP_SIZE"]
    method = options["METHOD"]
    modes_unitless = harm["modes_unitless"]
    gamma = harm["gamma"]
    q = harm["modes"]
    n_atom = mol.natom()

    for i in disp:
        disp_geom += modes_unitless[:,i].reshape(-1,3) * disp_size * disp[i]
    
    disp_mol = mol.clone()
    disp_mol.set_geometry(psi4.core.Matrix.from_array(disp_geom))
    disp_mol.reinterpret_coordentry(False)
    disp_mol.update_geometry()
    
    grad = psi4.gradient(method, molecule = disp_mol).np
    grad = grad.reshape(3*n_atom,1)

    gradQ = np.matmul(np.transpose(q), grad)
    gradQ = gradQ.reshape(3*n_atom)

    gradQ = np.einsum('i,i->i', gradQ, np.sqrt(gamma), optimize=True)

    return gradQ

def disp_hess(mol, disp, harm, options):
    disp_geom = mol.geometry().np
    disp_size = options["DISP_SIZE"]
    method = options["METHOD"]
    modes_unitless = harm["modes_unitless"]
    gamma = harm["gamma"]
    q = harm["modes"]

    for i in disp:
        disp_geom += modes_unitless[:,i].reshape(-1,3) * disp_size * disp[i]
    
    disp_mol = mol.clone()
    disp_mol.set_geometry(psi4.core.Matrix.from_array(disp_geom))
    disp_mol.reinterpret_coordentry(False)
    disp_mol.update_geometry()

    hess = psi4.hessian(method, molecule = disp_mol).np

    hessQ = np.matmul(np.transpose(q), np.matmul(hess,q))

    hessQ = np.einsum('ij,i,j->ij', hessQ, np.sqrt(gamma), np.sqrt(gamma), optimize=True)

    return hessQ

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

    phi_ijk /= wave_to_hartree
    phi_iijj /= wave_to_hartree 

    return phi_ijk, phi_iijj

def force_field_G(mol, harm, options):
    
    n_modes = harm["n_modes"]
    v_ind = harm["v_ind"]
    disp_size = options["DISP_SIZE"]
    wave_to_hartree = qcel.constants.get("inverse meter-hartree relationship") * 100
   
    phi_ijk = np.zeros((n_modes,n_modes,n_modes))
    phi_iijj = np.zeros((n_modes,n_modes))
    
    grad0 = disp_grad(mol, {0: 0}, harm, options)

    grad_p = np.zeros((n_modes,n_modes))
    grad_n = np.zeros((n_modes,n_modes))
    grad_pp = np.zeros((n_modes,n_modes,n_modes))
    grad_nn = np.zeros((n_modes,n_modes,n_modes))
    grad_3p = np.zeros((n_modes,n_modes))
    grad_3n = np.zeros((n_modes,n_modes))
    grad_np = np.zeros((n_modes,n_modes,n_modes))

    for i in v_ind:
        
        grad_p[:,i] = disp_grad(mol, {i: +1}, harm, options)
        grad_n[:,i] = disp_grad(mol, {i: -1}, harm, options)
        grad_3p[:,i] = disp_grad(mol, {i: +3}, harm, options)
        grad_3n[:,i] = disp_grad(mol, {i: -3}, harm, options)
        
        for j in v_ind:
            if i == j: continue
            
            if j > i:
                grad_pp[:,i,j] = disp_grad(mol, {i: +1, j: +1}, harm, options)
                grad_nn[:,i,j] = disp_grad(mol, {i: -1, j: -1}, harm, options)
                
            if j<i:
                grad_pp[:,i,j] = grad_pp[:,j,i]
                grad_nn[:,i,j] = grad_nn[:,j,i]
            
            grad_np[:,i,j] = disp_grad(mol, {i: -1, j: +1}, harm, options)
            
  
    
    for i in v_ind:
        
        phi_iijj[i,i] = ( grad_3p[i,i] - 3*grad_p[i,i] + 3*grad_n[i,i] - grad_3n[i,i] ) / (8 * disp_size**3)
        
        for j in v_ind:
            phi_ijk[i,j,j] = (grad_p[i,j] + grad_n[i,j] - 2*grad0[i]) / (disp_size**2)
            phi_ijk[j,j,i] = phi_ijk[i,j,j]
            phi_ijk[j,i,j] = phi_ijk[i,j,j]
            
            if i == j: continue
            phi_iijj[i,j] = (grad_pp[i,i,j] + grad_np[i,j,i] - 2*grad_p[i,i] - (grad_np[i,i,j] + grad_nn[i,i,j] - 2*grad_n[i,i])) / (2 * disp_size**3)
                
            
            for k in v_ind:
                if k==j or k==i: continue
                phi_ijk[i,j,k] = (grad_pp[i,j,k] + grad_nn[i,j,k] + 2*grad0[i] - (grad_p[i,j] + grad_p[i,k] + grad_n[i,j] + grad_n[i,k])) / (2 * disp_size**2)
            
            
    phi_ijk = phi_ijk / wave_to_hartree
    phi_iijj = phi_iijj / wave_to_hartree
    
    return phi_ijk, phi_iijj


def force_field_H(mol, harm, options):
    
    n_modes = harm["n_modes"]
    v_ind = harm["v_ind"]
    disp_size = options["DISP_SIZE"]
    wave_to_hartree = qcel.constants.get("inverse meter-hartree relationship") * 100
   
    phi_ijk = np.zeros((n_modes,n_modes,n_modes))
    phi_iijj = np.zeros((n_modes,n_modes))
    
    hess0 = disp_hess(mol, {0: 0}, harm, options)

    hess_p = np.zeros((n_modes,n_modes,n_modes))
    hess_n = np.zeros((n_modes,n_modes,n_modes))

    for i in v_ind:
        
        hess_p[i,:,:] = disp_hess(mol, {i: +1}, harm, options)
        hess_n[i,:,:] = disp_hess(mol, {i: -1}, harm, options) 
        
        phi_iijj[i,i] = (hess_p[i,i,i]  + hess_n[i,i,i] - 2 * hess0[i,i]) / (disp_size**2)
        phi_ijk[i,i,i] = (hess_p[i,i,i] - hess_n[i,i,i]) / (2 * disp_size)


    for i in v_ind:
        for j in v_ind:
        
            if (i != j):
                phi_iijj[i,j] = (hess_p[j,i,i] + hess_n[j,i,i] + hess_p[i,j,j] + hess_n[i,j,j] - 2*hess0[i,i] - 2*hess0[j,j]) / (2 * disp_size**2)
        
            for k in v_ind:
                if (k != i) or (k != j):
                    phi_ijk[i,j,k] = (hess_p[i,j,k] - hess_n[i,j,k] + hess_p[j,k,i] - hess_n[j,k,i] + hess_p[k,i,j] - hess_n[k,i,j]) / (6 * disp_size)

    phi_iijj = phi_iijj / wave_to_hartree
    phi_ijk = phi_ijk / wave_to_hartree
    
    return phi_ijk, phi_iijj 

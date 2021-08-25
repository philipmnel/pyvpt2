import psi4
import numpy as np
import qcelemental as qcel


def disp_energy(mol, disp, modes, disp_size=0.02, method='hf/sto-3g'):
    disp_geom = mol.geometry().np
    for i in disp:
        disp_geom += modes[:,i].reshape(-1,3) * disp_size * disp[i]
    
    disp_mol = mol.clone()
    disp_mol.set_geometry(psi4.core.Matrix.from_array(disp_geom))
    disp_mol.update_geometry()

    E = psi4.energy(method, molecule = disp_mol)
    return E

def force_field_E(mol, disp_size=0.02, method='hf/sto-3g'):

    # Get omegas and modes from harmonic analysis calc
    E0, wfn = psi4.frequency(method, molecule = mol, return_wfn = True)

    omega = wfn.frequency_analysis['omega'].data
    modes = wfn.frequency_analysis['x'].data
    kforce = wfn.frequency_analysis['k'].data
    trv = wfn.frequency_analysis['TRV'].data
    n_modes = len(trv)

    wave_to_hartree = qcel.constants.get("inverse meter-hartree relationship") * 100
    meter_to_bohr = qcel.constants.get("Bohr radius")
    joule_to_hartree = qcel.constants.get("hartree-joule relationship")
    mdyneA_to_hartreebohr = 100 * (meter_to_bohr**2) / (joule_to_hartree)

    omega = omega.real
    omega_au = omega * wave_to_hartree
    kforce_au = kforce * mdyneA_to_hartreebohr
    modes_unitless = np.copy(modes)
    v_ind = []

    for i in range(n_modes):
        if trv[i] == 'V' and omega[i] != 0.0:
            modes_unitless[:,i] *= np.sqrt(omega_au[i]/kforce_au[i])
            v_ind.append(i)
        else:
            modes_unitless[:,i] *= 0.0
    
    phi_ijk = np.zeros((n_modes,n_modes,n_modes))
    phi_iijj = np.zeros((n_modes,n_modes))

    for i in v_ind:

        E3p = disp_energy(mol, {i:3}, modes_unitless, disp_size, method)
        Ep = disp_energy(mol, {i:1}, modes_unitless, disp_size, method)
        En = disp_energy(mol, {i:-1}, modes_unitless, disp_size, method)
        E3n = disp_energy(mol, {i:-3}, modes_unitless, disp_size, method)

        phi_ijk[i,i,i] = ( E3p - 3*Ep + 3*En - E3n ) / (8 * disp_size**3)

        E2p = disp_energy(mol, {i:2}, modes_unitless, disp_size, method)
        E2n = disp_energy(mol, {i:-2}, modes_unitless, disp_size, method)

        phi_iijj[i,i] = ( E2p - 4*Ep + 6*E0 - 4*En + E2n) / (disp_size**4)
        

    for i in v_ind:
        for j in v_ind:

            if i==j: continue

            Epp = disp_energy(mol, {i:1, j:1}, modes_unitless, disp_size, method)
            Epn = disp_energy(mol, {i:1, j:-1}, modes_unitless, disp_size, method)
            Enp = disp_energy(mol, {i:-1, j:1}, modes_unitless, disp_size, method)
            Enn = disp_energy(mol, {i:-1, j:-1}, modes_unitless, disp_size, method)
            
            Eip = disp_energy(mol, {i:1}, modes_unitless, disp_size, method)
            Ejp = disp_energy(mol, {j:1}, modes_unitless, disp_size, method)
            Ein = disp_energy(mol, {i:-1}, modes_unitless, disp_size, method)
            Ejn = disp_energy(mol, {j:-1}, modes_unitless, disp_size, method)

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

                Eppp = disp_energy(mol, {i:1, j:1, k:1}, modes_unitless, disp_size, method) 
                Enpp = disp_energy(mol, {i:-1, j:1, k:1}, modes_unitless, disp_size, method) 
                Epnp = disp_energy(mol, {i:1, j:-1, k:1}, modes_unitless, disp_size, method) 
                Eppn = disp_energy(mol, {i:1, j:1, k:-1}, modes_unitless, disp_size, method) 
                Epnn = disp_energy(mol, {i:1, j:-1, k:-1}, modes_unitless, disp_size, method) 
                Ennp = disp_energy(mol, {i:-1, j:-1, k:1}, modes_unitless, disp_size, method) 
                Enpn = disp_energy(mol, {i:-1, j:1, k:-1}, modes_unitless, disp_size, method) 
                Ennn = disp_energy(mol, {i:-1, j:-1, k:-1}, modes_unitless, disp_size, method) 

                phi_ijk[i,j,k] = ( Eppp - Enpp - Epnp - Eppn + Epnn + Ennp + Enpn - Ennn ) / (8 * disp_size**3)
                phi_ijk[i,k,j] = phi_ijk[i,j,k]
                phi_ijk[j,k,i] = phi_ijk[i,j,k]
                phi_ijk[j,k,j] = phi_ijk[i,j,k]
                phi_ijk[k,i,j] = phi_ijk[i,j,k]
                phi_ijk[k,j,i] = phi_ijk[i,j,k]

    phi_ijk /= wave_to_hartree
    phi_iijj /= wave_to_hartree 

    return phi_ijk, phi_iijj
import psi4
import numpy as np
import quartic

def vpt2(mol, disp_size=0.02, method='hf/sto-3g'):

    E0, wfn = psi4.frequency(method, molecule = mol, return_wfn = True)

    omega = wfn.frequency_analysis['omega'].data
    omega = omega.real
    
    trv = wfn.frequency_analysis['TRV'].data
    n_modes = len(trv)

    v_ind = []

    for i in range(n_modes):
        if trv[i] == 'V' and omega[i] != 0.0:
            v_ind.append(i)

    phi_ijk, phi_iijj = quartic.force_field_E(mol,disp_size,method)

    chi = np.zeros((n_modes, n_modes))
    

    for i in v_ind:
        for j in v_ind:

            if i==j:
                chi[i,i] = phi_iijj[i,i]

                for k in v_ind:

                    chi[i,i] -= ( (8*omega[i]**2 - 3*omega[k]**2) * phi_ijk[i,i,k]**2 ) / (omega[k] * (4*omega[i]**2 - omega[k]**2))

                chi[i,i] /= 16

            else:
                chi[i,j] = phi_iijj[i,j]

                for k in v_ind:

                    chi[i,j] -= (phi_ijk[i,i,k] * phi_ijk[j,j,k]) / omega[k]

                for k in v_ind:

                    delta = (omega[i] + omega[j] - omega[k]) * (omega[i] + omega[j] + omega[k]) * (omega[i] - omega[j] + omega[k]) * (omega[i] - omega[j] - omega[k])

                    chi[i,j] += 2 * omega[k] * (omega[i]**2 + omega[j]**2 - omega[k]**2) * phi_ijk[i,j,k]**2 / delta

                chi[i,j] /= 4

    anharmonic = np.zeros(n_modes)
    
    for i in v_ind:

        anharmonic[i] = 2*chi[i,i]

        for j in v_ind:
            if j != i:
                anharmonic[i] += 0.5 * chi[i,j]

    return omega, anharmonic    
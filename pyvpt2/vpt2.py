import psi4
import numpy as np
import quartic
import qcelemental as qcel

def harmonic(mol, options):

    method = options["METHOD"]

    E0, wfn = psi4.frequency(method, molecule = mol, return_wfn = True)

    omega = wfn.frequency_analysis['omega'].data
    modes = wfn.frequency_analysis['x'].data
    kforce = wfn.frequency_analysis['k'].data
    trv = wfn.frequency_analysis['TRV'].data
    q = wfn.frequency_analysis['q'].data
    n_modes = len(trv)

    wave_to_hartree = qcel.constants.get("inverse meter-hartree relationship") * 100
    meter_to_bohr = qcel.constants.get("Bohr radius")
    joule_to_hartree = qcel.constants.get("hartree-joule relationship")
    mdyneA_to_hartreebohr = 100 * (meter_to_bohr**2) / (joule_to_hartree)

    omega = omega.real
    omega_au = omega * wave_to_hartree
    kforce_au = kforce * mdyneA_to_hartreebohr
    modes_unitless = np.copy(modes)
    gamma = omega_au / kforce_au
    v_ind = []

    for i in range(n_modes):
        if trv[i] == 'V' and omega[i] != 0.0:
            modes_unitless[:,i] *= np.sqrt(gamma[i])
            v_ind.append(i)
        else:
            modes_unitless[:,i] *= 0.0
    
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

def vpt2(mol, options=None):

    if options is None:
        options = {}

    options = {k.upper(): v for k, v in sorted(options.items())}

    if 'DISP_SIZE' not in options:
        options['DISP_SIZE'] = 0.02
    if 'METHOD' not in options:
        options['METHOD'] = 'SCF'
    if 'FD' not in options:
        options['FD'] = 'HESSIAN'

    harm = harmonic(mol, options)
    n_modes = harm["n_modes"]
    omega = harm["omega"]
    v_ind = harm["v_ind"]

    if (options['FD'] == 'ENERGY'):
        phi_ijk, phi_iijj = quartic.force_field_E(mol, harm, options)
    elif(options['FD'] == 'HESSIAN'):
        phi_ijk, phi_iijj = quartic.force_field_H(mol, harm, options)

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
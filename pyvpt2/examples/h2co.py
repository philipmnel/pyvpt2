import psi4
import numpy as np
from pyvpt2 import vpt2

psi4.set_memory('16gb')
psi4.core.set_num_threads(6)

mol = psi4.geometry("""
    C           -0.564304251071     0.179712875544    -0.000000000143
    O            0.564203907179    -0.179680931291     0.000000000145
    H           -1.117625698377     0.355928174750     0.924258616963
    H           -1.117625697898     0.355928174598    -0.924258617559
""")

psi4.set_options({'g_convergence': 'GAU_VERYTIGHT',
                'd_convergence': 1e-12,
                'e_convergence': 1e-12,
                'scf_type': 'direct',
                'basis': '6-31g*',
                'puream': True })

options = {'METHOD': 'HF/6-31g*',
           'FD':'HESSIAN',
           'DISP_SIZE': 0.1}

omega, anharmonic = vpt2(mol, options)

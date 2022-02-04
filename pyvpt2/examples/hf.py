import psi4
import numpy as np
from pyvpt2 import vpt2

psi4.set_memory('16gb')
psi4.core.set_num_threads(6)

mol = psi4.geometry("""
nocom
noreorient

H
F 1 R1

R1 = 0.920853
""")

psi4.set_options({'d_convergence': 1e-12,
                'e_convergence': 1e-12,
                'basis': '6-31g',
                'puream': True })

options = {'METHOD': 'HF/6-31g',
           'FD':'HESSIAN',
           'DISP_SIZE': 0.1}

results = vpt2(mol, options)

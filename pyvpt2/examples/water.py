import psi4
import numpy as np
from pyvpt2 import vpt2

psi4.set_memory('16gb')
psi4.core.set_num_threads(6)

mol = psi4.geometry("""
nocom
noreorient

O
H 1 R1
H 1 R1 2 A

R1 = 0.94731025924472878064
A = 105.50289

symmetry c2v
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

results = vpt2(mol, options)

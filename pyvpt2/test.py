import psi4
import numpy as np
from vpt2 import vpt2

psi4.set_memory('16gb')
psi4.core.set_num_threads(6)

mol = psi4.geometry("""
nocom
noreorient

O
H 1 R1
H 1 R2 2 A

R1 = 0.989409296024027
R2 = 0.989409296024027
A = 100.026881300680799

symmetry c1
""")

psi4.set_options({'g_convergence': 'GAU_TIGHT',
                'e_convergence': 1e-12,
                'd_convergence': 1e-10,
                'basis': 'sto-3g'})

E, wfn = psi4.optimize('hf', return_wfn=True)
mol.update_geometry()

options = {'METHOD': 'SCF',
           'FD':'HESSIAN',
           'DISP_SIZE': 0.005}

omega, anharmonic = vpt2(mol, options)

#print(omega)
#print(anharmonic)                                                     
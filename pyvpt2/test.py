import psi4
import numpy as np
from vpt2 import vpt2

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

psi4.set_options({'scf_type': 'pk',
                'e_convergence': 1e-12,
                'd_convergence': 1e-10,
                'basis': '6-31g'})

E, wfn = psi4.optimize('hf', return_wfn=True)
mol.update_geometry()

options = {'METHOD': 'SCF',
           'FD':'ENERGY',
           'DISP_SIZE': 0.02}

omega, anharmonic = vpt2(mol, options)

print(omega)
print(anharmonic)                                                     
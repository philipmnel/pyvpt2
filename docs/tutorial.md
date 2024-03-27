pyvpt2 is a python module for calculating anharmonic vibrational frequencies using second-order vibrational perturbation theory (VPT2). Harmonic frequencies and cubic/quartic force constants are calculated using psi4 as a quantum chemistry backend.

Example input:
```python
import psi4
import pyvpt2

mol = psi4.geometry("""
O
H 1 R
H 1 R 2 A
   
R = 0.9473
A =  105.5
""")

# set psi4 module options here
psi4.set_options({'e_convergence': 10,
                  'd_convergence': 10})

options = {'METHOD': 'scf/6-31g*',
            'FD': 'HESSIAN}

ret = pyvpt2.vpt2(mol, **options)
```

### Options list:
* `DISP_SIZE` (Default: 0.05) Displacement size used in finite-difference calculations.
* `METHOD` (Default: "SCF") Method formatted as string for psi4. In multilevel calculation, used for harmonic portion only.
* `METHOD_2` (Default: None) Method for cubic/quartic force constants. Used only for multilevel calculations.
* `FD` (Default: "HESSIAN") Level of finite-difference calculation. Choose highest analytical derivative available for chosen method. Options: "ENERGY", "GRADIENT", or "HESSIAN"   
* `FERMI` (Default: True) Deperturb Fermi resonances?
* `GVPT2` (Default: False) Diagonalize Fermi resonances? Requires `FERMI` to be enabled.
* `FERMI_OMEGA_THRESH` (Default: 200) Frequency difference threshold below which to deperturb resonances.
* `FERMI_K_THRESH` (Default: 1) Coupling threshold above which to depertub resonances.
* `RETURN_PLAN` (Default: False) Return a plan of tasks to be sent to a QCPortal client?

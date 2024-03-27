pyvpt2 is a python module for calculating anharmonic vibrational frequencies using second-order vibrational perturbation theory (VPT2). Harmonic frequencies and cubic/quartic force constants are calculated using psi4 as a quantum chemistry backend.

Example input:
```python
import qcelemental as qcel
import pyvpt2

mol = qcel.models.Molecule.from_data("""
O   0.0   0.0         -0.12126642
H   0.0  -1.42495308   0.96229308
H   0.0   1.42495308   0.96229308
""") 

# set method here
qc_model = {"method": "scf",
         "basis": "6-31g*"}

# set qc level options here
qc_kwargs = {'d_convergence': 1e-10,
            'e_convergence': 1e-10,
            }

# set vpt2 level options here
options = {'FD': 'HESSIAN',
            'DISP_SIZE': 0.05,
            'QC_PROGRAM': 'psi4',
            }

inp = {"molecule": mol,
        "input_specification": [{"model": qc_model,
                                "keywords": qc_kwargs}],
        "keywords": options}

results = pyvpt2.vpt2_from_schema(inp)
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

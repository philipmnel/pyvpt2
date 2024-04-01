### Basic Example
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
qc_kwargs = {"d_convergence": 1e-10,
            "e_convergence": 1e-10,
            }

# set vpt2 level options here
options = {"FD": 'HESSIAN',
            "DISP_SIZE": 0.05,
            "QC_PROGRAM": "psi4",
            }

inp = {"molecule": mol,
        "input_specification": [{"model": qc_model,
                                "keywords": qc_kwargs}],
        "keywords": options}

results = pyvpt2.vpt2_from_schema(inp)
```

pyVPT2 accepts QCSchema specification of the input molecule (`"molecule"`) and quantum chemistry model specification (`"input_specification"`).
pyVPT2 specific options are specified with the `"keywords"` section.
Details on specifying molecular geometries can be found in the [QCElemental documentation](https://molssi.github.io/QCElemental/model_molecule.html).
Molecular geometries should be tightly converged before frequency analysis. Choices for QCEngine intregrated geometry optimizers include [optking](https://github.com/psi-rking/optking) and [geomeTRIC](https://github.com/leeping/geomeTRIC).
Choices for QCEngine supported quantum chemistry programs can be found [here](https://molssi.github.io/QCEngine/program_overview.html).
It is very important to note that finite-difference calculations require tightly converged energies for numerical stability. It is highly advised to specify tight convergence criteria in the QC program keywords.

### Multilevel Computations

Because of the high cost of calculating the third and fourth derivatives required for VPT2, it is common to compute the anharmonic frequencies at a cheaper level of theory than the harmonic portion. 
It is advantageous to compute the anharmonic portion at the high-level geometry.
pyVPT2 can combine the harmonic and anharmonic portions at different levels of theory using the `MULTILEVEL` keyword, passing the different methods as list in `input_specification`.


Example input:
```python
import qcelemental as qcel
import pyvpt2

mol = qcel.models.Molecule.from_data("""
O   0.0   0.0         -0.12126642
H   0.0  -1.42495308   0.96229308
H   0.0   1.42495308   0.96229308
""") 

# set high-level method here
qc_model1 = {"method": "ccsd(t)",
         "basis": "cc-pvtz"}

# set low-level method here
qc_model2 = {"method": "mp2",
         "basis": "cc-pvtz"}

# set qc level options here
qc_kwargs = {"d_convergence": 1e-10,
            "e_convergence": 1e-10,
            }

# set vpt2 level options here
options = {"FD": 'HESSIAN',
            "DISP_SIZE": 0.05,
            "QC_PROGRAM": "psi4",
            "MULTILEVEL": True,
            }

inp = {"molecule": mol,
        "input_specification": [{"model": qc_model1,
                                "keywords": qc_kwargs},
                                {"model": qc_model2,
                                "keywords": qc_kwargs}],
        "keywords": options}

results = pyvpt2.vpt2_from_schema(inp)
```

### Distributed Computations with QCFractal Integration

If QCFractal is installed, one can distribute the finite-difference steps to a QCFractal server
by enabling the `RETURN_PLAN` keyword. 
For example:
```python
import qcelemental as qcel
import pyvpt2
from qcportal import PortalClient

client = PortalClient("https://[my qcfractal server here]")

mol = qcel.models.Molecule.from_data("""
O   0.0   0.0         -0.12126642
H   0.0  -1.42495308   0.96229308
H   0.0   1.42495308   0.96229308
""") 

# set high-level method here
qc_model1 = {"method": "ccsd(t)",
         "basis": "cc-pvtz"}

# set low-level method here
qc_model2 = {"method": "mp2",
         "basis": "cc-pvtz"}

# set qc level options here
qc_kwargs = {"d_convergence": 1e-10,
            "e_convergence": 1e-10,
            }

# set vpt2 level options here
options = {"FD": 'HESSIAN',
            "DISP_SIZE": 0.05,
            "QC_PROGRAM": "psi4",
            "MULTILEVEL": True,
            "RETURN_PLAN": True,
            }

inp = {"molecule": mol,
        "input_specification": [{"model": qc_model1,
                                "keywords": qc_kwargs},
                                {"model": qc_model2,
                                "keywords": qc_kwargs}],
        "keywords": options}

harmonic_plan = pyvpt2.vpt2_from_schema(inp)
harmonic_plan.compute(client=client)
harmonic_ret = harmonic_plan.get_results(client=client)

plan = pyvpt2.vpt2_from_harmonic(harmonic_ret, qc_spec=inp["input_specification"][1], **options)
plan.compute(client=client)
ret = plan.get_results(client=client)
results = pyvpt2.process_vpt2(ret, **options)
```


### Options list:
* `DISP_SIZE` (Default: 0.05) Displacement size used in finite-difference calculations.
* `FD` (Default: "HESSIAN") Level of finite-difference calculation. Choose highest analytical derivative available for chosen method. Options: "ENERGY", "GRADIENT", or "HESSIAN"   
* `FERMI` (Default: True) Deperturb Fermi resonances?
* `GVPT2` (Default: False) Diagonalize Fermi resonances? Requires `FERMI` to be enabled.
* `FERMI_OMEGA_THRESH` (Default: 200) Frequency difference threshold below which to deperturb resonances.
* `FERMI_K_THRESH` (Default: 1) Coupling threshold above which to depertub resonances.
* `RETURN_PLAN` (Default: False) Return a plan of tasks to be sent to a QCPortal client?
* `VPT2_OMEGA_THRESH` (Default: 1) Frequency below which to omit from VPT2 treatment`
* `QC_PROGRAM` (Default: "psi4") QC program to run 
* `MULTILEVEL` (Default: False) Use different levels of theory for harmonic and anharmonic portions
* `TASK_CONFIG` qcengine task configuration settings


### Troubleshooting

The most common issue encountered is the numerical stability of the finite-difference third/fourth derivatives.
Tight convergence of energies and tight DFT grids are usually required.

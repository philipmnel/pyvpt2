import logging

import psi4
from psi4.driver.driver_cbs import CompositeComputer, composite_procedures
from psi4.driver.driver_findif import FiniteDifferenceComputer
from psi4.driver.task_planner import expand_cbs_methods
from qcelemental.models import Molecule
from qcelemental.models.procedures import QCInputSpecification

from .quartic import QuarticComputer
from .task_base import AtomicComputer, BaseComputer

logger = logging.getLogger(f"psi4.{__name__}")

def hessian_planner(molecule: Molecule, qc_spec: QCInputSpecification, **kwargs) -> FiniteDifferenceComputer:
    """
    Generates computer for hessian finite difference calcutations

    Parameters
    ----------
    molecule: Molecule
        Input molecule
    qc_spec: QCInputSpecification
        Input specification

    Returns
    -------
    FiniteDifference
        Computer for finite difference calculations
    """
    # keywords are the psi4 option keywords

    # findif computer is expecting a psi4 molecule
    molecule = psi4.core.Molecule.from_schema(molecule.dict())

    keywords = qc_spec.keywords
    program = kwargs.get("QC_PROGRAM")
    if keywords == {} and program == "psi4":
        keywords = psi4.p4util.prepare_options_for_set_options()
    keywords.setdefault("function_kwargs", {})

    der_dict = {"ENERGY": 0, "GRADIENT": 1, "HESSIAN": 2}
    dermode = kwargs["FD"]
    driver = dermode.lower()
    dermode = [2, der_dict[dermode]]
    method = qc_spec.model.method
    basis = qc_spec.model.basis
    findif_kwargs = {"findif_stencil_size": 5, "findif_step_size": 0.005, }

    # Expand CBS methods
    method, basis, cbsmeta = expand_cbs_methods(method=method, basis=basis, driver=driver)
    if method in composite_procedures:
        findif_kwargs.update(cbsmeta)
        findif_kwargs.update({'cbs_metadata': composite_procedures[method](**kwargs)})
        method = 'cbs'

    packet = {"molecule": molecule, "driver": "hessian", "method": method, "basis": basis, "keywords": keywords, "program": program}

    if method == "cbs":
        if program not in ["psi4"]:
            raise AssertionError("Composite methods enabled only for psi4 currently")

        findif_kwargs.update(cbsmeta)

        if driver == "hessian":
            logger.info(f'PLANNING CBS:  packet={packet} kw={findif_kwargs}')
            return CompositeComputer(**packet, **findif_kwargs)

        else:
            logger.info(
                f'PLANNING FD(CBS):  dermode={dermode} packet={packet} kw={findif_kwargs}')
            return FiniteDifferenceComputer(**packet,
                                           findif_mode=dermode,
                                           computer=CompositeComputer,
                                           **findif_kwargs,)

    else:
        if driver == "hessian":
            logger.info(f'PLANNING ATOMIC:  packet={packet} kw={findif_kwargs}')
            return AtomicComputer(**packet, **findif_kwargs)

        else:
            logger.info(
                f'PLANNING FD:  dermode={dermode} packet={packet} kw={findif_kwargs}')
            return FiniteDifferenceComputer(**packet,
                                           findif_mode=dermode,
                                           computer=AtomicComputer,
                                           **findif_kwargs,)

def quartic_planner(molecule: Molecule, qc_spec: QCInputSpecification, **kwargs) -> QuarticComputer:
    """
    Generates computer for quartic finite difference calcutations

    Parameters
    ----------
    molecule: Molecule
        Input molecule
    qc_spec: QCInputSpecification
        Input specification

    Returns
    -------
    QuarticComputer
        Computer for finite difference calculations
    """
    # keywords are the psi4 option keywords

    keywords = qc_spec.keywords
    program = kwargs["options"]["QC_PROGRAM"]
    if keywords == {} and program == "psi4":
        keywords = psi4.p4util.prepare_options_for_set_options()

    driver = "hessian"
    dermode = kwargs["options"]["FD"]
    method = qc_spec.model.method
    basis = qc_spec.model.basis

    # Expand CBS methods
    method, basis, cbsmeta = expand_cbs_methods(method=method, basis=basis, driver=driver)
    if method in composite_procedures:
        kwargs.update(cbsmeta)
        kwargs.update({'cbs_metadata': composite_procedures[method](**kwargs)})
        method = 'cbs'

    packet = {"molecule": molecule, "driver": driver, "method": method, "basis": basis, "keywords": keywords, "program": program}

    if method == "cbs":
        if program not in ["psi4"]:
            raise AssertionError("Composite methods enabled only for psi4 currently")

        kwargs.update(cbsmeta)
        logger.info(
            f'PLANNING FD(CBS):  dermode={dermode} packet={packet} kw={kwargs}')
        return QuarticComputer(**packet, computer=CompositeComputer, **kwargs)

    else:
        logger.info(
            f'PLANNING FD:  dermode={dermode} keywords={keywords} kw={kwargs}')
        return QuarticComputer(**packet, **kwargs)

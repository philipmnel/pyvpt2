from typing import Any, Dict, Optional

try:
    from pydantic.v1 import Field, conlist
except ImportError:
    from pydantic import Field, conlist

from qcelemental.models import Molecule
from qcelemental.models.basemodels import ProtoModel
from qcelemental.models.common_models import Model, Provenance
from qcelemental.models.procedures import QCInputSpecification
from qcelemental.models.types import Array

#from pyvpt2 import __version__


def provenance_stamp() -> Dict[str, str]:
    """
    Return provenance stamp for pyvpt2
    """
    provenance = {
        "creator": "pyVPT2",
        "version": 0.0,
        "routine": "VPT2"
    }
    return provenance

class VPTInput(ProtoModel):
    molecule: Molecule = Field(..., description="Input molecule")
    keywords: Dict[str, Any] = Field({}, description="pyVPT2 keywords")
    input_specification: conlist(item_type=QCInputSpecification, min_items=1, max_items=2) = Field(..., description="QC input model")
    provenance: Provenance = Field(Provenance(**provenance_stamp()), description="Provenance")


class VPTResult(ProtoModel):
    molecule: Molecule = Field(..., description="The molecule used in the computation.")
    model: Model = Field(..., description="Quantum chemistry model.")
    keywords: Dict[str, Any] = Field({}, description="pyVPT2 keywords")
    omega: Array[float] = Field(..., description="Harmonic vibrational frequencies")
    nu: Array[float] = Field(..., description="VPT2 Anharmonic vibrational frequencies")
    harmonic_zpve: Array[float] = Field(..., description="Harmonic zero-point vibrational energy")
    anharmonic_zpve : Array[float] = Field(..., description="VPT2 Anharmonic zero-point vibrational energy")
    harmonic_intensity: Optional[Array[float]] = Field(..., description="Harmonic IR intensities")
    chi: Array[float] = Field(..., description="VPT2 anharmonicity constants")
    phi_ijk: Array[float] = Field(..., description="Cubic derivatives")
    phi_iijj: Array[float] = Field(..., description="Semi-diagonal quartic derivatives")
    extras: Dict[str, Any] = Field({}, description="Fun extras, e.g. depertubed freqs")
    provenance: Provenance = Field(..., description="Provenance")

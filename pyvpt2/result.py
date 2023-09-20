from typing import Any, Dict, Optional

try:
    from pydantic.v1 import Field
except ImportError:
    from pydantic import Field
from qcelemental.models import Molecule
from qcelemental.models.basemodels import ProtoModel
from qcelemental.models.common_models import Model
from qcelemental.models.types import Array


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

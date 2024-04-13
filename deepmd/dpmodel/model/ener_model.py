from .make_model import make_model
from .dp_model import DPModelCommon
from deepmd.dpmodel.atomic_model.dp_atomic_model import DPAtomicModel

class EnergyModel(DPModelCommon, make_model(DPAtomicModel)):
    pass
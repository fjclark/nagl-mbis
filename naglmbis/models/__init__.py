from naglmbis.models.base_model import ComputePartialPolarised, MBISChargeModel
from naglmbis.models.models import CHARGE_MODELS, load_charge_model

__all__ = [MBISChargeModel, ComputePartialPolarised, CHARGE_MODELS, load_charge_model]


def __getattr__(name):
    # the nagl based model needs dgl, so only import it when requested
    if name == "MBISGraphModel":
        from naglmbis.models._nagl_model import MBISGraphModel

        return MBISGraphModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

import os
from importlib.resources import files
from typing import Literal

# The ring sizes used by default by naglmbis.features.AtomInRingOfSize. Kept here,
# rather than in naglmbis.features, so they can be used without the ``bismuthadams1/nagl`` fork.
DEFAULT_RING_SIZES = [3, 4, 5, 6, 7, 8]


def get_model_weights(model_type: Literal["charge", "volume"], model_name: str) -> str:
    """
    Get the model weights from the naglmbis package.

    """

    fn = str(files("naglmbis") / "data" / "models" / model_type / model_name)
    if not os.path.exists(fn):
        raise ValueError(
            f"{model_name} does not exist. If you have just added it, you'll need to re-install."
        )
    return fn

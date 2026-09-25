import re
from typing import Literal

import torch
from openff.nagl import GNNModel

from naglmbis.models.base_model import MBISChargeModel
from naglmbis.utils import DEFAULT_RING_SIZES, get_model_weights

charge_weights = {
    "nagl-v1-mbis": {"checkpoint_path": "nagl-v1-mbis.ckpt"},
    "nagl-v1-mbis-dipole": {"checkpoint_path": "nagl-v1-mbis-dipole.ckpt"},
    "nagl-gas-charge-wb": {"checkpoint_path" : "nagl-gas-charge.ckpt"},
    "nagl-gas-charge-dipole-wb": {"checkpoint_path" : "nagl-gas-charge-dipole.ckpt"},
    "nagl-gas-charge-dipole-esp-wb-default":{"checkpoint_path":"nagl-gas-charge-dipole-esp.ckpt"},
    "nagl-water-charge-wb":  {"checkpoint_path" : "nagl-water-charge.ckpt"},
    "nagl-water-charge-dipole-wb":  {"checkpoint_path" : "nagl-water-charge-dipole.ckpt"},
    "nagl-water-charge-dipole-esp-wb-default":{"checkpoint_path":"nagl-water-charge-dipole-esp.ckpt"},
    "nagl-gas-esp-wb-2A": {"checkpoint_path":"nagl-gas-esp-2A.ckpt"},
    "nagl-gas-esp-wb-15A":{"checkpoint_path": "nagl-gas-esp-15A.ckpt"},
}

CHARGE_MODELS = Literal["nagl-v1-mbis-dipole",
                        "nagl-v1-mbis",
                        "nagl-gas-charge-wb",
                        "nagl-gas-charge-dipole-wb",
                        "nagl-gas-charge-dipole-esp-wb-default",
                        "nagl-water-charge-wb",
                        "nagl-water-charge-dipole-wb",
                        "nagl-water-charge-dipole-esp-wb-default",
                        "nagl-gas-esp-wb-2A",
                        "nagl-gas-esp-wb-15A",
                        ]


def _convert_layers(hidden_feats, activation, dropout, **extra) -> list[dict]:
    """Convert ``bismuthadams1/nagl`` fork layer settings to openff-nagl layer configs."""
    dropout = [0.0] * len(hidden_feats) if dropout is None else dropout
    return [
        {
            "hidden_feature_size": hidden_feature_size,
            "activation_function": act,
            "dropout": drop,
            **extra,
        }
        for hidden_feature_size, act, drop in zip(hidden_feats, activation, dropout, strict=True)
    ]


def _convert_atom_feature(feature: dict) -> list[dict]:
    """
    Convert a ``bismuthadams1/nagl`` fork atom feature config to the equivalent
    openff-nagl features.
    """
    feature_type = feature["type"]
    options = feature.keys() - {"type"}
    one_hot_names = {"element": "atomic_element", "connectivity": "atom_connectivity"}
    if feature_type in one_hot_names and options <= {"values"}:
        return [{"name": one_hot_names[feature_type], "categories": feature["values"]}]
    if feature_type == "ringofsize" and options <= {"ring_sizes"}:
        # naglmbis.features.AtomInRingOfSize, one column per ring size. The
        # checkpoints do not store the ring sizes, so they use the default.
        ring_sizes = feature.get("ring_sizes", DEFAULT_RING_SIZES)
        return [{"name": "atom_in_ring_of_size", "ring_size": n} for n in ring_sizes]
    raise NotImplementedError(f"Unsupported atom feature: {feature_type} {feature}")


def _convert_config(config: dict) -> dict:
    """
    Convert the model config of a checkpoint trained with the ``bismuthadams1/nagl``
    fork into an equivalent ``openff-nagl`` ``ModelConfig``.
    """
    model = config["model"]
    if model["bond_features"]:
        raise NotImplementedError("Bond features are not supported")

    convolution = model["convolution"]
    if convolution["type"] != "SAGEConv":
        raise NotImplementedError(f"Unsupported convolution: {convolution['type']}")

    readouts = {}
    for name, readout in model["readouts"].items():
        if readout["pooling"] != "atom" or readout["postprocess"] != "charges":
            raise NotImplementedError(f"Unsupported readout: {readout}")
        # openff-nagl appends the final (electronegativity, hardness) output layer
        # itself, so leave it out of the config
        readouts[name] = {
            "pooling": "atoms",
            "layers": _convert_layers(**readout["forward"])[:-1],
            "postprocess": "compute_partial_charges",
        }

    return {
        "version": "0.1",
        "atom_features": [
            converted
            for feature in model["atom_features"]
            for converted in _convert_atom_feature(feature)
        ],
        "bond_features": [],
        "convolution": {
            "architecture": "SAGEConv",
            "layers": _convert_layers(
                hidden_feats=convolution["hidden_feats"],
                activation=convolution["activation"],
                dropout=convolution["dropout"],
                aggregator_type=convolution.get("aggregator") or "mean",
            ),
        },
        "readouts": readouts,
    }


def _convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename the ``bismuthadams1/nagl`` fork weights to the equivalent openff-nagl weights."""

    def rename(key: str) -> str:
        key = re.sub(r"^convolution_module\.", "convolution_module.gcn_layers.", key)
        return re.sub(
            r"^(readout_modules\.[^.]+)\.forward_layers\.", r"\1.readout_layers.", key
        )

    return {rename(key): value for key, value in state_dict.items()}


def load_checkpoint(checkpoint_path: str) -> MBISChargeModel:
    """
    Load a model checkpoint trained with the ``bismuthadams1/nagl`` fork of NAGL.
    """
    model_data = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=True
    )
    gnn_model = GNNModel(
        config=_convert_config(model_data["hyper_parameters"]["config"])
    )
    gnn_model.load_state_dict(_convert_state_dict(model_data["state_dict"]))
    gnn_model.eval()
    return MBISChargeModel(gnn_model)


def load_charge_model(charge_model: CHARGE_MODELS) -> MBISChargeModel:
    """
    Load up one of the predefined charge models, this will load the weights and parameter settings.
    """
    weight_path = get_model_weights(
        model_type="charge", model_name=charge_weights[charge_model]["checkpoint_path"]
    )
    return load_checkpoint(weight_path)


# def load_volume_model(volume_model: VOLUME_MODELS) -> MBISGraphModel:
#     """
#     Load one of the predefined volume models, this will load the weights and parameter settings.
#     """
#     weight_path = get_model_weights(
#         model_type="volume", model_name=volume_weights[volume_model]["path"]
#     )
#     return volume_weights[volume_model]["model"].load_from_checkpoint(weight_path)

"""
Tests which only need the minimal ``inference`` environment (no ``bismuthadams1/nagl`` fork or dgl).
"""

import pytest
import torch
from rdkit import Chem

from naglmbis.models import load_charge_model
from naglmbis.models.base_model import ComputePartialPolarised
from naglmbis.models.models import charge_weights


@pytest.mark.parametrize("charge_model", list(charge_weights))
def test_load_all_charge_models(charge_model, methanol_rdkit):
    """Make sure every charge model in ``charge_weights`` loads and gives sensible charges."""
    model = load_charge_model(charge_model=charge_model)
    charges = model.compute_properties(molecule=methanol_rdkit)["mbis-charges"].detach()

    assert charges.shape == (6, 1)
    assert torch.isfinite(charges).all()
    # methanol is neutral
    assert charges.sum().item() == pytest.approx(0.0, abs=1e-4)


def test_partial_polarised(methanol_rdkit):
    """Make sure the polarised charges interpolate between the gas and water models."""
    model_gas = load_charge_model(charge_model="nagl-gas-charge-wb")
    model_water = load_charge_model(charge_model="nagl-water-charge-wb")
    gas = model_gas.compute_properties(molecule=methanol_rdkit)["mbis-charges"]
    water = model_water.compute_properties(molecule=methanol_rdkit)["mbis-charges"]

    for alpha, expected in [(0.0, gas), (1.0, water), (0.5, (gas + water) / 2)]:
        polarised = ComputePartialPolarised(
            model_gas=model_gas, model_water=model_water, alpha=alpha
        ).compute_polarised_charges(molecule=methanol_rdkit)
        assert torch.allclose(polarised, expected, atol=1e-6)


def test_radicals_not_supported():
    """The OpenFF toolkit can not parse radicals, make sure a clear error is raised."""
    from openff.toolkit.utils.exceptions import RadicalsNotSupportedError
    model = load_charge_model(charge_model="nagl-v1-mbis")
    with pytest.raises(RadicalsNotSupportedError):
        model.compute_properties(molecule=Chem.AddHs(Chem.MolFromSmiles("[CH3]")))


def test_fragments_predicted_separately():
    """Make sure each fragment of a salt gets the same charges as on its own."""
    model = load_charge_model(charge_model="nagl-v1-mbis")
    # the atoms of the two fragments are interleaved, as the hydrogens come last
    mixture = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-].O"))
    acetate_indices, water_indices = Chem.GetMolFrags(mixture)
    charges = model.compute_properties(molecule=mixture)["mbis-charges"]
    assert charges.shape == (10, 1)

    acetate = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
    water = Chem.AddHs(Chem.MolFromSmiles("O"))
    for indices, fragment, total in [
        (acetate_indices, acetate, -1.0),
        (water_indices, water, 0.0),
    ]:
        expected = model.compute_properties(molecule=fragment)["mbis-charges"]
        assert torch.allclose(charges[list(indices)], expected)
        assert charges[list(indices)].sum().item() == pytest.approx(total, abs=1e-5)


def test_model_in_eval_mode():
    """Make sure the loaded model stays in eval mode, even after converting a dgl
    model to openff-nagl's pure PyTorch layers, so dropout is not applied."""
    model = load_charge_model(charge_model="nagl-v1-mbis")
    assert not any(module.training for module in model.gnn_model.modules())


def test_latent_embeddings(methanol_rdkit):
    """Make sure the latent embeddings have one row per atom and each fragment of a
    mixture gets the same embeddings as on its own."""
    model = load_charge_model(charge_model="nagl-v1-mbis")
    hidden_size = model.gnn_model.config.convolution.layers[-1].hidden_feature_size
    embeddings = model.compute_latent_embeddings(molecule=methanol_rdkit)
    assert embeddings.shape == (6, hidden_size)
    assert torch.isfinite(embeddings).all()

    mixture = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-].O"))
    embeddings = model.compute_latent_embeddings(molecule=mixture)
    for indices, smiles in zip(Chem.GetMolFrags(mixture), ["CC(=O)[O-]", "O"]):
        fragment = Chem.AddHs(Chem.MolFromSmiles(smiles))
        expected = model.compute_latent_embeddings(molecule=fragment)
        assert torch.allclose(embeddings[list(indices)], expected)

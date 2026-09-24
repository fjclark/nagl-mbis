"""
Tests which only need the minimal ``inference`` environment (no OpenFF toolkit).
"""

import pytest
import torch

from naglmbis.models import load_charge_model
from naglmbis.models.base_model import ComputePartialPolarised
from naglmbis.models.models import charge_weights


@pytest.mark.parametrize("charge_model", list(charge_weights))
def test_load_all_charge_models(charge_model, methanol_rdkit):
    """Make sure every shipped charge model loads and gives sensible charges."""
    model = load_charge_model(charge_model=charge_model)
    charges = model.compute_properties(molecule=methanol_rdkit)["mbis-charges"].detach()

    assert charges.shape == (6, 1)
    assert torch.isfinite(charges).all()
    # methanol is neutral
    assert charges.sum().item() == pytest.approx(0.0, abs=1e-4)


def test_nagl_v1_mbis_reference(methanol_rdkit):
    """Make sure a molecule built with RDKit alone reproduces the reference charges."""
    model = load_charge_model(charge_model="nagl-v1-mbis")
    charges = model.compute_properties(molecule=methanol_rdkit)["mbis-charges"].detach()
    ref = torch.Tensor([[0.0835], [-0.6821], [0.0491], [0.0491], [0.0491], [0.4515]])
    assert torch.allclose(charges, ref, atol=1e-4)


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
    from rdkit import Chem

    model = load_charge_model(charge_model="nagl-v1-mbis")
    with pytest.raises(RadicalsNotSupportedError):
        model.compute_properties(molecule=Chem.AddHs(Chem.MolFromSmiles("[CH3]")))


def test_fragments_predicted_separately():
    """Make sure each fragment of a salt gets the same charges as on its own."""
    from rdkit import Chem

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

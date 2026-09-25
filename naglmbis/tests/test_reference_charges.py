"""
Check the models reproduce the charges of the original nagl + dgl implementation
(evaluated one fragment at a time), stored in ``data/reference_charges.npz`` (see
``data/generate_reference_charges.py``).
"""

import pathlib

import numpy as np
import pytest
from rdkit import Chem

from naglmbis.models.models import charge_weights, load_checkpoint
from naglmbis.utils import get_model_weights

REFERENCE_FILE = pathlib.Path(__file__).parent / "data" / "reference_charges.npz"
REFERENCES = np.load(REFERENCE_FILE)
SMILES = list(REFERENCES["smiles"])
FAILURES = set(REFERENCES["failures"])
CHECKPOINTS = sorted({key.split("|")[0] for key in REFERENCES.files if "|" in key})


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_reference_charges(checkpoint):
    """Make sure every molecule gives the same charges as the original implementation."""
    model = load_checkpoint(get_model_weights("charge", checkpoint))

    mismatches = []
    for i, smiles in enumerate(SMILES):
        key = f"{checkpoint}|{i}"
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))

        if key in FAILURES:
            # the original implementation could not handle this molecule
            with pytest.raises(ValueError):
                model.compute_properties(molecule)
            continue

        charges = model.compute_properties(molecule)["mbis-charges"].detach().numpy()
        reference = REFERENCES[key]
        assert charges.shape == reference.shape, smiles

        deviation = np.abs(charges - reference).max()
        if deviation > 1e-5:
            mismatches.append(f"{smiles}: max deviation {deviation:.2e}")

    assert not mismatches, "\n".join(mismatches)


def test_reference_file_covers_all_models():
    """Make sure every charge model in ``charge_weights`` has reference charges."""
    assert {w["checkpoint_path"] for w in charge_weights.values()} <= set(CHECKPOINTS)

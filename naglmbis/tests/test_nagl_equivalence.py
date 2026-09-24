"""
Compare the openff-nagl based models directly against the original nagl + dgl
implementation. These tests only run when the nagl fork and dgl are installed.
"""

import warnings

import numpy as np
import pytest
import torch
from rdkit import Chem

pytest.importorskip("dgl")
pytest.importorskip("nagl")

from nagl.features import AtomFeaturizer  # noqa: E402
from openff.nagl.molecule._graph.molecule import GraphMolecule  # noqa: E402
from openff.toolkit import Molecule  # noqa: E402

from naglmbis.models.models import load_checkpoint  # noqa: E402
from naglmbis.tests._original import (  # noqa: E402
    load_original_model,
    original_charges,
)
from naglmbis.tests.test_reference_charges import (  # noqa: E402
    CHECKPOINTS,
    FAILURES,
    SMILES,
)
from naglmbis.utils import get_model_weights  # noqa: E402


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_matches_original(checkpoint):
    """Make sure the atom features and charges match the original implementation."""
    path = get_model_weights("charge", checkpoint)
    original = load_original_model(path)
    model = load_checkpoint(path)

    for i, smiles in enumerate(SMILES):
        if f"{checkpoint}|{i}" in FAILURES:
            continue
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))

        # the atom feature vectors must be identical, including the column order
        original_features = AtomFeaturizer.featurize(
            molecule, original.config.model.atom_features
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*consists of more than one molecule"
            )
            off_molecule = Molecule.from_rdkit(
                molecule, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
        graph = GraphMolecule.from_openff(
            off_molecule, atom_features=model.gnn_model.config.atom_features
        )
        assert torch.equal(
            original_features.float(), graph.atom_features.float()
        ), smiles

        reference = original_charges(original, molecule)
        charges = model.compute_properties(molecule)["mbis-charges"]
        assert np.allclose(charges.numpy(), reference.numpy(), atol=1e-5), smiles

"""
Helpers to evaluate the models with the original nagl (``bismuthadams1/nagl``) + dgl
implementation, used to generate and check reference charges.
"""

import torch
from rdkit import Chem


def load_original_model(checkpoint_path):
    """Load a checkpoint with the original nagl based model."""
    from nagl.molecules import DGLMolecule
    from nagl.training import DGLMoleculeLightningModel

    class MBISGraphModel(DGLMoleculeLightningModel):
        def compute_properties(self, molecule: Chem.Mol) -> dict[str, torch.Tensor]:
            return self.forward(
                DGLMolecule.from_rdkit(
                    molecule,
                    self.config.model.atom_features,
                    self.config.model.bond_features,
                )
            )

    model_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = MBISGraphModel(**model_data["hyper_parameters"])
    model.load_state_dict(model_data["state_dict"])
    model.eval()
    return model


def original_charges(model, molecule: Chem.Mol) -> torch.Tensor:
    """
    Compute the charges of each fragment of the molecule separately with the original
    model, so the charges of each fragment sum to its own formal charge.
    """
    charges = torch.empty((molecule.GetNumAtoms(), 1))
    fragment_indices = Chem.GetMolFrags(molecule)
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    for indices, fragment in zip(fragment_indices, fragments, strict=True):
        with torch.no_grad():
            charges[list(indices)] = model.compute_properties(fragment)["mbis-charges"]
    return charges

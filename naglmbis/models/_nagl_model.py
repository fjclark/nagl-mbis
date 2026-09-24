# The original nagl (bismuthadams1/nagl) based model, which requires dgl. This is
# only needed for training; inference uses openff-nagl, see ``base_model.py``.

import torch
from nagl.molecules import DGLMolecule
from nagl.training import DGLMoleculeLightningModel
from rdkit import Chem


class MBISGraphModel(DGLMoleculeLightningModel):
    "A wrapper to make it easy to load and evaluate models"

    def compute_properties(self, molecule: Chem.Mol) -> dict[str, torch.Tensor]:
        dgl_molecule = DGLMolecule.from_rdkit(
            molecule, self.config.model.atom_features, self.config.model.bond_features
        )

        return self.forward(dgl_molecule)

    def return_dgl_molecule(self, molecule: Chem.Mol) -> DGLMolecule:

        return DGLMolecule.from_rdkit(
            molecule, self.config.model.atom_features, self.config.model.bond_features
        )

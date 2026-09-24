# models for the nagl run

import torch
from openff.nagl import GNNModel
from rdkit import Chem


class MBISChargeModel:
    """
    A wrapper to make it easy to evaluate the pre-trained models.

    The models were trained with the ``bismuthadams1/nagl`` fork of NAGL, but are
    evaluated with ``openff-nagl``'s pure PyTorch implementation so dgl is not needed.
    """

    def __init__(self, gnn_model: GNNModel):
        # always use openff-nagl's pure PyTorch layers, even if dgl is installed
        if gnn_model._is_dgl:
            gnn_model = gnn_model._as_nagl()
        self.gnn_model = gnn_model

    def compute_properties(self, molecule: Chem.Mol) -> dict[str, torch.Tensor]:
        """
        Compute the properties predicted by the model.

        Parameters
        ----------
        molecule: Chem.Mol
            The RDKit molecule to compute the properties for. The atom ordering and
            hydrogens are used as given. If the molecule contains multiple fragments
            (e.g. a salt), each is predicted separately so the charges of each
            fragment sum to its own formal charge. Radicals are not supported by the
            OpenFF toolkit.

        Returns
        -------
        dict[str, torch.Tensor]
            The predicted properties, e.g. ``"mbis-charges"`` with shape (n_atoms, 1).
        """
        from openff.nagl.molecule._graph.molecule import GraphMolecule
        from openff.toolkit import Molecule

        # Split the molecule into fragments ourselves rather than relying on
        # GNNModel.compute_properties, which can assign the charges of a fragment to
        # the wrong atoms when the fragment's atom indices are not contiguous.
        fragment_indices = Chem.GetMolFrags(molecule)
        fragments = Chem.GetMolFrags(molecule, asMols=True)

        properties = {}
        for indices, fragment in zip(fragment_indices, fragments, strict=True):
            off_fragment = Molecule.from_rdkit(
                fragment, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
            graph = GraphMolecule.from_openff(
                off_fragment,
                atom_features=self.gnn_model.config.atom_features,
                bond_features=self.gnn_model.config.bond_features,
            )
            for name, values in self.gnn_model.forward(graph).items():
                if name not in properties:
                    properties[name] = torch.empty(
                        (molecule.GetNumAtoms(), values.shape[1]), dtype=values.dtype
                    )
                properties[name][list(indices)] = values.detach()

        return properties


def __getattr__(name):
    # the nagl based model needs dgl, so only import it when requested
    if name == "MBISGraphModel":
        from naglmbis.models._nagl_model import MBISGraphModel

        return MBISGraphModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ComputePartialPolarised:
    "Compute the partially polarized properties based on a supplied scaling constant"
    def __init__(self,
                 model_gas: MBISChargeModel,
                 model_water: MBISChargeModel,
                 alpha: float =  0.5):
        """
        Parameters
        ----------
        model_gas: MBISChargeModel
            loaded graph model for the gas phase charges
        model_water: MBISChargeModel
            loaded graph model for the water based charges
        alpha: float
            weighting constant to weight each model
        """
        
        self.model_gas = model_gas
        self.model_water = model_water
        self.alpha = alpha
        
    def compute_polarised_charges(self, molecule: Chem.Mol) -> torch.Tensor:
        """Compute polarized charges based on an openff molecule input
        
        Parameters
        ----------
        molecule: Chem.Mol
            openff molecule to calculate the charges for
        
        Returns
        -------
        torch.Tensor
            weighted average partial charges
        """
        gas_charges = self.model_gas.compute_properties(
            molecule=molecule
        )["mbis-charges"]
        
        water_charges = self.model_water.compute_properties(
            molecule=molecule
        )["mbis-charges"]
        
        return (1-self.alpha) * gas_charges + self.alpha * water_charges
        
        

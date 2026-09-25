# models for the nagl run

import warnings
from typing import TYPE_CHECKING

import torch
from openff.nagl import GNNModel
from rdkit import Chem

if TYPE_CHECKING:
    from openff.nagl.molecule._graph.molecule import GraphMolecule


class MBISChargeModel:
    """
    A wrapper to make it easy to evaluate the pre-trained models.

    The models were trained with the ``bismuthadams1/nagl`` fork of NAGL, but are
    evaluated with ``openff-nagl``'s pure PyTorch implementation so dgl is not needed.
    """

    def __init__(self, gnn_model: GNNModel):
        # always use openff-nagl's pure PyTorch layers, even if dgl is installed
        if gnn_model._is_dgl:
            # the converted model is built fresh in training mode
            gnn_model = gnn_model._as_nagl().eval()
        self.gnn_model = gnn_model

    def _graph(self, molecule: Chem.Mol) -> "GraphMolecule":
        """Featurise an RDKit molecule into an openff-nagl graph."""
        from openff.nagl.molecule._graph.molecule import GraphMolecule
        from openff.toolkit import Molecule

        return GraphMolecule.from_openff(
            Molecule.from_rdkit(
                molecule, allow_undefined_stereo=True, hydrogens_are_explicit=True
            ),
            atom_features=self.gnn_model.config.atom_features,
            bond_features=self.gnn_model.config.bond_features,
        )

    def compute_properties(self, molecule: Chem.Mol) -> dict[str, torch.Tensor]:
        """
        Compute the properties predicted by the model.

        Each fragment (connected component) of the molecule is predicted separately,
        so the charges of each fragment sum to its own formal charge. Evaluating all
        fragments in one graph is not safe: message passing stays within each
        fragment, but the charge equilibration readout spreads the total charge over
        every atom in the graph, so charge leaks between fragments (e.g. acetate +
        water gives fragment charges of -1.09 and +0.09 rather than -1 and 0).

        We split the molecule ourselves rather than using
        ``GNNModel.compute_properties``, which also splits into fragments but can
        assign the charges of a fragment to the wrong atoms within that fragment.
        ``openff.nagl.toolkits.openff.split_up_molecule`` records each fragment's
        atom indices in the iteration order of a Python ``set``, but builds the
        fragment in the node order of a networkx subgraph, and the two orders can
        differ. For ``CCO.O`` the water atoms are recorded as ``[11, 10, 3]`` but
        the water fragment is built with its atoms ordered ``[3, 10, 11]``, so the
        oxygen's charge is given to a hydrogen (an error of 1.24 e). The total
        charge of each fragment is still correct, so this is easy to miss.

        Parameters
        ----------
        molecule: Chem.Mol
            The RDKit molecule to compute the properties for. The atom ordering and
            hydrogens are used as given. Radicals are not supported by the OpenFF
            toolkit.

        Returns
        -------
        dict[str, torch.Tensor]
            The predicted properties, e.g. ``"mbis-charges"`` with shape (n_atoms, 1).
        """
        results = {}
        fragment_indices = Chem.GetMolFrags(molecule)
        fragments = Chem.GetMolFrags(molecule, asMols=True)
        for indices, fragment in zip(fragment_indices, fragments, strict=True):
            for name, values in self.gnn_model.forward(self._graph(fragment)).items():
                if name not in results:
                    results[name] = torch.empty(
                        (molecule.GetNumAtoms(), values.shape[1]), dtype=values.dtype
                    )
                results[name][list(indices)] = values.detach()
        return results

    def compute_latent_embeddings(self, molecule: Chem.Mol) -> torch.Tensor:
        """
        Compute the latent atom embeddings, i.e. the output of the convolution
        layers which is fed to the readout layers.

        Unlike ``compute_properties`` the molecule does not need to be split into
        fragments, as message passing never crosses between fragments.

        Parameters
        ----------
        molecule: Chem.Mol
            The RDKit molecule, with the atom ordering and hydrogens used as given.

        Returns
        -------
        torch.Tensor
            The atom embeddings with shape (n_atoms, hidden_feature_size).
        """
        from openff.toolkit.utils.exceptions import MultipleComponentsInMoleculeWarning

        with warnings.catch_warnings():
            # fragments are never mixed by the convolution layers, so a molecule
            # with several fragments is safe here
            warnings.simplefilter("ignore", MultipleComponentsInMoleculeWarning)
            graph = self._graph(molecule)
        # the convolution module stores its output on the graph
        self.gnn_model.convolution_module(graph)
        return graph.graph.ndata[graph._graph_feature_name].detach()


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
        
        

"""
Generate the reference MBIS charges used by ``test_reference_charges.py``.

These were produced with the original nagl (``bismuthadams1/nagl``) + DGL implementation
of the models, and act as the ground truth for any other implementation. Molecules with
multiple fragments are evaluated one fragment at a time. Do not regenerate this file
with a new implementation.

Run from the repository root with ``pixi run python naglmbis/tests/data/generate_reference_charges.py``.
"""

import pathlib
import subprocess

import numpy as np
import torch
from rdkit import Chem

from naglmbis.tests._original import load_original_model, original_charges

MODEL_DIR = pathlib.Path(__file__).parents[2] / "data" / "models" / "charge"
OUTPUT = pathlib.Path(__file__).parent / "reference_charges.npz"

SMILES = [
    # simple neutral molecules
    "O",
    "CO",
    "CCO",
    "CC(=O)O",
    "CC#N",
    "CS",
    # charged species
    "CC(=O)[O-]",
    "C[NH3+]",
    "[NH3+]CC(=O)[O-]",
    "[O-]C(=O)C([O-])=O",
    "[NH3+]CC[NH3+]",
    "NC(N)=[NH2+]",
    "O=[N+]([O-])c1ccccc1",
    "[NH4+]",
    "[Cl-]",
    # rings of different sizes, fused, spiro and bridged systems
    "C1CC1",
    "C1CCC1",
    "C1CCCC1",
    "C1CCCCCC1",
    "C1CCCCCCC1",
    "C1CCCCCCCCC1",
    "c1ccc2ccccc2c1",
    "c1ccc2cccc-2cc1",
    "C1CCC2(C1)CCCCC2",
    "C1CC2CCC1C2",
    "C12C3C4C1C5C2C3C45",
    "C1C2CC3CC1CC(C2)C3",
    # aromatic and heteroaromatic rings, tautomers
    "c1ccccc1",
    "c1ccncc1",
    "c1cc[nH]c1",
    "c1c[nH]cn1",
    "c1nn[nH]n1",
    "c1ccsc1",
    "O=c1cccc[nH]1",
    "Oc1ccccn1",
    # stereochemistry
    "C[C@@H](N)C(=O)O",
    "CC(N)C(=O)O",
    "C/C=C/C",
    # halogens
    "Fc1ccccc1",
    "FC(F)(F)C(=O)O",
    "Clc1ccccc1",
    "CCBr",
    "Ic1ccccc1",
    # sulfur, phosphorus, boron, silicon
    "CS(C)=O",
    "CS(N)(=O)=O",
    "COP(=O)(OC)OC",
    "OP(=O)([O-])[O-]",
    "B(OC)(OC)OC",
    "OB(O)c1ccccc1",
    "C[Si](C)(C)C",
    # drug-like molecules
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
    # multiple fragments, each fragment is predicted separately
    "CC(=O)[O-].C[NH3+]",
    "CCO.O",
]


def main():
    arrays = {"smiles": np.array(SMILES)}
    failures = []

    for checkpoint in sorted(MODEL_DIR.glob("*.ckpt")):
        model = load_original_model(checkpoint)

        for i, smiles in enumerate(SMILES):
            molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
            try:
                charges = original_charges(model, molecule)
            except Exception:
                # e.g. elements or connectivities outside the model's one-hot vocabulary
                failures.append(f"{checkpoint.name}|{i}")
                continue
            arrays[f"{checkpoint.name}|{i}"] = charges.numpy().astype(np.float64)

    import dgl
    import nagl

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    arrays["failures"] = np.array(failures)
    arrays["provenance"] = np.array(
        f"naglmbis commit {commit}; nagl {nagl.__version__}; dgl {dgl.__version__}; "
        f"torch {torch.__version__}"
    )
    np.savez_compressed(OUTPUT, **arrays)
    print(f"wrote {len(arrays) - 3} reference charge sets, {len(failures)} failures")


if __name__ == "__main__":
    main()

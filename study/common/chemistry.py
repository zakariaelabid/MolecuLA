from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


PROPERTY_COLUMNS = [
    "MolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "cLogP",
    "TPSA",
    "HBD",
    "HBA",
    "NumRotatableBonds",
    "RingCount",
    "AromaticRingCount",
    "FractionCSP3",
    "NumSpiroAtoms",
    "NumBridgeheadAtoms",
    "BertzCT",
    "QED",
]

CONF_COLUMNS = [
    "selfies_len_tokens",
    "branch_token_count",
    "ring_token_count",
    "token_entropy",
]


def shannon_entropy(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    _, counts = np.unique(list(tokens), return_counts=True)
    probs = counts.astype(float) / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def build_confound_panel(selfies_values: Sequence[str]) -> pd.DataFrame:
    rows = []
    for value in selfies_values:
        toks = list(sf.split_selfies(str(value)))
        rows.append(
            {
                "selfies_len_tokens": len(toks),
                "branch_token_count": sum("Branch" in tok for tok in toks),
                "ring_token_count": sum("Ring" in tok for tok in toks),
                "token_entropy": shannon_entropy(toks),
            }
        )
    return pd.DataFrame(rows)


def _sa_score(mol):
    try:
        from rdkit.Contrib.SA_Score import sascorer

        return float(sascorer.calculateScore(mol))
    except Exception:
        return np.nan


def build_property_panel(smiles_values: Sequence[str], include_sa: bool = True) -> pd.DataFrame:
    rows = []
    for smiles in smiles_values:
        mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
        row = {column: np.nan for column in PROPERTY_COLUMNS}
        row["SA-score"] = np.nan
        row["is_rdkit_valid"] = bool(mol is not None)
        if mol is not None:
            row.update(
                {
                    "MolWt": Descriptors.MolWt(mol),
                    "ExactMolWt": Descriptors.ExactMolWt(mol),
                    "HeavyAtomCount": mol.GetNumHeavyAtoms(),
                    "cLogP": Descriptors.MolLogP(mol),
                    "TPSA": Descriptors.TPSA(mol),
                    "HBD": rdMolDescriptors.CalcNumHBD(mol),
                    "HBA": rdMolDescriptors.CalcNumHBA(mol),
                    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                    "RingCount": rdMolDescriptors.CalcNumRings(mol),
                    "AromaticRingCount": rdMolDescriptors.CalcNumAromaticRings(mol),
                    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
                    "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
                    "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    "BertzCT": Descriptors.BertzCT(mol),
                    "QED": Descriptors.qed(mol),
                }
            )
            if include_sa:
                row["SA-score"] = _sa_score(mol)
        rows.append(row)
    return pd.DataFrame(rows)


def build_combined_panel(df: pd.DataFrame, include_sa: bool = True) -> pd.DataFrame:
    properties = build_property_panel(df["smiles"], include_sa=include_sa)
    confounds = build_confound_panel(df["selfies"])
    return pd.concat([df[["smiles", "selfies"]].reset_index(drop=True), properties, confounds], axis=1)

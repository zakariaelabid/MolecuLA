from __future__ import annotations

import numpy as np
import pandas as pd
import selfies as sf
import torch
from rdkit import Chem

from .data import EOS_ID, PAD_ID, SOS_ID


def ids_to_selfies(ids, id2tok: dict[int | str, str]) -> str:
    toks = []
    for idx in ids:
        idx = int(idx)
        if idx in {PAD_ID, EOS_ID}:
            break
        if idx == SOS_ID:
            continue
        toks.append(id2tok.get(idx, id2tok.get(str(idx), "")))
    return "".join(toks)


def decode_token_ids(ids, id2tok: dict[int | str, str]) -> dict:
    selfies = ids_to_selfies(ids, id2tok)
    try:
        smiles = sf.decoder(selfies) if selfies else ""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        canonical = Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
        return {"selfies": selfies, "smiles": smiles, "canonical_smiles": canonical, "valid": mol is not None}
    except Exception:
        return {"selfies": selfies, "smiles": None, "canonical_smiles": None, "valid": False}


def decode_latents(model, Z: np.ndarray, id2tok: dict[int | str, str], device: str | torch.device = "cpu") -> pd.DataFrame:
    device = torch.device(device)
    model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for z in np.asarray(Z, dtype=np.float32):
            z_tensor = torch.as_tensor(z[None, :], dtype=torch.float32, device=device)
            decoded = model.decode(z_tensor).detach().cpu().numpy()[0]
            rows.append(decode_token_ids(decoded, id2tok))
    return pd.DataFrame(rows)


def run_latent_traversal(
    model,
    seed_z: np.ndarray,
    direction: np.ndarray,
    id2tok: dict[int | str, str],
    alphas: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    alphas = np.asarray(alphas if alphas is not None else np.linspace(-8.0, 8.0, 17), dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    path = np.stack([seed_z + alpha * direction for alpha in alphas]).astype(np.float32)
    decoded = decode_latents(model, path, id2tok, device=device)
    decoded.insert(0, "alpha", alphas)
    decoded.insert(1, "step", np.arange(len(alphas)))
    return decoded

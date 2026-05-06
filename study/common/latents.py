from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def load_checkpoint_state(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> dict:
    checkpoint_path = Path(checkpoint_path)
    try:
        obj = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(obj, dict):
        if isinstance(obj.get("model"), dict):
            return obj["model"]
        if isinstance(obj.get("model_state_dict"), dict):
            return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint payload type: {type(obj)!r}")


def encode_latents(model, token_ids: np.ndarray, device: str | torch.device = "cpu", batch_size: int = 256) -> np.ndarray:
    device = torch.device(device)
    model.to(device)
    model.eval()
    loader = DataLoader(torch.as_tensor(token_ids, dtype=torch.long), batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            encoded = model.encode(batch.to(device))
            mu = encoded[0] if isinstance(encoded, tuple) else encoded
            chunks.append(mu.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    model_file: str
    checkpoint: str | None
    hidden_size: int
    latent_size: int
    max_len: int
    attention_heads: int
    num_slots: int | None
    layers: int | None = None
    encoder_layers: int | None = None
    decoder_layers: int | None = None
    status: str = "available"

    @property
    def model_path(self) -> Path:
        return REPO_ROOT / self.model_file

    @property
    def checkpoint_path(self) -> Path | None:
        return None if self.checkpoint is None else REPO_ROOT / self.checkpoint


_SPECS = [
    {
        "name": "linear_attention",
        "description": "Non-autoregressive Transformer VAE with linear slot pooling and linear decoder cross-attention.",
        "model_file": "models/linear_attention_vae.py",
        "checkpoint": "checkpoints/linear_attention_h256_l512.pt",
        "hidden_size": 256,
        "latent_size": 512,
        "max_len": 77,
        "attention_heads": 8,
        "num_slots": 8,
        "layers": 1,
        "encoder_layers": None,
        "decoder_layers": None,
        "status": "available"
    },
    {
        "name": "simple_attention",
        "description": "Non-autoregressive Transformer VAE with standard multi-head slot pooling and standard cross-attention.",
        "model_file": "models/simple_attention_vae.py",
        "checkpoint": "checkpoints/simple_attention_h256_l256.pt",
        "hidden_size": 256,
        "latent_size": 256,
        "max_len": 77,
        "attention_heads": 8,
        "num_slots": 8,
        "layers": 1,
        "encoder_layers": None,
        "decoder_layers": None,
        "status": "available"
    },
    {
        "name": "autoregressive",
        "description": "Autoregressive Transformer VAE with slot encoder and causal Transformer decoder.",
        "model_file": "models/autoregressive_vae.py",
        "checkpoint": "checkpoints/H256-L256-3E-2D-Final-NoCorruption.pt",
        "hidden_size": 256,
        "latent_size": 256,
        "max_len": 77,
        "attention_heads": 8,
        "num_slots": 8,
        "layers": None,
        "encoder_layers": 3,
        "decoder_layers": 2,
        "status": "available"
    }
]

MODEL_SPECS: dict[str, ModelSpec] = {
    row["name"]: ModelSpec(**row) for row in _SPECS
}

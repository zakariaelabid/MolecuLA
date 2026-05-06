from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "smiles_selfies_full.csv"
DEFAULT_TOKENIZER = REPO_ROOT / "data" / "selfies_tokenizer.json"
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "MASK"]
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
MASK_ID = 3


def load_dataset(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    required = {"smiles", "selfies"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def build_or_load_tokenizer(
    df: pd.DataFrame | None = None,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER,
    save_if_missing: bool = False,
) -> dict:
    tokenizer_path = Path(tokenizer_path)
    if tokenizer_path.exists():
        return json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if df is None:
        df = load_dataset()
    tokens = df["selfies"].map(lambda value: list(sf.split_selfies(str(value))))
    vocab = SPECIAL_TOKENS + sorted({tok for seq in tokens for tok in seq})
    payload = {
        "vocab": vocab,
        "tok2id": {tok: idx for idx, tok in enumerate(vocab)},
        "id2tok": {str(idx): tok for idx, tok in enumerate(vocab)},
        "max_len": int(tokens.map(len).max() + 2),
        "special_tokens": SPECIAL_TOKENS,
    }
    if save_if_missing:
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def tokenize_selfies_column(df: pd.DataFrame, tokenizer: dict | None = None) -> tuple[np.ndarray, dict]:
    tokenizer = tokenizer or build_or_load_tokenizer(df)
    tok2id = tokenizer["tok2id"]
    max_len = int(tokenizer.get("max_len", 0))
    encoded = []
    for selfies_value in df["selfies"]:
        ids = [SOS_ID] + [tok2id[tok] for tok in sf.split_selfies(str(selfies_value))] + [EOS_ID]
        encoded.append(ids)
        max_len = max(max_len, len(ids))
    data = np.full((len(encoded), max_len), PAD_ID, dtype=np.int64)
    for row, ids in enumerate(encoded):
        data[row, : len(ids)] = ids
    tokenizer = dict(tokenizer)
    tokenizer["max_len"] = max_len
    return data, tokenizer


def make_splits(n_rows: int, seed: int = 42) -> dict[str, np.ndarray]:
    indices = np.arange(n_rows)
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=seed, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed, shuffle=True)
    return {"train": train_idx, "val": val_idx, "test": test_idx}

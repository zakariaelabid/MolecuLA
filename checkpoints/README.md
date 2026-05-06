# Checkpoints

Included:

- `linear_attention_h256_l512.pt`: linear-attention non-autoregressive model.
- `simple_attention_h256_l256.pt`: simple-attention `nat_model_h256_l256` model used by model comparison.

Missing by design:

- `H256-L256-3E-2D-Final-NoCorruption.pt`: autoregressive model checkpoint. The local source only contained a dummy placeholder, so it was not copied. Place the recovered AR checkpoint here before running AR notebooks or AR traversal studies.

Use Git LFS for `*.pt` files when publishing this repository.

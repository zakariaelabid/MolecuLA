# Molecules Meet Language: Confound-Aware Representation Learning and Chemical Property Steering in Transformer-VAE Latent Spaces

This repository is an anonymized, GitHub-ready export of a "Molecules Meet Language: Confound-Aware Representation Learning and Chemical Property Steering in Transformer-VAE Latent Spaces" paper built around Transformer VAEs trained on SELFIES strings. It preserves the runnable workflow and selected evidence without carrying local machine paths, private Git history, or large regenerated caches.

## Models

The export uses three canonical model identifiers:

| Identifier | Model file | Checkpoint | Status |
|---|---|---|---|
| `linear_attention` | `models/linear_attention_vae.py` | `checkpoints/linear_attention_h256_l512.pt` | included |
| `simple_attention` | `models/simple_attention_vae.py` | `checkpoints/simple_attention_h256_l256.pt` | included |
| `autoregressive` | `models/autoregressive_vae.py` | `checkpoints/H256-L256-3E-2D-Final-NoCorruption.pt` | included |

`simple_attention` corresponds to the `nat_model_h256_l256` model used by the model-comparison workflow.

## Repository Layout

```text
models/           Model definitions and registry
checkpoints/      Included model weights
data/             SELFIES/SMILES dataset and tokenizer
study/common/     Reusable data, chemistry, probe, latent, traversal, and plotting helpers
study/notebooks/  Five output-stripped workflow notebooks
study/results/    Curated compact figures, JSON summaries, and CSV tables
scripts/          Export verification script
```

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

RDKit installation is easiest through Conda or Mamba if pip wheels are unavailable on your platform:

```bash
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

## Data And Checkpoints

The dataset is expected at:

```text
data/smiles_selfies_full.csv
```

The included checkpoints are:

```text
checkpoints/linear_attention_h256_l512.pt
checkpoints/simple_attention_h256_l256.pt
checkpoints/H256-L256-3E-2D-Final-NoCorruption.pt
```


## Workflow

Run notebooks in this order:

1. `study/notebooks/00_data_tokenization_and_training.ipynb`
2. `study/notebooks/01_latent_quality_benchmarks_and_family_tests.ipynb`
3. `study/notebooks/02_linear_probes_panels_and_residuals.ipynb`
4. `study/notebooks/03_latent_traversals.ipynb`
5. `study/notebooks/04_mlp_probes_all_properties.ipynb`

The notebooks write regenerated panels, latents, direction arrays, and probe models under local output folders. These caches are intentionally not committed because the full panels and latent arrays are large and reproducible from the dataset and checkpoints.

## Results

Curated evidence is under `study/results/`:

- benchmark and family-retention metrics
- Step 3 confound summaries and compact R2/correlation tables
- Step 4 direction summaries and compact direction-quality tables
- single-property traversal summaries and representative figures
- MLP-vs-Ridge probe summaries
- ablation comparison summaries

The autoregressive result folder includes compact outputs from executed source notebooks: reconstruction and family-retention summaries, Ridge and linear-regression probe tables, MLP probe comparisons, missing-property probes, and copied traversal figures. The AR checkpoint is included under `checkpoints/`, so the AR notebooks can be rerun from the exported assets.

Large regenerated assets such as full panels, latent arrays, `.npz` direction caches, duplicate 300-550 MB CSVs, and MLP weight files are excluded.


# Autoregressive Results

This folder contains compact recovered results for the autoregressive VAE study. The local export did not include a real AR checkpoint, so these files were recovered from already executed notebook outputs and legacy traversal image folders rather than from a fresh rerun.

The included CSV and JSON files cover reconstruction quality, family retention, Ridge/linear-regression probes, MLP probes, missing-property probes, and copied traversal figures. Full panels, latent arrays, notebook outputs, generated probe weights, and large caches are intentionally excluded.

To reproduce the AR workflow from scratch, place the recovered checkpoint at:

```text
checkpoints/H256-L256-3E-2D-Final-NoCorruption.pt
```

Then run the study notebooks in the order described in the top-level README.

The maintenance script `scripts/enrich_autoregressive_results.py` can regenerate this folder from the original source workspace that still contains the executed AR notebooks and traversal images.

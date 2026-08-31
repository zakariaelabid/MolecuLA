# Forensic audit for a genuinely distinct MolecuLA Paper 2

**Repository audited:** `MolecuLA project`  
**Audit date:** 11 August 2026  
**Role:** senior ML-for-science / molecular generative-model assessment  
**Evidence convention:** numbers are reported only when I could trace them to the submitted paper, the review/rebuttal record, or a named repository artifact. “Main paper”, “appendix”, “rebuttal”, “repository-only”, “exploratory”, and “proposed” are kept separate.

## Scope and audit method

I inspected the repository manifest and source tree, the submitted Paper 1 PDF, the complete review/rebuttal PDF, the nine rebuttal evidence packages, the exported result tables/manifests, and the executed notebook archives. The export contains 772 tracked artifacts: 392 CSV files, 283 PNG files, 39 JSON files, 22 notebooks, 19 Python files, 5 Markdown files, 4 Parquet files, 3 checkpoints, 2 PDFs, 1 XML file, 1 text file, and 1 HTML file.

The two PDFs were rendered page-by-page and text-extracted with a PDF fallback workflow. One page of the paper contains unrelated instruction-like text embedded in the PDF; it was treated as untrusted source content and ignored. It has no bearing on the scientific claims below.

The most important limitation of the export is that full latent arrays, caches, and several large panels are intentionally omitted. Some autoregressive results are compact recoveries from executed notebooks rather than rerunnable end-to-end artifacts. This is adequate for an audit of the scientific record, but not equivalent to a clean replication package.

### Evidence grades used below

| Grade | Meaning |
|---|---|
| A | Main-paper result with a named table/figure and a matching compact repository artifact. |
| B | Rebuttal result with a protocol/data table or executable notebook, but incomplete independent replication or incomplete release. |
| C | Repository-only or archived exploratory result; useful for hypothesis generation, not yet a publication-grade claim. |
| P | Proposed experiment or interpretation; no result should be implied. |

Unless otherwise noted, paths below are relative to the repository root.

## 1. Executive scientific assessment

### Bottom line

There is a credible, scientifically distinct Paper 2 hiding in this project, but it is **not yet submission-ready as a standalone paper**. The strongest candidate is not “MolecuLA with more experiments.” It is:

> **When does a molecular latent representation that is easy to read become difficult or impossible to intervene on after decoding?**

The proposed Paper 2 should test whether reconstruction and post-hoc property predictability are poor proxies for **decoded actionability**. Here, actionability means a reproducible, property-responsive, valid, chemically local and non-collapsed change under a specified latent intervention protocol. This is an operational ML/chemistry construct, not a claim of biological efficacy or synthetic feasibility.

The repository already contains a striking version of the phenomenon:

* The SELFIES autoregressive model has high reconstruction and property readability, including residual readability after linear confound removal.
* The matched SMILES rebuttal experiment reports high reconstruction and substantial residual readability, but only 9.33% traversal validity in the review summary; the exported notebook has no clean checkpoint/table release and has split-protocol ambiguity.
* Archived corrected-validation runs for the linear and simple attention models show near-zero decoded movement at small standardized spans despite good probe R²: uniqueness fractions are roughly 0.007–0.009, path similarity is 1.0, and paths are often constant. These are repository-only findings and must be rerun before they can carry a paper claim.
* The E09 paired displacement study shows that the continuous OLS direction can retain substantial property ordering at much better local similarity than the adapted extreme-SVM/ChemSpacE baseline. The latter can produce larger nominal property changes, but in the native-L2 paired grid its similarity-constrained success is zero at Tanimoto cutoffs 0.2, 0.4, and 0.6 for the reported properties.
* E06 supplies signed, per-alpha curves showing an empirical locality regime near small displacements and a deterioration of similarity, scaffold retention, uniqueness, or validity as the displacement grows. It does not yet define a preregistered trust boundary.

Together, these support a **readability–actionability dissociation** and a **method-/representation-dependent trust-region hypothesis**. They do not yet support a universal law, a new generative architecture, a chemical-discovery claim, or the assertion that MolecuLA is a better optimizer than ChemSpacE.

### My recommendation

Proceed with Paper 2 **only if it is rebuilt around a new, held-out, factorial intervention benchmark**. Reusing the existing tables as the main contribution would look like an extended appendix or rebuttal. The decisive new work is a clean comparison of representation × generator × direction learner × displacement scale, with per-seed actionability curves and fixed similarity constraints.

**Current readiness:** approximately **5/10** for a distinct paper concept; **3–4/10** for a defensible submission today.  
**Potential after the must-have experiments:** approximately **8/10** for a strong specialist conference or journal paper, and potentially higher if the effect replicates across representations and model families.

### The strongest existing result

The strongest result is the **native-L2 paired E09 comparison**, because it puts two steering methods on the same frozen MolecuLA decoder, seed registry, candidate budget, displacement grid, and locality evaluation. Its central pattern is not “OLS wins every metric”; it is that **similar property ranking can coexist with radically different locality, scaffold retention, token-ceiling exposure, and constrained success**. That is the cleanest bridge from Paper 1’s readability story to Paper 2’s actionability story.

### The most important missing experiment

Rerun a preregistered, held-out **readability-to-actionability benchmark** with at least:

1. SELFIES and canonical SMILES under matched data/model budgets;
2. the autoregressive model plus at least two additional frozen generator conditions, with independent training seeds where feasible;
3. OLS/Ridge, ChemSpacE-style SVM, MLP local-gradient, and random-direction controls;
4. a displacement grid calibrated in both native latent L2 and standardized latent units;
5. a held-out seed registry not used for direction fitting or extreme-label selection; and
6. paired curves for property response, validity, uniqueness, Tanimoto locality, scaffold retention, SA/NP change, token ceiling, novelty, and constrained success.

Without that experiment, the central story is compelling but vulnerable to the objection that the apparent gap is an artifact of scale, decoder implementation, seed selection, or one representation.

## 2. What Paper 1 establishes—and what it does not

### Paper 1’s actual contribution

Paper 1, *Molecules Meet Language: Confound-Aware Representation Learning and Chemical Property Steering in Transformer-VAE Latent Spaces*, establishes a post-hoc analysis framework for a property-blind autoregressive Transformer-VAE trained on SELFIES:

1. Freeze the encoder/decoder after unsupervised training.
2. Fit linear probes from the latent mean to RDKit descriptors.
3. Treat probe coefficients as candidate global directions.
4. Remove a small panel of predictable representation confounds using statistical residualization.
5. Test raw and residual directions by decoding traversals.
6. Use nonlinear probes as a diagnostic for properties that are not adequately summarized by one global linear direction.

The strongest positive claim is deliberately narrower than “the latent space is disentangled”:

> Several computed molecular descriptors are linearly readable, remain at least partly readable after the reported confound controls, and show monotonic or directionally consistent behavior in decoded traversals for selected regimes.

### What Paper 1 establishes

* The autoregressive model is the best of the three released model conditions on the paper’s aggregate latent-probe metrics.
* Large structural descriptors such as HeavyAtomCount, MolecularWeight, BertzCT, RingCount, and AromaticRingCount are highly readable but strongly confounded with simple graph-size or sequence statistics.
* FractionCSP3, TPSA, cLogP, HBA, and SA retain meaningful residual predictability, although “meaningful” is descriptor- and protocol-dependent.
* HBD, QED, NumRotatableBonds, Spiro, and Bridge are substantially more nonlinear or weak under the linear probe; the MLP diagnostic recovers much more signal for several of them.
* Raw and residual directions are not interchangeable. Residualization reduces alignment with the reported confound subspace, sometimes substantially, but does not create geometric orthogonality in a strict causal or manifold sense.
* Decoded traversal is more informative than latent R² alone: the paper’s Appendix E.5 already shows plateaus, cliffs, saturation, repeated outputs, and nonmonotonic behavior for some directions.
* The model can generate valid and novel SELFIES molecules under the reported generation protocols, but validity and novelty do not establish chemical plausibility, synthetic accessibility, or usefulness for a design task.

### What Paper 1 leaves open

Paper 1 does not establish:

* that high probe R² predicts useful decoded intervention;
* that a latent direction has a representation-independent meaning;
* that a global direction remains locally smooth near individual molecules;
* that an intervention preserves a scaffold, functional group, or medicinal-chemistry rationale;
* that a single displacement scale is meaningful across models or latent normalizations;
* that the decoder has a well-defined trust region before saturation or attractor collapse;
* that residualization controls all structural mediators or nonlinear nuisance pathways;
* that SELFIES and SMILES produce the same intervention geometry;
* that the observations replicate on an independently trained model or dataset;
* that the directions support constrained design rather than post-hoc descriptor movement; or
* that the workflow improves a real biological, synthesis, or lead-optimization objective.

These are not minor omissions. They define the space in which a genuinely distinct Paper 2 can live.

## 3. Complete result inventory

### 3.1 Models and reconstruction/generation quality — Grade A, with provenance caveats

The repository contains three frozen conditions and three checkpoints:

| Condition | Checkpoint | Latent size | Reported test token accuracy | Reported test sequence accuracy | Other reported values |
|---|---|---:|---:|---:|---|
| Linear attention | `checkpoints/linear_attention_h256_l512.pt` | 512 | 0.999929 | 0.997671 | validity 0.9226; uniqueness 0.8071; novelty 0.9871; interpolation validity 1.0; family retention 0.8225 |
| Simple attention | `checkpoints/simple_attention_h256_l256.pt` | 256 | 0.998709 | 0.961443 | interpolation validity 1.0; family retention 0.6926; current compact phase-2 generation status is skipped |
| Autoregressive multi-slot | `checkpoints/H256-L256-3E-2D-Final-NoCorruption.pt` | 256 | 0.9951 | 0.9780 | generation validity 1.0; uniqueness 0.9964; novelty 0.9972; interpolation validity 1.0; family retention 0.8512 |

Sources: Paper 1 Table 2; `study/results/*/model_validation/metrics.json`; `README.md`.

There is an unresolved autoregressive provenance discrepancy. The compact headline metrics report 0.9951 token accuracy and 0.978 sequence accuracy, while `study/results/autoregressive/model_validation/tables/reconstruction_by_split.csv` reports lower split-level exact reconstruction values (train 0.974961, validation 0.965609, test 0.965068) and corresponding token values. These may be different evaluation definitions or recovered runs. Paper 2 must not combine them. The manuscript should state which checkpoint, split, decoder mode, and metric definition generated every headline number.

The simple-attention Paper 1 table includes older-looking generation values, whereas the current compact result marks phase 2 as skipped. Those values should be treated as historical unless regenerated from the released checkpoint.

### 3.2 Main linear probe results — Grade A

Paper 1 Table 3 reports ordinary linear raw latent-to-property R². The full table is reproduced here because it determines which properties are candidates for Paper 2 and which are primarily confound diagnostics.

| Property | Linear attention | Simple attention | Autoregressive |
|---|---:|---:|---:|
| HeavyAtomCount | 0.9562 | 0.9622 | 0.9750 |
| MolWt | 0.9136 | 0.9174 | 0.9480 |
| BertzCT | 0.9279 | 0.9305 | 0.9512 |
| RingCount | 0.8549 | 0.8491 | 0.9378 |
| AromaticRingCount | 0.8423 | 0.8498 | 0.8937 |
| cLogP | 0.6677 | 0.6830 | 0.7967 |
| TPSA | 0.6557 | 0.6994 | 0.8059 |
| HBA | 0.7071 | 0.7587 | 0.8008 |
| FractionCSP3 | 0.7815 | 0.7842 | 0.8951 |
| HBD | 0.2171 | 0.2589 | 0.4997 |
| Spiro | 0.0484 | 0.0509 | 0.0878 |
| Bridge | 0.0737 | 0.0767 | 0.1522 |
| QED | 0.3678 | 0.3465 | 0.4624 |
| NumRotatableBonds | 0.4734 | 0.4844 | 0.6914 |
| SA | 0.4566 | not consistently reported | 0.6412 |
| **Mean** | **0.5963** | **0.6072** | **0.7026** |

The simple-attention SA entry is not consistently displayed in the extracted paper table; it should be recovered from the original source before publication. Do not infer it from the mean.

Paper 1 Table 4 reports ordinary linear residual R²:

| Property | Linear attention | Simple attention | Autoregressive |
|---|---:|---:|---:|
| HeavyAtomCount | 0.3612 | 0.3811 | 0.6554 |
| MolWt | 0.3799 | 0.3109 | 0.5918 |
| BertzCT | 0.5475 | 0.5627 | 0.7009 |
| RingCount | 0.5143 | 0.5151 | 0.7944 |
| AromaticRingCount | 0.6067 | 0.6518 | 0.7559 |
| cLogP | 0.5406 | 0.5770 | 0.6861 |
| TPSA | 0.4653 | 0.5233 | 0.6759 |
| HBA | 0.4735 | 0.5404 | 0.6863 |
| FractionCSP3 | 0.6770 | 0.7098 | 0.8313 |
| HBD | 0.1782 | 0.2202 | 0.4624 |
| Spiro | 0.0318 | 0.0354 | 0.0728 |
| Bridge | 0.0472 | 0.0443 | 0.1207 |
| QED | 0.3011 | 0.2807 | 0.3907 |
| NumRotatableBonds | 0.2065 | 0.1992 | 0.4752 |
| SA | 0.3887 | 0.3848 | 0.6015 |
| **Mean** | **0.3813** | **0.3958** | **0.5668** |

These are probe metrics, not intervention metrics. Their main Paper 2 value is as the independent variable in a readability-to-actionability analysis.

### 3.3 Ridge auxiliary results — Grade A/B

Appendix E.4 reports autoregressive Ridge versus ordinary LinearRegression. Ridge improves both raw and residual R², especially residual scores for graph-size descriptors and chemically meaningful properties:

| Property | Raw linear → Ridge | Residual linear → Ridge |
|---|---:|---:|
| HeavyAtomCount | 0.9750 → 0.9884 | 0.6554 → 0.7692 |
| MolWt | 0.9480 → 0.9655 | 0.5918 → 0.7000 |
| BertzCT | 0.9512 → 0.9677 | 0.7009 → 0.7537 |
| RingCount | 0.9378 → 0.9567 | 0.7944 → 0.8506 |
| AromaticRingCount | 0.8937 → 0.9090 | 0.7559 → 0.7767 |
| cLogP | 0.7967 → 0.8442 | 0.6861 → 0.7552 |
| TPSA | 0.8059 → 0.8670 | 0.6759 → 0.7728 |
| HBA | 0.8008 → 0.8389 | 0.6759 → 0.7138 |
| FractionCSP3 | 0.8951 → 0.9153 | 0.8313 → 0.8542 |
| HBD | 0.4997 → 0.5535 | 0.4624 → 0.5211 |
| NumRotatableBonds | 0.6914 → 0.7640 | 0.4752 → 0.5960 |
| Spiro | 0.0878 → 0.1073 | 0.0728 → 0.0914 |
| Bridge | 0.1522 → 0.1878 | 0.1207 → 0.1556 |
| QED | 0.4624 → 0.5121 | 0.3907 → 0.4424 |

This is a warning for Paper 2: the direction learner itself changes the apparent intervention geometry. OLS versus Ridge must be held fixed and reported, not silently mixed.

### 3.4 Nonlinear probe results — Grade A/B

The autoregressive MLP comparison in Paper 1 Table 6/7 shows that some properties contain signal not captured by a single global linear direction:

| Property | AR linear raw | AR MLP raw | AR linear residual | AR MLP residual |
|---|---:|---:|---:|---:|
| HBD | 0.500 | 0.792 | 0.462 | 0.778 |
| NumRotatableBonds | 0.691 | 0.909 | 0.475 | 0.849 |
| Spiro | 0.088 | 0.628 | 0.073 | 0.608 |
| Bridge | 0.152 | 0.785 | 0.121 | 0.771 |
| QED | 0.462 | 0.803 | 0.391 | 0.771 |
| **Mean across the full reported panel** | — | **0.895** | — | **0.859** |

The corresponding simple-attention MLP means are 0.831 raw and 0.759 residual. The MLP gain is a diagnostic of nonlinear readability, not proof that MLP gradients produce valid or local decoded edits. Paper 1 explicitly leaves that intervention experiment open.

Appendix F.1.1 reports AR MLP minus linear ΔR². Group means are 0.061 raw / 0.152 residual for the monotonic group and 0.360 raw / 0.396 residual for the nonlinear group. The largest individual gains include Bridge 0.592 raw / 0.610 residual, Spiro 0.517 / 0.513, and QED 0.307 / 0.345. These values motivate a local-steering arm in Paper 2 but should not be presented as actionability evidence.

### 3.5 Confound and direction-alignment results — Grade A/B

The simple-attention artifact `study/results/simple_attention/confounds_step3/tables/r2_Z_to_C.csv` shows that the latent encodes the selected confounds strongly: test R² is 0.9906 for SELFIES length, 0.9474 for ring count, 0.9301 for branch count, and 0.8319 for token entropy. This is useful for diagnosing shortcut content, but it does not prove that all property directions are confound directions.

Appendix E.3 gives representative empirical-property versus direction-cosine pairs:

* MolWt/HAC: approximately 0.95 empirical correlation and 0.94 direction cosine.
* HAC/BertzCT: approximately 0.91 and 0.89.
* TPSA/HBA: approximately 0.80 and 0.82.
* FractionCSP3 versus aromaticity: approximately −0.67 and −0.68.
* Spiro/Bridge: empirical correlation approximately 0.028 but direction cosine approximately 0.60.
* Ring/Spiro: empirical correlation approximately 0.13 but direction cosine approximately 0.44.

The last two are particularly relevant: latent directions can be aligned even when the observed descriptor correlation is weak. That is a representation-geometry observation, not a guarantee of chemical co-editing.

### 3.6 Paper 1 traversal results — Grade A/B

Paper 1 Appendix E.5 reports that decoded traversals are architecture- and property-dependent. The linear-attention condition has high reconstruction/readability but can show plateaus, cliffs, saturation, repeated outputs, and nonmonotonicity. Simple attention is more responsive in some traversals but has saturation and bending behavior, including FractionCSP3 saturation, HBA downturn, and scale-dependent cLogP behavior.

Appendix E.2 adds monotonic traversal examples for BertzCT and HeavyAtomCount. These establish that some global directions work over selected ranges; they do not establish a general trust region.

### 3.7 Paper 1’s architecture-level summary

The paper’s most defensible architecture conclusion is:

> Better reconstruction does not guarantee better intervention, and the ranking of architectures depends on whether one measures readability, residual readability, decoded validity, uniqueness, or traversal responsiveness.

That sentence is already close to Paper 2. Paper 2 must make it quantitative, cross-validated, and central rather than leave it as an appendix observation.

## 4. Buried, rebuttal, and repository-only results

### 4.1 E02: novelty audit — Grade B

`rebuttal/E02_FPSim2_novelty/data/` provides a reproducible-looking novelty protocol for 5,000 generated samples:

* canonical RDKit SMILES, isomeric and canonical;
* Morgan fingerprints, radius 2, 2048 bits, no chirality;
* FPSim2 0.7.4 and RDKit 2025.9.3;
* comparison against the 635,522-molecule training partition;
* 4,983 distinct valid canonical structures and 17 duplicate excess samples.

At threshold 0.7, the structure-weighted result is 4,971/4,983 = 99.7592% novel. The structure-weighted nearest-training-similarity median is 0.2373, the 95th percentile is 0.3997, the 99th percentile is 0.5517, and the maximum is 1.0. The denominator audit includes a Wilson interval of approximately 0.9958–0.9986 for the sample-weighted threshold result.

This is evidence about novelty, not about local chemical actionability. A highly novel output can be a decoder attractor or a chemically unreasonable extrapolation.

### 4.2 E03: nonlinear nuisance residualization — Grade B

`rebuttal/E03_nonlinear_confounds/data/` is one of the better methodological additions. It uses five-fold cross-fitted nuisance residualization, with out-of-fold training residuals and a nuisance model trained on the full training set for validation/test prediction. The linear nuisance model uses standardized confounds and multi-output LinearRegression. The nonlinear nuisance model uses a ReLU MLP with hidden widths 64 and 32, batch size 1,024, early stopping, and a 75-iteration cap.

Held-out nonlinear-confound predictability is very small for most properties. The largest reported value is approximately 0.00907 for HeavyAtomCount; RingCount is approximately 0.00170, Exact/length-related entries are near 0.001, and MolWt is approximately 0.00089. After nonlinear residualization, latent R² remains:

| Property | R² after nonlinear residualization |
|---|---:|
| FractionCSP3 | 0.76064 |
| HBA | 0.66169 |
| BertzCT | 0.64784 |
| cLogP | 0.64142 |
| TPSA | 0.64121 |
| HeavyAtomCount | 0.56265 |

This supports the narrower claim that the selected four-confound panel does not explain away all held-out latent/property signal under the tested nonlinear nuisance model. It does **not** establish causal disentanglement, complete graph-size control, or chemical-mechanism control.

### 4.3 E04: residual traversal — Grade B

`rebuttal/E04_residual_traversal/data/` compares raw and residual directions over signed alpha curves with 50 seeds and all cached valid points. It explicitly avoids declaring a single automatic trust filter. Direction scaling is audited in original units, with unit-L2 traversal directions and alpha up to 150.

The headline median Spearman values are:

| Property | Raw direction on raw property | Residual direction on raw property | Residual direction on residual diagnostic |
|---|---:|---:|---:|
| FractionCSP3 | 0.9377 | 0.9117 | 0.8925 |
| HBA | 0.9547 | 0.9293 | 0.8810 |
| TPSA | 0.9297 | 0.9146 | 0.8650 |
| cLogP | 0.9348 | 0.9587 | 0.9258 |
| BertzCT | 0.9440 | 0.6894 | 0.7172 |
| HeavyAtomCount | 0.9734 | 0.6606 | 0.7463 |

Overall validity is approximately 1.0 for most rows; this is not surprising for SELFIES and should not be mistaken for local chemical fidelity. The most important result is the change in direction alignment: residualization reduces overlap with the confound subspace. For example, confound-subspace overlap drops from approximately 0.604 to 0.207 for HBA, from 0.604 to 0.120 for cLogP, from 0.911 to 0.293 for BertzCT, and from 0.975 to 0.127 for HeavyAtomCount.

The E04 data therefore supports a **mediator-reduction hypothesis**, not a claim of pure property disentanglement. Paper 2 can use it as a mechanistic covariate: a direction can remain property-responsive while changing its graph-size burden.

### 4.4 E05: graph and fragment changes — Grade B

`rebuttal/E05_chemical_changes/data/` contains 28,856 cached valid structures with 96 descriptor columns, a 50-seed representative-panel selection, a signed-alpha descriptor summary, and a large RDKit fragment summary. The descriptor cache includes molecular weight, carbon count, heavy-atom count, heteroatom count, ring counts, bond count, branch points, aromatic fractions, and fragment indicators.

The cache itself shows that several graph mediators are nearly redundant:

* MolWt versus calculated HeavyAtomCount: 0.9964;
* MolWt versus BondCount: 0.9938;
* MolWt versus CarbonCount: 0.9504;
* MolWt versus BranchPointCount: 0.3046;
* MolWt versus RingCount: 0.2192.

At the broad positive alpha=150 boundary, raw directions can make very large graph changes. For example, the raw cLogP direction has median changes of approximately +1,553 MolWt, +119.5 carbon atoms, +114 heavy atoms, −4.5 heteroatoms, −2 rings, and +111 bonds. The corresponding residual cLogP direction has approximately +435 MolWt, +37 carbon atoms, +34 heavy atoms, −5 heteroatoms, −3 rings, and +32.5 bonds. These are wide-boundary decoder trajectories, not proposed local edits.

The representative endpoint values quoted in the review (raw cLogP approximately +18.54 MolWt versus residual approximately +4.41 MolWt) appear to use a smaller alpha or a different representative panel than the alpha=150 summary. Both should not be merged. Paper 2 should report the exact alpha grid and endpoint definition with every graph-change number.

E05 is valuable because it makes “property steering” chemically inspectable. It also reveals the danger: a monotonic descriptor curve can be mediated by adding/removing large amounts of graph structure.

### 4.5 E06: signed traversal-quality curves — Grade B

`rebuttal/E06_traversal_quality/data/signed_alpha_traversal_quality.csv` contains 1,300 rows across 13 direction groups, signed alpha values, 50-seed aggregates, validity, canonical uniqueness, changed-from-seed fraction, seed Tanimoto statistics, scaffold retention/evaluable counts, training novelty, and token-ceiling-related information.

The most useful pattern is scale-dependent locality. Around small displacements near ±10.6 on the released grid, directions commonly retain seed similarity near 1.0 and scaffold retention roughly 0.84–0.96. Around ±19.7, seed similarity is roughly 0.5–0.64 and scaffold retention can fall to roughly 0.23–0.78. Around ±28.8, the regime becomes direction-dependent and scaffold retention can be near 0.05–0.44. At ±50, seed similarity is generally around 0.095–0.165 and scaffold retention is often zero. At ±150, unique fractions, validity, novelty, and token ceiling become highly direction-dependent.

Selected endpoint examples at alpha=+50 include:

* raw cLogP: validity 1.0, unique fraction 1.0, median seed Tanimoto approximately 0.143;
* raw TPSA: validity 1.0, unique fraction 1.0, median seed Tanimoto approximately 0.166;
* raw HBA: validity 1.0, unique fraction 1.0, median seed Tanimoto approximately 0.118;
* raw BertzCT: validity 1.0, unique fraction 1.0, median seed Tanimoto approximately 0.126.

The curve is the result. A single “trust alpha” would conceal the direction-specific tradeoff and should not be selected post hoc.

### 4.6 E07: ZINC250K transfer — Grade B/C

The ZINC rebuttal package includes a frozen-encoder notebook and a retraining notebook. Reported R² values are:

| Property | Frozen encoder + ZINC probes | Retrained same architecture |
|---|---:|---:|
| FractionCSP3 | 0.8875 | 0.8541 |
| HBA | 0.6579 | 0.5800 |
| TPSA | 0.6326 | 0.5952 |
| cLogP | 0.6020 | 0.6576 |
| BertzCT | 0.7753 | 0.7523 |
| HeavyAtomCount | 0.7260 | 0.6872 |

This is transfer evidence, not zero-shot transfer of original MolecuLA coefficients: the frozen encoder is paired with ZINC-specific probes. The notebook uses an 80/20 split and selects traversal seeds with an unseeded `np.random.randint` call. The retrained checkpoint is not in the export, and numerical traversal tables are not released; only plots/notebooks are available. Treat it as a useful replication lead, not a finished cross-dataset claim.

### 4.7 E08: matched SMILES representation — Grade B/C

The SMILES notebook is the strongest representation-dependence lead. The review reports approximately 97.65% reconstruction and only 9.33% traversal validity, with more than 90% of traversal candidates invalid. Among valid candidates, the reported property ordering is broadly sensible.

The notebook’s residual R² values include:

| Property | Raw R² | Residual R² |
|---|---:|---:|
| MolWt | 0.9043 | 0.6270 |
| HeavyAtomCount | 0.9705 | 0.6091 |
| cLogP | 0.3612 | 0.6921 |
| TPSA | 0.3786 | 0.7158 |
| HBD | 0.0604 | 0.6970 |
| HBA | 0.4089 | 0.6963 |
| NumRotatableBonds | 0.4112 | 0.5650 |
| RingCount | 0.7498 | 0.7017 |
| AromaticRingCount | 0.5467 | 0.7832 |
| FractionCSP3 | 0.3111 | 0.7837 |
| Spiro | 0.0176 | 0.1174 |
| Bridge | 0.0367 | 0.1254 |
| BertzCT | 0.8614 | 0.7504 |
| QED | 0.1479 | 0.3688 |
| SA | 0.0952 | 0.6947 |

The provenance is not yet clean enough for a headline. The notebook has an earlier data split and later per-target 80/20 split logic, the SMILES checkpoint is absent, and exact traversal CSVs are absent. This should be rerun in a single current script with an exported checkpoint, split manifest, and per-seed raw outputs. If it replicates, it is excellent evidence that **representation can preserve readability while changing decoded actionability**.

### 4.8 E09: ChemSpacE comparison — Grade B, strongest Paper 2 seed

E09 must be split into two different experiments.

#### Method-faithful prior evaluation

`rebuttal/E09_ChemSpace/code/chemspace_protocol_utils.py` implements a deterministic adaptation of the released ChemSpacE workflow: score valid prior decodes, define mean±one-standard-deviation tails, fit a linear SVM, and traverse 200 prior seeds over 21 points in [-1,1]. The code records a fixed shuffle seed because the historical release did not fix one.

This external-interface evaluation gives valid fractions of 1.0 and median Spearman values between approximately 0.149 and 0.372 across the six reported properties. It is a faithful baseline interface, but it is not the paired local comparison and should not be used to claim that OLS and ChemSpacE were evaluated on identical seed neighborhoods.

#### Paired native-L2 displacement evaluation

`rebuttal/E09_ChemSpace/data/definitive_paired_displacement/` compares MolecuLA continuous OLS with an adapted extreme-SVM method on the same frozen decoder, 50 encoded seeds, 21 displacement points, native latent L2, and common candidate accounting. The adaptation uses the top/bottom 200 eligible rows per class and is explicitly not a full ChemSpacE-supported generative model.

The native-L2 aggregate pattern is:

| Property | Method | Valid fraction | Canonical unique fraction | Median Spearman | Median seed similarity | Token-ceiling fraction |
|---|---|---:|---:|---:|---:|---:|
| FractionCSP3 | MolecuLA OLS | 1.000 | 0.883 | 0.938 | 0.080 | 0.095 |
| FractionCSP3 | adapted extreme-SVM | 1.000 | 0.291 | 0.919 | 0.022 | 0.874 |
| HBA | MolecuLA OLS | 1.000 | 0.749 | 0.955 | 0.084 | 0.167 |
| HBA | adapted extreme-SVM | 0.997 | 0.349 | 0.916 | 0.029 | 0.786 |
| TPSA | MolecuLA OLS | 1.000 | 0.810 | 0.936 | 0.090 | 0.110 |
| TPSA | adapted extreme-SVM | 1.000 | 0.449 | 0.837 | 0.036 | 0.860 |
| cLogP | MolecuLA OLS | 1.000 | 0.792 | 0.935 | 0.092 | 0.146 |
| cLogP | adapted extreme-SVM | 1.000 | 0.261 | 0.949 | 0.027 | 0.812 |
| BertzCT | MolecuLA OLS | 1.000 | 0.694 | 0.949 | 0.082 | 0.224 |
| BertzCT | adapted extreme-SVM | 0.919 | 0.167 | 0.726 | 0.013 | 0.610 |
| HeavyAtomCount | MolecuLA OLS | 0.970 | 0.656 | 0.968 | 0.086 | 0.221 |
| HeavyAtomCount | adapted extreme-SVM | 0.563 | 0.301 | 0.832 | 0.039 | 0.476 |

The exact current rows are in `compact_summary.csv`; the values above use canonical unique fraction where available. The OLS method is not uniformly superior in raw property effect, and the adapted SVM can be stronger for cLogP rank ordering. The scientific result is the tradeoff: **stronger nominal excursions are accompanied by more aggressive displacement, lower locality, lower scaffold evaluability, more token-ceiling exposure, and more SA deterioration**.

The constrained-design table is especially discriminating. For the adapted extreme-SVM in the native-L2 21-point paired grid, there are zero successful seeds at Tanimoto cutoffs 0.2, 0.4, and 0.6 for the listed properties. At cutoff 0, it can produce large improvements—for example, median improvements of approximately 150 HBA, 1,733.51 TPSA, 57.21 cLogP, and 10,732.90 BertzCT in the maximizing rows—but these are not local design successes. MolecuLA OLS retains nonzero success fractions at cutoffs .2, .4, and .6 for many properties; examples include cLogP maximizing .60, .34, and .06, and TPSA maximizing .80, .46, and .18.

The E09 claim-boundary table explicitly says not to claim that MolecuLA is a better optimizer than complete ChemSpacE-supported generative models, that SELFIES/RDKit validity establishes chemical plausibility, or that the paired extreme-SVM is a full ChemSpacE implementation. Paper 2 should preserve those qualifications verbatim in substance.

### 4.9 Archived corrected validation — Grade C, scientifically important

The source archive contains `single_property_direction_validation_corrected.ipynb`, `single_property_direction_validation_span_sweep_cLogP.ipynb`, and associated result tables. These are not in the main Paper 1 or official rebuttal result set, but they change the assessment of the project enough that they must be surfaced.

The protocol uses standardized latent directions, 12 seeds, dynamic-programming path selection, free-length/fixed-length decoding modes, and displacement spans. It checks validity, uniqueness, path constancy, adjacent Tanimoto, endpoint similarity, confound slopes, and geometry consistency.

For simple attention, the cLogP run reports raw probe R² 0.6824 and residual R² 0.5791, yet the corrected traversal summary has target Spearman approximately −0.0776, a slope near numerical zero, validity 1.0, uniqueness fraction approximately 0.00926, path-constant fraction approximately 0.8333, endpoint similarity 1.0, and adjacent Tanimoto 1.0. FractionCSP3 and TPSA have the same broad pattern: valid decodes, almost no unique movement, and often undefined Spearman because the decoded path is constant. HBA has an apparent Spearman of approximately 0.7274, but its near-zero slope and repeated paths make that number non-interpretable as useful control.

For linear attention, cLogP raw R² is approximately 0.7699 and residual R² approximately 0.6572, but x1/x4/x16 spans are nearly constant, with uniqueness around 0.00694 and path similarity 1.0. At x64, some rank correlations become nonzero, but results are not robust across free/fixed-length modes and can still have adjacent Tanimoto 1.0 or low uniqueness. The span-sweep FractionCSP3 and HBA runs show the same “no local movement until a much larger span” pattern.

These are potentially the most direct demonstrations of readability without control. They are not publication-grade yet because the result manifests contain old absolute paths, the runs are archived/compact rather than a clean current rerun, and some summary statistics are vulnerable to repeated-output artifacts. They should be treated as a **high-priority replication target**, not silently promoted to established evidence.

## 5. New phenomena that could support Paper 2

The following are candidates for new scientific phenomena. I distinguish what is already supported from what remains a hypothesis.

### 5.1 Readability–actionability dissociation — supported as a hypothesis; partly observed

**Claim:** held-out latent/property predictability can be high while decoded interventions produce repeated, constant, invalid, or chemically nonlocal outputs.

**Existing support:** Paper 1 Appendix E.5, E08 SMILES, E09 paired native-L2 results, and the repository-only corrected-validation runs. The corrected runs are the most direct but need rerunning. The E09 result is the most rigorously released.

**What would make it a publishable phenomenon:** a preregistered scatter/curve analysis showing that probe R², residual R², and MLP ΔR² do not reliably predict local actionability across held-out seeds and conditions. “Not reliably” must be quantified with confidence intervals and paired comparisons, not asserted from a few examples.

### 5.2 A local trust frontier — supported qualitatively; not yet calibrated

**Claim:** each representation/generator/direction/property combination has a displacement regime in which property response, validity, uniqueness, and locality coexist; beyond it, one or more quality dimensions fail.

**Existing support:** E06 signed curves and E09 similarity-constrained success tables. The frontier is direction-specific and may be asymmetric between positive and negative displacement.

**What remains open:** how to define the frontier without looking at the test curves first; whether it transfers to new seeds, properties, and datasets; whether native L2 or standardized latent units are the more stable coordinate; and whether the frontier is a smooth transition or a decoder phase change.

### 5.3 Intervention-method extremeness/locality tradeoff — strongest current phenomenon

**Claim:** a direction learner that is effective for global rank ordering or large property excursions can be worse for local, scaffold-preserving control.

**Existing support:** E09 paired native-L2 OLS versus adapted extreme-SVM. The SVM is more aggressive and often reaches the decoder token ceiling; OLS retains much better local similarity and constrained success. ChemSpacE’s own published literature also discusses out-of-distribution behavior and repeated outputs during extensive traversal, so Paper 2 must frame its contribution as a controlled comparative audit, not discovery of collapse itself.

**What remains open:** whether this tradeoff persists for Ridge, nonlinear local gradients, other decoders, other latent normalizations, and independent generators.

### 5.4 Representation-dependent intervention geometry — promising, incomplete

**Claim:** SELFIES and SMILES may preserve comparable property readability while inducing very different decoder validity and local continuity.

**Existing support:** E08’s reported 97.65% reconstruction versus 9.33% traversal validity, plus the high residual R² values. The evidence is incomplete because the SMILES checkpoint, clean split manifest, and numeric traversal tables are not in the export.

**What would make it strong:** the exact same seed registry, model budget, decoder mode, latent scaling, direction fit, and candidate count for SELFIES and SMILES, with validity and locality curves rather than a single aggregate validity number.

### 5.5 Residualization changes chemical mediation, not just statistics — supported in selected properties

**Claim:** residual directions can retain target-property ordering while reducing graph-size and sequence-confound alignment, changing the chemical edit profile.

**Existing support:** E03/E04/E05, especially the drop in confound-subspace overlap and the raw-versus-residual graph-change summaries.

**Risk:** this can be read as Paper 1 extended unless it is made subordinate to the actionability question. Paper 2 should treat residualization as one intervention factor and ask whether it improves the locality/effect Pareto frontier, not repackage it as the main contribution.

### 5.6 Nonlinear readability is not automatically local nonlinear control — proposed

**Claim:** an MLP’s superior held-out R² may reflect piecewise property encoding that has no single stable decoded intervention path.

**Existing support:** MLP ΔR² tables only. No MLP local-gradient decoded traversal is complete in the exported record.

**Status:** this is a strong secondary hypothesis, not an existing result. It should be tested after the linear/SVM/representation benchmark is stable.

## 6. Candidate Paper 2 framings

| Rank | Candidate framing | Why it is distinct | Existing evidence | Main risk | Decision |
|---:|---|---|---|---|---|
| 1 | **Readability is not controllability: actionable neighborhoods in molecular latent spaces** | Recasts Paper 1’s post-hoc readable axes as a testable intervention-validity problem; supports a cross-factor benchmark | E06, E08, E09, archived corrected runs | Could become a slogan unless the factorial benchmark is real | **Recommended** |
| 2 | **The locality–effect tradeoff of latent direction learners** | Makes OLS/Ridge/SVM/local gradients comparable under common decoder and constraints | E09 is already unusually strong | One decoder and an adapted SVM may look narrow | Strong component of #1 |
| 3 | **Residualization as mediator-aware molecular steering** | Connects confound removal to graph/fragment changes and constrained edits | E03–E05 | High overlap with Paper 1’s confound-aware thesis | Use as mechanism, not title |
| 4 | **Representation determines whether a readable molecular latent is actionable** | SELFIES/SMILES makes the result concrete and experimentally sharp | E08, archived runs | Current SMILES artifact is incomplete and validity may be representation syntax rather than geometry | Add as a major factor |
| 5 | **Local nonlinear fields for molecular property steering** | Builds on MLP ΔR² and Appendix F local gradients | Paper 1 MLP tables and proposal | No finished decoded MLP result; easy to overclaim | Defer unless fully executed |

### Recommended title and question

Working title: **When Molecular Latent Spaces Can Be Read but Not Intervened On**.

Primary question:

> **Which combinations of representation, generator, direction learner, and displacement regime produce a decoded molecular neighborhood that is simultaneously property-responsive, valid, unique, and chemically local?**

This question is stronger than “can we steer cLogP?” because Paper 1 already answers that selected descriptors can be steered in some regimes. It asks for the boundary conditions under which steering is trustworthy.

### The claim Paper 2 should make

If the experiments replicate the current pattern, the central claim should be:

> **Latent readability and decoded actionability are separable properties. In molecular Transformer-VAEs, the actionability profile is governed by representation, decoder geometry, direction learner, and displacement scale; evaluating only reconstruction, probe R², or unconstrained property gain can overstate controllability.**

This is a meaningful ML-for-science claim even if no new generative architecture is introduced.

## 7. Recommendation: standalone Paper 2 versus MolecuLA Extended

### Recommendation

Build a **standalone Paper 2** around the actionability benchmark. Do not submit a paper whose main novelty is simply:

* more properties in the existing probe table;
* another set of raw/residual traversal plots;
* the same E09 comparison with more screenshots;
* a second dataset without a held-out intervention protocol; or
* a larger MLP probe table without decoded local paths.

Those would be scientifically useful extensions but not a clean new paper.

### What makes it genuinely different

Paper 1 asks: **what information is encoded and can selected directions steer it?**

Paper 2 asks: **when does a readable direction survive the encoder–decoder interface as a trustworthy local intervention, and which factors determine failure?**

The unit of analysis changes from a property/direction to an **actionability profile** over displacement and locality constraints. The primary outputs become paired per-seed curves, failure modes, and cross-condition effect estimates. Paper 1’s probe tables become baselines/covariates, not the result.

### What can be reused responsibly

* the frozen checkpoints and training data split, if re-evaluated with explicit versioning;
* the Paper 1 probe and residualization code as baseline direction learners;
* E06 and E09 as pilot results and protocol starting points;
* the confound panel and RDKit descriptor cache as one of several quality panels;
* the novelty audit as a negative-control/secondary metric;
* Paper 1 figures only as motivation, with no duplicated main figure.

### What must be new

* a locked evaluation protocol and held-out seed registry;
* at least one independent representation or generator condition;
* a direct method comparison under common candidate budgets;
* local displacement curves, not only broad endpoint plots;
* statistical tests of readability versus actionability;
* clean manifests and rerunnable scripts for E08 and the archived corrected validation; and
* a new result table whose rows are representation/model/direction/scale conditions, not simply the same descriptor rows.

## 8. Existing evidence matrix

| Hypothesis / result | Main paper | Rebuttal | Repository-only | Evidence strength now | What it can support |
|---|---:|---:|---:|---|---|
| Selected properties are linearly readable | Yes | — | Matching tables | A | Baseline readability |
| Residual signal survives simple/nonlinear nuisance controls | Appendix / partial | E03 | Matching tables | B | Mediator-control covariate |
| Raw/residual directions can be monotonic after decoding | Appendix | E04 | Matching tables | B | Existence of selected steering regimes |
| Reconstruction/readability does not guarantee smooth traversal | Appendix E.5 | E06/E08 | Corrected validation | B/C | Central Paper 2 hypothesis |
| Small displacement can preserve local structure | Partial | E06/E09 | Corrected validation | B/C | Trust-region analysis |
| Larger property effect can be less local | Qualitative | E09 | Exact paired CSVs | B | Strongest Paper 2 pilot |
| ChemSpacE/extreme-SVM is more aggressive and collapses locality | — | E09 | Exact paired CSVs | B | Method tradeoff, with qualifications |
| SELFIES versus SMILES changes actionability | — | E08 | Incomplete notebook | C | Representation factor after rerun |
| ZINC transfer of readability | — | E07 | Incomplete notebooks | B/C | Replication lead, not final claim |
| MLP local decoded steering works | Proposed in appendix | — | No finished result | P | Must be tested or dropped |
| Molecular design usefulness / synthesis / biology | No | No | No | P | Out of scope until new task/oracle is added |

### Provenance issues that must be resolved before submission

1. **Autoregressive metric discrepancy:** headline `metrics.json` versus split-level reconstruction table.
2. **Simple-attention generation discrepancy:** Paper 1 table versus current phase-2 skipped status.
3. **Different probe conventions:** ordinary LinearRegression in Paper 1 tables, Ridge/RidgeCV in auxiliary notebooks, and different MLP architectures across public and archived notebooks.
4. **E08 split inconsistency:** initial data preparation and later per-target probe code use different split patterns.
5. **E07 randomization:** traversal seed selection is not explicitly seeded in the notebooks.
6. **Archived validation paths:** old absolute paths and compact output need a clean current rerun.
7. **E09 method distinction:** official method-faithful prior evaluation and paired adapted extreme-SVM comparison are not the same experiment.
8. **Missing full arrays/checkpoints:** the export is an audit snapshot, not a full replication bundle.

## 9. Critical missing experiments

### Must-have 1: clean intervention benchmark

Create one authoritative script/config that takes a frozen encoder/decoder, a direction learner, a property, an evaluation seed registry, and a displacement grid, then emits per-candidate rows. The row schema should include:

* model/representation/checkpoint hash;
* data split and seed ID;
* direction learner and fit-set hash;
* property target and residualization method;
* latent normalization and displacement units;
* alpha and signed side;
* decoded string and canonical molecular identifier;
* validity and decode status;
* property value and target effect;
* uniqueness and changed-from-seed flags;
* Morgan/Tanimoto similarity to seed;
* scaffold identity/retention;
* SA and NP-like auxiliary metrics;
* token length and max-token/ceiling flag;
* training novelty at the declared threshold; and
* error/failure reason.

No summary should be generated without retaining these raw rows.

### Must-have 2: factorial representation/generator study

Minimum viable design:

* **Representations:** SELFIES and canonical SMILES, using matched data and explicit tokenizer/decoder settings.
* **Generators:** the released autoregressive Transformer-VAE plus the released simple- and linear-attention conditions. If the latter two cannot be rerun consistently, use only conditions with verified checkpoints and label the study “decoder/architecture comparison,” not a broad family law.
* **Training seeds:** at least three independent seeds for the primary autoregressive condition, or a clearly justified fixed-checkpoint replication if compute prevents retraining.
* **Properties:** cLogP, TPSA, HBA, FractionCSP3, BertzCT, HeavyAtomCount, plus one nonlinear/weak property such as QED or HBD.
* **Direction learners:** ordinary OLS, Ridge with fixed alpha selected on training only, ChemSpacE-style SVM, random-direction null, and MLP local gradient if implemented cleanly.
* **Displacements:** native L2 and standardized-coordinate grids, with a pilot-derived conversion that is frozen before test evaluation.

The main statistical question is not whether every cell is significant. It is whether actionability metrics vary independently of readability and whether representation/model/direction interactions explain a substantial fraction of that variation.

### Must-have 3: held-out seeds and no label leakage

Use a three-way molecule registry:

1. direction-fitting molecules/latents;
2. calibration molecules used only to choose a displacement scale under a declared rule; and
3. final intervention seeds never used for direction fitting, extreme-label selection, or threshold selection.

For the main test, direction learners should not see the final seed molecules or their target properties. If the decoder is deterministic, the seed registry must still prevent exact duplicates from leaking into direction fitting.

### Must-have 4: actionability curves, not one score

Report per-alpha curves for:

* target-property rank correlation and signed slope;
* per-seed expected-sign fraction;
* validity;
* canonical uniqueness;
* changed-from-seed fraction;
* seed Tanimoto median and IQR;
* scaffold retention/evaluable fraction;
* token ceiling and output length;
* SA/NP-like changes;
* training novelty; and
* constrained success at Tanimoto cutoffs 0.2, 0.4, and 0.6.

Do not collapse these into one composite actionability score until the curves are shown. If a composite is useful, preregister the normalization and report the full vector beside it.

### Must-have 5: rerun the archived “constant path” finding

The corrected validation should be rerun from current repository paths with:

* the exact released checkpoint hash;
* 50 or more held-out seeds for the main analysis, not only 12;
* a dense but computationally bounded span sweep;
* fixed-length and free-length modes reported separately;
* repeated-output detection before calculating Spearman;
* exact raw candidate rows;
* independent geometry checks; and
* a predeclared rule that undefined rank correlation caused by constant paths is a failure mode, not a favorable result.

This is the experiment most likely to turn an intriguing repository observation into a decisive paper figure.

### Must-have 6: representation control

Rerun E08 with a single data-preparation script and one split manifest. Export:

* SMILES tokenizer and decoder checkpoint;
* exact training/evaluation rows or immutable row IDs;
* probe fit and residualization tables;
* per-seed traversal rows;
* invalid-string/failure categories; and
* matched SELFIES results generated by the same script.

The key comparison is not merely “SMILES has lower validity.” It is whether the two representations have comparable readability but different local actionability profiles after controlling for latent scale and decoder token limits.

### Important but second-wave experiments

* MLP local-gradient directions with a trust-region step rule and per-seed decoded paths.
* A graph/fragment mediator model that distinguishes size changes from functional-group changes.
* An independently trained molecular generator with a different decoder family, such as a graph or 3D model, if the research team can support the scope.
* ZINC250K transfer with fixed random seeds, a held-out traversal registry, and numeric tables rather than plots only.
* An oracle budget comparison against a real optimization method, only if Paper 2 wants to make a design-efficiency claim.
* A small externally meaningful objective, such as a calibrated QSPR or docking proxy, only after the actionability benchmark is stable. This is not needed for the core ML phenomenon but would strengthen a chemistry-facing journal submission.

### Experiments I would not prioritize now

* More descriptors from the same RDKit family without a new intervention question.
* More broad alpha=150 endpoint plots.
* A larger MLP probe table without decoded paths.
* Claiming biological relevance from computed descriptors.
* Adding ChemFlow as a baseline without a complete, fair, oracle-budget-matched implementation.
* A new model architecture before the existing actionability failure modes are quantified.

## 10. Full experimental roadmap

### Phase 0 — provenance freeze

**Deliverables:** a `paper2_protocol.yaml`, checkpoint hashes, split/seed registry, versioned environment, and a one-page metric dictionary.

**Stop/go gate:** every existing headline result can be regenerated from a named script or is explicitly marked historical/recovered. If the autoregressive metrics cannot be reconciled, choose one metric definition and report the discrepancy in supplementary material.

### Phase 1 — replication of the decisive pilots

1. Rerun the archived linear/simple constant-path validations.
2. Rerun E08 SELFIES/SMILES in one script.
3. Recompute E09 native-L2 paired summaries from raw candidate rows.
4. Add bootstrap confidence intervals and per-seed plots.

**Go gate:** at least two independent conditions reproduce high readability with weak or scale-dependent actionability, and E09’s locality/effect tradeoff survives exact recomputation.

### Phase 2 — controlled benchmark

Run the minimal factorial design with held-out seeds. Fit all directions on the same training rows and evaluate all methods under the same candidate budget. Freeze displacement-calibration rules before opening final test outputs.

**Primary figures:**

1. readability versus local actionability scatter, with points/intervals for every condition;
2. signed displacement curves for property response and locality;
3. Pareto frontiers of property improvement versus Tanimoto/scaffold retention;
4. representation × decoder heatmap of failure modes; and
5. method comparison showing OLS/Ridge/SVM/local-gradient behavior under common constraints.

### Phase 3 — mechanism and generalization

Use E03–E05-style mediator and fragment analyses to explain why the frontier moves. Add ZINC or an independent dataset only after the primary benchmark is locked. The point of transfer is not to collect another R² table; it is to test whether the actionability relationship generalizes.

### Phase 4 — chemistry-facing validation

If the benchmark reveals a stable local regime, test whether selecting candidates inside that regime improves a declared downstream proxy. The proxy must be independently justified, held out where possible, and clearly separated from the descriptor-level paper claim.

### Phase 5 — manuscript assembly

The main paper should contain the phenomenon and the causal factors. The full descriptor inventory, old Paper 1 tables, alternative probe conventions, historical E07/E08 plots, and all exploratory sweeps belong in a provenance-rich supplement or archive.

## 11. Literature-gap and novelty assessment

This is not a claim that no one has studied molecular latent traversal. They have. The novelty must be located at the level of **evaluation question and controlled comparison**.

### 11.1 Closest prior work

**Continuous molecular latent representations.** Gómez-Bombarelli et al. introduced a continuous molecular representation in which perturbation, interpolation, and gradient-based optimization can be performed in latent space. The relevant conceptual precedent is the idea that a continuous latent code enables molecular exploration, not the Paper 2 question of whether reconstruction/readability predicts safe decoded intervention. See [Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572).

**LIMO.** LIMO trains a VAE plus a differentiable property-prediction path and reverse-optimizes latent variables, using SELFIES to maintain validity. It is a targeted generation/optimization framework. It does not provide the proposed cross-factor audit of readability, locality, representation, decoder saturation, and intervention method. See [LIMO: Latent Inceptionism for Targeted Molecule Generation](https://proceedings.mlr.press/v162/eckmann22a/eckmann22a.pdf).

**ChemSpacE.** ChemSpacE identifies latent property directions with a linear SVM and frames the task as interactive molecular manipulation. It explicitly motivates smooth property and structure change and evaluates manipulation success. Its original paper is the closest methodological overlap, so Paper 2 must not claim novelty for linear latent directions, post-hoc direction learning, or interactive traversal. See the primary [ChemSpacE paper](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/62b40c367da6ce70d11c94e8/original/chem-spac-e-toward-steerable-and-interpretable-chemical-space-exploration.pdf).

**ChemFlow.** ChemFlow unifies latent traversal and optimization under flow dynamics and reports molecule manipulation, similarity-constrained optimization, and out-of-distribution behavior. Its NeurIPS paper explicitly notes that ChemSpacE outputs can converge to a few molecules and that unsupervised optimization degrades as similarity constraints increase. This means decoder collapse and locality degradation are already in the literature as observations. The distinct Paper 2 contribution would be to make them the object of a controlled **readability-to-actionability benchmark**, compare representations and direction learners under a common decoder, and quantify the frontier rather than proposing a new flow method. See [Navigating Chemical Space with Latent Flows](https://proceedings.neurips.cc/paper_files/paper/2024/file/6bbefb73c0ede70635823a18426b9208-Paper-Conference.pdf).

**Latent-space quality/continuity studies.** Prior work measures reconstruction, validity, continuity, interpolation, or latent topology. These are relevant quality axes, but they are usually not organized around the specific question of whether held-out property readability predicts a valid, local, unique, non-saturated decoded intervention. Paper 2 should cite this literature and avoid presenting “continuity” or “smoothness” as a newly invented concept.

### 11.2 Defensible gap statement

The defensible gap is:

> Existing molecular latent-space work demonstrates property optimization, manipulation, or continuity in individual model/protocol settings. There is still room for a systematic, representation- and direction-learner-aware audit that separates latent readability from decoded actionability and reports the tradeoff between property effect, validity, uniqueness, locality, scaffold retention, novelty, and decoder saturation on held-out seeds.

This is an **inference from the literature and the audited repository**, not a proof of exhaustive coverage. The final manuscript must update the search immediately before submission, including papers published after this audit date.

### 11.3 What would make the gap close

The gap disappears if a concurrent paper already provides the same factorial design, the same distinction between readability and decoded actionability, and comparable held-out locality/saturation metrics. Before submission, search recent work using terms including “molecular latent controllability,” “latent traversal actionability,” “representation intervention validity,” “molecule manipulation similarity constraint,” “decoder saturation molecular generation,” and “latent-space trust region.”

If a close paper appears, Paper 2 can still be valuable as a replication/reproducibility study, but its contribution and venue should be adjusted accordingly.

## 12. Overlap audit with Paper 1

### 12.1 Claim-level overlap

| Paper 1 claim | Paper 2 treatment | Overlap risk |
|---|---|---|
| Transformer-VAE latents encode RDKit properties | Use as baseline readability covariate | Low if no new contribution is claimed here |
| Confounds affect raw directions | Use residualization as one factor and mediator analysis | Medium; do not repeat the same confound story as the headline |
| Selected directions steer decoded properties | Treat as a necessary but insufficient condition | Low if the endpoint is actionability failure/frontier |
| MLPs reveal nonlinear readability | Test local MLP intervention as a secondary method | Low if decoded results are new; high if only the table is repeated |
| SELFIES validity/novelty | Use as controls, not chemical plausibility claims | Low |
| Architecture differences | Reframe as interaction with intervention quality | Medium; require new per-seed statistics |
| Raw/residual traversal plots | Recompute under held-out/actionability protocol | High if old plots are reused as main figures |

### 12.2 Data and figure reuse rules

Paper 2 can reuse the same training data and checkpoints because the scientific question is a new intervention evaluation, but it must:

* identify which results are reanalyses of Paper 1 checkpoints;
* keep the original Paper 1 tables in the supplement or cite them rather than duplicate them;
* give every reanalysis a new protocol name and script;
* avoid reusing the same figure with cosmetic changes;
* include a table mapping every Paper 2 result to its source artifact and whether it is new, reprocessed, or pilot; and
* disclose the relationship to the Paper 1 submission to the target venue.

### 12.3 Publication timing and policy risk

Paper 1 is still in the NeurIPS 2026 review cycle as of this audit date. A Paper 2 submission while Paper 1 is under review should be made only after checking the target conference’s current dual-submission and related-work rules and after ensuring that the manuscripts are not substantially overlapping. The NeurIPS 2026 call warns that irrelevant or duplicate papers can be desk-rejected and links its Main Track Handbook for dual-submission policies; see the [official NeurIPS 2026 call](https://neurips.cc/Conferences/2026/CallForPapers).

For a journal submission, Nature Machine Intelligence explicitly asks authors to disclose related manuscripts and prohibits significant overlap with another manuscript under consideration; see its [editorial policies](https://www.nature.com/natmachintell/editorial-policies). The safe practical rule is to finish the Paper 2 protocol and make its new question unmistakable before submitting anything.

## 13. Venue landscape and current official sources

### 13.1 Conference targets

| Venue | Current official information | Fit | Recommendation |
|---|---|---|---|
| **ICLR 2027** | Abstract deadline 11 September 2026 AOE; full paper deadline 16 September 2026 AOE; double blind. See [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines). | Excellent if framed as representation/generative-model behavior and the benchmark is complete. | **Best near-term conference target, but do not rush an underpowered paper.** |
| NeurIPS 2026 | Full-paper deadline was 6 May 2026; notification is 24 September 2026. See [NeurIPS 2026 Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers) and [dates](https://neurips.cc/Conferences/2026/Dates). | Scientifically excellent fit, but the deadline is closed. | Not available for Paper 2 this cycle. |
| NeurIPS 2027 | Official future-meetings page currently lists 2027 in Europe but does not provide a Paper 2 submission deadline. See [NeurIPS Future Meetings](https://neurips.cc/Conferences/FutureMeetings). | Strong if the cross-factor result is broad and methodologically rigorous. | Monitor when the official call appears. |
| ICML 2026 | Full-paper deadline was 28 January 2026; the call emphasizes original, rigorous ML contributions and application-driven ML. See [ICML 2026 Call for Papers](https://icml.cc/Conferences/2026/CallForPapers). | Good fit for a stronger general ML evaluation story. | Closed this cycle; next deadline not yet official in the audited sources. |

**Conference recommendation:** target ICLR 2027 only if Phase 1 and the minimal Phase 2 benchmark can be completed, audited, and written before the September deadlines. Otherwise, wait for the next NeurIPS/ICML cycle rather than submit a rebuttal-derived paper with unresolved provenance.

### 13.2 Journal targets

| Venue | Fit and current official scope | Recommendation |
|---|---|---|
| **Journal of Chemical Information and Modeling (JCIM)** | Its official scope includes AI/ML applied to chemical and biological data, computer-aided molecular design, computational methods, and analysis of chemical/physical data. See [JCIM aims and scope](https://pubs.acs.org/page/jcisd8/about.html). | **Best practical journal target** after a clean chemistry-facing benchmark. |
| Machine Learning: Science and Technology | Its scope bridges ML applications in science with ML advances motivated by physical insights and explicitly includes molecules/materials. See [MLST scope](https://publishingsupport.iopscience.iop.org/journals/machine-learning-science-and-technology/about-machine-learning-science-technology/). | Strong alternative if the manuscript emphasizes a general ML-for-science evaluation principle. |
| Nature Computational Science | It publishes fundamental/applied computational techniques and their cross-disciplinary application. See [journal information](https://www.nature.com/natcomputsci/journal-information). | Stretch target if the result generalizes broadly and the study has strong methodological completeness. |
| Nature Machine Intelligence | An Article must be a substantial novel study with a complex story and broad editorial relevance; the journal evaluates advance, soundness, evidence, and readership relevance. See [content types](https://www.nature.com/natmachintell/content) and [editorial process](https://www.nature.com/natmachintell/submission-guidelines/editorial-process). | Aspirational, not the best first target for the current one-dataset/one-family evidence. |

**Journal recommendation:** JCIM is the best realistic journal target; MLST is the best alternative if the paper becomes a general evaluation-method paper. Nature Machine Intelligence/Nature Computational Science should be treated as stretch submissions after independent model/representation replication, not as default targets.

## 14. Detailed Paper 2 blueprint

### 14.1 Proposed manuscript structure

**Title:** *When Molecular Latent Spaces Can Be Read but Not Intervened On*  
**Subtitle/alternative:** *A benchmark of decoded actionability across representations, generators, and latent direction learners*

**Abstract logic:**

1. Molecular generative models are commonly evaluated by reconstruction, validity, novelty, and property predictability.
2. These metrics do not test whether a post-hoc latent intervention produces a local, stable decoded change.
3. We introduce a held-out actionability benchmark over representation, generator, direction learner, and displacement scale.
4. We find [only after rerun: the validated empirical result], including the readability/actionability dissociation and locality/effect frontier.
5. We recommend reporting actionability curves and constrained success rather than unconstrained property gain alone.

### 14.2 Hypotheses to predeclare

* **H1 — Readability is insufficient:** held-out probe R² and residual R² are imperfect predictors of local decoded actionability.
* **H2 — Trust frontier:** actionability is nonmonotonic in displacement; small displacement can preserve locality but produce little effect, while larger displacement increases effect and decoder failure.
* **H3 — Direction learner interaction:** OLS/Ridge, SVM/extreme-label, and MLP local gradients have different effect/locality profiles even on the same decoder.
* **H4 — Representation interaction:** SELFIES and SMILES can have similar readability but different validity/continuity/actionability.
* **H5 — Residualization shifts mediation:** residual directions can reduce graph-size/sequence-confound alignment, but may also reduce target effect or change direction-specific locality.
* **H6 — Nonlinear readability does not imply global linear control:** MLP gains will not automatically yield a stable path under a single global direction.

H6 should be dropped if the MLP arm cannot be completed with the same leakage controls and per-seed raw outputs.

### 14.3 Primary analysis unit

The primary unit should be a **seed × direction × signed displacement point**, not an aggregate plot. Every aggregate should preserve seed-level uncertainty. Use hierarchical bootstrap or mixed-effects models with seed as a random effect and representation/generator/direction/displacement as fixed effects. For method comparisons on identical seed/alpha pairs, use paired bootstrap or paired permutation tests.

Do not treat 21 alphas from one seed as 21 independent samples. Do not average invalid rows away without reporting the denominator. Do not compute rank metrics on constant paths without an explicit failure label.

### 14.4 Suggested primary figures

1. **Readability/actionability map:** x-axis residual R² or MLP ΔR²; y-axis area under a predeclared local-actionability curve; point color for representation and shape for direction learner. Show confidence intervals and constant-path failures.
2. **Trust-region curves:** signed alpha on the x-axis; target effect, validity, uniqueness, Tanimoto, scaffold retention, and token ceiling in aligned panels.
3. **Pareto frontier:** property improvement versus Tanimoto similarity, with separate curves for OLS/Ridge/SVM/MLP and a random null.
4. **Representation × generator heatmap:** local actionability, constrained success, and failure-mode proportions.
5. **Chemical mediation panel:** raw versus residual directions with graph-size, heteroatom, ring, branch, and fragment changes.
6. **Per-seed examples:** a few representative paths plus failure cases, selected before inspecting the final target metric or clearly labeled as illustrative.

### 14.5 Suggested tables

* checkpoint/split/protocol registry;
* readability baseline table, with Paper 1 values clearly marked as reanalysis or inherited;
* actionability summary at predeclared local cutoffs;
* method-paired effect/locality differences with confidence intervals;
* invalid/repeated/token-ceiling failure taxonomy;
* ablation of latent normalization and displacement calibration; and
* exact data/code provenance for every figure.

### 14.6 What counts as a positive result

The paper does not require MolecuLA or OLS to win. A strong result could be:

* a replicated readability/actionability dissociation;
* a stable and interpretable trust frontier;
* a clear representation × decoder interaction;
* a method tradeoff where no method dominates all locality/effect axes; or
* a negative result showing that current probes cannot predict decoded intervention quality.

The null result that would weaken Paper 2 is also clear: if high R² reliably predicts local actionability across representations, models, methods, and held-out seeds, then the proposed gap is not a paper contribution. That is a reason to stop or reframe, not to search for a favorable property.

### 14.7 Operational definitions and guardrails

Use “actionable” only for the declared computational intervention metric. Avoid “chemically meaningful” unless supported by graph/functional-group analysis or an external chemical objective. Avoid “trustworthy” unless the paper defines a calibration rule and reports out-of-sample performance. Avoid “disentangled” unless the claim is narrowed to the tested confound subspace and statistical alignment.

The authors should not claim:

* biological activity;
* synthesizability;
* medicinal-chemistry usefulness;
* a universal molecular latent-space law;
* superiority over full ChemSpacE or ChemFlow;
* that SELFIES validity means chemical plausibility; or
* that a large descriptor excursion is a good molecular edit.

## 15. Immediate next actions

### In priority order

1. **Freeze a clean Paper 2 protocol and manifest.** Resolve checkpoint, split, probe, MLP, direction-scaling, and seed-registry conventions before running new analysis.
2. **Rerun the archived constant-path validation.** This is the highest information-value experiment because it tests the core hypothesis directly.
3. **Recompute E09 from raw rows.** Add paired confidence intervals and method-difference tests; retain the official-vs-adapted ChemSpacE distinction.
4. **Rerun E08 matched SELFIES/SMILES.** Export the missing checkpoint and numeric per-seed traversal table.
5. **Run the minimal factorial benchmark.** Do not expand properties or models until the primary actionability curves are stable.
6. **Add an explicit random-direction and permutation null.** This guards against property correlations and decoder drift being mistaken for steering.
7. **Decide on MLP scope.** Either execute local decoded MLP gradients with the same evaluation contract or remove MLP as a central claim.
8. **Write the paper around the failure boundary.** The main contribution should be the benchmark and the scientific conclusion, not the historical rebuttal chronology.
9. **Perform a fresh literature search immediately before submission.** Search for any 2026–2027 work on molecular latent controllability, actionability, latent trust regions, and decoder saturation.
10. **Choose the venue only after the go gate.** ICLR 2027 is the best near-term conference option; JCIM is the best realistic journal target; do not submit before the provenance and replication gates pass.

### Final verdict

* **BEST NEW SCIENTIFIC QUESTION:** Which representation × generator × direction learner × displacement regimes produce a decoded molecular neighborhood that is both property-responsive and chemically local?
* **WHY NOT “MOLECULA EXTENDED”:** the object of study changes from readable property axes to the out-of-sample actionability frontier and its failure mechanisms.
* **STRONGEST EXISTING RESULT:** E09’s paired native-L2 comparison: similar rank control can coexist with sharply different locality, uniqueness, scaffold retention, token-ceiling exposure, and constrained success.
* **MOST IMPORTANT MISSING EXPERIMENT:** a clean held-out readability-to-actionability factorial benchmark, beginning with rerun of the archived constant-path finding and matched SELFIES/SMILES validation.
* **BEST CONFERENCE TARGET:** ICLR 2027, only if the benchmark is completed before the official 11/16 September 2026 abstract/full-paper deadlines; otherwise the next NeurIPS/ICML cycle.
* **BEST JOURNAL TARGET:** JCIM for a chemistry-facing paper; MLST if the central result becomes a general ML-for-science evaluation principle.
* **READINESS:** conceptually viable, empirically promising, not yet submission-ready; approximately 5/10 now and 8/10 after the must-have reruns.
* **POTENTIAL:** high enough to justify the Paper 2 program. The project has a real phenomenon, but only a controlled replication can determine whether it is a robust scientific result or an artifact of latent scaling, decoder saturation, split leakage, or representation choice.

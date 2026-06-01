# Energy-adaptive retrieval in semantic fluency

This repository contains the minimal, reproducible analysis pipeline for a process model of animal semantic fluency. The central question is whether human fluency reflects an **energy-adaptive search process**: locally constrained retrieval most of the time, intermittent transitions to new semantic regions, and heavy-tailed retrieval difficulty.

The release includes the code and precomputed assets needed to reproduce the primary model comparison, M4b parameter inference, and posterior predictive checks.

---

## Core idea

Human semantic fluency is not treated only as a count of produced words. We focus on four process-level signatures:

| Signature | Statistic | Interpretation |
|---|---|---|
| Local semantic movement | `mean_adjacent_distance` | Consecutive responses stay close in embedding space |
| Variable search scale | `sd_adjacent_distance` | Transition distances fluctuate across a list |
| Intermittent switching | `switch_rate_distance` | Some transitions are relatively large within a list |
| Heavy-tailed retrieval difficulty | `mu_latency_median_norm` | Retrieval latency has a broad, heavy-tailed distribution |

The latency exponent is not used to fit the primary semantic-search model. It is used as an **out-of-target posterior predictive check**: a successful process model should generate heavy-tailed retrieval dynamics as a consequence of adaptive search, rather than by directly fitting timing.

---

## Model family

The repository compares six retrieval families:

| Model | Description |
|---|---|
| `M0_frequency` | Static frequency baseline |
| `M1_crw` | Correlated random walk in embedding space |
| `M2_crw_jump` | Correlated random walk with occasional long jumps |
| `M3_fixed_T` | Energy-guided proposal with fixed temperature |
| `M4_adaptive_T` | Temperature adapts to local energy gaps |
| `M4b_adaptive_T_global` | Adaptive temperature plus temperature-gated global restarts |

The winning model, **M4b**, combines adaptive temperature with adaptive global restarts. The mechanism is intended to capture how retrieval can remain locally constrained while occasionally escaping depleted or difficult semantic neighbourhoods.

---

## Repository layout

```text
semantic_github_minimal/
├── Model_Core/              # core model and analysis package
│   ├── config.py             # ModelParams, ProjectPaths, load_config
│   ├── sampler.py            # list generation and MH-style acceptance
│   ├── baselines.py          # M0–M4b model generators
│   ├── proposal.py           # local/global proposal kernels
│   ├── energy.py             # lexical energy landscape
│   ├── latency_budget.py     # retrieval attempts and latency budget
│   ├── simulate.py           # load assets and run simulations
│   ├── metrics.py            # semantic-trajectory statistics
│   ├── levy_flight.py        # heavy-tail latency exponent
│   ├── lopez_features.py     # primary process signatures and diagnostics
│   ├── abcrf_priors.py       # prior distributions over model families
│   └── lopez_reference.py    # reference-table builder
│
├── data/
│   ├── processed/animals_fluency_oaf.csv
│   ├── stats/
│   ├── figures/animals_oaf_human_fluency_signatures.{png,pdf}
│   ├── embeddings/
│   ├── vocab/
│   └── llm_scores/
│
├── Model_compare/
│   ├── 01_make_reference.py
│   ├── 02_run_abcrf.py
│   ├── 03_plot_comparison.py
│   ├── data/
│   └── figures/
│
├── abc/
│   ├── 02_run_abc_ppc.py
│   ├── 03_plot_ppc_forest.py
│   ├── 04_plot_mu.py
│   ├── data/
│   └── figures/
│
├── config/
│   ├── default_params.json
│   └── fitted_m4b_oaf.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Create a clean Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the repository root so that `Model_Core/` is importable and the `data/` directory is found.

---

## Reproducing the analysis

### A. Model comparison

The model-comparison stage generates prior-predictive simulations from the candidate model families and classifies the human summary vector using ABC random forests.

```bash
# Build the prior-predictive reference table
python Model_compare/01_make_reference.py

# Quick smoke test
python Model_compare/01_make_reference.py --n-draws 4 --n-lists 5

# ABC-RF model comparison
python Model_compare/02_run_abcrf.py

# Plot posterior model probabilities and feature importance
python Model_compare/03_plot_comparison.py
```

Precomputed reference tables are included in `Model_compare/data/` so that the full simulation does not need to be rerun before inspecting the main results.

### B. M4b inference and posterior predictive checks

After model comparison, the M4b model is fit by rejection ABC and evaluated with posterior predictive checks.

```bash
# Rejection ABC for M4b and posterior predictive simulations
python abc/02_run_abc_ppc.py

# Posterior predictive forest plot for process signatures
python abc/03_plot_ppc_forest.py

# Heavy-tail latency diagnostic
python abc/04_plot_mu.py
```

To regenerate the M4b reference table:

```bash
python Model_compare/01_make_reference.py \
  --models M4b_adaptive_T_global \
  --n-draws 1000 \
  --out-prefix abc/data/animals_oaf_lopez_style_reference_M4b_wideprior_1000x50_cap150
```

---

## Summary statistics

### Primary process signatures

The paper-level narrative focuses on four process signatures:

| Statistic | Role |
|---|---|
| `mean_adjacent_distance` | Mean semantic step size between consecutive responses |
| `sd_adjacent_distance` | Variability in semantic step size |
| `switch_rate_distance` | Fraction of within-list transitions above the list-specific 80th percentile |
| `mu_latency_median_norm` | Scale-free heavy-tail exponent of pooled retrieval latency |

Embedding distances are computed from unit-normalized sentence-embedding vectors:

```text
d(i,j) = sqrt(2 - 2*cos(theta_ij))
```

The switch-rate statistic is intentionally scale-free: the threshold is computed separately within each list, so it measures the relative frequency of large transitions rather than an absolute semantic-distance cutoff.

### Secondary diagnostics

The following statistics are computed and reported as diagnostics, but are not used as the main evidential basis for the energy-adaptive search claim:

| Statistic | Diagnostic role |
|---|---|
| `p90_adjacent_distance` | Magnitude of rare large semantic jumps |
| `n_relevant_bigrams_norm` | Length-detrended count of human-typical associative couples |
| `log_frequency_likelihood` | Empirical lexical-frequency typicality relative to human response frequencies |
| `mean_adjacent_similarity` | Distance-derived similarity diagnostic, nearly collinear with mean distance |
| `mu_latency_raw_pooled` | Raw latency exponent; reported only as robustness because human latency is in seconds and model latency is in retrieval turns |
| `list_length` | Stopping or continuation diagnostic |

This separation is important. The main claim is not that the model reproduces every descriptive statistic exactly. The goal is to test whether an adaptive energy-based retrieval process captures the core human pattern: local semantic movement, intermittent switching, and heavy-tailed retrieval difficulty.

---

## Data assets

The repository uses precomputed candidate-set assets:

- unit-normalized sentence-embedding vectors (`data/embeddings/animals_embeddings.npy`, ~15 MB),
- kNN index files,
- candidate vocabulary,
- LLM prompt scores,
- human animal-fluency lists.

The pairwise distance matrix is **not committed**: it is derived on the fly at load time
from the embeddings (`distance_sq = 2 - 2·a·b`, `distance = sqrt(distance_sq)`). This keeps
every file well under 25 MB (repo ≈ 22 MB), so no Git LFS is required. If you happen to
have `animals_distance_sq.npy` / `animals_distance_matrix.npy` locally, dropping them into
`data/embeddings/` is also supported (they are used directly when present).

The simulation pipeline reads these assets directly and does not require running the
original LLM scoring or embedding-generation pipeline (no `torch` / `transformers`).

---

## Outputs

Main tables are written to:

```text
Model_compare/data/
abc/data/
data/stats/
```

Main figures are written to:

```text
Model_compare/figures/
abc/figures/
data/figures/
```

The most important output figures are:

- human semantic-fluency signatures,
- ABC-RF posterior model probabilities,
- ABC-RF feature importance,
- M4b posterior predictive process-signature forest,
- M4b heavy-tail latency diagnostic.

---

## Interpretation of the released results

The intended interpretation is deliberately process-level. M4b should be read as a best-supported process approximation rather than a fully calibrated generator of every human statistic.

In posterior predictive checks, M4b captures central semantic-trajectory signatures and generates heavy-tailed latency dynamics without fitting the latency exponent directly. Remaining mismatches, including list length, rare-word production, large-jump magnitude, and associative richness, identify mechanisms that are not yet fully specified: stopping, richer lexical representations, and higher-order associative structure.

---

## Reproducibility notes

- Random seeds are set inside the simulation scripts where applicable.
- Precomputed reference tables are included for inspection and plotting.
- No Git LFS needed: the only sizable asset is the 15 MB embeddings file; the large distance matrix is recomputed from it at load time.
- The release is intentionally minimal: development notebooks, exploratory scripts, and unused intermediate outputs are excluded.

---

## Citation

If using this repository, please cite the associated manuscript/preprint when available.

```bibtex

```

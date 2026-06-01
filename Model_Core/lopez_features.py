"""López-style group-level feature names (shared across ABC / ABC-RF scripts)."""

from __future__ import annotations

# Primary 4-feature set: 3 embedding-space trajectory stats + 1 associative-structure
# metric (nRB, CN couples; Guimera et al. 2026).
# p90_adjacent_distance is NOT a primary feature: the upper-tail jump magnitude is
# sensitive to the embedding/candidate environment, so it is kept only as a diagnostic.
# log_frequency_likelihood (fLL) is NOT a primary feature either: only 47.6% of human
# word *types* (81.6% of tokens) are in the v5000 LLM-probe vocab, so the LLM-prior
# frequency likelihood is confounded with vocabulary coverage. fLL is a secondary
# lexical diagnostic computed from the full human empirical word frequency.
LOPEZ_SEMANTIC_FEATURES = [
    "mean_adjacent_distance",
    "sd_adjacent_distance",
    "switch_rate_distance",
    "n_relevant_bigrams_norm",
]

# Computed and reported, but excluded from the ABC target (mean_adjacent_similarity is
# ~collinear with mean_adjacent_distance; p90 is environment-sensitive).
LOPEZ_DIAGNOSTIC_SEMANTIC = ["p90_adjacent_distance", "mean_adjacent_similarity"]

# Reported in PPC / figures as a secondary diagnostic, not used for ABC selection.
LOPEZ_DIAGNOSTIC_LEXICAL = ["log_frequency_likelihood"]

# Human descriptive: pooled LBN on raw units (seconds or turns, min cutoff 1).
MU_LATENCY_RAW = "mu_latency_raw_pooled"

# Scale-normalized pooled LBN (robustness / timing diagnostic).
MU_LATENCY_MEDIAN_NORM = "mu_latency_median_norm"

MU_LATENCY_TAIL50 = "mu_latency_tail50"
MU_LATENCY_TAIL75 = "mu_latency_tail75"

# Primary model comparison: semantic + lexical trajectory only (no timing μ).
LOPEZ_FEATURES_PRIMARY = list(LOPEZ_SEMANTIC_FEATURES)

# Robustness: add latency exponent variants.
LOPEZ_FEATURES_WITH_MEDIAN_NORM_MU = LOPEZ_SEMANTIC_FEATURES + [MU_LATENCY_MEDIAN_NORM]
LOPEZ_FEATURES_RAW_MU = LOPEZ_SEMANTIC_FEATURES + [MU_LATENCY_RAW]

# Aliases
LOPEZ_FEATURES_NO_MU = LOPEZ_FEATURES_PRIMARY
LOPEZ_FEATURES_MAIN = LOPEZ_FEATURES_PRIMARY

# Reported in PPC / figures but not used for ABC rejection or ABC-RF.
LOPEZ_FEATURES_DIAGNOSTIC_TIMING = [MU_LATENCY_MEDIAN_NORM, MU_LATENCY_RAW]

# Legacy alias (deprecated for ABC); maps to raw pooled name after refactor.
MU_LATENCY_LEGACY = "mu_latency_pooled"

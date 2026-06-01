#!/usr/bin/env python3
"""Train ABC-RF on the Lopez-style group-level reference table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Model_Core.lopez_features import (
    LOPEZ_FEATURES_PRIMARY,
    LOPEZ_FEATURES_RAW_MU,
    LOPEZ_FEATURES_WITH_MEDIAN_NORM_MU,
)

DEFAULT_FEATURES = list(LOPEZ_FEATURES_PRIMARY)


def main() -> None:
    parser = argparse.ArgumentParser(description="ABC-RF for Lopez-style reference stats.")
    parser.add_argument(
        "--prefix",
        default="Model_compare/data/animals_oaf_lopez_style_reference_wideprior_200x50_cap150",
        help="Reference prefix written by 01_make_reference.py",
    )
    parser.add_argument("--n-trees", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument(
        "--tag",
        default="primary_4feat",
        help="Suffix for outputs: *_abcrf_{tag}_*.",
    )
    args = parser.parse_args()

    prefix = ROOT / args.prefix
    ref_path = prefix.with_name(prefix.name + "_model_reference.csv")
    human_path = prefix.with_name(prefix.name + "_human_group_stats.csv")
    tag = args.tag if args.tag.startswith("_") or not args.tag else f"_{args.tag}"
    out_prefix = prefix.with_name(prefix.name + f"_abcrf{tag}")

    ref = pd.read_csv(ref_path)
    human = pd.read_csv(human_path).iloc[0].to_dict()

    missing = [f for f in args.features if f not in ref.columns or f not in human]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    clean = ref.dropna(subset=args.features + ["model_id"]).copy()
    if clean["model_id"].nunique() < 2:
        raise RuntimeError("Reference table has fewer than two valid model classes.")

    X = clean[args.features].to_numpy(float)
    y = clean["model_id"].to_numpy(str)
    human_x = pd.DataFrame([{f: human[f] for f in args.features}], columns=args.features)
    if human_x.isna().any(axis=None):
        raise RuntimeError(f"Human feature vector has NaN: {human_x.to_dict(orient='records')[0]}")

    clf = RandomForestClassifier(
        n_estimators=args.n_trees,
        random_state=args.seed,
        oob_score=True,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X, y)

    posterior = pd.DataFrame(
        {
            "model_id": clf.classes_,
            "posterior_vote_probability": clf.predict_proba(human_x.to_numpy(float))[0],
        }
    ).sort_values("posterior_vote_probability", ascending=False)
    selected_model = str(posterior.iloc[0]["model_id"])

    oob_pred = clf.classes_[np.argmax(clf.oob_decision_function_, axis=1)]
    oob_error_indicator = (oob_pred != y).astype(float)
    err_rf = RandomForestRegressor(
        n_estimators=args.n_trees,
        random_state=args.seed + 1,
        oob_score=True,
        n_jobs=-1,
    )
    err_rf.fit(X, oob_error_indicator)
    local_error = float(err_rf.predict(human_x.to_numpy(float))[0])
    local_error = float(np.clip(local_error, 0.0, 1.0))
    selected_post_prob = 1.0 - local_error

    importance = pd.DataFrame(
        {
            "feature": args.features,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    cm = pd.DataFrame(
        confusion_matrix(y, oob_pred, labels=clf.classes_),
        index=clf.classes_,
        columns=clf.classes_,
    )

    posterior_path = out_prefix.with_name(out_prefix.name + "_posterior.csv")
    importance_path = out_prefix.with_name(out_prefix.name + "_feature_importance.csv")
    confusion_path = out_prefix.with_name(out_prefix.name + "_oob_confusion.csv")
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.json")

    posterior.to_csv(posterior_path, index=False)
    importance.to_csv(importance_path, index=False)
    cm.to_csv(confusion_path)
    meta = {
        "features": args.features,
        "human_features": {f: float(human[f]) for f in args.features},
        "n_trees": args.n_trees,
        "seed": args.seed,
        "reference_path": str(ref_path),
        "human_path": str(human_path),
        "n_reference_rows": int(len(ref)),
        "n_valid_rows": int(len(clean)),
        "dropped_rows_due_to_nan": int(len(ref) - len(clean)),
        "models": sorted(clean["model_id"].unique().tolist()),
        "oob_accuracy": float(clf.oob_score_),
        "oob_error_rate": float(1.0 - clf.oob_score_),
        "selected_model": selected_model,
        "selected_model_vote_probability": float(posterior.iloc[0]["posterior_vote_probability"]),
        "selected_model_local_error": local_error,
        "selected_model_abcrf_posterior_probability": selected_post_prob,
        "error_regression_oob_score": float(err_rf.oob_score_),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Posterior vote probabilities")
    print(posterior.to_string(index=False))
    print(
        "\nSelected model ABC-RF posterior probability "
        f"(second-stage error RF): {selected_post_prob:.3f}"
    )
    print(f"Selected model local error: {local_error:.3f}")
    print("\nFeature importance")
    print(importance.to_string(index=False))
    print(f"\nOOB accuracy: {clf.oob_score_:.3f}")
    print(f"OOB error rate: {1.0 - clf.oob_score_:.3f}")
    print(f"Valid rows: {len(clean)}/{len(ref)}")
    print(f"Wrote {posterior_path}")
    print(f"Wrote {importance_path}")
    print(f"Wrote {confusion_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()

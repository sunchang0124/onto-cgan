"""
Tune downstream classifiers on seen-disease training data.
==========================================================

Runs GridSearchCV (inner 5-fold stratified CV) on the full dataset
(seen + unseen diseases) to find the best hyperparameters for each
classifier, optimizing directly for the actual classification task.

Usage
-----
    python evaluation/tune_classifiers.py                        # AML (default)
    python evaluation/tune_classifiers.py --disease psc
    python evaluation/tune_classifiers.py --disease cgan_519_diagnosis

Outputs
-------
    output/<disease>/best_classifier_params.json   — best params (copy into evaluate.py)
    output/<disease>/tuning_cv_results.csv         — full grid results for inspection
"""

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils import ROOT, load_data, prepare_features, CONFIGS

# ── Parameter grids ───────────────────────────────────────────────────────────

PARAM_GRIDS = {
    'RandomForest': {
        'estimator': RandomForestClassifier(
            class_weight='balanced', random_state=42, n_jobs=-1),
        'grid': {
            'n_estimators':    [300, 500],
            'max_depth':       [None, 10, 20],
            'max_features':    ['sqrt', 'log2'],
            'min_samples_leaf':[1, 2, 4],
            'criterion':       ['gini', 'entropy'],
        },
    },
    'KNN': {
        'estimator': KNeighborsClassifier(),
        'grid': {
            'n_neighbors': [3, 5, 7, 11, 21, 51, 101],
            'weights':     ['uniform', 'distance'],
            'metric':      ['euclidean', 'manhattan'],
        },
    },
    'GaussianNB': {
        'estimator': GaussianNB(),
        'grid': {
            'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        },
    },
}

SCORING = 'balanced_accuracy'


def tune(disease: str, n_splits: int, random_state: int) -> None:
    cfg = CONFIGS[disease]
    out_dir = ROOT / "output" / disease
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDisease  : {disease}  (unseen = {cfg.unseen_label})")
    print(f"Scoring  : {SCORING}  |  inner CV = {n_splits}-fold")

    df_full, feat_cols = load_data(cfg)

    print(f"Full data: {len(df_full)} rows  |  "
          f"{df_full[cfg.label_col].nunique()} classes  |  "
          f"{len(feat_cols)} features")
    print(f"Class distribution:\n"
          f"{df_full[cfg.label_col].value_counts().to_string()}\n")

    X, y = prepare_features(df_full, feat_cols, label_col=cfg.label_col)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_params_all = {}
    all_result_dfs  = []

    for clf_name, spec in PARAM_GRIDS.items():
        estimator = spec['estimator']
        grid      = spec['grid']
        n_combos  = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"{'='*70}")
        print(f"{clf_name}: {n_combos} combinations × {n_splits} folds "
              f"= {n_combos * n_splits} fits")

        gs = GridSearchCV(
            estimator   = estimator,
            param_grid  = grid,
            cv          = cv,
            scoring     = SCORING,
            n_jobs      = -1,
            refit       = True,
            return_train_score = False,
            verbose     = 1,
        )
        gs.fit(X, y)

        best_params_all[clf_name] = gs.best_params_
        print(f"\n  Best score : {gs.best_score_:.4f} ({SCORING})")
        print(f"  Best params: {gs.best_params_}")

        result_df = pd.DataFrame(gs.cv_results_)
        result_df.insert(0, 'classifier', clf_name)
        all_result_dfs.append(result_df)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = out_dir / "best_classifier_params.json"
    with open(json_path, 'w') as f:
        json.dump(best_params_all, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Saved best params → {json_path}")

    # ── Save full CV results ──────────────────────────────────────────────────
    csv_path = out_dir / "tuning_cv_results.csv"
    pd.concat(all_result_dfs, ignore_index=True).to_csv(csv_path, index=False)
    print(f"Saved full CV results → {csv_path}")

    # ── Print copy-paste block for evaluate.py ────────────────────────────────
    print(f"\n{'='*70}")
    print("Copy-paste this CLASSIFIERS block into evaluate.py:\n")
    _print_classifiers_block(best_params_all)


def _print_classifiers_block(best: dict) -> None:
    rf  = best.get('RandomForest', {})
    knn = best.get('KNN', {})
    nb  = best.get('GaussianNB', {})

    rf_lines = [f"        {k}={repr(v)}," for k, v in sorted(rf.items())]
    rf_lines += ["        class_weight='balanced', random_state=42, n_jobs=-1,"]

    knn_lines = [f"        {k}={repr(v)}," for k, v in sorted(knn.items())]

    nb_lines = [f"        {k}={repr(v)}," for k, v in sorted(nb.items())]

    print("CLASSIFIERS = {")
    print("    'RandomForest': RandomForestClassifier(")
    for l in rf_lines:
        print(l)
    print("    ),")
    print("    'KNN': KNeighborsClassifier(")
    for l in knn_lines:
        print(l)
    print("    ),")
    print("    'GaussianNB': GaussianNB(")
    for l in nb_lines:
        print(l)
    print("    ),")
    print("}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease',      type=str, default='aml',
                        choices=list(CONFIGS.keys()),
                        help='Which experiment config to tune on (default: aml)')
    parser.add_argument('--n-splits',     type=int, default=5,
                        help='Inner CV folds for GridSearchCV (default: 5)')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    tune(args.disease, args.n_splits, args.random_state)

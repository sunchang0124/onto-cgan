"""
Onto-CGAN ZSL Utility Evaluation
=================================

Loads trained GAN models (pkl), samples synthetic AML data on-the-fly,
and evaluates classification utility and distributional similarity.

Evaluation has two independent parts:
  1. Statistical analysis  — all 199 real AML vs synthetic AML (run once, no splits)
  2. ML classifier utility — single-seed or k-fold CV over seen/unseen splits

Single-seed run (80/20 seen split, all AML in test):
    python evaluation/evaluate.py --model-dir output/synthetic
    python evaluation/evaluate.py --model-dir output/synthetic --include-s3

Cross-validation (StratifiedKFold over full dataset, all 199 AML evaluated):
    python evaluation/evaluate.py --model-dir output/synthetic --cv
    python evaluation/evaluate.py --model-dir output/synthetic --cv --n-splits 5

Outputs (both modes):
    stat_similarity_S2_{mode}.csv    — KS + Wasserstein per feature (all real AML)
    kde_comparison.png               — distribution plots (all real AML)

Outputs (single seed):
    evaluation_results.csv           — ML utility table

Outputs (cross-validation):
    evaluation_results_all_folds.csv — per-fold results (includes 'fold' column)
    evaluation_results_cv.csv        — mean ± std across folds
"""

import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
)

from utils import (
    ROOT, EMB_DIR, UNSEEN_IRI, UNSEEN_LABEL, LABEL_COL,
    load_and_split, load_kfold, load_data, prepare_features, ALL_FEAT_COLS,
)


N_SYN  = 1340   # 5 × total AML count (268)
MODES  = ["ontology", "decoder", "decoder_contrast", "decoder_var"]

CLASSIFIERS = {
    'RandomForest': RandomForestClassifier(
                        n_estimators=500, class_weight='balanced',
                        criterion='gini', random_state=42, n_jobs=-1),
    'KNN':          KNeighborsClassifier(n_neighbors=100),
    'GaussianNB':   GaussianNB(var_smoothing=1e-8),
}


# ══════════════════════════════════════════════════════════════════════════════
# Classifier helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_classifier(clf_template, train_df, feat_cols, extra_df=None):
    import copy
    clf = copy.deepcopy(clf_template)
    combined = pd.concat([train_df, extra_df], ignore_index=True) \
               if extra_df is not None and len(extra_df) > 0 else train_df
    X, y = prepare_features(combined, feat_cols)
    clf.fit(X, y)
    return clf


def evaluate_on_unseen(clf, test_df, feat_cols, unseen_label=UNSEEN_LABEL):
    X_test, y_test = prepare_features(test_df, feat_cols)
    classes = list(clf.classes_)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    y_bin = (y_test == unseen_label).astype(int)
    if unseen_label in classes:
        y_score = y_proba[:, classes.index(unseen_label)]
    else:
        y_score = np.zeros(len(y_test))

    rec  = recall_score(y_test, y_pred, labels=[unseen_label], average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred, labels=[unseen_label], average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, labels=[unseen_label], average='macro', zero_division=0)

    if y_bin.sum() > 0 and y_score.sum() > 0:
        roc_auc  = roc_auc_score(y_bin, y_score)
        avg_prec = average_precision_score(y_bin, y_score)
    else:
        roc_auc = avg_prec = 0.5

    try:
        roc_auc_multi = roc_auc_score(y_test, y_proba, multi_class='ovo',
                                      labels=classes, average='macro')
    except Exception:
        roc_auc_multi = float('nan')

    tp = int((y_pred[y_bin == 1] == unseen_label).sum())
    return {
        'recall':                 round(rec, 4),
        'precision':              round(prec, 4),
        'f1':                     round(f1, 4),
        'roc_auc_aml':            round(roc_auc, 4),
        'pr_auc_aml':             round(avg_prec, 4),
        'roc_auc_multiclass':     round(roc_auc_multi, 4),
        'n_unseen_test':          int(y_bin.sum()),
        'n_correctly_identified': tp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Statistical similarity + KDE plot
# ══════════════════════════════════════════════════════════════════════════════

def compute_stat_similarity(real_df, synth_df, feat_cols):
    rows = []
    for col in [c for c in feat_cols if c in real_df.columns and c in synth_df.columns]:
        r = real_df[col].dropna().values.astype(float)
        s = synth_df[col].dropna().values.astype(float)
        if len(r) == 0 or len(s) == 0:
            continue
        ks_stat, ks_p = ks_2samp(r, s)
        rows.append({
            'Feature':      col,
            'Real mean':    round(r.mean(), 4),
            'Synth mean':   round(s.mean(), 4),
            'Real std':     round(r.std(), 4),
            'Synth std':    round(s.std(), 4),
            'KS statistic': round(ks_stat, 4),
            'KS p-value':   round(ks_p, 4),
            'Wasserstein':  round(wasserstein_distance(r, s), 4),
        })
    return pd.DataFrame(rows).set_index('Feature')


def plot_kde(real_aml, real_other, synth_dict, feat_cols, save_path, modes):
    colors = {
        'Other diseases':   'steelblue',
        'Real AML':         'black',
        'ontology':         '#1a5276',
        'prior':            '#1e8449',
        'decoder':          '#d35400',
        'decoder_contrast': '#c0392b',
        'decoder_var':      '#6c3483',
    }
    linestyles = {
        'ontology':         '--',
        'prior':            '--',
        'decoder':          '--',
        'decoder_contrast': '-.',
        'decoder_var':      (0, (3, 1, 1, 1)),
    }
    n_cols = 2
    n_rows = (len(feat_cols) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()
    for i, feat in enumerate(feat_cols):
        ax = axes[i]
        sns.kdeplot(real_other[feat].dropna(), ax=ax, color=colors['Other diseases'],
                    linewidth=1.5, label='Other diseases', fill=True, alpha=0.15)
        sns.kdeplot(real_aml[feat].dropna(), ax=ax, color=colors['Real AML'],
                    linewidth=2.5, label='Real AML', fill=True, alpha=0.15)
        for mode in modes:
            s = synth_dict.get(mode)
            if s is not None and feat in s.columns:
                sns.kdeplot(s[feat].dropna(), ax=ax,
                            color=colors.get(mode, 'grey'),
                            linewidth=1.5, linestyle=linestyles.get(mode, '--'),
                            label=f'Synth {mode}')
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=10,
               bbox_to_anchor=(0.98, 0.02), framealpha=0.9)
    fig.suptitle('Distribution: Other diseases vs Real AML vs Synthetic AML',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Statistical analysis (run once — independent of train/test splits)
# ══════════════════════════════════════════════════════════════════════════════

def run_stat_analysis(gan_models, base_dir):
    """Compare all real AML vs synthetic AML per mode. Runs once, no splits.

    Uses the full clean cohort (199 AML after multi-disease exclusion) so the
    statistical comparison is not biased by a particular fold or seed.
    Saves stat_similarity_S2_{mode}.csv and kde_comparison.png to base_dir.
    """
    df_full, clf_feat_cols = load_data()
    real_aml   = df_full[df_full[LABEL_COL] == UNSEEN_LABEL][clf_feat_cols]
    real_other = df_full[df_full[LABEL_COL] != UNSEEN_LABEL][clf_feat_cols]
    print(f"\n  Statistical analysis: {len(real_aml)} real AML patients")

    synth_dict     = {}
    available_modes = []
    for mode, gan_model in gan_models.items():
        synth = gan_model.sample(N_SYN, unseen_rds=[UNSEEN_IRI])
        synth = synth.dropna(thresh=16)
        synth_features = synth.drop(columns=['IRI'], errors='ignore')
        synth_unseen = synth_features[synth_features[LABEL_COL] == UNSEEN_LABEL].copy() \
                       if LABEL_COL in synth_features.columns else synth_features.copy()
        synth_feats = synth_unseen[[c for c in clf_feat_cols if c in synth_unseen.columns]]

        sim = compute_stat_similarity(real_aml, synth_feats, clf_feat_cols)
        sim.to_csv(base_dir / f"stat_similarity_S2_{mode}.csv")
        print(f"  Saved → stat_similarity_S2_{mode}.csv")

        synth_dict[mode] = synth_feats
        available_modes.append(mode)

    plot_kde(real_aml, real_other, synth_dict, clf_feat_cols,
             base_dir / "kde_comparison.png", available_modes)


# ══════════════════════════════════════════════════════════════════════════════
# ML classifier evaluation (one split — one fold or one seed)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_split(seen_train, unseen_train, test_df, clf_feat_cols,
                   gan_models, include_s3):
    """Sample from GANs, train classifiers, return list of result dicts."""

    available_modes     = []
    synth_features_dict = {}

    for mode, gan_model in gan_models.items():
        synth = gan_model.sample(N_SYN, unseen_rds=[UNSEEN_IRI])
        synth = synth.dropna(thresh=16)
        synth_features = synth.drop(columns=['IRI'], errors='ignore')
        synth_unseen = synth_features[synth_features[LABEL_COL] == UNSEEN_LABEL].copy() \
                       if LABEL_COL in synth_features.columns else synth_features.copy()
        synth_features_dict[mode] = (synth_features, synth_unseen)
        available_modes.append(mode)

    all_results = []
    for clf_name, clf_template in CLASSIFIERS.items():
        # S1
        clf_s1 = train_classifier(clf_template, seen_train, clf_feat_cols)
        res = evaluate_on_unseen(clf_s1, test_df, clf_feat_cols)
        res.update({'Classifier': clf_name, 'Scenario': 'S1 — Baseline'})
        all_results.append(res)

        # S2
        for mode in available_modes:
            synth_features, _ = synth_features_dict[mode]
            clf_s2 = train_classifier(clf_template, seen_train, clf_feat_cols,
                                      extra_df=synth_features)
            res = evaluate_on_unseen(clf_s2, test_df, clf_feat_cols)
            res.update({'Classifier': clf_name, 'Scenario': f'S2 — {mode}'})
            all_results.append(res)

        # S3
        if include_s3 and len(unseen_train) > 0:
            clf_s3 = train_classifier(clf_template, seen_train, clf_feat_cols,
                                      extra_df=unseen_train)
            res = evaluate_on_unseen(clf_s3, test_df, clf_feat_cols)
            res.update({'Classifier': clf_name, 'Scenario': 'S3 — Upper bound (real)'})
            all_results.append(res)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════

METRIC_COLS = ['Recall (AML)', 'Precision (AML)', 'F1 (AML)',
               'ROC-AUC (AML)', 'PR-AUC (AML)', 'ROC-AUC (multiclass OvO)']


def results_to_df(all_results, fold=None):
    rows = []
    for r in all_results:
        row = {
            'Classifier':               r['Classifier'],
            'Scenario':                 r['Scenario'],
            'Recall (AML)':             r['recall'],
            'Precision (AML)':          r['precision'],
            'F1 (AML)':                 r['f1'],
            'ROC-AUC (AML)':            r['roc_auc_aml'],
            'PR-AUC (AML)':             r['pr_auc_aml'],
            'ROC-AUC (multiclass OvO)': r['roc_auc_multiclass'],
            'TP':                       r['n_correctly_identified'],
            'N unseen test':            r['n_unseen_test'],
        }
        if fold is not None:
            row['fold'] = fold
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_cv(all_fold_dfs):
    """Mean ± std across folds for each classifier × scenario × metric."""
    combined = pd.concat(all_fold_dfs, ignore_index=True)
    agg = combined.groupby(['Classifier', 'Scenario'])[METRIC_COLS].agg(['mean', 'std'])
    agg.columns = [f"{m} {s}" for m, s in agg.columns]
    for m in METRIC_COLS:
        agg[m] = agg.apply(
            lambda row: f"{row[f'{m} mean']:.4f} ± {row[f'{m} std']:.4f}", axis=1)
    return agg[METRIC_COLS]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(model_dir, include_s3=False, cv=False, n_splits=5,
         random_state=42, out_dir=None, stat_only=False, ml_only=False):
    model_dir = Path(model_dir)

    if out_dir:
        base_dir = Path(out_dir)
    elif cv:
        tag = "with_s3" if include_s3 else "no_split"
        base_dir = ROOT / "output" / f"{tag}_cv{n_splits}fold_seed{random_state}"
    else:
        tag = "with_s3" if include_s3 else "no_split"
        base_dir = ROOT / "output" / f"{tag}_seed{random_state}"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model dir  : {model_dir}")
    print(f"Output dir : {base_dir}")
    print(f"CV mode    : {cv}  (n_splits={n_splits} | random_state={random_state})")
    print(f"Include S3 : {include_s3}")

    # ── Load GAN models once ──────────────────────────────────────────────────
    gan_models = {}
    for mode in MODES:
        pkl_path = model_dir / f"gan_model_S2_{mode}.pkl"
        if not pkl_path.exists():
            print(f"\n  WARNING: {pkl_path} not found — skipping '{mode}'")
            continue
        print(f"Loading '{mode}' model...")
        with open(pkl_path, 'rb') as f:
            gan_models[mode] = pickle.load(f)

    # ── Part 1: Statistical analysis (once, all real AML vs synthetic) ──────────
    if not ml_only:
        print(f"\n{'='*80}")
        print("Part 1: Statistical similarity (all real AML vs synthetic AML)")
        print(f"{'='*80}")
        run_stat_analysis(gan_models, base_dir)

    if stat_only:
        return

    # ── Part 2: ML classifier evaluation (single-seed or CV) ─────────────────
    print(f"\n{'='*80}")
    print("Part 2: ML classifier utility evaluation")
    print(f"{'='*80}")
    all_fold_dfs = []

    if cv:
        # ── Cross-validation mode ─────────────────────────────────────────────
        for fold, seen_train, unseen_train, test_df, clf_feat_cols in \
                load_kfold(n_splits=n_splits, include_s3=include_s3,
                           random_state=random_state):

            print(f"\n{'='*60}  Fold {fold+1}/{n_splits}  {'='*60}")
            results = evaluate_split(seen_train, unseen_train, test_df,
                                     clf_feat_cols, gan_models, include_s3)
            fold_df = results_to_df(results, fold=fold + 1)
            all_fold_dfs.append(fold_df)

            # Print fold summary
            for _, row in fold_df[fold_df['Classifier'] == 'RandomForest'].iterrows():
                print(f"  RF  {row['Scenario']:35s}  "
                      f"Recall={row['Recall (AML)']:.4f}  "
                      f"F1={row['F1 (AML)']:.4f}  "
                      f"ROC-AUC={row['ROC-AUC (AML)']:.4f}  "
                      f"TP={row['TP']}/{row['N unseen test']}")

        # Save per-fold + aggregated
        all_folds_df = pd.concat(all_fold_dfs, ignore_index=True)
        all_folds_df.to_csv(base_dir / "evaluation_results_all_folds.csv", index=False)
        print(f"\nSaved per-fold results → {base_dir}/evaluation_results_all_folds.csv")

        cv_table = aggregate_cv(all_fold_dfs)
        cv_table.to_csv(base_dir / "evaluation_results_cv.csv")
        print(f"Saved CV summary       → {base_dir}/evaluation_results_cv.csv")

        print(f"\n{'='*80}")
        print(f"Cross-validation summary ({n_splits}-fold, random_state={random_state})")
        print(f"{'='*80}")
        print(cv_table.to_string())

    else:
        # ── Single-seed mode ──────────────────────────────────────────────────
        seen_train, unseen_train, test_df, clf_feat_cols = load_and_split(
            include_s3=include_s3, random_state=random_state)

        results = evaluate_split(seen_train, unseen_train, test_df,
                                 clf_feat_cols, gan_models, include_s3)

        table = results_to_df(results).set_index(['Classifier', 'Scenario'])
        print(f"\n{'='*80}")
        print(table.to_string())
        table.to_csv(base_dir / "evaluation_results.csv")
        print(f"\nSaved → {base_dir}/evaluation_results.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',    type=str, required=True)
    parser.add_argument('--include-s3',   action='store_true')
    parser.add_argument('--cv',           action='store_true',
                        help='Run stratified k-fold CV over full dataset')
    parser.add_argument('--n-splits',     type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--out-dir',      type=str, default=None)
    parser.add_argument('--stat-only',    action='store_true',
                        help='Run statistical analysis only (skip ML classifiers)')
    parser.add_argument('--ml-only',      action='store_true',
                        help='Run ML classifier evaluation only (skip statistical analysis)')
    args = parser.parse_args()
    main(
        model_dir=args.model_dir,
        include_s3=args.include_s3,
        cv=args.cv,
        n_splits=args.n_splits,
        random_state=args.random_state,
        out_dir=args.out_dir,
        stat_only=args.stat_only,
        ml_only=args.ml_only,
    )

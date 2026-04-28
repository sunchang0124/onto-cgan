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
    ROOT, EMB_DIR,
    load_and_split, load_kfold, load_data, prepare_features,
    CONFIGS, ExperimentConfig,
)


N_SYN_MULTIPLIER = 5   # synthetic rows = N_SYN_MULTIPLIER × real unseen count


CLASSIFIERS = {
    'RandomForest': RandomForestClassifier(
                        n_estimators=500, class_weight='balanced',
                        criterion='gini', random_state=42, n_jobs=-1),
    'KNN':          KNeighborsClassifier(n_neighbors=100),
    'GaussianNB':   GaussianNB(var_smoothing=1e-8),
}


def _build_classifiers_from_params(params: dict) -> dict:
    """Reconstruct CLASSIFIERS from a tune_classifiers.py best-params JSON.

    Grid params come from the JSON; fixed params (class_weight, random_state,
    n_jobs for RF) are re-applied here so the JSON stays minimal.
    Any classifier missing from the JSON falls back to the default in CLASSIFIERS.
    """
    clfs = {}
    if 'RandomForest' in params:
        clfs['RandomForest'] = RandomForestClassifier(
            **params['RandomForest'],
            class_weight='balanced', random_state=42, n_jobs=-1)
    else:
        clfs['RandomForest'] = CLASSIFIERS['RandomForest']

    if 'KNN' in params:
        clfs['KNN'] = KNeighborsClassifier(**params['KNN'])
    else:
        clfs['KNN'] = CLASSIFIERS['KNN']

    if 'GaussianNB' in params:
        clfs['GaussianNB'] = GaussianNB(**params['GaussianNB'])
    else:
        clfs['GaussianNB'] = CLASSIFIERS['GaussianNB']

    return clfs


# ══════════════════════════════════════════════════════════════════════════════
# Post-hoc filters
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# Classifier helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_classifier(clf_template, train_df, feat_cols, label_col, extra_df=None):
    import copy
    combined = pd.concat([train_df, extra_df], ignore_index=True) \
               if extra_df is not None and len(extra_df) > 0 else train_df
    X, y = prepare_features(combined, feat_cols, label_col=label_col)
    clf = copy.deepcopy(clf_template)
    clf.fit(X, y)
    return clf


def evaluate_on_unseen(clf, test_df, feat_cols, label_col, unseen_label):
    X_test, y_test = prepare_features(test_df, feat_cols, label_col=label_col)
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

    y_pred_bin = (y_pred == unseen_label).astype(int)
    tp = int(((y_bin == 1) & (y_pred_bin == 1)).sum())
    fp = int(((y_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_bin == 1) & (y_pred_bin == 0)).sum())
    tn = int(((y_bin == 0) & (y_pred_bin == 0)).sum())

    from sklearn.metrics import confusion_matrix as _cm
    conf_mat = _cm(y_test, y_pred, labels=classes)

    return {
        'recall':                 round(rec, 4),
        'precision':              round(prec, 4),
        'f1':                     round(f1, 4),
        'roc_auc_aml':            round(roc_auc, 4),
        'pr_auc_aml':             round(avg_prec, 4),
        'roc_auc_multiclass':     round(roc_auc_multi, 4),
        'n_unseen_test':          int(y_bin.sum()),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'conf_matrix':            conf_mat,
        'conf_classes':           classes,
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
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index('Feature')


def plot_kde(real_aml, real_other, synth_dict, feat_cols, save_path, modes,
             unseen_label="unseen"):
    colors = {
        'Other diseases':        'steelblue',
        f'Real {unseen_label}':  'black',
        'ontology':              '#1a5276',
        'prior':                 '#1e8449',
        'decoder':               '#d35400',
        'decoder_contrast':      '#c0392b',
        'decoder_var':           '#6c3483',

    }
    linestyles = {
        'ontology':              '--',
        'prior':                 '--',
        'decoder':               '--',
        'decoder_contrast':      '-.',
        'decoder_var':           (0, (3, 1, 1, 1)),

    }
    n_cols = 2
    n_rows = (len(feat_cols) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()
    for i, feat in enumerate(feat_cols):
        ax = axes[i]
        sns.kdeplot(real_other[feat].dropna(), ax=ax, color=colors['Other diseases'],
                    linewidth=1.5, label='Other diseases', fill=True, alpha=0.15)
        sns.kdeplot(real_aml[feat].dropna(), ax=ax, color=colors[f'Real {unseen_label}'],
                    linewidth=2.5, label=f'Real {unseen_label}', fill=True, alpha=0.15)
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
    fig.suptitle(f'Distribution: Other diseases vs Real {unseen_label} vs Synthetic {unseen_label}',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Statistical analysis (run once — independent of train/test splits)
# ══════════════════════════════════════════════════════════════════════════════

def run_stat_analysis(gan_models, base_dir, cfg, n_syn):
    """Compare all real unseen vs synthetic unseen per mode. Runs once, no splits."""
    df_full, clf_feat_cols = load_data(cfg)
    real_unseen = df_full[df_full[cfg.label_col] == cfg.unseen_label][clf_feat_cols]
    real_other  = df_full[df_full[cfg.label_col] != cfg.unseen_label][clf_feat_cols]
    print(f"\n  Statistical analysis: {len(real_unseen)} real {cfg.unseen_label} patients")

    # dropna threshold: keep rows with at least half of feature columns non-null
    dropna_thresh = max(1, len(clf_feat_cols) // 2)

    synth_dict      = {}
    available_modes = []
    for mode, gan_model in gan_models.items():
        synth = gan_model.sample(n_syn, unseen_rds=[cfg.unseen_iri])
        synth = synth.dropna(thresh=dropna_thresh)
        synth_features = synth.drop(columns=['IRI'], errors='ignore')
        synth_unseen = synth_features[
            synth_features[cfg.label_col] == cfg.unseen_label].copy() \
            if cfg.label_col in synth_features.columns else synth_features.copy()
        synth_feats = synth_unseen[
            [c for c in clf_feat_cols if c in synth_unseen.columns]]

        sim = compute_stat_similarity(real_unseen, synth_feats, clf_feat_cols)
        sim.to_csv(base_dir / f"stat_similarity_S2_{mode}.csv")
        print(f"  Saved → stat_similarity_S2_{mode}.csv")

        synth_dict[mode] = synth_feats
        available_modes.append(mode)


    plot_kde(real_unseen, real_other, synth_dict, clf_feat_cols,
             base_dir / "kde_comparison.png", available_modes,
             unseen_label=cfg.unseen_label)


# ══════════════════════════════════════════════════════════════════════════════
# Random noise baseline
# ══════════════════════════════════════════════════════════════════════════════

def generate_random_unseen(seen_train, feat_cols, label_col, unseen_label,
                           n_syn, random_state=42):
    """Generate n_syn rows of per-feature uniform random values labeled as the unseen disease.

    Each feature is sampled uniformly within [min, max] observed in seen_train.
    This serves as a lower bound for GAN utility: if the GAN cannot beat random
    noise, it is adding no useful signal.
    """
    rng = np.random.default_rng(random_state)
    data = {}
    for col in feat_cols:
        col_vals = seen_train[col].dropna() if col in seen_train.columns else pd.Series(dtype=float)
        if len(col_vals) == 0:
            data[col] = np.zeros(n_syn)
        else:
            data[col] = rng.uniform(col_vals.min(), col_vals.max(), size=n_syn)
    df = pd.DataFrame(data)
    df[label_col] = unseen_label
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ML classifier evaluation (one split — one fold or one seed)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_split(seen_train, unseen_train, test_df, clf_feat_cols,
                   gan_models, include_s3, cfg, n_syn, classifiers=None,
                   random_state=42):
    """Sample from GANs, train classifiers, return list of result dicts."""
    if classifiers is None:
        classifiers = CLASSIFIERS

    # dropna threshold: keep rows with at least half of feature columns non-null
    dropna_thresh = max(1, len(clf_feat_cols) // 2)

    available_modes     = []
    synth_features_dict = {}

    for mode, gan_model in gan_models.items():
        import random as _random
        import torch as _torch
        _random.seed(random_state)
        np.random.seed(random_state)
        _torch.manual_seed(random_state)
        synth = gan_model.sample(n_syn, unseen_rds=[cfg.unseen_iri])
        synth = synth.dropna(thresh=dropna_thresh)
        synth_features = synth.drop(columns=['IRI'], errors='ignore')
        synth_unseen = synth_features[
            synth_features[cfg.label_col] == cfg.unseen_label].copy() \
            if cfg.label_col in synth_features.columns else synth_features.copy()
        synth_features_dict[mode] = (synth_features, synth_unseen)
        available_modes.append(mode)


    # Build random baseline once (same n_syn, independent of classifier)
    random_unseen = generate_random_unseen(
        seen_train, clf_feat_cols, cfg.label_col, cfg.unseen_label, n_syn,
        random_state=random_state)

    all_results = []
    for clf_name, clf_template in classifiers.items():
        # S1 — Random noise (uniform per-feature within seen-train range)
        clf_rand = train_classifier(clf_template, seen_train, clf_feat_cols,
                                    label_col=cfg.label_col,
                                    extra_df=random_unseen)
        res = evaluate_on_unseen(clf_rand, test_df, clf_feat_cols,
                                 label_col=cfg.label_col,
                                 unseen_label=cfg.unseen_label)
        res.update({'Classifier': clf_name, 'Scenario': 'S1 — Random noise'})
        all_results.append(res)

        # S2
        for mode in available_modes:
            synth_features, _ = synth_features_dict[mode]
            clf_s2 = train_classifier(clf_template, seen_train, clf_feat_cols,
                                      label_col=cfg.label_col,
                                      extra_df=synth_features)
            res = evaluate_on_unseen(clf_s2, test_df, clf_feat_cols,
                                     label_col=cfg.label_col,
                                     unseen_label=cfg.unseen_label)
            res.update({'Classifier': clf_name, 'Scenario': f'S2 — {mode}'})
            all_results.append(res)

        # S3
        if include_s3 and len(unseen_train) > 0:
            clf_s3 = train_classifier(clf_template, seen_train, clf_feat_cols,
                                      label_col=cfg.label_col,
                                      extra_df=unseen_train)
            res = evaluate_on_unseen(clf_s3, test_df, clf_feat_cols,
                                     label_col=cfg.label_col,
                                     unseen_label=cfg.unseen_label)
            res.update({'Classifier': clf_name, 'Scenario': 'S3 — Upper bound (real)'})
            all_results.append(res)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════

METRIC_COLS = ['Recall', 'Precision', 'F1',
               'ROC-AUC (unseen)', 'PR-AUC (unseen)', 'ROC-AUC (multiclass OvO)']


def results_to_df(all_results, fold=None):
    rows = []
    for r in all_results:
        row = {
            'Classifier':               r['Classifier'],
            'Scenario':                 r['Scenario'],
            'Recall':                   r['recall'],
            'Precision':                r['precision'],
            'F1':                       r['f1'],
            'ROC-AUC (unseen)':         r['roc_auc_aml'],
            'PR-AUC (unseen)':          r['pr_auc_aml'],
            'ROC-AUC (multiclass OvO)': r['roc_auc_multiclass'],
            'TP':                       r['tp'],
            'FP':                       r['fp'],
            'FN':                       r['fn'],
            'TN':                       r['tn'],
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

def save_confusion_matrices(all_results, base_dir):
    """Save one confusion-matrix CSV per classifier × scenario (single-seed only)."""
    for r in all_results:
        clf_name = r['Classifier']
        safe_scenario = (r['Scenario']
                         .replace(' — ', '_').replace('—', '_')
                         .replace(' ', '_').replace('(', '').replace(')', ''))
        classes = r['conf_classes']
        mat     = r['conf_matrix']
        df_cm   = pd.DataFrame(mat, index=classes, columns=classes)
        df_cm.index.name   = 'actual \\ predicted'
        fname = f"confusion_matrix_{clf_name}_{safe_scenario}.csv"
        df_cm.to_csv(base_dir / fname)
        print(f"  Saved → {fname}")


_KNOWN_EMBEDDINGS = {'gensim', 'hit', 'sapbert', 'biolord', 'pubmedbert'}


def _emb_suffix_from_dirname(dir_name: str) -> str:
    """Return '_hit', '_biolord', etc. if the directory name ends with a known
    embedding name, otherwise return ''.

    Handles names like 'synthetic_epoch_2000_hit' → '_hit',
    'synthetic_epoch_2000' → '', 'synthetic_epoch_3000_biolord' → '_biolord'.
    """
    for emb in _KNOWN_EMBEDDINGS:
        if dir_name == emb or dir_name.endswith(f"_{emb}"):
            return f"_{emb}"
    return ""


def _discover_models(model_dirs: list) -> dict:
    """Scan one or more directories for gan_model_S2_*.pkl files.

    Embedding names are extracted from directory names
    (e.g. synthetic_epoch_2000_hit → suffix '_hit') and appended to the
    mode key, so decoder_hit, decoder_biolord, etc. are treated as
    distinct modes rather than overwriting each other.

    When the same mode+embedding key appears in multiple dirs, the one from
    the lexicographically latest directory name wins (higher epoch = later).
    Returns {mode: Path} ordered by mode name.
    """
    found: dict = {}
    for d in sorted(model_dirs):  # sort so later epochs overwrite earlier ones
        d = Path(d)
        if not d.is_dir():
            continue
        emb_sfx = _emb_suffix_from_dirname(d.name)
        for pkl in sorted(d.glob("gan_model_S2_*.pkl")):
            base_mode = pkl.stem.replace("gan_model_S2_", "")
            mode = base_mode + emb_sfx
            found[mode] = pkl
    return dict(sorted(found.items()))


def _load_gan_models(mode_paths: dict, cfg=None) -> dict:
    """Load each pkl into memory, remapping tensors to CPU.

    For 'prior' mode models saved without a clinical prior, inject cfg.clinical_prior
    post-load so the unseen disease gets a meaningful conditioning signal.
    """
    import io, torch

    class _CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            return super().find_class(module, name)

    gan_models = {}
    for mode, pkl_path in mode_paths.items():
        print(f"  Loading '{mode}' from {pkl_path.parent.name}/...")
        with open(pkl_path, 'rb') as f:
            model = _CPUUnpickler(f).load()
        inner = getattr(model, '_model', model)
        if hasattr(inner, 'set_device'):
            inner.set_device(torch.device('cpu'))

        if mode == "prior" and cfg is not None and cfg.clinical_prior:
            _inject_clinical_prior(model, cfg.clinical_prior)

        gan_models[mode] = model
    return gan_models


def _inject_clinical_prior(model, clinical_prior: dict) -> None:
    """Walk model hierarchy to find the synthesizer and inject the clinical prior."""
    obj = model
    for _ in range(5):  # at most 5 levels deep
        synthesizer = getattr(obj, '_model', None)
        if synthesizer is None:
            break
        if hasattr(synthesizer, '_clinical_prior_normalised') and \
                hasattr(synthesizer, '_cont_cols') and \
                hasattr(synthesizer, '_clinical_mean_center'):
            if not getattr(synthesizer, '_clinical_prior_raw', None):
                prior_vec = np.array(
                    [clinical_prior.get(c, 0.0) for c in synthesizer._cont_cols],
                    dtype='float32')
                synthesizer._clinical_prior_normalised = (
                    (prior_vec - synthesizer._clinical_mean_center)
                    / synthesizer._clinical_mean_scale
                )
                synthesizer._clinical_prior_raw = clinical_prior
                print(f"    → Injected clinical prior ({len(clinical_prior)} features).")
            return
        obj = synthesizer


def main(model_dirs, include_s3=False, cv=False, n_splits=5,
         random_state=42, out_dir=None, stat_only=False, ml_only=False,
         disease="aml", load_params=None):
    cfg = CONFIGS[disease]
    if isinstance(model_dirs, (str, Path)):
        model_dirs = [model_dirs]

    if out_dir:
        base_dir = Path(out_dir)
    elif cv:
        tag = "with_s3" if include_s3 else "no_split"
        base_dir = ROOT / "output" / disease / f"{tag}_cv{n_splits}fold_seed{random_state}"
    else:
        tag = "with_s3" if include_s3 else "no_split"
        base_dir = ROOT / "output" / disease / f"{tag}_seed{random_state}"
    base_dir.mkdir(parents=True, exist_ok=True)

    df_full, _ = load_data(cfg)
    n_unseen = (df_full[cfg.label_col] == cfg.unseen_label).sum()
    n_syn    = N_SYN_MULTIPLIER * n_unseen

    # ── Dataset statistics ────────────────────────────────────────────────────
    disease_counts = (df_full[cfg.label_col].value_counts()
                      .rename_axis('label').reset_index(name='n_patients'))
    disease_counts['pct_of_total'] = (
        disease_counts['n_patients'] / len(df_full) * 100).round(2)
    stats_path = base_dir / "data_stats.csv"
    disease_counts.to_csv(stats_path, index=False)

    print(f"Disease    : {disease}  ({cfg.unseen_label})")
    print(f"Unseen N   : {n_unseen}  →  n_syn = {N_SYN_MULTIPLIER}× = {n_syn}")
    print(f"Model dirs : {[str(d) for d in model_dirs]}")
    print(f"Output dir : {base_dir}")
    print(f"CV mode    : {cv}  (n_splits={n_splits} | random_state={random_state})")
    print(f"Include S3 : {include_s3}")

    # ── Resolve classifier hyperparameters ────────────────────────────────────
    if load_params is not None:
        import json
        params_path = Path(load_params) if load_params is not True \
                      else ROOT / "output" / disease / "best_classifier_params.json"
        if not params_path.exists():
            raise FileNotFoundError(
                f"--load-params: {params_path} not found. "
                f"Run tune_classifiers.py --disease {disease} first.")
        with open(params_path) as f:
            best_params = json.load(f)
        classifiers = _build_classifiers_from_params(best_params)
        print(f"Classifiers: loaded from {params_path}")
        for name, clf in classifiers.items():
            print(f"  {name}: {clf.get_params()}")
    else:
        classifiers = CLASSIFIERS
        print(f"Classifiers: using defaults (no --load-params)")

    # ── Discover and load GAN models ─────────────────────────────────────────
    mode_paths = _discover_models(model_dirs)
    if not mode_paths:
        print(f"ERROR: no gan_model_S2_*.pkl files found in: {model_dirs}")
        return
    print(f"\nFound {len(mode_paths)} mode(s): {list(mode_paths.keys())}")
    gan_models = _load_gan_models(mode_paths, cfg=cfg)

    # ── Part 1: Statistical analysis ─────────────────────────────────────────
    if not ml_only:
        print(f"\n{'='*80}")
        print(f"Part 1: Statistical similarity (all real {cfg.unseen_label} vs synthetic)")
        print(f"{'='*80}")
        run_stat_analysis(gan_models, base_dir, cfg, n_syn)

    if stat_only:
        return

    # ── Part 2: ML classifier evaluation (single-seed or CV) ─────────────────
    print(f"\n{'='*80}")
    print("Part 2: ML classifier utility evaluation")
    print(f"{'='*80}")
    all_fold_dfs = []

    if cv:
        for fold, seen_train, unseen_train, test_df, clf_feat_cols in \
                load_kfold(cfg, n_splits=n_splits, include_s3=include_s3,
                           random_state=random_state):

            print(f"\n{'='*60}  Fold {fold+1}/{n_splits}  {'='*60}")
            results = evaluate_split(seen_train, unseen_train, test_df,
                                     clf_feat_cols, gan_models, include_s3,
                                     cfg=cfg, n_syn=n_syn,
                                     classifiers=classifiers,
                                     random_state=random_state)
            fold_df = results_to_df(results, fold=fold + 1)
            all_fold_dfs.append(fold_df)

            for _, row in fold_df[fold_df['Classifier'] == 'RandomForest'].iterrows():
                print(f"  RF  {row['Scenario']:35s}  "
                      f"Recall={row['Recall']:.4f}  "
                      f"F1={row['F1']:.4f}  "
                      f"ROC-AUC={row['ROC-AUC (unseen)']:.4f}  "
                      f"TP={row['TP']}/{row['N unseen test']}")

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
        seen_train, unseen_train, test_df, clf_feat_cols = load_and_split(
            cfg, include_s3=include_s3, random_state=random_state)

        results = evaluate_split(seen_train, unseen_train, test_df,
                                 clf_feat_cols, gan_models, include_s3,
                                 cfg=cfg, n_syn=n_syn,
                                 classifiers=classifiers,
                                 random_state=random_state)

        table = results_to_df(results).set_index(['Classifier', 'Scenario'])
        print(f"\n{'='*80}")
        print(table.to_string())
        table.to_csv(base_dir / "evaluation_results.csv")
        print(f"\nSaved → {base_dir}/evaluation_results.csv")

        print("\nSaving confusion matrices...")
        save_confusion_matrices(results, base_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',    type=str, nargs='+', required=True,
                        help='One or more directories containing gan_model_S2_*.pkl files. '
                             'Models from later (lexicographically) dirs take precedence.')
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
    parser.add_argument('--disease',      type=str, default='aml',
                        choices=list(CONFIGS.keys()),
                        help='Experiment to evaluate (default: aml)')
    parser.add_argument('--load-params',  nargs='?', const=True, default=None,
                        metavar='PATH',
                        help='Load tuned classifier params from JSON produced by '
                             'tune_classifiers.py. Omit PATH to use the default '
                             'output/<disease>/best_classifier_params.json, or '
                             'supply an explicit path.')
    args = parser.parse_args()
    main(
        model_dirs=args.model_dir,
        include_s3=args.include_s3,
        cv=args.cv,
        n_splits=args.n_splits,
        random_state=args.random_state,
        out_dir=args.out_dir,
        stat_only=args.stat_only,
        ml_only=args.ml_only,
        disease=args.disease,
        load_params=args.load_params,
    )

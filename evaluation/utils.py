"""
Shared constants and data utilities for generate.py and evaluate.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR  = DATA_DIR / "ontology_emb"

# ── Disease mapping (7 diseases — matches Evaluation.ipynb) ───────────────────
ICD_TO_IRI = {
    2050.0: "http://www.orpha.net/ORDO/Orphanet_519",    # AML (unseen)
    2040.0: "http://www.orpha.net/ORDO/Orphanet_513",    # ALL
    2041.0: "http://www.orpha.net/ORDO/Orphanet_67038",  # B-cell lymphoma
    2387.0: "http://www.orpha.net/ORDO/Orphanet_52688",  # MDS
    2007.0: "http://www.orpha.net/ORDO/Orphanet_544",    # DLBCL
    2005.0: "http://www.orpha.net/ORDO/Orphanet_46135",  # CNS lymphoma
    2028.0: "http://www.orpha.net/ORDO/Orphanet_207046", # Malignant lymphoma with neuropathy
}
ICD_TO_LABEL = {k: f"ORDO.{v.split('/')[-1]}" for k, v in ICD_TO_IRI.items()}
UNSEEN_IRI   = "http://www.orpha.net/ORDO/Orphanet_519"
UNSEEN_LABEL = "ORDO.Orphanet_519"
LABEL_COL    = 'icd_code'

# 14 features — matches Evaluation.ipynb selected_col (excluding icd_code/gender)
ALL_FEAT_COLS = [
    'Neutrophils', 'White Blood Cells', 'Monocytes', 'Platelet Count',
    'Hematocrit', 'anchor_age', 'Red Blood Cells', 'PT', 'INR(PT)',
    'MCHC', 'Eosinophils', 'Hemoglobin', 'Basophils', 'RDW',
]


def load_data():
    """
    Load and clean MIMICDATA_MoreDisease_all_test.csv.
    Returns the full cleaned DataFrame and clf_feat_cols.
    """
    df = pd.read_csv(DATA_DIR / "MIMICDATA_MoreDisease_all_test.csv",
                     index_col="Unnamed: 0")
    df['icd_code'] = df['icd_code'].apply(
        lambda x: np.floor(x / 10) if x > 9999 else np.floor(x))
    n_raw = len(df)
    df = df.drop_duplicates()
    # One record per patient per disease — removes multi-visit duplicates explicitly.
    n_after_exact = len(df)
    df = df.drop_duplicates(subset=['subject_id', 'icd_code'], keep='first') \
        if 'subject_id' in df.columns else df
    n_after_patient = len(df)
    print(f"  Deduplication: {n_raw} raw → {n_after_exact} (exact) → {n_after_patient} (per patient×disease)")
    df.replace({'gender': 0}, 'Male',   inplace=True)
    df.replace({'gender': 1}, 'Female', inplace=True)
    df = df[df["icd_code"].isin(ICD_TO_IRI.keys())]

    clf_feat_cols = [c for c in ALL_FEAT_COLS if c in df.columns]
    keep_cols = [c for c in ['subject_id', 'icd_code', 'gender'] + clf_feat_cols if c in df.columns]
    df = df[keep_cols].copy()
    df.insert(0, 'IRI', df['icd_code'].map(ICD_TO_IRI))
    df.replace({'icd_code': ICD_TO_LABEL}, inplace=True)

    # ── Multi-disease patient exclusion ──────────────────────────────────────
    # Patients with records under BOTH AML and a seen disease are ambiguous
    # (their seen-disease rows carry AML signal; their AML rows have a mixed
    # clinical profile). Remove ALL their rows here so the exclusion applies
    # equally to GAN training, classifier training, and evaluation.
    if 'subject_id' in df.columns:
        aml_pids  = set(df.loc[df[LABEL_COL] == UNSEEN_LABEL, 'subject_id'].dropna())
        seen_pids = set(df.loc[df[LABEL_COL] != UNSEEN_LABEL, 'subject_id'].dropna())
        multi_pids = aml_pids & seen_pids
        n_before = len(df)
        df = df[~df['subject_id'].isin(multi_pids)].copy()
        removed = n_before - len(df)
        print(f"  Multi-disease exclusion: {len(multi_pids)} patients appear under both "
              f"AML and a seen disease — removed all {removed} of their rows")
        df = df.drop(columns='subject_id')

    df = df.reset_index(drop=True)
    return df, clf_feat_cols


def load_and_split(include_s3=False, random_state=42):
    """
    Split data into seen_train, unseen_train (S3 only), and test_df.

    Returns
    -------
    seen_train, unseen_train, test_df : pd.DataFrame
    clf_feat_cols : list[str]
    """
    df, clf_feat_cols = load_data()

    # Multi-disease patients are already removed in load_data().
    # subject_id is dropped there; seen_df and unseen_df are already clean.
    seen_df   = df[df[LABEL_COL] != UNSEEN_LABEL].copy()
    unseen_df = df[df[LABEL_COL] == UNSEEN_LABEL].copy()

    rng = np.random.default_rng(random_state)

    if include_s3:
        n_unseen = len(unseen_df)
        n_unseen_train = int(np.floor(n_unseen * 0.7))
        idx = rng.permutation(n_unseen)
        unseen_train = unseen_df.iloc[idx[:n_unseen_train]].reset_index(drop=True)
        unseen_test  = unseen_df.iloc[idx[n_unseen_train:]].reset_index(drop=True)
    else:
        unseen_train = unseen_df.iloc[0:0].copy()
        unseen_test  = unseen_df.reset_index(drop=True)

    seen_idx    = rng.permutation(len(seen_df))
    n_seen_test = max(1, int(np.floor(len(seen_df) * 0.2)))
    seen_train  = seen_df.iloc[seen_idx[n_seen_test:]].reset_index(drop=True)
    seen_test   = seen_df.iloc[seen_idx[:n_seen_test]].reset_index(drop=True)

    test_df = pd.concat([seen_test, unseen_test], ignore_index=True)

    print(f"\nData split summary (random_state={random_state}):")
    print(f"  Seen train   : {len(seen_train)} rows, {seen_train[LABEL_COL].nunique()} diseases")
    print(f"  Unseen train : {len(unseen_train)} rows ({UNSEEN_LABEL}) — used only in S3")
    print(f"  Test set     : {len(test_df)} rows "
          f"({len(unseen_test)} unseen + {len(seen_test)} seen)")

    return seen_train, unseen_train, test_df, clf_feat_cols


def load_kfold(n_splits=5, include_s3=False, random_state=42):
    """Stratified k-fold CV splits over the full dataset (seen + AML).

    AML patients are never in seen_train (ZSL constraint). In each fold:
      - seen_train  : seen-disease rows in train index (AML patients excluded)
      - unseen_train: AML rows in train index (used only for S3 upper bound)
      - test_df     : all rows in test index (seen + AML)

    Across all folds every AML patient appears in exactly one test set,
    so the full clean AML cohort (199 after multi-disease exclusion) is evaluated.

    Yields (fold_idx, seen_train, unseen_train, test_df, clf_feat_cols).
    """
    from sklearn.model_selection import StratifiedKFold

    df, clf_feat_cols = load_data()

    # Multi-disease patients are already removed in load_data().
    seen_df   = df[df[LABEL_COL] != UNSEEN_LABEL].reset_index(drop=True)
    unseen_df = df[df[LABEL_COL] == UNSEEN_LABEL].reset_index(drop=True)

    # Stratified k-fold on the full dataset so AML is proportionally distributed
    full_df = pd.concat([seen_df, unseen_df], ignore_index=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(skf.split(full_df, full_df[LABEL_COL])):
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        test_df  = full_df.iloc[test_idx].reset_index(drop=True)

        # AML rows in train index → unseen_train (S3) or dropped (S1/S2)
        unseen_train = train_df[train_df[LABEL_COL] == UNSEEN_LABEL].reset_index(drop=True) \
                       if include_s3 else train_df.iloc[0:0].copy()
        seen_train   = train_df[train_df[LABEL_COL] != UNSEEN_LABEL].reset_index(drop=True)

        n_aml_test  = (test_df[LABEL_COL] == UNSEEN_LABEL).sum()
        n_seen_test = (test_df[LABEL_COL] != UNSEEN_LABEL).sum()
        print(f"\n  Fold {fold+1}/{n_splits}: "
              f"seen_train={len(seen_train)}  "
              f"unseen_train={len(unseen_train)}  "
              f"test={len(test_df)} ({n_aml_test} AML + {n_seen_test} seen)")

        yield fold, seen_train, unseen_train, test_df, clf_feat_cols


def prepare_features(df, feat_cols):
    """Return float feature matrix X and label array y, with MICE imputation."""
    from impyute.imputation.cs import mice
    cols = [c for c in feat_cols if c in df.columns]
    X = df[cols].copy()
    X['gender'] = (df['gender'] == 'Female').astype(float)
    num_cols = [c for c in X.columns if c != 'gender']
    try:
        imputed = mice(X[num_cols].values)
        X[num_cols] = imputed
    except Exception:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    return X.astype(float), df[LABEL_COL].values

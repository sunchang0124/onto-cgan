"""
Shared constants and data utilities for generate.py and evaluate.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR  = DATA_DIR / "ontology_emb"

# ── Metadata columns in cgan-format CSVs (never treated as features) ──────────
_CGAN_META = {"IRI", "label", "gender", "age"}


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """All disease-specific settings for one onto_cGAN experiment."""
    name: str
    # Path to input CSV
    data_path: Path
    # IRI of the withheld (ZSL) disease
    unseen_iri: str
    # Column that holds the disease label in the CSV
    label_col: str
    # Feature columns for GAN training + classifier.
    # None → auto-detect: every column that is not a meta column.
    feat_cols: Optional[list] = None
    # ICD code → IRI mapping for legacy MIMIC CSVs (AML only).
    # If None the CSV is assumed to already contain IRI + label columns.
    icd_to_iri: Optional[dict] = None
    # Clinical prior dict for 'prior' conditioning mode (optional)
    clinical_prior: Optional[dict] = None

    @property
    def unseen_label(self) -> str:
        """Derive label string from IRI  e.g. Orphanet_171 → ORDO.Orphanet_171."""
        return "ORDO." + self.unseen_iri.split("/")[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Generic config builder — no pre-registration needed
# ─────────────────────────────────────────────────────────────────────────────

def build_config(data_path: str | Path,
                 unseen_iri: str,
                 label_col: str = "label",
                 name: str = None) -> ExperimentConfig:
    """Build an ExperimentConfig from a CSV path and unseen IRI.

    Works with any cgan-format CSV (IRI | label | [gender] | [age] | features).
    Feature columns are auto-detected (everything that is not a meta column).

    unseen_iri can be:
      - a full IRI:   "http://www.orpha.net/ORDO/Orphanet_519"
      - Orphanet_XXX: "Orphanet_519"
      - a plain ID:   "519"

    Examples
    --------
    cfg = build_config(
        "/path/to/cgan_99819_diagnosis.csv",
        unseen_iri="519",
    )
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    iri = _expand_iri(unseen_iri)

    return ExperimentConfig(
        name      = name or p.stem,
        data_path = p,
        unseen_iri = iri,
        label_col  = label_col,
        feat_cols  = None,  # auto-detected by _load_cgan
    )


def _expand_iri(raw: str) -> str:
    """Accept a full IRI, Orphanet_XXX, or a plain numeric ID."""
    raw = raw.strip()
    if raw.startswith("http"):
        return raw
    if raw.startswith("Orphanet_"):
        return f"http://www.orpha.net/ORDO/{raw}"
    try:
        int(raw)
        return f"http://www.orpha.net/ORDO/Orphanet_{raw}"
    except ValueError:
        raise ValueError(
            f"Cannot parse unseen IRI: {raw!r}. "
            "Provide a full IRI, 'Orphanet_XXX', or a plain numeric ID."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Named disease configurations (pre-registered experiments)
# ─────────────────────────────────────────────────────────────────────────────

_AML_FEAT_COLS = [
    'Neutrophils', 'White Blood Cells', 'Monocytes', 'Platelet Count',
    'Hematocrit', 'anchor_age', 'Red Blood Cells', 'PT', 'INR(PT)',
    'MCHC', 'Eosinophils', 'Hemoglobin', 'Basophils', 'RDW',
]

_AML_ICD_TO_IRI = {
    2050.0: "http://www.orpha.net/ORDO/Orphanet_519",    # AML  (unseen)
    2040.0: "http://www.orpha.net/ORDO/Orphanet_513",    # ALL
    2041.0: "http://www.orpha.net/ORDO/Orphanet_67038",  # B-cell lymphoma
    2387.0: "http://www.orpha.net/ORDO/Orphanet_52688",  # MDS
    2007.0: "http://www.orpha.net/ORDO/Orphanet_544",    # DLBCL
    2005.0: "http://www.orpha.net/ORDO/Orphanet_46135",  # CNS lymphoma
    2028.0: "http://www.orpha.net/ORDO/Orphanet_207046", # Malignant lymphoma with neuropathy
}

_QUERY_DATA_DIR = ROOT.parent / "data_query" / "Data"

# ── Clinical priors: observed feature means from real target patients ─────────
_PRIOR_519 = {
    'Alanine Aminotransferase (ALT)': 43.2173,
    'Albumin': 3.5667,
    'Alkaline Phosphatase': 96.4603,
    'Asparate Aminotransferase (AST)': 49.0889,
    'Basophils': 0.2959,
    'Eosinophils': 0.9762,
    'Glucose': 127.3754,
    'Hematocrit': 26.0305,
    'Hemoglobin': 8.6883,
    'Lactate Dehydrogenase (LD)': 638.4277,
    'Lymphocytes': 31.6965,
    'MCH': 31.8625,
    'MCV': 95.4206,
    'Monocytes': 8.6587,
    'Neutrophils': 28.186,
    'PT': 15.0424,
    'Phosphate': 3.4308,
    'Platelet Count': 96.2414,
    'RDW': 17.4925,
    'Red Blood Cells': 2.7599,
    'White Blood Cells': 25.7477,
}

_PRIOR_54057 = {
    'Bicarbonate': 22.0682,
    'Calcium, Total': 8.5977,
    'Creatinine': 3.3068,
    'Glucose': 140.6705,
    'Hematocrit': 28.5307,
    'Hemoglobin': 9.3205,
    'L': 15.4217,
    'MCH': 30.0159,
    'MCHC': 32.6557,
    'MCV': 91.9886,
    'Magnesium': 2.2159,
    'PT': 13.9512,
    'PTT': 31.6881,
    'Phosphate': 4.5739,
    'Platelet Count': 119.5341,
    'Potassium': 4.1136,
    'RDW': 16.6205,
    'RDW-SD': 54.533,
    'Red Blood Cells': 3.1275,
    'Urea Nitrogen': 45.5682,
    'White Blood Cells': 9.8307,
}

_PRIOR_85443 = {
    'Bicarbonate': 24.5924,
    'Calcium, Total': 8.9898,
    'Chloride': 102.3579,
    'Creatinine': 1.3486,
    'Glucose': 129.3549,
    'Hematocrit': 36.6158,
    'Hemoglobin': 12.037,
    'MCH': 30.0929,
    'MCHC': 32.8448,
    'MCV': 91.668,
    'PT': 14.785,
    'PTT': 33.0117,
    'Phosphate': 3.4966,
    'Platelet Count': 213.999,
    'Potassium': 4.1477,
    'RDW': 14.5587,
    'Red Blood Cells': 4.0159,
    'Sodium': 139.0209,
    'Urea Nitrogen': 24.6769,
    'White Blood Cells': 9.0894,
}

_PRIOR_99819 = {
    'Creatinine': 0.752,
    'H': 17.66,
    'Hematocrit': 33.9549,
    'Hemoglobin': 11.1216,
    'I': 0.79,
    'L': 15.8211,
    'MCH': 28.5772,
    'MCHC': 32.7415,
    'MCV': 87.2581,
    'Platelet Count': 231.8814,
    'RDW': 14.369,
    'RDW-SD': 45.1441,
    'Red Blood Cells': 3.9104,
    'Urea Nitrogen': 12.3636,
    'White Blood Cells': 10.7197,
}

CONFIGS: dict[str, ExperimentConfig] = {
    "aml": ExperimentConfig(
        name           = "aml",
        data_path      = DATA_DIR / "MIMICDATA_MoreDisease_all_test.csv",
        unseen_iri     = "http://www.orpha.net/ORDO/Orphanet_519",
        label_col      = "icd_code",
        feat_cols      = _AML_FEAT_COLS,
        icd_to_iri     = _AML_ICD_TO_IRI,
    ),
    "psc": ExperimentConfig(
        name       = "psc",
        data_path  = ROOT.parent / "data_query" / "Data" / "cgan_psc_union.csv",
        unseen_iri = "http://www.orpha.net/ORDO/Orphanet_171",
        label_col  = "label",
        feat_cols  = None,
    ),
    "cgan_519_diagnosis": ExperimentConfig(
        name           = "cgan_519_diagnosis",
        data_path      = _QUERY_DATA_DIR / "cgan_519_diagnosis.csv",
        unseen_iri     = "http://www.orpha.net/ORDO/Orphanet_519",
        label_col      = "label",
        feat_cols      = None,
        clinical_prior = _PRIOR_519,
    ),
    "cgan_54057_diagnosis": ExperimentConfig(
        name           = "cgan_54057_diagnosis",
        data_path      = _QUERY_DATA_DIR / "cgan_54057_diagnosis.csv",
        unseen_iri     = "http://www.orpha.net/ORDO/Orphanet_54057",
        label_col      = "label",
        feat_cols      = None,
        clinical_prior = _PRIOR_54057,
    ),
    "cgan_85443_diagnosis": ExperimentConfig(
        name           = "cgan_85443_diagnosis",
        data_path      = _QUERY_DATA_DIR / "cgan_85443_diagnosis.csv",
        unseen_iri     = "http://www.orpha.net/ORDO/Orphanet_85443",
        label_col      = "label",
        feat_cols      = None,
        clinical_prior = _PRIOR_85443,
    ),
    "cgan_99819_diagnosis": ExperimentConfig(
        name           = "cgan_99819_diagnosis",
        data_path      = _QUERY_DATA_DIR / "cgan_99819_diagnosis.csv",
        unseen_iri     = "http://www.orpha.net/ORDO/Orphanet_99819",
        label_col      = "label",
        feat_cols      = None,
        clinical_prior = _PRIOR_99819,
    ),
}

# ── Backward-compatible aliases (keeps old imports working without changes) ────
_DEFAULT_CFG  = CONFIGS["aml"]
UNSEEN_IRI    = _DEFAULT_CFG.unseen_iri
UNSEEN_LABEL  = _DEFAULT_CFG.unseen_label
LABEL_COL     = _DEFAULT_CFG.label_col
ALL_FEAT_COLS = _AML_FEAT_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg: ExperimentConfig = _DEFAULT_CFG):
    """Load and clean the experiment dataset.

    Legacy (AML) format: normalises ICD codes, maps to IRI/label,
    excludes multi-disease patients.
    cgan format (PSC etc.): reads directly — IRI + label already present.

    Returns (df, feat_cols).
    """
    if cfg.icd_to_iri is not None:
        return _load_legacy(cfg)
    return _load_cgan(cfg)


def _load_legacy(cfg: ExperimentConfig):
    """Load AML-style MIMIC CSV with raw ICD codes."""
    icd_to_iri   = cfg.icd_to_iri
    icd_to_label = {k: "ORDO." + v.split("/")[-1] for k, v in icd_to_iri.items()}

    df = pd.read_csv(cfg.data_path, index_col="Unnamed: 0")
    df[cfg.label_col] = df[cfg.label_col].apply(
        lambda x: np.floor(x / 10) if x > 9999 else np.floor(x))
    n_raw = len(df)
    df = df.drop_duplicates()
    n_after_exact = len(df)
    if 'subject_id' in df.columns:
        df = df.drop_duplicates(subset=['subject_id', cfg.label_col], keep='first')
    n_after_patient = len(df)
    print(f"  Deduplication: {n_raw} raw → {n_after_exact} (exact) "
          f"→ {n_after_patient} (per patient×disease)")
    df.replace({'gender': 0}, 'Male',   inplace=True)
    df.replace({'gender': 1}, 'Female', inplace=True)
    df = df[df[cfg.label_col].isin(icd_to_iri.keys())]

    feat_cols = [c for c in (cfg.feat_cols or []) if c in df.columns]
    keep_cols = [c for c in ['subject_id', cfg.label_col, 'gender'] + feat_cols
                 if c in df.columns]
    df = df[keep_cols].copy()
    df.insert(0, 'IRI', df[cfg.label_col].map(icd_to_iri))
    df[cfg.label_col] = df[cfg.label_col].map(icd_to_label)

    # Multi-disease patient exclusion
    if 'subject_id' in df.columns:
        unseen_pids = set(df.loc[df[cfg.label_col] == cfg.unseen_label,
                                 'subject_id'].dropna())
        seen_pids   = set(df.loc[df[cfg.label_col] != cfg.unseen_label,
                                 'subject_id'].dropna())
        multi_pids  = unseen_pids & seen_pids
        n_before = len(df)
        df = df[~df['subject_id'].isin(multi_pids)].copy()
        removed = n_before - len(df)
        print(f"  Multi-disease exclusion: {len(multi_pids)} patients appear under both "
              f"{cfg.unseen_label} and a seen disease — removed {removed} rows")
        df = df.drop(columns='subject_id')

    df = df.reset_index(drop=True)
    return df, feat_cols


def _load_cgan(cfg: ExperimentConfig):
    """Load cgan-format CSV — IRI and label columns are already present."""
    df = pd.read_csv(cfg.data_path)
    n_raw = len(df)

    if cfg.feat_cols is not None:
        feat_cols = [c for c in cfg.feat_cols if c in df.columns]
    else:
        feat_cols = [c for c in df.columns if c not in _CGAN_META]

    keep_cols = [c for c in ['IRI', cfg.label_col, 'gender'] + feat_cols
                 if c in df.columns]
    df = df[keep_cols].copy()

    # Exclude patients whose feature values appear under more than one disease
    # label — identical lab profiles with different labels are ambiguous and
    # would leak information across seen/unseen splits.
    feature_key_cols = [c for c in feat_cols if c in df.columns]
    if feature_key_cols:
        df['_feat_hash'] = pd.util.hash_pandas_object(df[feature_key_cols], index=False)
        n_labels_per_hash = df.groupby('_feat_hash')[cfg.label_col].nunique()
        ambiguous_hashes = n_labels_per_hash[n_labels_per_hash > 1].index
        before = len(df)
        df = df[~df['_feat_hash'].isin(ambiguous_hashes)].drop(columns='_feat_hash').copy()
        removed = before - len(df)
        if removed > 0:
            print(f"  Ambiguous exclusion: removed {removed} rows where identical "
                  f"feature values appear under multiple disease labels")

    print(f"  Loaded : {n_raw} → {len(df)} rows | "
          f"{df[cfg.label_col].nunique()} diseases | "
          f"{len(feat_cols)} features")
    print(f"  Labels : {sorted(df[cfg.label_col].unique().tolist())}")

    df = df.reset_index(drop=True)
    return df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Splits
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(cfg: ExperimentConfig = _DEFAULT_CFG,
                   include_s3=False, random_state=42):
    """80/20 seen split; all unseen in test (S1/S2) or 70% in train (S3)."""
    df, feat_cols = load_data(cfg)

    seen_df   = df[df[cfg.label_col] != cfg.unseen_label].copy()
    unseen_df = df[df[cfg.label_col] == cfg.unseen_label].copy()
    rng = np.random.default_rng(random_state)

    if include_s3:
        n_unseen_train = int(np.floor(len(unseen_df) * 0.7))
        idx = rng.permutation(len(unseen_df))
        unseen_train = unseen_df.iloc[idx[:n_unseen_train]].reset_index(drop=True)
        unseen_test  = unseen_df.iloc[idx[n_unseen_train:]].reset_index(drop=True)
    else:
        unseen_train = unseen_df.iloc[0:0].copy()
        unseen_test  = unseen_df.reset_index(drop=True)

    seen_idx    = rng.permutation(len(seen_df))
    n_seen_test = max(1, int(np.floor(len(seen_df) * 0.2)))
    seen_train  = seen_df.iloc[seen_idx[n_seen_test:]].reset_index(drop=True)
    seen_test   = seen_df.iloc[seen_idx[:n_seen_test]].reset_index(drop=True)
    test_df     = pd.concat([seen_test, unseen_test], ignore_index=True)

    print(f"\nData split summary (random_state={random_state}):")
    print(f"  Seen train   : {len(seen_train)} rows, "
          f"{seen_train[cfg.label_col].nunique()} diseases")
    print(f"  Unseen train : {len(unseen_train)} rows "
          f"({cfg.unseen_label}) — used only in S3")
    print(f"  Test set     : {len(test_df)} rows "
          f"({len(unseen_test)} unseen + {len(seen_test)} seen)")

    return seen_train, unseen_train, test_df, feat_cols


def load_kfold(cfg: ExperimentConfig = _DEFAULT_CFG,
               n_splits=5, include_s3=False, random_state=42):
    """Stratified k-fold CV splits over the full dataset (seen + unseen).

    Unseen patients are never in seen_train (ZSL constraint).
    Yields (fold_idx, seen_train, unseen_train, test_df, feat_cols).
    """
    from sklearn.model_selection import StratifiedKFold

    df, feat_cols = load_data(cfg)

    seen_df   = df[df[cfg.label_col] != cfg.unseen_label].reset_index(drop=True)
    unseen_df = df[df[cfg.label_col] == cfg.unseen_label].reset_index(drop=True)
    full_df   = pd.concat([seen_df, unseen_df], ignore_index=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(
            skf.split(full_df, full_df[cfg.label_col])):
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        test_df  = full_df.iloc[test_idx].reset_index(drop=True)

        unseen_train = (train_df[train_df[cfg.label_col] == cfg.unseen_label]
                        .reset_index(drop=True)) if include_s3 \
                       else train_df.iloc[0:0].copy()
        seen_train   = (train_df[train_df[cfg.label_col] != cfg.unseen_label]
                        .reset_index(drop=True))

        n_unseen_test = (test_df[cfg.label_col] == cfg.unseen_label).sum()
        n_seen_test   = (test_df[cfg.label_col] != cfg.unseen_label).sum()
        print(f"\n  Fold {fold+1}/{n_splits}: "
              f"seen_train={len(seen_train)}  "
              f"unseen_train={len(unseen_train)}  "
              f"test={len(test_df)} ({n_unseen_test} unseen + {n_seen_test} seen)")

        yield fold, seen_train, unseen_train, test_df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df, feat_cols, label_col=None):
    """Return float feature matrix X and label array y, with MICE imputation.

    label_col defaults to 'icd_code' for backward compatibility.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    if label_col is None:
        label_col = LABEL_COL  # backward-compatible default

    cols = [c for c in feat_cols if c in df.columns]
    X = df[cols].copy()
    if 'gender' in df.columns:
        X['gender'] = (df['gender'] == 'Female').astype(float)
    num_cols = [c for c in X.columns if c != 'gender']
    try:
        imputer = IterativeImputer(max_iter=10, random_state=0)
        imputed = imputer.fit_transform(X[num_cols].values)
        X[num_cols] = imputed
    except Exception:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    return X.astype(float), df[label_col].values

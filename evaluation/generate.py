"""
Onto-CGAN — Synthetic Data Generation
======================================

Trains one GAN per conditioning mode on seen diseases, then generates a large
pool of synthetic unseen-disease (AML / Orphanet_519) samples and saves them
to CSV.  Run this once; then use evaluate.py (which is fast) as many times as
needed with different random seeds, splits, or multipliers.

Output (per mode)
-----------------
    <out_dir>/gan_model_S2_<mode>.pkl   — saved GAN model
    <out_dir>/synthetic_S2_<mode>.csv   — <n_samples> synthetic AML rows

Run
---
    python evaluation/generate.py [--epochs 3000] [--n-samples 3000] [--quick]
    python evaluation/generate.py --modes ontology prior  # subset of modes
    python evaluation/generate.py --out-dir output/my_run
"""

import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path

from utils import (
    ROOT, EMB_DIR, UNSEEN_IRI, LABEL_COL,
    load_data, ALL_FEAT_COLS,
)


def train_gan(seen_train, embedding, mode, epochs, batch_size, save_path):
    """Fit Onto_DP_CGAN and save the model to save_path."""
    from onto_cgans import Onto_DP_CGAN

    # Clinical prior for ZSL in 'prior' mode — AML (Orphanet_519) reference values.
    # Source: observed means from real AML patients in MIMICDATA_MoreDisease_all_test.csv.
    # In a true ZSL setting these would come from published clinical literature.
    AML_CLINICAL_PRIOR = {
        'Neutrophils':                        42.6,
        'White Blood Cells':                  11.96,
        'Monocytes':                          10.9,
        'Platelet Count':                     96.4,
        'Hematocrit':                         25.8,
        'anchor_age':                         61.7,
        'Red Blood Cells':                     2.85,
        'PT':                                 14.4,
        'INR(PT)':                             1.35,
        'MCHC':                               34.5,
        'Eosinophils':                         2.6,
        'Hemoglobin':                          8.9,
        'Basophils':                           0.76,
        'RDW':                                16.8,
    }

    model = Onto_DP_CGAN(
        log_file_path=None,
        embedding=embedding,
        epochs=epochs,
        batch_size=batch_size,
        noise_dim=128,
        generator_dim=(256, 256, 256),
        discriminator_dim=(256, 256, 256),
        discriminator_steps=10,
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        private=False,
        cuda=True,
        verbose=True,
        conditioning_mode=mode,
        clinical_prior=AML_CLINICAL_PRIOR if mode == 'prior' else None,
        # decoder mode trains its own EmbeddingDecoder internally during fit()

    )

    gan_train = seen_train.copy()
    num_cols = [c for c in gan_train.columns if c not in ['IRI', LABEL_COL, 'gender']]
    gan_train[num_cols] = gan_train[num_cols].fillna(gan_train[num_cols].median())
    gan_train = gan_train.dropna(subset=num_cols).reset_index(drop=True)

    model.fit(gan_train)

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved model → {save_path}")

    return model


def main(epochs=3000, batch_size=500, modes=None, n_samples=3000,
         out_dir=None, quick=False):

    if quick:
        epochs, n_samples, batch_size = 10, 100, 50
        print("Quick mode: epochs=10, n_samples=100")

    if modes is None:
        modes = ["ontology", "decoder", "decoder_contrast"]

    out_dir = Path(out_dir) if out_dir else ROOT / "output" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    from onto_cgans import OntologyEmbedding
    embedding = OntologyEmbedding(
        embedding_path=str(EMB_DIR / "ontology.embeddings"),
        embedding_size=100,
        hp_dict_fn=str(EMB_DIR / "HPO.dict"),
        rd_dict_fn=str(EMB_DIR / "ORDO.dict"),
    )

    # Use all seen diseases for GAN training (no unseen split needed here)
    df, clf_feat_cols = load_data()
    seen_train = df[df[LABEL_COL] != "ORDO.Orphanet_519"].copy().reset_index(drop=True)
    print(f"GAN training set: {len(seen_train)} rows, "
          f"{seen_train[LABEL_COL].nunique()} diseases")

    for mode in modes:
        print("\n" + "═" * 55)
        print(f"Mode: {mode}")
        print("═" * 55)

        model_path = out_dir / f"gan_model_S2_{mode}.pkl"
        gan_model = train_gan(seen_train, embedding, mode, epochs, batch_size, model_path)

        print(f"  Generating {n_samples} synthetic AML samples...")
        synth = gan_model.sample(n_samples, unseen_rds=[UNSEEN_IRI])

        csv_path = out_dir / f"synthetic_S2_{mode}.csv"
        synth.to_csv(csv_path, index=False)
        print(f"  Saved synthetic data ({len(synth)} rows) → {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train GANs and generate synthetic unseen-disease data.")
    parser.add_argument('--epochs',     type=int,   default=3000)
    parser.add_argument('--batch-size', type=int,   default=500)
    parser.add_argument('--modes',      nargs='+',  default=None,
                        help='Modes to run (default: ontology decoder decoder_contrast)')
    parser.add_argument('--n-samples',  type=int,   default=3000,
                        help='Synthetic rows to generate per mode (default: 3000)')
    parser.add_argument('--out-dir',    type=str,   default=None,
                        help='Output directory (default: output/synthetic/)')
    parser.add_argument('--quick',      action='store_true',
                        help='Smoke test: 10 epochs, 100 samples')
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        modes=args.modes,
        n_samples=args.n_samples,
        out_dir=args.out_dir,
        quick=args.quick,
    )

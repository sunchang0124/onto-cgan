"""
Onto-CGAN — Synthetic Data Generation
======================================

Trains one GAN per conditioning mode on seen diseases and saves each model to
a pkl file.  Sampling is intentionally NOT done here — use evaluate.py instead,
which loads saved models and samples as many times as needed.

Output (per mode)
-----------------
    <out_dir>/gan_model_S2_<mode>.pkl   — saved GAN model

Usage
-----
Interactive (prompts for all settings):
    python3.11 generate.py

Named experiment (pre-registered in utils.CONFIGS):
    python3.11 generate.py --disease psc

Generic CSV (any cgan-format file):
    python3.11 generate.py \\
        --data-path /path/to/cgan_99819_diagnosis.csv \\
        --unseen-iri 519 \\
        --epochs 2000 --batch-size 500

Quick smoke test:
    python3.11 generate.py --data-path /path/to/data.csv --unseen-iri 519 --quick

Conditioning modes (default: ontology decoder decoder_contrast):
    python3.11 generate.py --data-path /path/to/data.csv --unseen-iri 519 \\
        --modes ontology decoder
"""

import argparse
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path

from utils import (
    ROOT, EMB_DIR,
    load_data, CONFIGS, build_config, _expand_iri,
)

_DEFAULT_MODES = ["ontology", "decoder"]

_EMBEDDING_CHOICES = ["gensim", "hit", "sapbert", "biolord", "pubmedbert"]

_OWL_MODELS = {
    "hit":       "Hierarchy-Transformers/HiT-MiniLM-L12-SnomedCT",
    "sapbert":   "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "biolord":   "FremyCompany/BioLORD-2023",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}


# ─────────────────────────────────────────────────────────────────────────────
# Interactive prompt
# ─────────────────────────────────────────────────────────────────────────────

def _ask(prompt_text, default=None, validate=None):
    """Prompt the user; return default if Enter is pressed with no input."""
    hint = f" [{default}]" if default is not None else ""
    while True:
        val = input(f"  {prompt_text}{hint}: ").strip()
        val = val if val else default
        if val is None:
            print("    (required — please enter a value)")
            continue
        if validate:
            err = validate(val)
            if err:
                print(f"    {err}")
                continue
        return val


def _validate_path(p):
    if not Path(p).exists():
        return f"File not found: {p}"


def _validate_iri(raw):
    try:
        _expand_iri(raw)
    except ValueError as e:
        return str(e)


def _validate_int(val):
    try:
        int(val)
    except ValueError:
        return f"Expected an integer, got: {val!r}"


def _validate_embedding(name):
    if name not in _EMBEDDING_CHOICES:
        return f"Expected one of {_EMBEDDING_CHOICES}, got: {name!r}"


def _build_embedding(embedding_name: str):
    """Instantiate the embedding model by name."""
    from onto_cgans import make_embedding_model
    if embedding_name == "gensim":
        from onto_cgans import OntologyEmbedding
        return OntologyEmbedding(
            embedding_path=str(EMB_DIR / "ontology.embeddings"),
            embedding_size=100,
            hp_dict_fn=str(EMB_DIR / "HPO.dict"),
            rd_dict_fn=str(EMB_DIR / "ORDO.dict"),
        )
    owl_path   = str(EMB_DIR / "hpObo_hoom_ordo.owl")
    model_name = _OWL_MODELS[embedding_name]
    cache_path = str(EMB_DIR / f"cache_{embedding_name}.json")
    return make_embedding_model(
        owl_path=owl_path,
        model_name=model_name,
        include_definition=True,
        cache_path=cache_path,
    )


def _interactive_prompt():
    """Ask for all required settings interactively. Returns a namespace."""
    print()
    print("── Onto-CGAN Generation ─────────────────────────────────────────────")
    print("  Press Enter to accept the default shown in [brackets].")
    print()

    data_path  = _ask("Dataset CSV path", validate=_validate_path)
    unseen_iri = _ask(
        "Unseen disease IRI / Orphanet ID  (e.g. 519, Orphanet_519, or full IRI)",
        validate=_validate_iri,
    )
    label_col  = _ask("Label column name", default="label")
    epochs     = int(_ask("Epochs", default="2000", validate=_validate_int))
    batch_size = int(_ask("Batch size", default="500", validate=_validate_int))

    modes_raw  = _ask(
        "Conditioning modes (space-separated)",
        default=" ".join(_DEFAULT_MODES),
    )
    modes = modes_raw.split()

    embedding = _ask(
        f"Embedding ({'/'.join(_EMBEDDING_CHOICES)})",
        default="gensim",
        validate=_validate_embedding,
    )

    stem    = Path(data_path).stem
    default_out_dir = f"output/{stem}/synthetic_epoch_{epochs}"
    if embedding != "gensim":
        default_out_dir = f"{default_out_dir}_{embedding}"
    out_dir = _ask("Output directory", default=default_out_dir)

    print()
    return argparse.Namespace(
        data_path  = data_path,
        unseen_iri = unseen_iri,
        label_col  = label_col,
        epochs     = epochs,
        batch_size = batch_size,
        modes      = modes,
        embedding  = embedding,
        out_dir    = out_dir,
        disease    = None,
        quick      = False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_gan(seen_train, embedding, mode, epochs, batch_size, save_path,
              clinical_prior=None, label_col='label'):
    """Fit Onto_DP_CGAN and save the model to save_path."""
    from onto_cgans import Onto_DP_CGAN

    transformer_path = save_path.parent.parent / 'fitted_transformer.pkl'

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
        clinical_prior=clinical_prior if mode == 'prior' else None,
        saved_transformer=str(transformer_path),
    )

    gan_train = seen_train.copy()
    num_cols = [c for c in gan_train.columns if c not in ['IRI', label_col, 'gender']]
    gan_train[num_cols] = gan_train[num_cols].fillna(gan_train[num_cols].median())
    gan_train = gan_train.dropna(subset=num_cols).reset_index(drop=True)

    model.fit(gan_train)

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved model → {save_path}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Resolve experiment config ────────────────────────────────────────────
    if args.disease:
        if args.disease not in CONFIGS:
            print(f"Unknown --disease {args.disease!r}. "
                  f"Known: {list(CONFIGS.keys())}")
            sys.exit(1)
        cfg = CONFIGS[args.disease]
    else:
        cfg = build_config(
            data_path  = args.data_path,
            unseen_iri = args.unseen_iri,
            label_col  = getattr(args, 'label_col', 'label'),
        )

    epochs         = args.epochs
    batch_size     = args.batch_size
    modes          = args.modes or _DEFAULT_MODES
    embedding_name = getattr(args, 'embedding', 'gensim') or 'gensim'

    if embedding_name not in _EMBEDDING_CHOICES:
        print(f"Unknown --embedding {embedding_name!r}. "
              f"Choices: {_EMBEDDING_CHOICES}")
        sys.exit(1)

    if args.quick:
        epochs, batch_size = 10, 50
        print("Quick mode: epochs=10, batch_size=50")

    synth_tag = f"synthetic_epoch_{epochs}"
    if embedding_name != "gensim":
        synth_tag = f"{synth_tag}_{embedding_name}"
    out_dir = Path(args.out_dir) if args.out_dir \
              else ROOT / "output" / cfg.name / synth_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print(f"  Disease / config : {cfg.name}")
    print(f"  Dataset          : {cfg.data_path}")
    print(f"  Unseen IRI       : {cfg.unseen_iri}")
    print(f"  Unseen label     : {cfg.unseen_label}")
    print(f"  Epochs           : {epochs}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Modes            : {modes}")
    print(f"  Embedding        : {embedding_name}")
    print(f"  Output           : {out_dir}")
    print("═" * 60)
    print()

    # ── Load embedding ───────────────────────────────────────────────────────
    embedding = _build_embedding(embedding_name)

    # ── Load data ────────────────────────────────────────────────────────────
    df, clf_feat_cols = load_data(cfg)
    seen_train = df[df[cfg.label_col] != cfg.unseen_label].copy().reset_index(drop=True)
    print(f"GAN training set: {len(seen_train)} rows, "
          f"{seen_train[cfg.label_col].nunique()} diseases")

    # ── Train one model per mode ─────────────────────────────────────────────
    for mode in modes:
        print("\n" + "─" * 55)
        print(f"  Mode: {mode}")
        print("─" * 55)

        model_path = out_dir / f"gan_model_S2_{mode}.pkl"
        if model_path.exists():
            print(f"  Skipping — model already exists: {model_path}")
            continue
        train_gan(
            seen_train, embedding, mode, epochs, batch_size,
            model_path,
            clinical_prior=cfg.clinical_prior,
            label_col=cfg.label_col,
        )

    print("\nDone. Run evaluate.py to sample and score the saved models.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train Onto-CGAN models (one per conditioning mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Interactive — prompts for all settings
  python3.11 generate.py

  # Generic CSV
  python3.11 generate.py --data-path /path/to/cgan_99819_diagnosis.csv --unseen-iri 519

  # Named experiment
  python3.11 generate.py --disease psc --epochs 2000 --batch-size 500

  # Subset of modes
  python3.11 generate.py --data-path /path/to/data.csv --unseen-iri 519 --modes ontology decoder

  # Alternate local embedding backend
  python3.11 generate.py --data-path /path/to/data.csv --unseen-iri 519 --embedding hit
""",
    )

    # ── Source: named config OR generic CSV (mutually exclusive) ────────────
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        '--disease', type=str, default=None,
        metavar='NAME',
        help=f'Named experiment from CONFIGS (choices: {list(CONFIGS.keys())})',
    )
    src.add_argument(
        '--data-path', type=str, default=None,
        metavar='PATH',
        help='Path to a cgan-format CSV (IRI | label | features)',
    )

    parser.add_argument(
        '--unseen-iri', type=str, default=None,
        metavar='IRI',
        help='Orphanet ID (e.g. 519), Orphanet_519, or full IRI of the withheld disease',
    )
    parser.add_argument(
        '--label-col', type=str, default='label',
        help='Label column name in the CSV (default: label)',
    )
    parser.add_argument('--epochs',     type=int,  default=2000)
    parser.add_argument('--batch-size', type=int,  default=500)
    parser.add_argument(
        '--modes', nargs='+', default=None,
        help=f'Conditioning modes to run (default: {_DEFAULT_MODES})',
    )
    parser.add_argument(
        '--embedding', type=str, default='gensim',
        choices=_EMBEDDING_CHOICES,
        help='Embedding backend to use for conditioning (default: gensim)',
    )
    parser.add_argument(
        '--out-dir', type=str, default=None,
        help='Output directory (default: output/<name>/synthetic_epoch_<epochs>[_<embedding>]/)',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Smoke test: epochs=10, batch_size=50',
    )

    args = parser.parse_args()

    # If no source was given, run the interactive prompt
    if args.disease is None and args.data_path is None:
        args = _interactive_prompt()
    elif args.data_path is not None and args.unseen_iri is None:
        parser.error("--data-path requires --unseen-iri")

    main(args)

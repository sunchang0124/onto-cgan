#!/usr/bin/env python3.11
"""Interactive CLI to run onto-cGAN evaluation for any trained disease.

Usage:
    python3.11 evaluation/run_evaluation.py
    python3.11 evaluation/run_evaluation.py --non-interactive \
        --disease cgan_85443_diagnosis --cv --n-splits 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"
EVAL_SCRIPT = Path(__file__).parent / "evaluate.py"

ORDO_PREFIX = "http://www.orpha.net/ORDO/Orphanet_"

# Known ORDO codes for cgan_<id>_diagnosis stems
_STEM_TO_ORDO: dict[str, str] = {
    "cgan_519_diagnosis":   "519",
    "cgan_54057_diagnosis": "54057",
    "cgan_85443_diagnosis": "85443",
    "cgan_99819_diagnosis": "99819",
}

_DATA_DIR = ROOT.parent / "data_query" / "Data"


def prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value if value else default


def prompt_yes_no(text: str, default: bool = True) -> bool:
    label = "Y/n" if default else "y/N"
    value = input(f"{text} [{label}]: ").strip().lower()
    return default if not value else value in {"y", "yes"}


def prompt_int(text: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = prompt(text, str(default))
        try:
            v = int(raw)
            if v >= minimum:
                return v
            print(f"  Please enter a value >= {minimum}.")
        except ValueError:
            print("  Please enter an integer.")


def discover_diseases() -> list[dict]:
    """Scan output/ for disease dirs that have any synthetic*/ folder with pkl files.

    Models may be spread across multiple epoch folders (e.g. synthetic_epoch_2000/
    and synthetic_epoch_3000/). All such dirs are collected and passed together so
    evaluate.py can load every available mode.
    """
    diseases = []
    if not OUTPUT_DIR.exists():
        return diseases
    for disease_dir in sorted(OUTPUT_DIR.iterdir()):
        if not disease_dir.is_dir():
            continue
        # Collect all synthetic*/ subdirs that contain at least one pkl
        epoch_dirs = sorted(
            d for d in disease_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("synthetic")
            and list(d.glob("gan_model_S2_*.pkl"))
        )
        if not epoch_dirs:
            continue
        # Union of all modes across all epoch dirs (later epoch overrides)
        mode_to_dir: dict = {}
        for d in epoch_dirs:
            for pkl in sorted(d.glob("gan_model_S2_*.pkl")):
                mode = pkl.stem.replace("gan_model_S2_", "")
                mode_to_dir[mode] = d  # later epoch dirs overwrite earlier ones
        modes = sorted(mode_to_dir.keys())
        stem = disease_dir.name
        ordo = _STEM_TO_ORDO.get(stem, "")
        data_path = _DATA_DIR / f"{stem}.csv"
        diseases.append({
            "name":       stem,
            "model_dirs": epoch_dirs,   # list of all epoch dirs
            "modes":      modes,
            "ordo":       ordo,
            "data_path":  data_path,
        })
    return diseases


def pick_disease_interactive(diseases: list[dict]) -> dict:
    print("\nAvailable trained diseases:")
    print(f"  {'#':>3}  {'Disease':35s}  {'ORDO':>8}  {'Modes'}")
    print(f"  {'─'*70}")
    for i, d in enumerate(diseases, start=1):
        print(f"  {i:>3}  {d['name']:35s}  {d['ordo']:>8}  {', '.join(d['modes'])}")
    print()
    while True:
        raw = prompt("Select disease by number").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(diseases):
                return diseases[idx]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(diseases)}.")


def build_command(disease: dict, cv: bool, n_splits: int,
                  stat_only: bool, ml_only: bool,
                  include_s3: bool, random_state: int) -> list[str]:
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model-dir", *[str(d) for d in disease["model_dirs"]],
        "--disease",   disease["name"],
    ]
    if cv:
        cmd += ["--cv", "--n-splits", str(n_splits)]
    if stat_only:
        cmd += ["--stat-only"]
    if ml_only:
        cmd += ["--ml-only"]
    if include_s3:
        cmd += ["--include-s3"]
    if random_state != 42:
        cmd += ["--random-state", str(random_state)]
    return cmd


def run_interactive() -> None:
    diseases = discover_diseases()
    if not diseases:
        print(f"No trained models found under {OUTPUT_DIR}.")
        sys.exit(1)

    disease = pick_disease_interactive(diseases)
    print(f"\nSelected: {disease['name']}  (ORDO {disease['ordo']})")
    for d in disease["model_dirs"]:
        print(f"  Model dir : {d}")
    print(f"  Modes     : {', '.join(disease['modes'])}")

    cv = prompt_yes_no("\nRun cross-validation (k-fold CV)?", default=False)
    n_splits = prompt_int("Number of CV folds", default=5) if cv else 5

    print("\nEvaluation scope:")
    print("  1. Both statistical + ML (default)")
    print("  2. Statistical analysis only")
    print("  3. ML classifiers only")
    scope_raw = prompt("Choose scope", "1").strip()
    stat_only = scope_raw == "2"
    ml_only   = scope_raw == "3"

    include_s3 = prompt_yes_no("Include S3 (upper bound with real unseen samples)?", default=False)

    random_state = 42
    if prompt_yes_no("Change random seed? (default 42)", default=False):
        random_state = prompt_int("Random seed", default=42, minimum=0)

    cmd = build_command(disease, cv, n_splits, stat_only, ml_only, include_s3, random_state)

    print("\n" + "=" * 60)
    print("Running:")
    print("  " + " ".join(cmd))
    print("=" * 60 + "\n")

    subprocess.run(cmd, check=True)


def run_non_interactive(args: argparse.Namespace) -> None:
    diseases = discover_diseases()
    match = next((d for d in diseases if d["name"] == args.disease), None)
    if match is None:
        available = [d["name"] for d in diseases]
        print(f"Disease '{args.disease}' not found. Available: {available}")
        sys.exit(1)

    cmd = build_command(
        match,
        cv=args.cv,
        n_splits=args.n_splits,
        stat_only=args.stat_only,
        ml_only=args.ml_only,
        include_s3=args.include_s3,
        random_state=args.random_state,
    )
    print("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive launcher for onto-cGAN evaluation."
    )
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip prompts; requires --disease.")
    parser.add_argument("--disease", type=str, default=None,
                        help="Disease stem (e.g. cgan_85443_diagnosis).")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--include-s3", action="store_true")
    parser.add_argument("--stat-only", action="store_true")
    parser.add_argument("--ml-only", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.non_interactive:
        if not args.disease:
            parser.error("--non-interactive requires --disease.")
        run_non_interactive(args)
    else:
        run_interactive()


if __name__ == "__main__":
    main()

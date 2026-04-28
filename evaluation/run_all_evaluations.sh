#!/usr/bin/env bash
# Merge pkl files from multiple epoch directories and evaluate all modes.
#
# Usage:
#   bash run_all_evaluations.sh                  # evaluate all four diseases
#   bash run_all_evaluations.sh --cv             # with 5-fold cross-validation
#   bash run_all_evaluations.sh --n-splits 3     # custom folds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/../output"
EXTRA_ARGS=("$@")

# If --n-splits was given without --cv, inject --cv so evaluate.py actually runs CV
has_cv=false
for arg in "${EXTRA_ARGS[@]}"; do [[ "$arg" == "--cv" ]] && has_cv=true; done
for arg in "${EXTRA_ARGS[@]}"; do [[ "$arg" == "--n-splits" ]] && has_cv=true; done
$has_cv || true  # reset exit code
if [[ " ${EXTRA_ARGS[*]} " == *" --n-splits "* ]] && [[ " ${EXTRA_ARGS[*]} " != *" --cv "* ]]; then
    EXTRA_ARGS=("--cv" "${EXTRA_ARGS[@]}")
fi

DISEASES=(
    # "cgan_519_diagnosis"
    # "cgan_54057_diagnosis"
    "cgan_85443_diagnosis"
    "cgan_99819_diagnosis"
)

for disease in "${DISEASES[@]}"; do
    disease_dir="$OUTPUT_DIR/$disease"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Disease: $disease"
    echo "════════════════════════════════════════════════════════════"

    # Collect all synthetic_epoch_* dirs from the main dir only.
    # evaluate.py extracts the embedding suffix from each dir name, so
    # passing dirs directly avoids the filename-collision problem that the
    # old symlink-merge approach had.
    epoch_dirs=()
    for src_dir in "$disease_dir"/synthetic_epoch_* "$disease_dir"/synthetic; do
        [ -d "$src_dir" ] || continue
        epoch_dirs+=("$src_dir")
        echo "  Model dir: $src_dir"
    done

    if [ ${#epoch_dirs[@]} -eq 0 ]; then
        echo "  WARNING: no synthetic_epoch_* directories found — skipping."
        continue
    fi

    echo ""
    echo "  Running evaluation across ${#epoch_dirs[@]} model dir(s)..."
    python3.11 "$SCRIPT_DIR/evaluate.py" \
        --model-dir "${epoch_dirs[@]}" \
        --disease "$disease" \
        "${EXTRA_ARGS[@]}"

    # Visualize all saved result CSVs for this disease
    mapfile -t result_files < <(find "$disease_dir" \
        \( -name "evaluation_results.csv" -o -name "evaluation_results_all_folds.csv" \) \
        -type f 2>/dev/null | sort)

    if [ ${#result_files[@]} -eq 0 ]; then
        echo "  WARNING: no result CSVs found — skipping visualization."
    else
        for results_csv in "${result_files[@]}"; do
            echo ""
            echo "  Generating plots for: $(basename "$(dirname "$results_csv")")/$(basename "$results_csv")"
            python3.11 "$SCRIPT_DIR/visualize_results.py" --results "$results_csv"
        done
    fi
done

echo ""
echo "All evaluations done."

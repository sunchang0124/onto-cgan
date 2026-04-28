#!/usr/bin/env bash
# Run onto-CGAN generation for defined diseases across all embedding methods.
#
# Usage:
#   bash run_all_diseases.sh
#   bash run_all_diseases.sh --quick
#   bash run_all_diseases.sh --epochs 1000 --batch-size 256
#   bash run_all_diseases.sh --modes ontology decoder

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../data_query/Data"

EMBEDDINGS=(
    gensim
    hit
    # sapbert
    # biolord
    pubmedbert
)

# Disease IRIs to run (unseen IRI = the number in the filename)
DISEASES=(
    99819
    # 519
    # 85443
)

EXTRA_ARGS=("$@")

TOTAL_DISEASES=${#DISEASES[@]}
TOTAL_EMBEDDINGS=${#EMBEDDINGS[@]}
TOTAL_RUNS=$((TOTAL_DISEASES * TOTAL_EMBEDDINGS))
RUN_COUNT=0

for iri in "${DISEASES[@]}"; do
    csv="$DATA_DIR/cgan_${iri}_diagnosis.csv"
    stem="cgan_${iri}_diagnosis"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Disease : $stem  (unseen IRI: $iri)"
    echo "  Embeddings: ${EMBEDDINGS[*]}"
    echo "════════════════════════════════════════════════════════════"

    for embedding in "${EMBEDDINGS[@]}"; do
        RUN_COUNT=$((RUN_COUNT + 1))

        echo ""
        echo "────────────────────────────────────────────────────────────"
        echo "  Run $RUN_COUNT / $TOTAL_RUNS"
        echo "  Disease   : $stem"
        echo "  Embedding : $embedding"
        echo "────────────────────────────────────────────────────────────"

        python3.11 "$SCRIPT_DIR/generate.py" \
            --data-path "$csv" \
            --unseen-iri "$iri" \
            --embedding "$embedding" \
            "${EXTRA_ARGS[@]}"
    done
done

echo ""
echo "All $TOTAL_RUNS generation runs done."
echo "Run evaluate.py for each output directory to sample and score."

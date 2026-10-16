#!/bin/bash
# Re-run the test protocol for trained predictors (optionally with a negative control).
#   bash scripts/evaluate.sh CONFIG                                  # main protocol
#   TAG=brainvista OUT_TAG=circular bash scripts/evaluate.sh CONFIG --stim_control circular
set -euo pipefail
source "$1"; shift; source "$(dirname "$0")/_common.sh"
TAG=${TAG:-brainvista}; OUT_TAG=${OUT_TAG:-$TAG}

for SUBJ in "${SUBJECTS[@]}"; do
  data_args "$SUBJ"
  $PY -u evaluate.py "${DATA_ARGS[@]}" "${STIM_ARGS[@]}" \
    --predictor_ckpt "$CKPT_ROOT/$SUBJ/predictor_$TAG/predictor_best.pt" --run_list_file "$TEST_LIST" \
    --horizons 1 10 20 --cls_pool_size "$CLS_POOL" --out_dir "$OUT_ROOT/eval_$OUT_TAG/$SUBJ" "$@"
done
$PY analysis/merge_eval_summaries.py --inputs "$OUT_ROOT"/eval_$OUT_TAG/*/eval_summary.json \
  --out "$OUT_ROOT/eval_$OUT_TAG/eval_summary_all.json"

#!/bin/bash
# Model-based analyses (paper §4.4): history-token occlusion and Spatial Mixer attention.
#   bash scripts/analysis.sh CONFIG
set -euo pipefail
source "$1"; shift; source "$(dirname "$0")/_common.sh"
TAG=${TAG:-brainvista}

for SUBJ in "${SUBJECTS[@]}"; do
  data_args "$SUBJ"
  CK="$CKPT_ROOT/$SUBJ/predictor_$TAG/predictor_best.pt"
  $PY -u analysis/history_occlusion.py "${DATA_ARGS[@]}" "${STIM_ARGS[@]}" \
    --predictor_ckpt "$CK" --run_list_file "$TEST_LIST" --plot --out_dir "$OUT_ROOT/occlusion/$SUBJ"
  $PY -u analysis/mixer_attention.py "${DATA_ARGS[@]}" "${STIM_ARGS[@]}" \
    --predictor_ckpt "$CK" --run_list_file "$TEST_LIST" --horizon 20 --plot \
    --out_dir "$OUT_ROOT/mixer_attention/$SUBJ"
done

#!/bin/bash
# Long-horizon rollout inference (paper Eq. 2): trajectories + Spatial Mixer attention per run.
#   bash scripts/inference.sh CONFIG [--anchors tile|first|all] [--horizon 20]
set -euo pipefail
source "$1"; shift; source "$(dirname "$0")/_common.sh"
TAG=${TAG:-brainvista}

for SUBJ in "${SUBJECTS[@]}"; do
  data_args "$SUBJ"
  $PY -u inference.py "${DATA_ARGS[@]}" "${STIM_ARGS[@]}" \
    --predictor_ckpt "$CKPT_ROOT/$SUBJ/predictor_$TAG/predictor_best.pt" --run_list_file "$TEST_LIST" \
    --horizon 20 --anchors tile --out_dir "$OUT_ROOT/predictions_$TAG" "$@"
done

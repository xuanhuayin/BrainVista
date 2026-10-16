#!/bin/bash
# Stage 2 (paper §3.5, Eq. 15) + held-out evaluation, per subject.
#   bash scripts/train_predictor.sh scripts/configs/algonauts2025.sh
# Ablations: TAG=mask_causal bash scripts/train_predictor.sh CONFIG --mask causal
set -euo pipefail
source "$1"; shift; source "$(dirname "$0")/_common.sh"
TAG=${TAG:-brainvista}

for SUBJ in "${SUBJECTS[@]}"; do
  data_args "$SUBJ"
  SAVE="$CKPT_ROOT/$SUBJ/predictor_$TAG"
  echo "========== predictor $SUBJ ($TAG) =========="
  $PY -u train_predictor.py "${DATA_ARGS[@]}" "${STIM_ARGS[@]}" \
    --train_list_file "$TRAIN_LIST" --val_list_file "$VAL_LIST" --test_list_file "$TEST_LIST" \
    --epochs 40 --batch_size 32 --lr 2e-4 --weight_decay 1e-2 --grad_clip 0.5 \
    --d_model 256 --nhead 8 --nlayers 4 --dropout 0.1 --L 40 \
    --mask s2b --mixer attn --spatial_dz 128 --spatial_nhead 4 --spatial_nlayers 1 \
    --val_horizon 20 --select_metric p_corr \
    --save_dir "$SAVE" --seed "$SEED" "$@"
  $PY -u evaluate.py "${DATA_ARGS[@]}" "${STIM_ARGS[@]}" \
    --predictor_ckpt "$SAVE/predictor_best.pt" --run_list_file "$TEST_LIST" \
    --horizons 1 10 20 --cls_pool_size "$CLS_POOL" --cls_ks 10 \
    --out_dir "$OUT_ROOT/eval_$TAG/$SUBJ"
done

$PY analysis/merge_eval_summaries.py --inputs "$OUT_ROOT"/eval_$TAG/*/eval_summary.json \
  --out "$OUT_ROOT/eval_$TAG/eval_summary_all.json"

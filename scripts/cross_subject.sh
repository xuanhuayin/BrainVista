#!/bin/bash
# Single-source cross-subject transfer on Algonauts 2025 (paper Appendix E.3): a model trained on
# SRC is evaluated on TGT without fine-tuning. TGT's tokens are exported with SRC's tokenizers,
# normalized with TGT's own training-run statistics.
#   bash scripts/cross_subject.sh scripts/configs/algonauts2025.sh
set -euo pipefail
source "$1"; shift; source "$(dirname "$0")/_common.sh"
TAG=${TAG:-brainvista}

for SRC in "${SUBJECTS[@]}"; do
  for TGT in "${SUBJECTS[@]}"; do
    [[ "$SRC" == "$TGT" ]] && continue
    ZX="$WORK_ROOT/z_cross/${SRC}_to_${TGT}"
    for NET in "${NETWORKS[@]}"; do
      $PY -u export_z.py --fmri_root "$FMRI_ROOT" --network "$NET" --subjects "$TGT" \
        --run_list_file "$TEST_LIST" --train_list_file "$TRAIN_LIST" \
        --ckpt "$CKPT_ROOT/$SRC/ckpt_tokenizer_$NET/best.pt" --out_root "$ZX"
    done
    $PY -u evaluate.py --fmri_root "$FMRI_ROOT" --z_root "$ZX" --subjects "$TGT" --parcels "${NETWORKS[@]}" \
      --tokenizer_ckpt_dir "$CKPT_ROOT/$SRC" --norm_train_list_file "$TRAIN_LIST" "${STIM_ARGS[@]}" \
      --predictor_ckpt "$CKPT_ROOT/$SRC/predictor_$TAG/predictor_best.pt" --run_list_file "$TEST_LIST" \
      --horizons 20 --cls_pool_size "$CLS_POOL" --out_dir "$OUT_ROOT/cross_subject/${SRC}_to_${TGT}"
  done
done

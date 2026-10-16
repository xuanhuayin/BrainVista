#!/bin/bash
# Stage 1 (paper §3.5, Eq. 14): one network-wise tokenizer per subject and network, then
# export of the brain region tokens for all splits.
#   bash scripts/train_tokenizer.sh scripts/configs/algonauts2025.sh [extra train_tokenizer.py args]
set -euo pipefail
source "$1"; shift; source "$(dirname "$0")/_common.sh"

for SUBJ in "${SUBJECTS[@]}"; do
  for NET in "${NETWORKS[@]}"; do
    CK="$CKPT_ROOT/$SUBJ/ckpt_tokenizer_$NET"
    echo "========== tokenizer $SUBJ / $NET =========="
    $PY -u train_tokenizer.py --fmri_root "$FMRI_ROOT" --network "$NET" --subjects "$SUBJ" \
      --train_list_file "$TRAIN_LIST" --val_list_file "$VAL_LIST" \
      --z_ratio 0.8 --hidden 768 --dropout 0.1 --seg_len 64 \
      --epochs 40 --batch_size 64 --lr 3e-4 --weight_decay 1e-2 \
      --save_dir "$CK" --seed "$SEED" "$@"
    $PY -u export_z.py --fmri_root "$FMRI_ROOT" --network "$NET" --subjects "$SUBJ" \
      --run_list_file "$TRAIN_LIST" "$VAL_LIST" "$TEST_LIST" \
      --ckpt "$CK/best.pt" --out_root "$Z_ROOT"
  done
done

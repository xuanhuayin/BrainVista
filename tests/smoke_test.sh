#!/bin/bash
# End-to-end smoke test on synthetic data: tokenizers -> export -> predictor -> eval -> inference.
# Usage: bash tests/smoke_test.sh [work_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

WORK="${1:-$(mktemp -d)}"
PY="${PYTHON:-python}"
NETS=(NetA NetB NetC)
SUBJ=(sub-01 sub-02)
L=8

"$PY" tests/make_synthetic_data.py --out "$WORK/data"
D="$WORK/data"

for NET in "${NETS[@]}"; do
  "$PY" train_tokenizer.py --fmri_root "$D/fmri" --network "$NET" --subjects "${SUBJ[@]}" \
    --train_list_file "$D/train.txt" --val_list_file "$D/val.txt" \
    --hidden 64 --seg_len 16 --epochs 3 --batch_size 16 --workers 0 \
    --save_dir "$WORK/ckpt/ckpt_tokenizer_$NET"
  "$PY" export_z.py --fmri_root "$D/fmri" --network "$NET" --subjects "${SUBJ[@]}" \
    --run_list_file "$D/train.txt" "$D/val.txt" "$D/test.txt" \
    --ckpt "$WORK/ckpt/ckpt_tokenizer_$NET/best.pt" --out_root "$WORK/z"
done

COMMON=(--fmri_root "$D/fmri" --z_root "$WORK/z" --subjects "${SUBJ[@]}" --parcels "${NETS[@]}"
        --tokenizer_ckpt_dir "$WORK/ckpt"
        --video_root "$D/features/video" --audio_root "$D/features/audio" --text_root "$D/features/text")

"$PY" train_predictor.py "${COMMON[@]}" \
  --train_list_file "$D/train.txt" --val_list_file "$D/val.txt" --test_list_file "$D/test.txt" \
  --L $L --d_model 32 --nhead 4 --nlayers 2 --spatial_dz 16 --spatial_nhead 2 \
  --epochs 3 --batch_size 16 --workers 0 --val_horizon 5 --save_dir "$WORK/ckpt/predictor"

"$PY" evaluate.py "${COMMON[@]}" --predictor_ckpt "$WORK/ckpt/predictor/predictor_best.pt" \
  --run_list_file "$D/test.txt" --horizons 1 5 --cls_pool_size 50 --out_dir "$WORK/eval"

"$PY" inference.py "${COMMON[@]}" --predictor_ckpt "$WORK/ckpt/predictor/predictor_best.pt" \
  --run_list_file "$D/test.txt" --horizon 5 --anchors tile --out_dir "$WORK/pred"

echo "SMOKE TEST PASSED -> $WORK"

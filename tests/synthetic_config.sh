# Config for running the scripts/ pipeline on tests/make_synthetic_data.py output.
DATASET=synthetic
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the synthetic data directory}
FMRI_ROOT=$DATA_ROOT/fmri
VIDEO_ROOT=$DATA_ROOT/features/video
AUDIO_ROOT=$DATA_ROOT/features/audio
TEXT_ROOT=$DATA_ROOT/features/text
TRAIN_LIST=$DATA_ROOT/train.txt
VAL_LIST=$DATA_ROOT/val.txt
TEST_LIST=$DATA_ROOT/test.txt
SUBJECTS=(sub-01 sub-02)
NETWORKS=(NetA NetB NetC)
MODALITIES=(video audio text)
TR=1.0
RATIO_V=3; RATIO_A=3; RATIO_W=3
FEATURE_ID_REGEX=''
CLS_POOL=50
WORK_ROOT=${WORK_ROOT:?set WORK_ROOT}
Z_ROOT=$WORK_ROOT/z_tokens
CKPT_ROOT=$WORK_ROOT/checkpoints
OUT_ROOT=$WORK_ROOT/outputs
SEED=${SEED:-2025}

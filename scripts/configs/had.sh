# HAD (human action clips), TR = 2.0 s, fsLR surface -> Schaefer-1000 -> Yeo-7; no audio stream.
DATASET=had
DATA_ROOT=${DATA_ROOT:-/path/to/had}

FMRI_ROOT=$DATA_ROOT/fmri_networks
VIDEO_ROOT=$DATA_ROOT/features/video
AUDIO_ROOT=''
TEXT_ROOT=$DATA_ROOT/features/text          # clip-level action descriptions (visibility = annotations)
TRAIN_LIST=$DATA_ROOT/splits/train.txt
VAL_LIST=$DATA_ROOT/splits/val.txt
TEST_LIST=$DATA_ROOT/splits/test.txt

SUBJECTS=($(printf "sub-%02d " $(seq 1 30)))
NETWORKS=(Vis SomMot DorsAttn SalVentAttn Limbic Cont Default)
MODALITIES=(video text)
TR=2.0
RATIO_V=4; RATIO_A=0; RATIO_W=4             # 0.5 s stride; audio inactive
FEATURE_ID_REGEX=''
CLS_POOL=300

WORK_ROOT=${WORK_ROOT:-./work/$DATASET}
Z_ROOT=$WORK_ROOT/z_tokens
CKPT_ROOT=$WORK_ROOT/checkpoints
OUT_ROOT=$WORK_ROOT/outputs
SEED=${SEED:-2025}

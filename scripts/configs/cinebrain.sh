# CineBrain, TR = 0.8 s, dataset-defined visual / auditory ROI voxels -> training-set PCA (512 each).
DATASET=cinebrain
DATA_ROOT=${DATA_ROOT:-/path/to/cinebrain}

FMRI_ROOT=$DATA_ROOT/fmri_pca               # Visual|Auditory/<subj>/<rid>.npy, (T, 512) from preprocessing/train_pca.py
VIDEO_ROOT=$DATA_ROOT/features/video
AUDIO_ROOT=$DATA_ROOT/features/audio
TEXT_ROOT=$DATA_ROOT/features/text
TRAIN_LIST=$DATA_ROOT/splits/train.txt
VAL_LIST=$DATA_ROOT/splits/val.txt
TEST_LIST=$DATA_ROOT/splits/test.txt

SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
NETWORKS=(Visual Auditory)
MODALITIES=(video audio text)
TR=0.8
RATIO_V=2; RATIO_A=2; RATIO_W=2             # 0.4 s stride
FEATURE_ID_REGEX=''
CLS_POOL=400

WORK_ROOT=${WORK_ROOT:-./work/$DATASET}
Z_ROOT=$WORK_ROOT/z_tokens
CKPT_ROOT=$WORK_ROOT/checkpoints
OUT_ROOT=$WORK_ROOT/outputs
SEED=${SEED:-2025}

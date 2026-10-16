# Algonauts 2025 (CNeuroMod Friends / Movie10), TR = 1.49 s, Schaefer-1000 -> Yeo-7.
# Source this file from the scripts in scripts/. Edit DATA_ROOT and the run lists.
DATASET=algonauts2025
DATA_ROOT=${DATA_ROOT:-/path/to/algonauts2025}

FMRI_ROOT=$DATA_ROOT/fmri_networks          # <network>/<subj>/<rid>.npy  (preprocessing/split_schaefer_networks.py)
VIDEO_ROOT=$DATA_ROOT/features/video        # <feature id>.npy            (V-JEPA 2, compressed to 512)
AUDIO_ROOT=$DATA_ROOT/features/audio        # (Wav2Vec2-BERT, compressed to 256)
TEXT_ROOT=$DATA_ROOT/features/text          # (Llama 3.2 3B, compressed to 512)
TRAIN_LIST=$DATA_ROOT/splits/train.txt      # "subj rid" per line, official run-level split
VAL_LIST=$DATA_ROOT/splits/val.txt
TEST_LIST=$DATA_ROOT/splits/test.txt

SUBJECTS=(sub-01 sub-02 sub-03 sub-05)
NETWORKS=(Vis SomMot DorsAttn SalVentAttn Limbic Cont Default)
MODALITIES=(video audio text)
TR=1.49
RATIO_V=3; RATIO_A=3; RATIO_W=3             # samples per TR (~0.5 s stride)
FEATURE_ID_REGEX='task-[A-Za-z0-9]+'        # ses-003_task-s01e02a -> task-s01e02a (shared stimulus features)
CLS_POOL=200

WORK_ROOT=${WORK_ROOT:-./work/$DATASET}
Z_ROOT=$WORK_ROOT/z_tokens
CKPT_ROOT=$WORK_ROOT/checkpoints
OUT_ROOT=$WORK_ROOT/outputs
SEED=${SEED:-2025}

# Shared argument arrays for the BrainVista shell scripts. Sourced after a dataset config.
PY=${PYTHON:-python}
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STIM_ARGS=(--modalities "${MODALITIES[@]}" --ratio_v "$RATIO_V" --ratio_a "$RATIO_A" --ratio_w "$RATIO_W"
           --tr "$TR")
[[ -n "${VIDEO_ROOT:-}" ]] && STIM_ARGS+=(--video_root "$VIDEO_ROOT")
[[ -n "${AUDIO_ROOT:-}" ]] && STIM_ARGS+=(--audio_root "$AUDIO_ROOT")
[[ -n "${TEXT_ROOT:-}"  ]] && STIM_ARGS+=(--text_root "$TEXT_ROOT")
[[ -n "${FEATURE_ID_REGEX:-}" ]] && STIM_ARGS+=(--feature_id_regex "$FEATURE_ID_REGEX")

# data_args SUBJ -> per-subject data / tokenizer arguments
data_args() {
  DATA_ARGS=(--fmri_root "$FMRI_ROOT" --z_root "$Z_ROOT" --subjects "$1" --parcels "${NETWORKS[@]}"
             --tokenizer_ckpt_dir "$CKPT_ROOT/$1")
}

#!/bin/bash
# Design ablations and controls reported in the paper (Algonauts 2025 by default).
# Each variant trains a separate predictor (same frozen tokenizers) and evaluates it.
#   bash scripts/ablations.sh scripts/configs/algonauts2025.sh
set -euo pipefail
CONFIG="$1"
run() { TAG="$1" bash "$(dirname "$0")/train_predictor.sh" "$CONFIG" "${@:2}"; }

# Temporal mask (Table: mask ablation) and Spatial Mixer (Table: mixer ablation).
run mask_causal        --mask causal
run mask_bidirectional --mask bidirectional
run mixer_none         --mixer none
run mixer_linear       --mixer linear
run causal_no_mixer    --mask causal --mixer none            # joint mask x mixer control

# Input sources (Table: input-source ablation).
run brain_only         --input_mode brain_only
run stim_only          --input_mode stim_only

# Stimulus access: strictly earlier stimuli s^{<u}, and an earlier-cutoff lag sweep in TR steps.
run stim_strict        --stim_offset 1
for k in 2 3 4 5; do run "stim_lag_$k" --stim_offset "$k"; done

# Stimulus construction (Table: HRF / FIR).
run stim_hrf           --stim_transform hrf
run stim_fir           --stim_transform fir --fir_lags 4

# Training-data scaling (complete units).
for f in 0.25 0.5 0.75; do run "data_$f" --train_fraction "$f"; done

# Seeds for the main model.
for s in 2026 2027 2028 2029; do
  TAG="seed_$s" SEED="$s" bash "$(dirname "$0")/train_predictor.sh" "$CONFIG"
done

# Negative controls on the main model (evaluation only).
TAG=brainvista OUT_TAG=circular  bash "$(dirname "$0")/evaluate.sh" "$CONFIG" --stim_control circular
TAG=brainvista OUT_TAG=cross_run bash "$(dirname "$0")/evaluate.sh" "$CONFIG" --stim_control cross_run

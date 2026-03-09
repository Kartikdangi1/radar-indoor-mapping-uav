#!/usr/bin/env bash
# =============================================================================
#  eval/eval_config.sh  —  Single place to configure the evaluation pipeline.
#  Sourced by eval/launch_eval.sh and eval/run_full_eval.sh.
# =============================================================================

# ---------------------------------------------------------------------------
# Playback rate  (2.0 = 2× real-time → 247 s bag ≈ 124 s wall-clock)
# ---------------------------------------------------------------------------
RATE=2.5

# ---------------------------------------------------------------------------
# Map post-processing filters (applied on top of auto_enhanced)
# Kernel must be an odd integer ≥ 1.  Set to 0 to disable a filter.
# ---------------------------------------------------------------------------
MEDIAN_K=3      # median blur kernel size
GAUSSIAN_K=3    # Gaussian blur kernel size

# ---------------------------------------------------------------------------
# Map resolution — room uses 0.11, corridor uses 0.21
# This applies to room map post-processing (enhancement, analysis)
# ---------------------------------------------------------------------------
MAP_RESOLUTION=0.11

# ---------------------------------------------------------------------------
# Actual room dimensions (measured ground truth) — used for coverage metrics
# ---------------------------------------------------------------------------
ROOM_W_M=8.05
ROOM_H_M=16.05

# ---------------------------------------------------------------------------
# Dual-lap drift visualisation
# Set to the Unix timestamp (float) where the second lap starts.
# Leave empty ("") to disable the dual-lap figure.
# "2026-03-07 6:10:32.755 PM CET"  →  CET = UTC+1  →  Unix 1772903432.755
# ---------------------------------------------------------------------------
LAP_SPLIT_TIMESTAMP="1772903432.755"

# ---------------------------------------------------------------------------
# Experiment bags
# ---------------------------------------------------------------------------
ROOM_BAG_PATH="$HOME/ws/NewRec/recording_20260228_142454/recording_20260228_142454_0.mcap"
CORRIDOR_BAG_PATH="$HOME/ws/NewRec/recording_20260307_170719/recording_20260307_170719_0.mcap"

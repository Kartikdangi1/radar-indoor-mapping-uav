#!/usr/bin/env bash
# =============================================================================
# 06_run_ablation.sh
# Orchestrates the 7 ablation experiments (A1–A7) defined in thesis Table 5.2.
#
# Each run:
#   1. Patches the relevant YAML config to disable the feature
#   2. Launches the full pipeline
#   3. Plays the bag
#   4. Saves the final map to eval/maps/ablation/<label>.pgm
#   5. Restores the YAML
#
# Usage:
#   source /opt/ros/humble/setup.bash
#   cd /home/kartik/Thesis/fradar/radar-indoor-mapping-uav
#   ./eval/06_run_ablation.sh --bag /path/to/bag.mcap [--run A1,A2,A7]
#
# After all runs:
#   python3 eval/05_analyze_map.py --ablation-dir eval/maps/ablation/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAPS_DIR="$REPO_ROOT/eval/maps/ablation"
LOG_DIR="$REPO_ROOT/eval/logs/ablation"
BAG_PATH=""
RUN_FILTER=""   # e.g. "A1,A3,A7"  (empty = run all)

# Config file paths
OCC_CFG="$REPO_ROOT/src/temporal_radar_mapping/config/temporal_radar_occupancy_config.yaml"
SLAM_CFG="$REPO_ROOT/src/fused_odometry/config/radar_scan_slam.yaml"

# Bag play timeout (seconds) — slightly longer than bag duration
BAG_TIMEOUT=130

# Map saver wait (seconds after bag ends)
MAP_SAVE_WAIT=10

# ===========================================================================
# Argument parsing
# ===========================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bag)     BAG_PATH="$2"; shift 2 ;;
        --run)     RUN_FILTER="$2"; shift 2 ;;
        --timeout) BAG_TIMEOUT="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$BAG_PATH" ]]; then
    echo "Usage: $0 --bag /path/to/bag.mcap [--run A1,A3]"
    exit 1
fi

if [[ ! -f "$BAG_PATH" && ! -d "$BAG_PATH" ]]; then
    echo "ERROR: Bag not found: $BAG_PATH"
    exit 1
fi

mkdir -p "$MAPS_DIR" "$LOG_DIR"

# ===========================================================================
# Helpers
# ===========================================================================

# Backup a config file
backup_config() {
    local cfg="$1"
    cp "$cfg" "${cfg}.ablation_backup"
}

# Restore a config file from backup
restore_config() {
    local cfg="$1"
    if [[ -f "${cfg}.ablation_backup" ]]; then
        mv "${cfg}.ablation_backup" "$cfg"
        echo "  Restored: $cfg"
    fi
}

# Patch a YAML key: sed-based single-value replacement
# Usage: patch_yaml <file> <key> <new_value>
patch_yaml() {
    local file="$1" key="$2" val="$3"
    # Matches: "    key: something" and replaces value
    sed -i "s|^\(\s*${key}:\s*\).*$|\1${val}|" "$file"
    echo "  Patched $key → $val in $(basename $file)"
}

# Run one ablation
run_ablation() {
    local label="$1"
    local description="$2"

    # Check filter
    if [[ -n "$RUN_FILTER" ]]; then
        if [[ ! "$RUN_FILTER" == *"$label"* ]]; then
            echo "[$label] Skipped (not in --run filter)"
            return
        fi
    fi

    local map_out="$MAPS_DIR/${label}.pgm"
    if [[ -f "$map_out" ]]; then
        echo "[$label] Already exists, skipping: $map_out"
        return
    fi

    echo ""
    echo "========================================================"
    echo "[$label] $description"
    echo "========================================================"

    local log_launch="$LOG_DIR/${label}_launch.log"
    local log_bag="$LOG_DIR/${label}_bag.log"
    local log_save="$LOG_DIR/${label}_save.log"

    # Launch pipeline in background
    echo "  Starting pipeline..."
    ros2 launch temporal_radar_mapping radar_mapping.launch.py \
        use_sim_time:=true \
        >"$log_launch" 2>&1 &
    LAUNCH_PID=$!

    # Wait for pipeline to initialise
    sleep 8

    # Play bag
    echo "  Playing bag (timeout=${BAG_TIMEOUT}s)..."
    timeout "$BAG_TIMEOUT" ros2 bag play "$BAG_PATH" --clock \
        >"$log_bag" 2>&1 || true
    echo "  Bag finished."

    # Wait for final map to be processed
    sleep "$MAP_SAVE_WAIT"

    # Save map via service call
    echo "  Saving map..."
    ros2 service call /save_map std_srvs/srv/Trigger \
        >"$log_save" 2>&1 || echo "  [WARN] Map save service call failed"

    # Find the most recently written PGM
    LATEST_PGM=$(find "$HOME/maps" -name "*.pgm" -newer "$LOG_DIR" 2>/dev/null | sort -t_ | tail -1 || true)
    if [[ -z "$LATEST_PGM" ]]; then
        LATEST_PGM=$(find "$HOME/maps" -name "*.pgm" 2>/dev/null | sort | tail -1 || true)
    fi
    if [[ -n "$LATEST_PGM" ]]; then
        cp "$LATEST_PGM" "$map_out"
        echo "  Map saved: $map_out"
    else
        echo "  [WARN] No PGM found in ~/maps — check map_saver output"
    fi

    # Kill pipeline
    echo "  Stopping pipeline..."
    kill "$LAUNCH_PID" 2>/dev/null || true
    sleep 3

    echo "[$label] Done."
}

# ===========================================================================
# Baseline (no modifications)
# ===========================================================================
echo "Running BASELINE (all features enabled)..."
run_ablation "baseline" "Baseline — all features enabled"

# ===========================================================================
# A1: Temporal decay disabled
# Patch: decay_factor = 1.0, static_decay_factor = 1.0, dynamic_decay_factor = 1.0
# ===========================================================================
backup_config "$OCC_CFG"
patch_yaml "$OCC_CFG" "decay_factor"         "1.0"
patch_yaml "$OCC_CFG" "static_decay_factor"  "1.0"
patch_yaml "$OCC_CFG" "dynamic_decay_factor" "1.0"
run_ablation "A1" "Temporal decay disabled (all cells uniform decay = 1.0)"
restore_config "$OCC_CFG"

# ===========================================================================
# A2: Multipath rejection disabled
# Patch: enable_multipath_rejection = false
# ===========================================================================
backup_config "$OCC_CFG"
patch_yaml "$OCC_CFG" "enable_multipath_rejection" "false"
run_ablation "A2" "Multipath rejection disabled"
restore_config "$OCC_CFG"

# ===========================================================================
# A3: Per-ray occlusion filtering disabled
# Patch: enable_per_ray_occlusion = false
# ===========================================================================
backup_config "$OCC_CFG"
patch_yaml "$OCC_CFG" "enable_per_ray_occlusion" "false"
run_ablation "A3" "Per-ray occlusion filtering disabled"
restore_config "$OCC_CFG"

# ===========================================================================
# A4: Map-based occlusion filtering disabled
# Patch: enable_map_occlusion = false
# ===========================================================================
backup_config "$OCC_CFG"
patch_yaml "$OCC_CFG" "enable_map_occlusion" "false"
run_ablation "A4" "Map-based occlusion filtering disabled"
restore_config "$OCC_CFG"

# ===========================================================================
# A5: Dynamic object rejection disabled
# Patch: enable_dynamic_filter = false
# ===========================================================================
backup_config "$OCC_CFG"
patch_yaml "$OCC_CFG" "enable_dynamic_filter" "false"
run_ablation "A5" "Dynamic object rejection disabled"
restore_config "$OCC_CFG"

# ===========================================================================
# A6: Inter-scan ego-motion compensation disabled
# Patch: enable_motion_compensation = false
# ===========================================================================
backup_config "$OCC_CFG"
patch_yaml "$OCC_CFG" "enable_motion_compensation" "false"
run_ablation "A6" "Inter-scan ego-motion compensation disabled"
restore_config "$OCC_CFG"

# ===========================================================================
# A7: GICP drift correction disabled (SLAM node)
# Patch: radar_scan_slam.yaml — set max_fitness_score to 0 (never accepts)
# This prevents the SLAM node from publishing any map→odom correction.
# ===========================================================================
backup_config "$SLAM_CFG"
patch_yaml "$SLAM_CFG" "max_fitness_score" "0.0"
run_ablation "A7" "GICP drift correction disabled (SLAM node rejects all registrations)"
restore_config "$SLAM_CFG"

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "========================================================"
echo "Ablation runs complete."
echo "Maps saved to: $MAPS_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Next step:"
echo "  python3 eval/05_analyze_map.py --ablation-dir $MAPS_DIR"
echo "========================================================"

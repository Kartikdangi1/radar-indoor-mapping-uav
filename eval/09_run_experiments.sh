#!/usr/bin/env bash
# =============================================================================
#  eval/09_run_experiments.sh
#  ⭐ MAIN THESIS EVALUATION SCRIPT — runs ALL 5 experiments (both bags)
#
#  Comprehensive evaluation: room baseline, room ablation, corridor full+laps.
#  Runs headless; no RViz. Collects all metrics and generates all thesis figures.
#
#  Prerequisites:
#    source /opt/ros/humble/setup.bash
#    source install/setup.bash
#    colcon build --symlink-install
#
#  Usage:
#    ./eval/09_run_experiments.sh
#
#  Experiments (sequential):
#    Exp 1  — Room bag, all features ON    → eval/maps/baseline.pgm
#    Exp 2  — Room bag, occlusion OFF      → eval/maps/experiments/occlusion_off.pgm
#    Exp 3A — Corridor full (both laps)    → eval/maps/experiments/corridor_full.pgm
#    Exp 3B — Corridor lap 1 only          → eval/maps/experiments/corridor_lap1.pgm
#    Exp 3C — Corridor lap 2 only          → eval/maps/experiments/corridor_lap2.pgm
#
#  Output:
#    eval/maps/baseline.pgm                — room baseline occupancy map
#    eval/maps/experiments/                — ablation + corridor maps
#    eval/data/                            — extracted odometry, IMU, radar
#    eval/data/corridor/                   — corridor-specific data (odom, perf)
#    eval/data/performance_metrics.json    — corridor performance data
#    eval/figures/thesis/                  — all thesis PDFs + .tex snippets
#    eval/maps/experiments/logs/           — per-experiment logs
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load shared config (defines ROOM_BAG_PATH, CORRIDOR_BAG_PATH,
# LAP_SPLIT_TIMESTAMP, etc.)
source "$SCRIPT_DIR/eval_config.sh"

# Define EXPERIMENTS_DIR after WS is set
EXPERIMENTS_DIR="$WS/eval/maps/experiments"
MAP_OUTPUT_DIR="$HOME/Thesis/maps"
LOG="$EXPERIMENTS_DIR/logs"
PYTHON="${PYTHON:-python3}"
IMU_WARMUP_S=14   # seconds for IMU gyro bias calibration
RATE="${RATE:-1.0}"  # bag playback rate (override via env: RATE=2.0 ./09_run_experiments.sh)

# ---------------------------------------------------------------------------
# Scenario config file sets
# ---------------------------------------------------------------------------
# Room experiments (Exp 1, 2) — 8×16m indoor room
ROOM_FUSED_ODOM_CFG="fused_odom_config_room.yaml"
ROOM_PREPROCESSING_CFG="radar_preprocessing_mapping_room.yaml"
ROOM_SLAM_CFG="radar_scan_slam_room.yaml"
ROOM_OCCUPANCY_CFG="temporal_radar_occupancy_config_room.yaml"

# Corridor experiments (Exp 3A/B/C) — longer corridor with two laps
CORRIDOR_FUSED_ODOM_CFG="fused_odom_config_corridor.yaml"
CORRIDOR_PREPROCESSING_CFG="radar_preprocessing_mapping_corridor.yaml"
CORRIDOR_SLAM_CFG="radar_scan_slam_corridor.yaml"
CORRIDOR_OCCUPANCY_CFG="temporal_radar_occupancy_config_corridor.yaml"

# Active config set (set before calling run_pipeline_and_save)
ACTIVE_FUSED_ODOM_CFG=""
ACTIVE_PREPROCESSING_CFG=""
ACTIVE_SLAM_CFG=""
ACTIVE_OCCUPANCY_CFG=""

mkdir -p "$EXPERIMENTS_DIR" "$LOG" "$MAP_OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[$(date '+%H:%M:%S')] $*"; }
warn()  { echo "[$(date '+%H:%M:%S')] WARN: $*" >&2; }
die()   { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; exit 1; }

require_bag() { [[ -f "$1" ]] || die "Bag not found: $1"; }

kill_ros_nodes() {
    pkill -f "temporal_radar_occupancy_node" 2>/dev/null || true
    pkill -f "fused_odom_node"               2>/dev/null || true
    pkill -f "radar_scan_slam_node"          2>/dev/null || true
    pkill -f "radar_preprocessing_node"      2>/dev/null || true
    pkill -f "radar_mapping.launch"          2>/dev/null || true
    pkill -f "map_saver"                     2>/dev/null || true
    pkill -f "rosbag2_player"               2>/dev/null || true
    sleep 3
}

# ---------------------------------------------------------------------------
# run_pipeline_and_save <map_name> <map_out.pgm> [extra bag args...]
#   BAG_FILE must be set by caller.
#   For lap 1 (time-limited play): pass --duration-s <seconds> as first
#   extra argument (handled specially below, not forwarded to ros2 bag play).
# ---------------------------------------------------------------------------
run_pipeline_and_save() {
    local map_name="$1"
    local map_out="$2"
    shift 2
    local bag_extra_args=("$@")

    # Extract --lap-duration-s if present (used for lap1 time limit via timeout)
    local lap_duration_s=""
    local fwd_args=()
    local i=0
    while [[ $i -lt ${#bag_extra_args[@]} ]]; do
        if [[ "${bag_extra_args[$i]}" == "--lap-duration-s" ]]; then
            i=$(( i + 1 ))
            lap_duration_s="${bag_extra_args[$i]}"
        else
            fwd_args+=("${bag_extra_args[$i]}")
        fi
        i=$(( i + 1 ))
    done

    info "────────────────────────────────────────────────────"
    info "Run: $map_name"
    info "  bag:  $BAG_FILE"
    info "  configs: fused_odom=${ACTIVE_FUSED_ODOM_CFG}  preprocessing=${ACTIVE_PREPROCESSING_CFG}"
    info "           slam=${ACTIVE_SLAM_CFG}  occupancy=${ACTIVE_OCCUPANCY_CFG}"
    [[ -n "$lap_duration_s" ]] && info "  duration: ${lap_duration_s}s (timeout)"
    [[ ${#fwd_args[@]} -gt 0 ]] && info "  extra bag args: ${fwd_args[*]}"
    info "────────────────────────────────────────────────────"

    kill_ros_nodes

    # Launch pipeline in background — use active config set
    ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true \
        fused_odom_config_file:="${ACTIVE_FUSED_ODOM_CFG}" \
        mapping_preprocessing_config_file:="${ACTIVE_PREPROCESSING_CFG}" \
        slam_config_file:="${ACTIVE_SLAM_CFG}" \
        mapping_config_file:="${ACTIVE_OCCUPANCY_CFG}" \
        > "$LOG/${map_name}_pipeline.log" 2>&1 &
    local PIPELINE_PID=$!
    info "  Pipeline PID=$PIPELINE_PID — waiting ${IMU_WARMUP_S}s for IMU warmup..."
    sleep "$IMU_WARMUP_S"

    # Start map_saver in background
    ros2 run temporal_radar_mapping map_saver.py \
        --ros-args \
        -p use_sim_time:=true \
        -p "output_dir:=${MAP_OUTPUT_DIR}" \
        -p "map_name:=${map_name}" \
        > "$LOG/${map_name}_mapsaver.log" 2>&1 &
    local MAP_SAVER_PID=$!
    info "  Map saver PID=$MAP_SAVER_PID"

    # Play bag — use timeout for lap1 duration limit
    if [[ -n "$lap_duration_s" ]]; then
        # lap_duration_s is bag-time; timeout is wall-clock → divide by rate
        local wall_timeout
        wall_timeout=$($PYTHON -c "import math; print(math.ceil(${lap_duration_s} / ${RATE}))")
        info "  Playing bag for ${lap_duration_s}s bag-time (${wall_timeout}s wall-clock at ${RATE}x) ..."
        timeout "${wall_timeout}s" \
            ros2 bag play "$BAG_FILE" --clock --rate "$RATE" "${fwd_args[@]}" \
            > "$LOG/${map_name}_bag.log" 2>&1 || true   # timeout exit code != 0 is fine
    else
        info "  Playing full bag at ${RATE}x ..."
        ros2 bag play "$BAG_FILE" --clock --rate "$RATE" "${fwd_args[@]}" \
            > "$LOG/${map_name}_bag.log" 2>&1 || warn "bag play exited non-zero"
    fi

    info "  Bag done — saving map ..."
    sleep 3
    ros2 service call /save_map std_srvs/srv/Trigger "{}" 2>/dev/null || warn "save_map service unavailable"
    sleep 4

    # Copy the latest matching PGM
    local LATEST
    LATEST=$(ls "$MAP_OUTPUT_DIR/${map_name}"*.pgm 2>/dev/null | sort | tail -1 || true)
    if [[ -n "$LATEST" ]]; then
        cp "$LATEST" "$map_out"
        info "  Map saved: $map_out  (from $LATEST)"
    else
        warn "No PGM found matching ${MAP_OUTPUT_DIR}/${map_name}*.pgm"
    fi

    kill "$MAP_SAVER_PID" 2>/dev/null || true
    kill "$PIPELINE_PID"  2>/dev/null || true
    kill_ros_nodes
    info "  Done: $map_name"
    echo ""
}

# ---------------------------------------------------------------------------
# Compute lap 1 duration from MCAP metadata and LAP_SPLIT_TIMESTAMP
# ---------------------------------------------------------------------------
compute_lap1_duration() {
    local bag="$1"
    local split_ts="$2"
    $PYTHON -c "
import rosbag2_py, sys
try:
    info = rosbag2_py.Info()
    meta = info.read_metadata('$bag', 'mcap')
    start_ns = meta.starting_time.nanoseconds
    split_ts  = float('$split_ts')
    duration  = split_ts - start_ns / 1e9
    if duration <= 0:
        sys.exit(1)
    print(f'{duration:.3f}')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
}

# ============================================================================
# Main
# ============================================================================
info "=== Experiment pipeline start ==="
info "  WS:           $WS"
info "  Room bag:     $ROOM_BAG_PATH"
info "  Corridor bag: $CORRIDOR_BAG_PATH"
info "  Output:       $EXPERIMENTS_DIR"
echo ""

# Kill any stale ROS processes before starting
kill_ros_nodes

# ── Experiment 1 — Room bag, all features ON (→ baseline.pgm) ─────────────
require_bag "$ROOM_BAG_PATH"
info "=== Exp 1: Room baseline (all features ON) ==="
ACTIVE_FUSED_ODOM_CFG="$ROOM_FUSED_ODOM_CFG"
ACTIVE_PREPROCESSING_CFG="$ROOM_PREPROCESSING_CFG"
ACTIVE_SLAM_CFG="$ROOM_SLAM_CFG"
ACTIVE_OCCUPANCY_CFG="$ROOM_OCCUPANCY_CFG"
BAG_FILE="$ROOM_BAG_PATH"
run_pipeline_and_save "exp1_room_baseline" "$WS/eval/maps/baseline.pgm"

# ── Experiment 2 — Room bag, occlusion OFF ────────────────────────────────
info "=== Exp 2: Room bag, occlusion OFF ==="
ACTIVE_FUSED_ODOM_CFG="$ROOM_FUSED_ODOM_CFG"
ACTIVE_PREPROCESSING_CFG="$ROOM_PREPROCESSING_CFG"
ACTIVE_SLAM_CFG="$ROOM_SLAM_CFG"
ACTIVE_OCCUPANCY_CFG="$ROOM_OCCUPANCY_CFG"
CONFIG="$WS/src/temporal_radar_mapping/config/${ROOM_OCCUPANCY_CFG}"
cp "$CONFIG" "$CONFIG.exp_backup"

# Use function trap — variable expansion must happen at definition time
trap 'cp "$CONFIG.exp_backup" "$CONFIG" && rm -f "$CONFIG.exp_backup" || true' RETURN

sed -i 's/^\(\s*enable_per_ray_occlusion:\s*\).*/\1false/' "$CONFIG"
sed -i 's/^\(\s*enable_map_occlusion:\s*\).*/\1false/'      "$CONFIG"
info "  Occlusion flags patched to false"

BAG_FILE="$ROOM_BAG_PATH"
run_pipeline_and_save "exp2_occlusion_off" "$EXPERIMENTS_DIR/occlusion_off.pgm"

# Clean up (trap will also run)
trap - RETURN
cp "$CONFIG.exp_backup" "$CONFIG" && rm -f "$CONFIG.exp_backup" || warn "Could not restore config"
info "  Config restored."
echo ""

# ── Experiment 3A — Full corridor run (perf + odom) ──────────────────────
info "=== Exp 3A: Full corridor run ==="
require_bag "$CORRIDOR_BAG_PATH"
mkdir -p "$WS/eval/data/corridor"
ACTIVE_FUSED_ODOM_CFG="$CORRIDOR_FUSED_ODOM_CFG"
ACTIVE_PREPROCESSING_CFG="$CORRIDOR_PREPROCESSING_CFG"
ACTIVE_SLAM_CFG="$CORRIDOR_SLAM_CFG"
ACTIVE_OCCUPANCY_CFG="$CORRIDOR_OCCUPANCY_CFG"

# Start performance monitor before pipeline
CORRIDOR_DURATION_S=$($PYTHON -c "
import rosbag2_py
info = rosbag2_py.Info()
meta = info.read_metadata('$CORRIDOR_BAG_PATH', 'mcap')
print(f'{meta.duration.nanoseconds / 1e9:.1f}')
" 2>/dev/null || echo "300")
CORRIDOR_WALL_S=$($PYTHON -c "import math; print(math.ceil(${CORRIDOR_DURATION_S} / ${RATE}))")
info "  Corridor bag duration: ${CORRIDOR_DURATION_S}s bag-time  (${CORRIDOR_WALL_S}s wall-clock at ${RATE}x)"

# Kill stale nodes, start pipeline, start perf monitor, play bag
kill_ros_nodes
ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true \
    fused_odom_config_file:="${CORRIDOR_FUSED_ODOM_CFG}" \
    mapping_preprocessing_config_file:="${CORRIDOR_PREPROCESSING_CFG}" \
    slam_config_file:="${CORRIDOR_SLAM_CFG}" \
    mapping_config_file:="${CORRIDOR_OCCUPANCY_CFG}" \
    > "$LOG/exp3a_pipeline.log" 2>&1 &
PIPELINE_PID=$!
info "  Pipeline PID=$PIPELINE_PID — waiting ${IMU_WARMUP_S}s ..."
sleep "$IMU_WARMUP_S"

ros2 run temporal_radar_mapping map_saver.py \
    --ros-args -p use_sim_time:=true \
    -p "output_dir:=${MAP_OUTPUT_DIR}" \
    -p "map_name:=exp3a_corridor_full" \
    > "$LOG/exp3a_mapsaver.log" 2>&1 &
MAP_SAVER_PID=$!

$PYTHON "$SCRIPT_DIR/03_collect_live_performance.py" \
    --duration "$CORRIDOR_WALL_S" \
    --out "$WS/eval/data/corridor/performance_metrics.json" \
    > "$LOG/exp3a_perf.log" 2>&1 &
PERF_PID=$!
info "  Perf monitor PID=$PERF_PID"

BAG_FILE="$CORRIDOR_BAG_PATH"
info "  Playing full corridor bag at ${RATE}x ..."
ros2 bag play "$BAG_FILE" --clock --rate "$RATE" > "$LOG/exp3a_bag.log" 2>&1 || warn "bag play exited non-zero"

sleep 3
ros2 service call /save_map std_srvs/srv/Trigger "{}" 2>/dev/null || warn "save_map unavailable"
sleep 4

wait "$PERF_PID" 2>/dev/null || warn "perf monitor exited non-zero"

LATEST=$(ls "$MAP_OUTPUT_DIR/exp3a_corridor_full"*.pgm 2>/dev/null | sort | tail -1 || true)
[[ -n "$LATEST" ]] && { cp "$LATEST" "$EXPERIMENTS_DIR/corridor_full.pgm"; info "  Saved: corridor_full.pgm"; } || warn "No corridor_full PGM found"

kill "$MAP_SAVER_PID" 2>/dev/null || true
kill "$PIPELINE_PID"  2>/dev/null || true
kill_ros_nodes

# Extract corridor odom + radar data for trajectory figures
info "  Extracting bag data ..."
$PYTHON "$SCRIPT_DIR/01_extract_bag_data.py" \
    --bag "$CORRIDOR_BAG_PATH" \
    --out "$WS/eval/data/corridor" \
    > "$LOG/exp3a_extract.log" 2>&1 || warn "01_extract_bag_data.py failed"
info "  Bag data extracted to eval/data/corridor/"

# Copy corridor perf metrics to top-level for thesis figure generation
if [[ -f "$WS/eval/data/corridor/performance_metrics.json" ]]; then
    cp "$WS/eval/data/corridor/performance_metrics.json" "$WS/eval/data/performance_metrics.json"
    info "  Perf metrics copied to eval/data/performance_metrics.json"
fi
echo ""

# ── Experiment 3B — Corridor lap 1 ───────────────────────────────────────
info "=== Exp 3B: Corridor lap 1 ==="
ACTIVE_FUSED_ODOM_CFG="$CORRIDOR_FUSED_ODOM_CFG"
ACTIVE_PREPROCESSING_CFG="$CORRIDOR_PREPROCESSING_CFG"
ACTIVE_SLAM_CFG="$CORRIDOR_SLAM_CFG"
ACTIVE_OCCUPANCY_CFG="$CORRIDOR_OCCUPANCY_CFG"
LAP1_DURATION_S=$(compute_lap1_duration "$CORRIDOR_BAG_PATH" "$LAP_SPLIT_TIMESTAMP") \
    || { warn "Cannot compute lap1 duration — skipping 3B"; LAP1_DURATION_S=""; }

if [[ -n "$LAP1_DURATION_S" ]]; then
    info "  Lap 1 duration: ${LAP1_DURATION_S}s"
    BAG_FILE="$CORRIDOR_BAG_PATH"
    run_pipeline_and_save "exp3b_corridor_lap1" "$EXPERIMENTS_DIR/corridor_lap1.pgm" \
        --lap-duration-s "$LAP1_DURATION_S"
else
    warn "Skipping Exp 3B"
fi

# ── Experiment 3C — Corridor lap 2 ───────────────────────────────────────
info "=== Exp 3C: Corridor lap 2 ==="
ACTIVE_FUSED_ODOM_CFG="$CORRIDOR_FUSED_ODOM_CFG"
ACTIVE_PREPROCESSING_CFG="$CORRIDOR_PREPROCESSING_CFG"
ACTIVE_SLAM_CFG="$CORRIDOR_SLAM_CFG"
ACTIVE_OCCUPANCY_CFG="$CORRIDOR_OCCUPANCY_CFG"
if [[ -n "$LAP1_DURATION_S" ]]; then
    info "  Lap 2 starts at offset: ${LAP1_DURATION_S}s"
    BAG_FILE="$CORRIDOR_BAG_PATH"
    run_pipeline_and_save "exp3c_corridor_lap2" "$EXPERIMENTS_DIR/corridor_lap2.pgm" \
        --start-offset "$LAP1_DURATION_S"
else
    warn "Skipping Exp 3C (no lap split)"
fi

# ── Final thesis figure assembly ──────────────────────────────────────────
info "=== Generating all thesis figures ==="
LAP_SPLIT_ARG=""
[[ -n "$LAP_SPLIT_TIMESTAMP" ]] && LAP_SPLIT_ARG="--lap-split $LAP_SPLIT_TIMESTAMP"
$PYTHON "$SCRIPT_DIR/07_generate_thesis_figures.py" \
    --data "$WS/eval/data" \
    --maps "$WS/eval/maps" \
    --perf "$WS/eval/data/performance_metrics.json" \
    --out  "$WS/eval/figures/thesis" \
    --figs "$WS/eval/figures" \
    $LAP_SPLIT_ARG \
    2>&1 | tee "$LOG/gen_figures.log"

info "=== All experiments complete ==="
echo ""
info "Results:"
ls "$EXPERIMENTS_DIR"/*.pgm 2>/dev/null | xargs -I{} echo "  {}" || true
echo ""
info "Thesis figures: $WS/eval/figures/thesis/"
ls "$WS/eval/figures/thesis/" 2>/dev/null | xargs -I{} echo "  {}" || true

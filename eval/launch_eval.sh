#!/usr/bin/env bash
# =============================================================================
#  eval/launch_eval.sh
#  Full headless evaluation orchestrator — runs ALL 5 thesis experiments.
#  No RViz. No gnome-terminal. Fully autonomous.
#
#  Prerequisites:
#    source /opt/ros/humble/setup.bash
#    (build is handled automatically by this script)
#
#  Usage:
#    ./eval/launch_eval.sh
#
#  Experiments (sequential):
#    Exp 1  — Room bag, all features ON    → eval/maps/baseline.pgm
#    Exp 2  — Room bag, occlusion OFF      → eval/maps/experiments/occlusion_off.pgm
#    Exp 3B — Corridor lap 1 only          → eval/maps/experiments/corridor_lap1.pgm
#    Exp 3C — Corridor lap 2 only          → eval/maps/experiments/corridor_lap2.pgm
#    Exp 3A — Corridor full (both laps)    → eval/maps/experiments/corridor_full.pgm
# =============================================================================

set -o pipefail

command -v ros2 >/dev/null || { echo "ERROR: source /opt/ros/humble/setup.bash first"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load shared config — sets ROOM_BAG_PATH, CORRIDOR_BAG_PATH, RATE, MAP_RESOLUTION,
# LAP_SPLIT_TIMESTAMP, MEDIAN_K, GAUSSIAN_K, ROOM_W_M, ROOM_H_M
[[ -f "$SCRIPT_DIR/eval_config.sh" ]] && source "$SCRIPT_DIR/eval_config.sh"

# Defaults for any vars not set by eval_config.sh
RATE="${RATE:-2.0}"
MEDIAN_K="${MEDIAN_K:-6}"
GAUSSIAN_K="${GAUSSIAN_K:-6}"
MAP_RESOLUTION="${MAP_RESOLUTION:-0.21}"
ROOM_W_M="${ROOM_W_M:-0}"
ROOM_H_M="${ROOM_H_M:-0}"
ROOM_BAG_PATH="${ROOM_BAG_PATH:-$HOME/ws/NewRec/recording_20260228_142454/recording_20260228_142454_0.mcap}"
CORRIDOR_BAG_PATH="${CORRIDOR_BAG_PATH:-$HOME/ws/NewRec/recording_20260307_170719/recording_20260307_170719_0.mcap}"

EVAL_DIR="$WS/eval"
MAP_OUTPUT_DIR="$HOME/Thesis/maps"
EXPERIMENTS_DIR="$EVAL_DIR/maps/experiments"
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_DIR="$EXPERIMENTS_DIR/logs"
PYTHON=python3
IMU_WARMUP_S=14

PERF_OUTPUT="$EVAL_DIR/data/performance_metrics.json"

PIPELINE_PID="" MAPSAVER_PID="" PERF_PID=""

# ---------------------------------------------------------------------------
# Colours + helpers
# ---------------------------------------------------------------------------
R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m'
B='\033[1;34m' C='\033[0;36m' M='\033[0;35m' N='\033[0m'
log()    { echo -e "${C}[$(date '+%H:%M:%S')]${N} $*"; }
ok()     { echo -e "${G}  ✓${N} $*"; }
warn()   { echo -e "${Y}  ⚠${N} $*"; }
die()    { echo -e "${R}  ✗${N} $*"; exit 1; }
banner() { echo -e "\n${M}══════════════════════════════════════════${N}"; \
           echo -e "${M}  $*${N}"; \
           echo -e "${M}══════════════════════════════════════════${N}"; }

# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------
_spin_pid=""
spin_start() {
    local msg="$1"
    (while true; do
        for c in '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏'; do
            printf "\r${C}  %s${N} %s " "$c" "$msg"; sleep 0.1
        done
    done) &
    _spin_pid=$!
}
spin_stop() {
    [[ -n "$_spin_pid" ]] && kill "$_spin_pid" 2>/dev/null; _spin_pid=""
    printf "\r%-70s\r" " "
}

# ---------------------------------------------------------------------------
# Kill all ROS pipeline nodes
# ---------------------------------------------------------------------------
kill_ros_nodes() {
    pkill -f "temporal_radar_occupancy_node" 2>/dev/null || true
    pkill -f "fused_odom_node"               2>/dev/null || true
    pkill -f "radar_scan_slam_node"          2>/dev/null || true
    pkill -f "radar_preprocessing_node"      2>/dev/null || true
    pkill -f "radar_mapping.launch"          2>/dev/null || true
    pkill -f "map_saver"                     2>/dev/null || true
    pkill -f "rosbag2_player"               2>/dev/null || true
    pkill -f "rosbag2_recorder"             2>/dev/null || true
    sleep 3
}

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
cleanup() {
    spin_stop
    log "Cleaning up..."
    [[ -n "$PIPELINE_PID" ]] && kill "$PIPELINE_PID" 2>/dev/null || true
    [[ -n "$MAPSAVER_PID" ]] && kill "$MAPSAVER_PID" 2>/dev/null || true
    [[ -n "$PERF_PID"     ]] && kill "$PERF_PID"     2>/dev/null || true
    kill_ros_nodes
    log "Done."
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# run_one_experiment  — launch pipeline, play bag, save map, copy to dest
#
#  Usage: run_one_experiment <map_name> <dest_pgm> \
#             <fused_odom_cfg> <preprocessing_cfg> <slam_cfg> <occ_cfg> \
#             <bag_path> [--lap-duration-s N | --start-offset N] [--extract-to DIR]
#
#  --extract-to DIR  Records live pipeline topics (odom, radar, loop closures)
#                    to a bag during the run, then extracts NPZ files to DIR.
#                    Sensor topics (IMU, raw radar) are also extracted from the
#                    original bag. Use for Exp 1 (room) and Exp 3A (corridor full).
# ---------------------------------------------------------------------------
run_one_experiment() {
    local map_name="$1"
    local map_dest="$2"
    local fused_cfg="$3"
    local preproc_cfg="$4"
    local slam_cfg="$5"
    local occ_cfg="$6"
    local bag="$7"
    shift 7

    # Parse optional extra args
    local lap_duration_s="" start_offset="" extract_to=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --lap-duration-s) lap_duration_s="$2"; shift 2 ;;
            --start-offset)   start_offset="$2";   shift 2 ;;
            --extract-to)     extract_to="$2";     shift 2 ;;
            *) warn "run_one_experiment: unknown arg $1"; shift ;;
        esac
    done

    log "────────────────────────────────────────────────────"
    log "Exp: $map_name"
    log "  bag: $bag"
    log "  fused_odom=${fused_cfg}  preprocessing=${preproc_cfg}"
    log "  slam=${slam_cfg}  occupancy=${occ_cfg}"
    [[ -n "$lap_duration_s" ]] && log "  mode: timeout ${lap_duration_s}s bag-time"
    [[ -n "$start_offset"   ]] && log "  mode: start-offset ${start_offset}s"
    log "────────────────────────────────────────────────────"

    # Clean up stale PGMs from previous runs so they don't get copied on failure
    rm -f "$MAP_OUTPUT_DIR/${map_name}"_*.pgm 2>/dev/null || true

    kill_ros_nodes

    # Launch pipeline headlessly in background
    ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true \
        fused_odom_config_file:="${fused_cfg}" \
        mapping_preprocessing_config_file:="${preproc_cfg}" \
        slam_config_file:="${slam_cfg}" \
        mapping_config_file:="${occ_cfg}" \
        > "$LOG_DIR/${map_name}_pipeline.log" 2>&1 &
    PIPELINE_PID=$!
    log "  Pipeline PID=$PIPELINE_PID — waiting ${IMU_WARMUP_S}s for IMU warmup..."
    sleep "$IMU_WARMUP_S"

    # Start map_saver
    ros2 run temporal_radar_mapping map_saver.py \
        --ros-args -p use_sim_time:=true \
        -p "output_dir:=${MAP_OUTPUT_DIR}" \
        -p "map_name:=${map_name}" \
        > "$LOG_DIR/${map_name}_mapsaver.log" 2>&1 &
    MAPSAVER_PID=$!
    log "  Map saver PID=$MAPSAVER_PID"

    # Wait for /save_map service to appear (up to 20s)
    local svc_ready=0
    for i in $(seq 1 20); do
        ros2 service list 2>/dev/null | grep -qF "/save_map" && { svc_ready=1; break; }
        sleep 1
    done
    [[ $svc_ready -eq 1 ]] && ok "/save_map service ready" || warn "/save_map not found — save may fail"

    # Capture live pipeline output topics directly to NPZ (no intermediate bag)
    local RECORD_PID=""
    if [[ -n "$extract_to" ]]; then
        mkdir -p "$extract_to"
        $PYTHON "$EVAL_DIR/capture_pipeline_data.py" \
            --out "$extract_to" \
            > "$LOG_DIR/${map_name}_capture.log" 2>&1 &
        RECORD_PID=$!
        log "  Data capture PID=$RECORD_PID → $extract_to"
    fi

    # Play bag (background+kill for lap1, start-offset for lap2, full play otherwise)
    if [[ -n "$lap_duration_s" ]]; then
        local wall_s
        wall_s=$($PYTHON -c "import math; print(math.ceil(${lap_duration_s} / ${RATE}))")
        log "  Playing bag for ${lap_duration_s}s bag-time (${wall_s}s wall-clock at ${RATE}x) ..."
        # Run in background then kill after wall_s seconds — avoids `timeout` disrupting
        # the ROS DDS network which would take the map_saver down with the bag player
        ros2 bag play "$bag" --clock --rate "$RATE" \
            > "$LOG_DIR/${map_name}_bag.log" 2>&1 &
        local LAP_BAG_PID=$!
        sleep "${wall_s}"
        log "  Lap duration reached — stopping bag..."
        kill "$LAP_BAG_PID" 2>/dev/null || true
        wait "$LAP_BAG_PID" 2>/dev/null || true
    elif [[ -n "$start_offset" ]]; then
        log "  Playing bag from offset ${start_offset}s at ${RATE}x ..."
        ros2 bag play "$bag" --clock --rate "$RATE" --start-offset "$start_offset" \
            > "$LOG_DIR/${map_name}_bag.log" 2>&1 || warn "bag play exited non-zero"
    else
        log "  Playing full bag at ${RATE}x ..."
        ros2 bag play "$bag" --clock --rate "$RATE" \
            > "$LOG_DIR/${map_name}_bag.log" 2>&1 || warn "bag play exited non-zero"
    fi

    log "  Bag done — saving map..."
    # After a timeout-killed bag play, give nodes extra time to settle before saving
    if [[ -n "$lap_duration_s" ]]; then
        sleep 8
    else
        sleep 3
    fi

    # Retry /save_map up to 3 times; use a 15s timeout per call to avoid hanging
    # ROS2 service call format: std_srvs.srv.Trigger_Response(success=True, ...)
    local save_ok=0 svc_out
    for attempt in 1 2 3; do
        svc_out=$(timeout 15s ros2 service call /save_map std_srvs/srv/Trigger "{}" 2>/dev/null || true)
        echo "$svc_out" | grep -q "success=True" && { save_ok=1; break; }
        warn "/save_map attempt $attempt did not confirm success — retrying..."
        sleep 3
    done
    [[ $save_ok -eq 1 ]] && ok "Map saved (service confirmed)" || warn "/save_map did not confirm success"
    sleep 2

    # Validate output PGM and copy to destination
    local LATEST
    LATEST=$(ls "$MAP_OUTPUT_DIR/${map_name}"_*.pgm 2>/dev/null | sort | tail -1 || true)
    if [[ -n "$LATEST" ]]; then
        local sz
        sz=$(stat -c%s "$LATEST" 2>/dev/null || echo 0)
        if [[ $sz -lt 50000 ]]; then
            warn "Map is suspiciously small (${sz} bytes) — may be empty grid; check pipeline logs"
        fi
        cp "$LATEST" "$map_dest"
        ok "Map → $map_dest  (${sz} bytes)"
    else
        warn "No PGM found for $map_name — check $LOG_DIR/${map_name}_mapsaver.log"
    fi

    # Stop capture script (SIGTERM triggers save) then extract sensor topics from original bag
    if [[ -n "$RECORD_PID" ]]; then
        kill "$RECORD_PID" 2>/dev/null || true
        wait "$RECORD_PID" 2>/dev/null || true
        log "  Extracting sensor topics (IMU, raw radar) from original bag..."
        $PYTHON "$EVAL_DIR/01_extract_bag_data.py" \
            --bag "$bag" --out "$extract_to" \
            2>&1 | grep -E "samples|Saved|WARN" || true
        ok "Data captured → $extract_to"
    fi

    kill "${MAPSAVER_PID:-}" 2>/dev/null || true
    kill "${PIPELINE_PID:-}"  2>/dev/null || true
    MAPSAVER_PID="" PIPELINE_PID=""
    kill_ros_nodes
    echo ""
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
banner "RADAR EVAL  ${DATE}"

mkdir -p "$LOG_DIR" "$MAP_OUTPUT_DIR" "$EXPERIMENTS_DIR" \
         "$EVAL_DIR/data" "$EVAL_DIR/data/corridor" \
         "$EVAL_DIR/maps" "$EVAL_DIR/figures/thesis"

[[ -f "$ROOM_BAG_PATH" ]]     || die "Room bag not found: $ROOM_BAG_PATH"
[[ -f "$CORRIDOR_BAG_PATH" ]] || die "Corridor bag not found: $CORRIDOR_BAG_PATH"

log "Room bag:     $ROOM_BAG_PATH"
log "Corridor bag: $CORRIDOR_BAG_PATH"
log "Rate:         ${RATE}x"
log "Log dir:      $LOG_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. Kill stale processes
# ---------------------------------------------------------------------------
log "[1] Killing stale ROS processes..."
kill_ros_nodes
ok "Stale processes cleared"

# ---------------------------------------------------------------------------
# 2. Build
# ---------------------------------------------------------------------------
log "[2] Building workspace..."
cd "$WS"
spin_start "colcon build --symlink-install"
if ! colcon build --symlink-install > "$LOG_DIR/build.log" 2>&1; then
    spin_stop; die "Build failed — see $LOG_DIR/build.log"
fi
spin_stop
source "$WS/install/setup.bash"
ok "Build successful"

# ---------------------------------------------------------------------------
# 3. Exp 1 — Room baseline (all features ON)
# ---------------------------------------------------------------------------
banner "Exp 1: Room baseline (all features ON)"
run_one_experiment "exp1_room_baseline" "$EVAL_DIR/maps/baseline.pgm" \
    "fused_odom_config_room.yaml" \
    "radar_preprocessing_mapping_room.yaml" \
    "radar_scan_slam_room.yaml" \
    "temporal_radar_occupancy_config_room.yaml" \
    "$ROOM_BAG_PATH" \
    --extract-to "$EVAL_DIR/data"

# ---------------------------------------------------------------------------
# 4. Exp 2 — Room bag, occlusion OFF
# ---------------------------------------------------------------------------
banner "Exp 2: Room bag, occlusion OFF"
ROOM_OCC_CFG="$WS/src/temporal_radar_mapping/config/temporal_radar_occupancy_config_room.yaml"
cp "$ROOM_OCC_CFG" "$ROOM_OCC_CFG.bak"
# On EXIT, restore config then run standard cleanup
trap 'cp "$ROOM_OCC_CFG.bak" "$ROOM_OCC_CFG" 2>/dev/null; rm -f "$ROOM_OCC_CFG.bak"; cleanup' EXIT
sed -i 's/^\(\s*enable_per_ray_occlusion:\s*\).*/\1false/' "$ROOM_OCC_CFG"
sed -i 's/^\(\s*enable_map_occlusion:\s*\).*/\1false/'      "$ROOM_OCC_CFG"
log "  Occlusion flags patched to false"

run_one_experiment "exp2_occlusion_off" "$EXPERIMENTS_DIR/occlusion_off.pgm" \
    "fused_odom_config_room.yaml" \
    "radar_preprocessing_mapping_room.yaml" \
    "radar_scan_slam_room.yaml" \
    "temporal_radar_occupancy_config_room.yaml" \
    "$ROOM_BAG_PATH"

cp "$ROOM_OCC_CFG.bak" "$ROOM_OCC_CFG" && rm -f "$ROOM_OCC_CFG.bak"
trap cleanup EXIT   # restore standard cleanup trap
ok "Occlusion config restored"

# ---------------------------------------------------------------------------
# 5. Compute corridor lap 1 duration from bag metadata + split timestamp
# ---------------------------------------------------------------------------
log "Computing corridor lap 1 duration..."
LAP1_DURATION_S=""
if [[ -n "${LAP_SPLIT_TIMESTAMP:-}" ]]; then
    LAP1_DURATION_S=$($PYTHON -c "
import rosbag2_py, sys
try:
    info = rosbag2_py.Info()
    meta = info.read_metadata('$CORRIDOR_BAG_PATH', 'mcap')
    start_ns = meta.starting_time.nanoseconds
    split_ts  = float('$LAP_SPLIT_TIMESTAMP')
    duration  = split_ts - start_ns / 1e9
    if duration <= 0:
        sys.exit(1)
    print(f'{duration:.3f}')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null) || { warn "Cannot compute lap 1 duration — skipping lap split experiments"; LAP1_DURATION_S=""; }
fi
[[ -n "$LAP1_DURATION_S" ]] && ok "Lap 1 duration: ${LAP1_DURATION_S}s" || warn "Lap 1 duration unknown"

# Corridor total duration (wall-clock estimate for perf monitor)
CORRIDOR_DURATION_S=$($PYTHON -c "
import rosbag2_py, sys
try:
    info = rosbag2_py.Info()
    meta = info.read_metadata('$CORRIDOR_BAG_PATH', 'mcap')
    print(f'{meta.duration.nanoseconds / 1e9:.1f}')
except:
    print('400')
" 2>/dev/null || echo "400")
CORRIDOR_WALL_S=$($PYTHON -c "import math; print(math.ceil(${CORRIDOR_DURATION_S} / ${RATE}))")
log "Corridor bag: ${CORRIDOR_DURATION_S}s bag-time → ~${CORRIDOR_WALL_S}s wall-clock at ${RATE}x"

# ---------------------------------------------------------------------------
# 6. Exp 3B — Corridor lap 1 only
# ---------------------------------------------------------------------------
if [[ -n "$LAP1_DURATION_S" ]]; then
    banner "Exp 3B: Corridor lap 1"
    run_one_experiment "exp3b_corridor_lap1" "$EXPERIMENTS_DIR/corridor_lap1.pgm" \
        "fused_odom_config_corridor.yaml" \
        "radar_preprocessing_mapping_corridor.yaml" \
        "radar_scan_slam_corridor.yaml" \
        "temporal_radar_occupancy_config_corridor.yaml" \
        "$CORRIDOR_BAG_PATH" --lap-duration-s "$LAP1_DURATION_S"
else
    warn "Skipping Exp 3B (lap 1 duration unknown)"
fi

# ---------------------------------------------------------------------------
# 7. Exp 3C — Corridor lap 2 only
# ---------------------------------------------------------------------------
if [[ -n "$LAP1_DURATION_S" ]]; then
    banner "Exp 3C: Corridor lap 2"
    run_one_experiment "exp3c_corridor_lap2" "$EXPERIMENTS_DIR/corridor_lap2.pgm" \
        "fused_odom_config_corridor.yaml" \
        "radar_preprocessing_mapping_corridor.yaml" \
        "radar_scan_slam_corridor.yaml" \
        "temporal_radar_occupancy_config_corridor.yaml" \
        "$CORRIDOR_BAG_PATH" --start-offset "$LAP1_DURATION_S"
else
    warn "Skipping Exp 3C (lap 1 duration unknown)"
fi

# ---------------------------------------------------------------------------
# 8. Exp 3A — Corridor full (both laps) + performance monitor
# ---------------------------------------------------------------------------
banner "Exp 3A: Corridor full run"

# Start perf monitor before pipeline launch
$PYTHON "$EVAL_DIR/03_collect_live_performance.py" \
    --duration $(( CORRIDOR_WALL_S + IMU_WARMUP_S + 30 )) \
    --out "$EVAL_DIR/data/corridor/performance_metrics.json" \
    > "$LOG_DIR/exp3a_perf.log" 2>&1 &
PERF_PID=$!
log "  Perf monitor PID=$PERF_PID"

run_one_experiment "exp3a_corridor_full" "$EXPERIMENTS_DIR/corridor_full.pgm" \
    "fused_odom_config_corridor.yaml" \
    "radar_preprocessing_mapping_corridor.yaml" \
    "radar_scan_slam_corridor.yaml" \
    "temporal_radar_occupancy_config_corridor.yaml" \
    "$CORRIDOR_BAG_PATH" \
    --extract-to "$EVAL_DIR/data/corridor"

log "Waiting for perf monitor to finish (max 30s)..."
_w=0
while kill -0 "${PERF_PID:-}" 2>/dev/null && (( _w < 30 )); do
    sleep 1; _w=$(( _w + 1 ))
done
kill -9 "${PERF_PID:-}" 2>/dev/null || true   # SIGKILL — rclpy ignores SIGTERM when stuck
PERF_PID=""
ok "Perf monitor done"

# Copy corridor perf metrics to top-level data dir
if [[ -f "$EVAL_DIR/data/corridor/performance_metrics.json" ]]; then
    cp "$EVAL_DIR/data/corridor/performance_metrics.json" "$PERF_OUTPUT"
    ok "Corridor perf metrics → $PERF_OUTPUT"
fi

# ---------------------------------------------------------------------------
# 9. Offline evaluation
# ---------------------------------------------------------------------------
banner "OFFLINE EVALUATION"

# 9a. Room bag data extraction
log "[9a] Extracting room bag data..."
$PYTHON "$EVAL_DIR/01_extract_bag_data.py" --bag "$ROOM_BAG_PATH" --out "$EVAL_DIR/data" \
    2>&1 | tee "$LOG_DIR/extract_bag.log" | grep -E "Saved|WARN|ERROR" || true
ok "Room bag data extracted"

# 9b. Trajectory plots
if [[ -f "$EVAL_DIR/data/odom.npz" ]]; then
    log "[9b] Plotting trajectory..."
    LAP_SPLIT_ARG=""
    [[ -n "${LAP_SPLIT_TIMESTAMP:-}" ]] && LAP_SPLIT_ARG="--lap-split ${LAP_SPLIT_TIMESTAMP}"
    $PYTHON "$EVAL_DIR/02_plot_trajectory.py" \
        --data "$EVAL_DIR/data" --out "$EVAL_DIR/figures" \
        --room-w "$ROOM_W_M" --room-h "$ROOM_H_M" \
        ${LAP_SPLIT_ARG} \
        2>&1 | tee "$LOG_DIR/plot_trajectory.log" | grep -E "Saved|WARN" || true
    ok "Trajectory figures → eval/figures/"
else
    warn "[9b] odom.npz missing — trajectory plots skipped"
fi

# 9c. Performance plots
if [[ -f "$PERF_OUTPUT" ]]; then
    log "[9c] Plotting performance metrics..."
    $PYTHON "$EVAL_DIR/04_plot_performance.py" \
        --data "$EVAL_DIR/data" --out "$EVAL_DIR/figures" \
        2>&1 | tee "$LOG_DIR/plot_perf.log" | grep -E "Saved|Summary|mean" || true
    ok "Performance figures → eval/figures/"
fi

# 9d. Baseline map quality
if [[ -f "$EVAL_DIR/maps/baseline.pgm" ]]; then
    log "[9d] Analysing map quality..."
    $PYTHON "$EVAL_DIR/05_analyze_map.py" \
        --map "$EVAL_DIR/maps/baseline.pgm" --out "$EVAL_DIR/figures" \
        --resolution "$MAP_RESOLUTION" \
        --room-w "$ROOM_W_M" --room-h "$ROOM_H_M" \
        2>&1 | tee "$LOG_DIR/analyze_map.log" | grep -E "Saved|metrics|occ|coverage|room" || true
    ok "Map analysis → eval/figures/"

    # 9e. Map enhancement
    log "[9e] Enhancing map (median_k=${MEDIAN_K}, gaussian_k=${GAUSSIAN_K})..."
    $PYTHON "$EVAL_DIR/08_enhance_map.py" \
        --map        "$EVAL_DIR/maps/baseline.pgm" \
        --out-dir    "$EVAL_DIR/maps/enhanced" \
        --resolution "$MAP_RESOLUTION" \
        --median-k   "$MEDIAN_K" \
        --gaussian-k "$GAUSSIAN_K" \
        2>&1 | tee "$LOG_DIR/enhance_map.log" | grep -E "\[|Saved|Summary" || true
    ok "Enhanced maps → eval/maps/enhanced/"
fi

# 9f. Assemble thesis figures
log "[9f] Assembling thesis figures..."
LAP_SPLIT_ARG=""
[[ -n "${LAP_SPLIT_TIMESTAMP:-}" ]] && LAP_SPLIT_ARG="--lap-split ${LAP_SPLIT_TIMESTAMP}"
$PYTHON "$EVAL_DIR/07_generate_thesis_figures.py" \
    --data "$EVAL_DIR/data" \
    --maps "$EVAL_DIR/maps" \
    --perf "$PERF_OUTPUT" \
    --out  "$EVAL_DIR/figures/thesis" \
    --figs "$EVAL_DIR/figures" \
    $LAP_SPLIT_ARG \
    2>&1 | tee "$LOG_DIR/gen_figures.log" | grep -E "fig|Copied|SKIP|table" || true
ok "Thesis figures → eval/figures/thesis/"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
banner "COMPLETE ✓ FULL EVALUATION"
TOTAL_TIME=$SECONDS
echo ""
printf "  ${G}%-28s${N} %s\n" "Total runtime:"  "${TOTAL_TIME}s"
printf "  ${G}%-28s${N} %s\n" "Logs:"            "$LOG_DIR/"
printf "  ${G}%-28s${N} %s\n" "Eval data:"       "eval/data/"
printf "  ${G}%-28s${N} %s\n" "Maps:"            "eval/maps/"
printf "  ${G}%-28s${N} %s\n" "Thesis figures:"  "eval/figures/thesis/"
echo ""

# Trajectory stats
if [[ -f "$EVAL_DIR/figures/trajectory_stats.json" ]]; then
    echo -e "  ${B}Trajectory${N}"
    $PYTHON -c "
import json
s = json.load(open('$EVAL_DIR/figures/trajectory_stats.json'))
print(f\"    Distance:  {s.get('total_distance_m', 0):.1f} m\")
print(f\"    Drift:     {s.get('start_to_end_drift_m', 0):.2f} m  ({s.get('drift_percent', 0):.1f}%)\")
print(f\"    Duration:  {s.get('duration_s', 0):.0f} s\")
print(f\"    Speed avg: {s.get('mean_speed_m_s', 0):.2f} m/s  max: {s.get('max_speed_m_s', 0):.2f} m/s\")
" 2>/dev/null || true
    echo ""
fi

# Map metrics
if [[ -f "$EVAL_DIR/figures/map_metrics_baseline.json" ]]; then
    echo -e "  ${B}Map Quality (baseline)${N}"
    $PYTHON -c "
import json
m = json.load(open('$EVAL_DIR/figures/map_metrics_baseline.json'))
print(f\"    Explored:  {m.get('explored_m2', 0):.1f} m²\")
print(f\"    Occupied:  {m.get('occupied_fraction', 0)*100:.1f}%\")
print(f\"    Isolated:  {m.get('n_isolated_points', 0)} noise points\")
print(f\"    Wall:      {m.get('wall_length_m', 0):.1f} m\")
" 2>/dev/null || true
    echo ""
fi

# Performance
if [[ -f "$PERF_OUTPUT" ]]; then
    echo -e "  ${B}Performance${N}"
    $PYTHON -c "
import json
d = json.load(open('$PERF_OUTPUT'))
lat = d.get('latency', {})
for stage, s in lat.items():
    if s.get('count', 0) > 0:
        print(f\"    {stage:<22} {s['mean_ms']:.1f} ms  (p95={s['p95_ms']:.1f} ms)\")
" 2>/dev/null || true
    echo ""
fi

# Thesis figures list
echo -e "  ${B}Thesis figures${N}"
ls "$EVAL_DIR/figures/thesis/" 2>/dev/null | while read f; do
    printf "    %s\n" "$f"
done
echo ""
log "All done. Check eval/figures/thesis/ for PDF outputs."

#!/usr/bin/env bash
# =============================================================================
#  eval/run_full_eval.sh
#  Full pipeline run + simultaneous evaluation data collection + offline analysis
#
#  Mirrors run_full_test.sh but adds:
#    • 03_collect_live_performance.py running in background during bag playback
#    • After bag: extracts bag data, plots all figures, saves LaTeX tables
#
#  Usage:
#    source /opt/ros/humble/setup.bash
#    cd /home/kartik/Thesis/fradar/radar-indoor-mapping-uav
#    ./eval/run_full_eval.sh [BAG_PATH] [RATE]
#
#  Defaults:
#    BAG_PATH = ~/ws/NewRec/recording_20260228_142454/
#    RATE     = 2.0
#
#  Outputs:
#    eval/data/               — extracted numpy data + performance metrics
#    eval/maps/baseline.pgm   — final occupancy map
#    eval/figures/            — all PDF plots
#    eval/figures/thesis/     — thesis-ready figures + LaTeX snippets
#    docs/test/               — test reports (same as run_full_test.sh)
# =============================================================================

set -eo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BAG_PATH="${1:-$HOME/ws/NewRec/recording_20260228_142454/}"
RATE="${2:-2.0}"
WS=/home/kartik/Thesis/fradar/radar-indoor-mapping-uav
EVAL_DIR="$WS/eval"
MAP_OUTPUT_DIR="$HOME/Thesis/maps"
RESULTS_DIR="$WS/docs/test"
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
MAP_NAME="eval_run_${DATE}"
LOG_DIR=/tmp/radar_eval_${DATE}

# Bag duration & timing (matches run_full_test.sh)
BAG_DURATION_S=247
BAG_WALL_S=$(echo "$BAG_DURATION_S / $RATE" | bc)
IMU_WARMUP_WALL_S=12

# Performance monitor duration = bag wall-clock time (self-expires before bag ends,
# calls _save_results() cleanly, then we kill it after the script is done)
PERF_DURATION_S=$(( BAG_WALL_S ))

PYTHON=python3

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
log()   { echo -e "\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m"; }
ok()    { echo -e "\033[1;32m  ✓ $*\033[0m"; }
warn()  { echo -e "\033[1;33m  ⚠ $*\033[0m"; }
die()   { echo -e "\033[1;31m  ✗ $*\033[0m"; exit 1; }
banner(){ echo -e "\033[1;35m\n  ══ $* ══\033[0m"; }

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
PIPELINE_PID="" MAP_SAVER_PID="" BAG_PID="" PERF_PID=""

cleanup() {
    log "Cleaning up all background processes..."
    kill "${PIPELINE_PID:-}" "${MAP_SAVER_PID:-}" "${BAG_PID:-}" "${PERF_PID:-}" 2>/dev/null || true
    sleep 1
    pkill -f "temporal_radar_occupancy_node" 2>/dev/null || true
    pkill -f "fused_odom_node"               2>/dev/null || true
    pkill -f "radar_scan_slam_node"          2>/dev/null || true
    pkill -f "radar_preprocessing_node"      2>/dev/null || true
    pkill -f "rviz2"                         2>/dev/null || true
    pkill -f "map_saver.py"                  2>/dev/null || true
    pkill -f "rosbag2_player"               2>/dev/null || true
    pkill -f "collect_live_performance"      2>/dev/null || true
    sleep 2
    log "Cleanup done."
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
log "=================================================================="
log "  RADAR MAPPING  FULL EVAL RUN   ${DATE}"
log "  Bag:  ${BAG_PATH}"
log "  Rate: ${RATE}x  (wall-clock bag time ≈ ${BAG_WALL_S}s)"
log "=================================================================="

mkdir -p "$LOG_DIR" "$MAP_OUTPUT_DIR" "$RESULTS_DIR" \
         "$EVAL_DIR/data" "$EVAL_DIR/maps" "$EVAL_DIR/figures/thesis" \
         "$EVAL_DIR/logs"

[[ -d "$BAG_PATH" ]] || die "Bag path not found: $BAG_PATH"
[[ -d "$WS" ]]       || die "Workspace not found: $WS"

# ---------------------------------------------------------------------------
# 1. Kill stale processes
# ---------------------------------------------------------------------------
log "[1/10] Killing any stale ROS processes..."
pkill -f "temporal_radar_occupancy_node" 2>/dev/null || true
pkill -f "fused_odom_node"               2>/dev/null || true
pkill -f "radar_scan_slam_node"          2>/dev/null || true
pkill -f "radar_preprocessing_node"      2>/dev/null || true
pkill -f "rviz2"                         2>/dev/null || true
pkill -f "map_saver.py"                  2>/dev/null || true
pkill -f "rosbag2_player"               2>/dev/null || true
sleep 3
ok "Stale processes cleared"

# ---------------------------------------------------------------------------
# 2. Build
# ---------------------------------------------------------------------------
log "[2/10] Building workspace..."
cd "$WS"
source /opt/ros/humble/setup.bash
if colcon build --symlink-install 2>&1 | tee "$LOG_DIR/build.log" | grep -qE "^(Failed|ERROR)"; then
    die "Build failed — see $LOG_DIR/build.log"
fi
source "$WS/install/setup.bash"
ok "Build successful"

# ---------------------------------------------------------------------------
# 3. Launch pipeline (no RViz — headless for eval)
# ---------------------------------------------------------------------------
log "[3/10] Launching radar mapping pipeline (headless)..."
ros2 launch temporal_radar_mapping radar_mapping.launch.py \
    use_sim_time:=true \
    > "$LOG_DIR/pipeline.log" 2>&1 &
PIPELINE_PID=$!

log "  Waiting 6s for nodes to initialize..."
sleep 6

NODES=$(ros2 node list 2>/dev/null || true)
for NODE in fused_odom_node radar_preprocessing_node radar_scan_slam_node temporal_radar_occupancy; do
    if echo "$NODES" | grep -q "$NODE"; then
        ok "$NODE"
    else
        warn "$NODE not found (may still be starting)"
    fi
done

# ---------------------------------------------------------------------------
# 4. Launch map saver
# ---------------------------------------------------------------------------
log "[4/10] Starting map saver..."
ros2 run temporal_radar_mapping map_saver.py \
    --ros-args \
    -p use_sim_time:=true \
    -p "output_dir:=${MAP_OUTPUT_DIR}" \
    -p "map_name:=${MAP_NAME}" \
    > "$LOG_DIR/map_saver.log" 2>&1 &
MAP_SAVER_PID=$!
sleep 2
ok "Map saver ready → ${MAP_OUTPUT_DIR}/${MAP_NAME}.*"

# ---------------------------------------------------------------------------
# 5. Start bag playback
# ---------------------------------------------------------------------------
log "[5/10] Starting bag playback at ${RATE}x speed..."
ros2 bag play "$BAG_PATH" --clock --rate "$RATE" \
    > "$LOG_DIR/bag_play.log" 2>&1 &
BAG_PID=$!
ok "Bag playing (PID: $BAG_PID)"

BAG_START_WALL=$SECONDS

# ---------------------------------------------------------------------------
# 6. Start live performance monitor IN PARALLEL with bag
# ---------------------------------------------------------------------------
log "[6/10] Starting live performance monitor (${PERF_DURATION_S}s window)..."
PERF_OUTPUT="$EVAL_DIR/data/performance_metrics.json"
$PYTHON "$EVAL_DIR/03_collect_live_performance.py" \
    --duration "$PERF_DURATION_S" \
    --out "$PERF_OUTPUT" \
    > "$LOG_DIR/perf_monitor.log" 2>&1 &
PERF_PID=$!
ok "Performance monitor started (PID: $PERF_PID) → $PERF_OUTPUT"

# ---------------------------------------------------------------------------
# 7. IMU warmup then run existing test scripts
# ---------------------------------------------------------------------------
log "[7/10] Waiting ${IMU_WARMUP_WALL_S}s for IMU warmup..."
sleep "$IMU_WARMUP_WALL_S"

log "  Verifying fused odometry..."
if timeout 8 ros2 topic hz /fused_odom/odometry 2>/dev/null | grep -q "average rate"; then
    ok "Fused odometry is flowing"
else
    warn "Fused odometry not yet detected (may still be warming up)"
fi

log "[7/10] Running quick pipeline check (~25s)..."
$PYTHON "$WS/test_pipeline.py" 2>&1 | tee /tmp/quick_test_output.txt
ok "Quick test complete"

log "[7/10] Running integration test (~40s)..."
$PYTHON "$WS/test_pipeline_integration.py" 2>&1 | tee /tmp/integration_test_output.txt
ok "Integration test complete"

# ---------------------------------------------------------------------------
# 8. Wait for bag to finish, save map
# ---------------------------------------------------------------------------
ELAPSED_SINCE_BAG=$(( SECONDS - BAG_START_WALL ))
REMAINING=$(( BAG_WALL_S - ELAPSED_SINCE_BAG ))
if (( REMAINING > 0 )); then
    log "[8/10] Waiting ${REMAINING}s for bag to finish..."
    sleep "$REMAINING"
fi

log "[8/10] Bag finished. Saving final map..."
ros2 service call /save_map std_srvs/srv/Trigger "{}" 2>/dev/null \
    || warn "/save_map service not available"
sleep 5

# Find the saved map
LATEST_PGM=""
if ls "${MAP_OUTPUT_DIR}/${MAP_NAME}"*.pgm 2>/dev/null | grep -q pgm; then
    LATEST_PGM=$(ls "${MAP_OUTPUT_DIR}/${MAP_NAME}"*.pgm 2>/dev/null | sort | tail -1)
    ok "Map file: $LATEST_PGM"
    cp "$LATEST_PGM" "$EVAL_DIR/maps/baseline.pgm"
    ok "Copied to eval/maps/baseline.pgm"
else
    warn "No .pgm map file found — analysis will be skipped"
fi

# Copy test reports
cp /tmp/quick_test_output.txt       "${RESULTS_DIR}/quick_test_${DATE}.txt"    2>/dev/null || true
cp /tmp/integration_test_output.txt "${RESULTS_DIR}/integration_test_${DATE}.txt" 2>/dev/null || true
[[ -f "$WS/pipeline_validation_report.txt" ]] && \
    cp "$WS/pipeline_validation_report.txt" "${RESULTS_DIR}/validation_report_${DATE}.txt"

# ---------------------------------------------------------------------------
# 9. Stop live nodes, wait for performance monitor to flush
# ---------------------------------------------------------------------------
log "[9/10] Stopping pipeline and flushing performance monitor..."
# Kill pipeline nodes (SIGTERM is fine for ROS nodes)
kill "${PIPELINE_PID:-}" "${MAP_SAVER_PID:-}" "${BAG_PID:-}" 2>/dev/null || true

# Send SIGINT to the performance monitor — this triggers Python's
# except KeyboardInterrupt handler which calls _save_results() and writes
# both the JSON and the NPZ signal files before exiting.
if [[ -n "${PERF_PID:-}" ]]; then
    # The monitor self-expires at BAG_WALL_S and calls _save_results() in its
    # finally block. If it's already exited, great. If still running (shouldn't
    # happen unless bag played very fast), send SIGTERM — the finally block fires.
    if kill -0 "${PERF_PID}" 2>/dev/null; then
        log "  Performance monitor still running — waiting up to 15s for self-exit..."
        PERF_WAIT=0
        while kill -0 "${PERF_PID}" 2>/dev/null && (( PERF_WAIT < 15 )); do
            sleep 1; PERF_WAIT=$(( PERF_WAIT + 1 ))
        done
        kill "${PERF_PID}" 2>/dev/null || true
        sleep 2
    fi
fi

if [[ -f "$PERF_OUTPUT" ]]; then
    ok "Performance metrics saved: $PERF_OUTPUT"
    # Check NPZ files
    NPZ_COUNT=$(ls "$EVAL_DIR/data/"*.npz 2>/dev/null | wc -l)
    ok "Signal NPZ files: ${NPZ_COUNT} files in eval/data/"
else
    warn "Performance metrics JSON not found — monitor may have been killed too early"
fi

sleep 2
pkill -f "temporal_radar_occupancy_node" 2>/dev/null || true
pkill -f "fused_odom_node"               2>/dev/null || true
pkill -f "radar_scan_slam_node"          2>/dev/null || true
pkill -f "radar_preprocessing_node"      2>/dev/null || true
pkill -f "map_saver.py"                  2>/dev/null || true
pkill -f "collect_live_performance"      2>/dev/null || true

# ---------------------------------------------------------------------------
# 10. Offline evaluation (all plots from live-recorded data)
# ---------------------------------------------------------------------------
banner "OFFLINE EVALUATION"

# 10a. Live-recorded signal data (odom.npz etc.) was written by the
#      performance monitor during the run — no bag re-reading needed.
#      Also extract IMU + LiDAR height from the raw bag (no ROS topics needed).
log "[10/10] Extracting IMU/LiDAR data from raw bag..."
if $PYTHON "$EVAL_DIR/01_extract_bag_data.py" \
        --bag "$BAG_PATH" \
        --out "$EVAL_DIR/data" \
        2>&1 | tee "$LOG_DIR/extract_bag.log"; then
    ok "Raw bag data (IMU, LiDAR) extracted to eval/data/"
else
    warn "Bag extraction had errors — check $LOG_DIR/extract_bag.log"
fi

# 10b. Trajectory + sensor plots (uses odom.npz saved by performance monitor)
if [[ -f "$EVAL_DIR/data/odom.npz" ]]; then
    log "  Plotting trajectory and sensor data..."
    $PYTHON "$EVAL_DIR/02_plot_trajectory.py" \
        --data "$EVAL_DIR/data" \
        --out  "$EVAL_DIR/figures" \
        2>&1 | tee "$LOG_DIR/plot_trajectory.log"
    ok "Trajectory figures → eval/figures/"
else
    warn "No odometry data (odom.npz) — trajectory plots skipped"
    warn "  This means /fused_odom/odometry was not received during the run"
fi

# 10c. Performance plots
if [[ -f "$PERF_OUTPUT" ]]; then
    log "  Plotting performance metrics..."
    $PYTHON "$EVAL_DIR/04_plot_performance.py" \
        --data "$EVAL_DIR/data" \
        --out  "$EVAL_DIR/figures" \
        2>&1 | tee "$LOG_DIR/plot_performance.log"
    ok "Performance figures → eval/figures/"
else
    warn "No performance metrics file — performance plots skipped"
fi

# 10d. Map quality analysis (raw baseline)
if [[ -f "$EVAL_DIR/maps/baseline.pgm" ]]; then
    log "  Analysing occupancy map quality..."
    $PYTHON "$EVAL_DIR/05_analyze_map.py" \
        --map "$EVAL_DIR/maps/baseline.pgm" \
        --out "$EVAL_DIR/figures" \
        2>&1 | tee "$LOG_DIR/analyze_map.log"
    ok "Map analysis → eval/figures/"
else
    warn "No baseline.pgm — map analysis skipped"
fi

# 10e. Map enhancement (auto-enhance + median + gaussian)
if [[ -f "$EVAL_DIR/maps/baseline.pgm" ]]; then
    log "  Running map enhancement (auto / median / gaussian)..."
    $PYTHON "$EVAL_DIR/08_enhance_map.py" \
        --map        "$EVAL_DIR/maps/baseline.pgm" \
        --out-dir    "$EVAL_DIR/maps/enhanced" \
        --resolution 0.07 \
        2>&1 | tee "$LOG_DIR/enhance_map.log"
    ok "Enhanced maps → eval/maps/enhanced/"

    # Analyse each enhanced variant and produce per-variant figures
    for VARIANT in auto_enhanced median gaussian; do
        VMAP="$EVAL_DIR/maps/enhanced/${VARIANT}.pgm"
        if [[ -f "$VMAP" ]]; then
            $PYTHON "$EVAL_DIR/05_analyze_map.py" \
                --map "$VMAP" \
                --out "$EVAL_DIR/figures" \
                2>&1 >> "$LOG_DIR/analyze_map.log"
        fi
    done
    ok "Enhanced map analyses → eval/figures/"
else
    warn "No baseline.pgm — map enhancement skipped"
fi

# 10f. Assemble thesis figures
log "  Assembling final thesis figures..."
$PYTHON "$EVAL_DIR/07_generate_thesis_figures.py" \
    --data "$EVAL_DIR/data" \
    --maps "$EVAL_DIR/maps" \
    --perf "$PERF_OUTPUT" \
    --out  "$EVAL_DIR/figures/thesis" \
    --figs "$EVAL_DIR/figures" \
    2>&1 | tee "$LOG_DIR/generate_figures.log"
ok "Thesis figures → eval/figures/thesis/"

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
log "=================================================================="
log "  EVAL RUN COMPLETE   ${DATE}"
log "=================================================================="
echo ""
echo "  Test reports:"
echo "    Quick check:       ${RESULTS_DIR}/quick_test_${DATE}.txt"
echo "    Integration test:  ${RESULTS_DIR}/integration_test_${DATE}.txt"
echo "    Validation report: ${RESULTS_DIR}/validation_report_${DATE}.txt"
echo ""
echo "  Evaluation data:"
echo "    Numpy data:        eval/data/"
echo "    Performance JSON:  ${PERF_OUTPUT}"
echo "    Baseline map:      eval/maps/baseline.pgm"
echo ""
echo "  Figures:"
echo "    All figures:       eval/figures/"
echo "    Thesis-ready:      eval/figures/thesis/"
echo "    Run logs:          ${LOG_DIR}/"
echo ""

# Print test pass/fail summary
echo "  --- Test Results ---"
grep -E "(PASS|FAIL|WARN|✓|✗)" /tmp/integration_test_output.txt 2>/dev/null | tail -15 || true

# Print trajectory stats if available
if [[ -f "$EVAL_DIR/figures/trajectory_stats.json" ]]; then
    echo ""
    echo "  --- Trajectory Stats ---"
    $PYTHON -c "
import json
with open('$EVAL_DIR/figures/trajectory_stats.json') as f:
    s = json.load(f)
print(f'    Total distance:  {s.get(\"total_distance_m\", \"?\"):.1f} m')
print(f'    Duration:        {s.get(\"duration_s\", \"?\"):.1f} s')
print(f'    End-to-end drift:{s.get(\"start_to_end_drift_m\", \"?\"):.2f} m ({s.get(\"drift_percent\", \"?\"):.1f}%)')
print(f'    Loop closures:   {s.get(\"n_loop_closures\", 0)}')
print(f'    Mean speed:      {s.get(\"mean_speed_m_s\", \"?\"):.2f} m/s')
" 2>/dev/null || true
fi

# Print map metrics if available
if [[ -f "$EVAL_DIR/figures/map_metrics_baseline.json" ]]; then
    echo ""
    echo "  --- Map Metrics ---"
    $PYTHON -c "
import json
with open('$EVAL_DIR/figures/map_metrics_baseline.json') as f:
    m = json.load(f)
print(f'    Explored area:   {m.get(\"explored_m2\", \"?\"):.1f} m²')
print(f'    Occupied cells:  {m.get(\"occupied_fraction\", 0)*100:.1f}%')
print(f'    Free cells:      {m.get(\"free_fraction\", 0)*100:.1f}%')
print(f'    Isolated points: {m.get(\"n_isolated_points\", \"?\")} (noise proxy)')
print(f'    Wall segments:   {m.get(\"n_wall_segments\", \"?\")}')
print(f'    Est. wall length:{m.get(\"wall_length_m\", \"?\"):.1f} m')
" 2>/dev/null || true
fi

echo ""
log "All done."

#!/usr/bin/env bash
# =============================================================================
#  run_full_test.sh — Full radar mapping pipeline test orchestrator
#
#  Launches the pipeline, plays the bag, runs tests during playback, saves map.
#
#  Usage:
#    ./run_full_test.sh [BAG_PATH] [RATE]
#
#  Defaults:
#    BAG_PATH = ~/ws/NewRec/recording_20260228_142454/
#    RATE     = 2.0
# =============================================================================

set -eo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BAG_PATH="${1:-$HOME/ws/NewRec/recording_20260228_142454/}"
RATE="${2:-2.0}"
WS=/home/kartik/Thesis/fradar/radar-indoor-mapping-uav
MAP_OUTPUT_DIR="$HOME/Thesis/maps"
RESULTS_DIR="$WS/docs/test"
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
MAP_NAME="full_test_${DATE}"
LOG_DIR=/tmp/radar_test_${DATE}

# Bag duration in seconds (247s / RATE = wall-clock wait time)
BAG_DURATION_S=247
BAG_WALL_S=$(echo "$BAG_DURATION_S / $RATE" | bc)

# IMU warmup at real-time = 2400 samples / 480 Hz = 5s; add 5s buffer
IMU_WARMUP_WALL_S=12

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo -e "\033[1;36m[$(date '+%H:%M:%S')] $*\033[0m"; }
ok()   { echo -e "\033[1;32m  ✓ $*\033[0m"; }
warn() { echo -e "\033[1;33m  ⚠ $*\033[0m"; }
die()  { echo -e "\033[1;31m  ✗ $*\033[0m"; exit 1; }

cleanup() {
    log "Cleaning up all background processes..."
    kill "${PIPELINE_PID:-}" "${MAP_SAVER_PID:-}" "${BAG_PID:-}" 2>/dev/null || true
    sleep 1
    # Kill any remaining ROS nodes from this workspace
    pkill -f "temporal_radar_occupancy_node" 2>/dev/null || true
    pkill -f "fused_odom_node"               2>/dev/null || true
    pkill -f "radar_scan_slam_node"          2>/dev/null || true
    pkill -f "radar_preprocessing_node"      2>/dev/null || true
    pkill -f "rviz2"                         2>/dev/null || true
    pkill -f "map_saver.py"                  2>/dev/null || true
    pkill -f "rosbag2_player"               2>/dev/null || true
    sleep 2
    log "Cleanup done."
}

wait_for_topic() {
    local topic=$1
    local timeout_s=${2:-30}
    local start=$SECONDS
    while (( SECONDS - start < timeout_s )); do
        local count
        count=$(ros2 topic hz "$topic" --window 5 2>/dev/null | grep -c "average rate" || true)
        if (( count > 0 )); then return 0; fi
        sleep 1
    done
    return 1
}

# ---------------------------------------------------------------------------
# Step 0: Pre-flight checks
# ---------------------------------------------------------------------------
log "==================================================================="
log "  RADAR MAPPING FULL TEST   ${DATE}"
log "  Bag: ${BAG_PATH}"
log "  Rate: ${RATE}x  (wall-clock bag time ≈ ${BAG_WALL_S}s)"
log "==================================================================="

mkdir -p "$LOG_DIR" "$MAP_OUTPUT_DIR" "$RESULTS_DIR"

[[ -d "$BAG_PATH" ]] || die "Bag path not found: $BAG_PATH"
[[ -d "$WS" ]]       || die "Workspace not found: $WS"

# ---------------------------------------------------------------------------
# Step 1: Kill any stale processes
# ---------------------------------------------------------------------------
log "[1/8] Killing any stale ROS processes..."
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
# Step 2: Build
# ---------------------------------------------------------------------------
log "[2/8] Building workspace..."
cd "$WS"
source /opt/ros/humble/setup.bash
if colcon build --symlink-install 2>&1 | tee "$LOG_DIR/build.log" | grep -qE "^(Failed|ERROR)"; then
    die "Build failed — check $LOG_DIR/build.log"
fi
source "$WS/install/setup.bash"
ok "Build successful"

# ---------------------------------------------------------------------------
# Step 3: Launch pipeline
# ---------------------------------------------------------------------------
log "[3/8] Launching radar mapping pipeline..."
ros2 launch temporal_radar_mapping radar_mapping.launch.py \
    use_sim_time:=true launch_rviz:=true \
    > "$LOG_DIR/pipeline.log" 2>&1 &
PIPELINE_PID=$!
trap cleanup EXIT

log "  Waiting for nodes to initialize (6s)..."
sleep 6

# Verify core nodes are up
NODES=$(ros2 node list 2>/dev/null || true)
for NODE in fused_odom_node radar_preprocessing_node radar_scan_slam_node temporal_radar_occupancy; do
    if echo "$NODES" | grep -q "$NODE"; then
        ok "$NODE"
    else
        warn "$NODE not found — pipeline may still be starting"
    fi
done

# ---------------------------------------------------------------------------
# Step 4: Launch map saver
# ---------------------------------------------------------------------------
log "[4/8] Starting map saver..."
ros2 run temporal_radar_mapping map_saver.py \
    --ros-args \
    -p use_sim_time:=true \
    -p "output_dir:=${MAP_OUTPUT_DIR}" \
    -p "map_name:=${MAP_NAME}" \
    > "$LOG_DIR/map_saver.log" 2>&1 &
MAP_SAVER_PID=$!
sleep 2
ok "Map saver ready (will save to ${MAP_OUTPUT_DIR}/${MAP_NAME}.*)"

# ---------------------------------------------------------------------------
# Step 5: Start bag playback
# ---------------------------------------------------------------------------
log "[5/8] Starting bag playback at ${RATE}x speed..."
ros2 bag play "$BAG_PATH" --clock --rate "$RATE" \
    > "$LOG_DIR/bag_play.log" 2>&1 &
BAG_PID=$!
ok "Bag playing (PID: $BAG_PID)"

# ---------------------------------------------------------------------------
# Step 6: Wait for IMU warmup then run tests
# ---------------------------------------------------------------------------
log "[6/8] Waiting ${IMU_WARMUP_WALL_S}s for IMU warmup..."
sleep "$IMU_WARMUP_WALL_S"

# Confirm fused odometry is flowing
log "  Checking fused odometry is live..."
if timeout 8 ros2 topic hz /fused_odom/odometry 2>/dev/null | grep -q "average rate"; then
    ok "Fused odometry is flowing"
else
    warn "Fused odometry not detected — tests may show partial results"
fi

log "[6/8] Running quick pipeline check (20s)..."
python3 "$WS/test_pipeline.py" 2>&1 | tee /tmp/quick_test_output.txt
ok "Quick test complete"

log "[6/8] Running integration test (30s)..."
python3 "$WS/test_pipeline_integration.py" 2>&1 | tee /tmp/integration_test_output.txt
ok "Integration test complete"

# ---------------------------------------------------------------------------
# Step 7: Wait for bag to finish, then save map
# ---------------------------------------------------------------------------
# Calculate remaining wait: bag has been playing since step 5
# We've used IMU_WARMUP_WALL_S + ~50s for tests = ~62s
# Total bag wall time = BAG_WALL_S; remaining ≈ BAG_WALL_S - 62
ELAPSED_SINCE_BAG=$((IMU_WARMUP_WALL_S + 55))
REMAINING=$((BAG_WALL_S - ELAPSED_SINCE_BAG))
if (( REMAINING > 0 )); then
    log "[7/8] Waiting ${REMAINING}s for bag to finish..."
    sleep "$REMAINING"
fi

log "[7/8] Bag playback complete. Saving map..."
ros2 service call /save_map std_srvs/srv/Trigger "{}" 2>/dev/null || warn "/save_map service not available"
sleep 3

# Check map files
if ls "${MAP_OUTPUT_DIR}/${MAP_NAME}"*.pgm 2>/dev/null | grep -q pgm; then
    ok "Map saved: $(ls ${MAP_OUTPUT_DIR}/${MAP_NAME}*.pgm 2>/dev/null)"
else
    warn "No .pgm map file found — map saver may not have received a /map message"
    ls "$MAP_OUTPUT_DIR/" 2>/dev/null | grep full_test || true
fi

# ---------------------------------------------------------------------------
# Step 8: Copy results and final summary
# ---------------------------------------------------------------------------
log "[8/8] Saving test results to ${RESULTS_DIR}/"
cp /tmp/quick_test_output.txt       "${RESULTS_DIR}/quick_test_${DATE}.txt"
cp /tmp/integration_test_output.txt "${RESULTS_DIR}/integration_test_${DATE}.txt"

if [[ -f pipeline_validation_report.txt ]]; then
    cp pipeline_validation_report.txt "${RESULTS_DIR}/validation_report_${DATE}.txt"
fi

echo ""
log "==================================================================="
log "  TEST COMPLETE"
log "==================================================================="
echo "  Quick test:       ${RESULTS_DIR}/quick_test_${DATE}.txt"
echo "  Integration test: ${RESULTS_DIR}/integration_test_${DATE}.txt"
echo "  Map output:       ${MAP_OUTPUT_DIR}/${MAP_NAME}.*"
echo "  Pipeline logs:    ${LOG_DIR}/"
echo ""
grep -E "(PASS|FAIL|WARN|✓|✗)" /tmp/integration_test_output.txt | tail -20 || true

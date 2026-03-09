#!/usr/bin/env bash
# =============================================================================
# run_ablations.sh — Run A1, A3, A5 ablation experiments
#
# Each ablation:
#   1. Patches the relevant YAML config
#   2. Launches the full ROS pipeline (use_sim_time:=true)
#   3. Replays the bag with --clock
#   4. Captures the final /map → PGM  and  /fused_odom/odometry → NPZ
#   5. Restores the original config
#
# Usage:
#   ./eval/run_ablations.sh
#
# Prerequisites:
#   source /opt/ros/humble/setup.bash
#   source install/setup.bash
#   (run from repo root)
# =============================================================================

set -eo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BAG="/home/kartik/ws/NewRec/recording_20260228_142454/recording_20260228_142454_0.mcap"
OCC_CFG="$REPO_ROOT/src/temporal_radar_mapping/config/temporal_radar_occupancy_config.yaml"
FUSED_CFG="$REPO_ROOT/src/fused_odometry/config/fused_odom_config.yaml"
EVAL_DIR="$REPO_ROOT/eval"
OUT_DIR="$EVAL_DIR/data/ablation"
ROOM_W=8.05
ROOM_H=16.05

mkdir -p "$OUT_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
source_ros() {
    source /opt/ros/humble/setup.bash
    source "$REPO_ROOT/install/setup.bash"
}

backup_cfg() {
    cp "$OCC_CFG"   "$OCC_CFG.bak"
    cp "$FUSED_CFG" "$FUSED_CFG.bak"
}

restore_cfg() {
    [[ -f "$OCC_CFG.bak"   ]] && mv "$OCC_CFG.bak"   "$OCC_CFG"
    [[ -f "$FUSED_CFG.bak" ]] && mv "$FUSED_CFG.bak" "$FUSED_CFG"
    echo "[restore] original configs restored"
}

# Restore configs on unexpected exit (SIGINT, error, etc.)
trap restore_cfg EXIT

# ── Run one ablation ──────────────────────────────────────────────────────────
# $1 = ablation id (a1 / a3 / a5)
# $2 = human label
run_ablation() {
    local ID="$1"
    local LABEL="$2"
    local ABLATION_OUT="$OUT_DIR/$ID"
    mkdir -p "$ABLATION_OUT"

    echo ""
    echo "============================================================"
    echo "  Running ablation $ID: $LABEL"
    echo "  Output → $ABLATION_OUT"
    echo "============================================================"

    # ── Launch pipeline ────────────────────────────────────────────────────
    echo "[launch] Starting pipeline..."
    source_ros
    ros2 launch temporal_radar_mapping radar_mapping.launch.py \
        use_sim_time:=true \
        &> "$ABLATION_OUT/pipeline.log" &
    PIPELINE_PID=$!
    echo "[launch] pipeline PID=$PIPELINE_PID"

    # Give nodes time to start
    sleep 8

    # ── Bag play ───────────────────────────────────────────────────────────
    echo "[bag] Playing bag..."
    source_ros
    ros2 bag play "$BAG" --clock --rate 1.0 \
        &> "$ABLATION_OUT/bag.log" &
    BAG_PID=$!
    echo "[bag] bag PID=$BAG_PID"

    # ── Data capture ───────────────────────────────────────────────────────
    echo "[capture] Subscribing to /map and /fused_odom/odometry..."
    source_ros
    python3 "$EVAL_DIR/capture_ablation_data.py" \
        --out "$ABLATION_OUT" \
        --duration 280 \
        &> "$ABLATION_OUT/capture.log" &
    CAPTURE_PID=$!

    # ── Wait for bag to finish ─────────────────────────────────────────────
    echo "[wait] Waiting for bag to finish (max 300s)..."
    wait $BAG_PID 2>/dev/null || true
    echo "[wait] Bag finished."

    # Give occupancy grid time to flush final map update
    sleep 5

    # ── Stop capture + pipeline ───────────────────────────────────────────
    kill $CAPTURE_PID 2>/dev/null || true
    kill $PIPELINE_PID 2>/dev/null || true
    # Kill any lingering ros2 nodes
    pkill -f "temporal_radar_occupancy" 2>/dev/null || true
    pkill -f "fused_odom_node"          2>/dev/null || true
    pkill -f "radar_preprocessing"      2>/dev/null || true
    pkill -f "radar_scan_slam"          2>/dev/null || true
    sleep 3

    echo "[done] $ID outputs saved to $ABLATION_OUT"
}

# ─────────────────────────────────────────────────────────────────────────────
# A1: Disable temporal decay
# ─────────────────────────────────────────────────────────────────────────────
backup_cfg

python3 - "$OCC_CFG" << 'PYEOF'
import sys, re
path = sys.argv[1]
text = open(path).read()
# Only patch the three TEMPORAL decay_factor lines (not range_decay_factor)
text = re.sub(r'^(\s*decay_factor:\s*)0\.\d+',         r'\g<1>1.0', text, flags=re.MULTILINE)
text = re.sub(r'^(\s*static_decay_factor:\s*)0\.\d+',  r'\g<1>1.0', text, flags=re.MULTILINE)
text = re.sub(r'^(\s*dynamic_decay_factor:\s*)0\.\d+', r'\g<1>1.0', text, flags=re.MULTILINE)
# Set a very large decay_delay so it effectively never triggers
text = re.sub(r'^(\s*decay_delay_sec:\s*)\d+\.?\d*',   r'\g<1>9999.0', text, flags=re.MULTILINE)
open(path, 'w').write(text)
print("[A1] Temporal decay disabled (temporal decay factors → 1.0, delay → 9999s)")
PYEOF

run_ablation "a1" "No temporal decay"
restore_cfg

# ─────────────────────────────────────────────────────────────────────────────
# A3: Disable per-ray occlusion filtering
# ─────────────────────────────────────────────────────────────────────────────
backup_cfg

python3 - "$OCC_CFG" << 'PYEOF'
import sys, re
path = sys.argv[1]
text = open(path).read()
text = re.sub(r'(enable_per_ray_occlusion:\s*)true', r'\g<1>false', text)
text = re.sub(r'(enable_map_occlusion:\s*)true',     r'\g<1>false', text)
open(path, 'w').write(text)
print("[A3] Per-ray + map occlusion filtering disabled")
PYEOF

run_ablation "a3" "No per-ray occlusion filtering"
restore_cfg

# ─────────────────────────────────────────────────────────────────────────────
# A5: Disable dynamic object rejection (RANSAC side)
# ─────────────────────────────────────────────────────────────────────────────
backup_cfg

python3 - "$FUSED_CFG" << 'PYEOF'
import sys, re
path = sys.argv[1]
text = open(path).read()
# Set dynamic_removal_thresh to 0.0 to disable the filter
text = re.sub(r'(dynamic_removal_thresh:\s*)[\d.]+', r'\g<1>0.0', text)
open(path, 'w').write(text)
print("[A5] Dynamic rejection disabled (dynamic_removal_thresh → 0.0)")
PYEOF

# Also disable the occupancy-level dynamic filter
python3 - "$OCC_CFG" << 'PYEOF'
import sys, re
path = sys.argv[1]
text = open(path).read()
text = re.sub(r'(enable_dynamic_filter:\s*)true', r'\g<1>false', text)
open(path, 'w').write(text)
print("[A5] Occupancy dynamic filter also disabled")
PYEOF

run_ablation "a5" "No dynamic rejection"
restore_cfg

# ─────────────────────────────────────────────────────────────────────────────
# Post-process all ablation outputs
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Post-processing ablation outputs"
echo "============================================================"
source_ros

for ID in a1 a3 a5; do
    ABLATION_OUT="$OUT_DIR/$ID"
    if [ -f "$ABLATION_OUT/map.pgm" ]; then
        echo "[analyze] $ID — running map analysis..."
        python3 "$EVAL_DIR/05_analyze_map.py" \
            --map "$ABLATION_OUT/map.pgm" \
            --room-w $ROOM_W --room-h $ROOM_H \
            --out "$ABLATION_OUT" \
            2>&1 | tail -5
        # 05_analyze_map saves as map_metrics_map.json when input is map.pgm
        # rename to the canonical name expected by the summary script
        [[ -f "$ABLATION_OUT/map_metrics_map.json" ]] && \
            cp "$ABLATION_OUT/map_metrics_map.json" "$ABLATION_OUT/map_metrics.json"
    else
        echo "[warn] $ID — map.pgm not found, skipping analysis"
    fi

    if [ -f "$ABLATION_OUT/odom.npz" ]; then
        echo "[plot] $ID — plotting trajectory..."
        python3 "$EVAL_DIR/02_plot_trajectory.py" \
            --data "$ABLATION_OUT" \
            --out  "$ABLATION_OUT" \
            --room-w $ROOM_W --room-h $ROOM_H \
            2>&1 | tail -5
    else
        echo "[warn] $ID — odom.npz not found, skipping trajectory plot"
    fi
done

# ── Summary table ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Ablation summary"
echo "============================================================"
python3 - "$OUT_DIR" "$EVAL_DIR/data/map_metrics_baseline.json" << 'PYEOF'
import sys, json
from pathlib import Path

out_dir = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])

rows = []

if baseline_path.exists():
    b = json.loads(baseline_path.read_text())
    rows.append(("Baseline",       b.get("occupied_fraction",0)*100,
                  b.get("n_isolated_points",0),
                  b.get("wall_length_m",0),
                  b.get("room_coverage_pct",0)))

label_map = {"a1": "A1: No temporal decay",
             "a3": "A3: No occlusion filter",
             "a5": "A5: No dynamic rejection"}

for ablation_id, label in label_map.items():
    p = out_dir / ablation_id / "map_metrics.json"
    if p.exists():
        m = json.loads(p.read_text())
        rows.append((label,
                     m.get("occupied_fraction",0)*100,
                     m.get("n_isolated_points",0),
                     m.get("wall_length_m",0),
                     m.get("room_coverage_pct",0)))
    else:
        rows.append((label, float("nan"), float("nan"), float("nan"), float("nan")))

header = f"{'Variant':<35} {'Occ%':>6} {'Isolated':>9} {'Wall(m)':>8} {'Cov%':>6}"
print(header)
print("-" * len(header))
for row in rows:
    print(f"{row[0]:<35} {row[1]:>6.1f} {row[2]:>9} {row[3]:>8.1f} {row[4]:>6.1f}")

# Save as JSON for thesis table generation
summary = {}
for row in rows:
    summary[row[0]] = {
        "occupied_pct":    round(row[1], 2),
        "isolated_points": row[2],
        "wall_length_m":   round(row[3], 1),
        "coverage_pct":    round(row[4], 1),
    }
out_path = out_dir / "ablation_summary.json"
out_path.write_text(json.dumps(summary, indent=2))
print(f"\nSaved summary → {out_path}")
PYEOF

echo ""
echo "All done. Results in $OUT_DIR"

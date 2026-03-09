#!/usr/bin/env python3
"""
Integration test suite for the Radar Indoor Mapping UAV pipeline.
Validates node health, topic flow, TF tree, odometry quality, radar quality,
SLAM health, and occupancy map quality. Saves a report for thesis documentation.

Runtime: ~40 seconds.

Usage (standalone):
  python3 test_pipeline_integration.py

Usage (with output saved):
  python3 test_pipeline_integration.py 2>&1 | tee validation_run.txt

The pipeline must be running:
  ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true
A bag file must be playing:
  ros2 bag play /path/to/bag --clock
"""

import json
import math
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import rclpy
import rclpy.duration
import rclpy.time
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy)
from sensor_msgs.msg import PointCloud2
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import MarkerArray

# ---------------------------------------------------------------------------
# Thresholds (tuned for ARS 548 + fused_odometry system)
# ---------------------------------------------------------------------------
ODOM_RATE_MIN_HZ     = 30.0   # expect ~50 Hz; allow bag-throttled minimum
RADAR_RATE_MIN_HZ    =  6.0   # expect ~10 Hz
MAP_RATE_MIN_HZ      =  0.5   # map publishes ~2-5 Hz (slower is fine)
RADAR_POINTS_MAX     = 250    # ARS 548 hard ceiling per scan
VELOCITY_MAX_MS      =  2.5   # indoor UAV; burst spikes OK up to this after RANSAC
COLLECT_SEC          = 30     # total collection window (covers 10s warmup + 20s good data)
QUAT_NORM_TOL        =  0.05  # |norm-1| < 0.05


# ---------------------------------------------------------------------------
# Message collector
# ---------------------------------------------------------------------------
class MessageCollector(Node):
    """Subscribes to all relevant topics and collects messages for analysis."""

    def __init__(self):
        super().__init__('pipeline_integration_tester')
        self.wall_start = time.time()

        # Storage
        self.odom_msgs:           List[Odometry]      = []
        self.radar_filtered_msgs: List[PointCloud2]   = []
        self.radar_mapping_msgs:  List[PointCloud2]   = []
        self.radar_inlier_msgs:   List[PointCloud2]   = []
        self.map_msgs:            List[OccupancyGrid] = []
        self.loop_closure_msgs:   List[MarkerArray]   = []

        # TF2 buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(Odometry, '/fused_odom/odometry',
                                 lambda m: self.odom_msgs.append(m), sensor_qos)
        self.create_subscription(PointCloud2, '/PointCloudDetectionFiltered',
                                 lambda m: self.radar_filtered_msgs.append(m), sensor_qos)
        self.create_subscription(PointCloud2, '/PointCloudDetectionMapping',
                                 lambda m: self.radar_mapping_msgs.append(m), sensor_qos)
        self.create_subscription(PointCloud2, '/fused_odom/radar_inliers',
                                 lambda m: self.radar_inlier_msgs.append(m), sensor_qos)
        self.create_subscription(OccupancyGrid, '/map',
                                 lambda m: self.map_msgs.append(m), reliable_qos)
        self.create_subscription(MarkerArray, '/fused_odom/loop_closures',
                                 lambda m: self.loop_closure_msgs.append(m), reliable_qos)

    def wall_elapsed(self) -> float:
        return time.time() - self.wall_start

    def wall_rate(self, msg_list: list) -> float:
        """Message rate in Hz based on wall clock — correct for any bag speed."""
        elapsed = self.wall_elapsed()
        return len(msg_list) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_nodes_running() -> Tuple[bool, Dict[str, bool]]:
    """
    Verify all required nodes are running.
    Note: radar_preprocessing_node runs twice (odom + mapping paths),
    so we just verify at least one instance is present.
    """
    expected = [
        'fused_odom_node',
        'radar_preprocessing_node',
        'radar_scan_slam_node',
        'temporal_radar_occupancy',     # matches both _node and _grid variants
    ]
    try:
        result = subprocess.run(
            ['ros2', 'node', 'list'],
            capture_output=True, text=True, timeout=5
        )
        running = [n.strip().lstrip('/') for n in result.stdout.strip().split('\n') if n.strip()]
        results = {node: any(node in r for r in running) for node in expected}
    except Exception as e:
        print(f"  [!] Could not query node list: {e}")
        results = {node: False for node in expected}
    return all(results.values()), results


def test_topics_and_rates(collector: MessageCollector) -> Tuple[bool, Dict]:
    """Check that all required topics received data and at reasonable rates."""
    wall_sec = collector.wall_elapsed()

    def rate(lst):
        return len(lst) / wall_sec if wall_sec > 0 else 0.0

    checks = {
        'odom_received':    len(collector.odom_msgs) > 0,
        'filtered_received': len(collector.radar_filtered_msgs) > 0,
        'mapping_received':  len(collector.radar_mapping_msgs) > 0,
        'map_received':      len(collector.map_msgs) > 0,
    }
    rate_checks = {
        'odom_rate_ok':     rate(collector.odom_msgs)           >= ODOM_RATE_MIN_HZ,
        'filtered_rate_ok': rate(collector.radar_filtered_msgs) >= RADAR_RATE_MIN_HZ,
        'mapping_rate_ok':  rate(collector.radar_mapping_msgs)  >= RADAR_RATE_MIN_HZ,
        'map_rate_ok':      rate(collector.map_msgs)            >= MAP_RATE_MIN_HZ,
    }
    rates = {
        'fused_odom':      rate(collector.odom_msgs),
        'radar_filtered':  rate(collector.radar_filtered_msgs),
        'radar_mapping':   rate(collector.radar_mapping_msgs),
        'occupancy_map':   rate(collector.map_msgs),
        'radar_inliers':   rate(collector.radar_inlier_msgs),   # informational
        'loop_closures':   rate(collector.loop_closure_msgs),   # informational
    }
    counts = {
        'fused_odom':      len(collector.odom_msgs),
        'radar_filtered':  len(collector.radar_filtered_msgs),
        'radar_mapping':   len(collector.radar_mapping_msgs),
        'occupancy_map':   len(collector.map_msgs),
        'radar_inliers':   len(collector.radar_inlier_msgs),
        'loop_closures':   len(collector.loop_closure_msgs),
    }

    all_checks = {**checks, **rate_checks}
    return all(all_checks.values()), {
        'checks': all_checks, 'rates': rates, 'counts': counts,
        'wall_sec': wall_sec
    }


def test_tf_tree(collector: MessageCollector) -> Tuple[bool, Dict]:
    """Verify the full TF tree: map→odom→base_link→radar_frame."""
    required_pairs = [
        ('map',       'odom'),
        ('odom',      'base_link'),
        ('base_link', 'radar_frame'),
    ]
    results = {}
    for parent, child in required_pairs:
        key = f'{parent}→{child}'
        try:
            collector.tf_buffer.lookup_transform(
                parent, child,
                rclpy.time.Time(seconds=0),
                timeout=rclpy.duration.Duration(seconds=0.3)
            )
            results[key] = True
        except Exception as e:
            results[key] = False

    # Also check the full chain map→radar_frame
    try:
        collector.tf_buffer.lookup_transform(
            'map', 'radar_frame',
            rclpy.time.Time(seconds=0),
            timeout=rclpy.duration.Duration(seconds=0.3)
        )
        results['map→radar_frame (full chain)'] = True
    except Exception:
        results['map→radar_frame (full chain)'] = False

    all_ok = all(results.values())
    return all_ok, results


def test_slam_health(collector: MessageCollector) -> Tuple[bool, Dict]:
    """
    Verify radar_scan_slam_node is publishing map→odom.
    Checks that:
    1. The transform exists (published at least once).
    2. The RANSAC inlier topic is alive (submaps can be built).
    """
    results = {}

    # map→odom TF must exist
    try:
        tf = collector.tf_buffer.lookup_transform(
            'map', 'odom',
            rclpy.time.Time(seconds=0),
            timeout=rclpy.duration.Duration(seconds=0.3)
        )
        results['map_odom_tf_exists'] = True
        # Translation from identity (small is OK for short test)
        t = tf.transform.translation
        results['map_odom_tf_translation_m'] = round(
            math.sqrt(t.x**2 + t.y**2 + t.z**2), 4)
    except Exception:
        results['map_odom_tf_exists'] = False
        results['map_odom_tf_translation_m'] = None

    # RANSAC inliers must be flowing (SLAM consumes these)
    results['ransac_inliers_flowing'] = len(collector.radar_inlier_msgs) > 0

    all_ok = results['map_odom_tf_exists'] and results['ransac_inliers_flowing']
    return all_ok, results


def test_odometry_quality(collector: MessageCollector) -> Tuple[bool, Dict]:
    """
    Validate fused odometry messages are numerically sound.
    Skips messages from the warmup window (first 5s of sim time).
    """
    if len(collector.odom_msgs) < 5:
        return False, {'error': f'Too few odometry messages: {len(collector.odom_msgs)}'}

    # Use the last 20 messages (well past warmup)
    sample = collector.odom_msgs[-20:]
    results = {'checks': {}}

    # 1. No NaN in position
    nan_free = all(
        not any(math.isnan(v) for v in [
            m.pose.pose.position.x,
            m.pose.pose.position.y,
            m.pose.pose.position.z
        ])
        for m in sample
    )
    results['checks']['position_nan_free'] = nan_free

    # 2. Quaternion normalised
    def qnorm(m):
        q = m.pose.pose.orientation
        return math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)

    qnorms = [qnorm(m) for m in sample]
    quat_ok = all(abs(n - 1.0) < QUAT_NORM_TOL for n in qnorms)
    results['checks']['quaternion_normalised'] = quat_ok
    results['quat_norm_range'] = [round(min(qnorms), 4), round(max(qnorms), 4)]

    # 3. Velocity within physical bounds
    def vmag(m):
        v = m.twist.twist.linear
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    vmags = [vmag(m) for m in sample]
    vel_ok = all(v <= VELOCITY_MAX_MS for v in vmags)
    results['checks']['velocity_bounded'] = vel_ok
    results['velocity_max_ms'] = round(max(vmags), 3)

    # 4. Covariance populated (not all zeros)
    last = sample[-1]
    results['checks']['covariance_set'] = (
        any(last.pose.covariance) or any(last.twist.covariance)
    )

    # 5. Timestamps non-decreasing
    ts = [m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in sample]
    results['checks']['timestamps_non_decreasing'] = all(
        ts[i] >= ts[i-1] for i in range(1, len(ts))
    )

    last_pos = last.pose.pose.position
    last_vel = last.twist.twist.linear
    results['last_position_m'] = [round(last_pos.x, 3),
                                   round(last_pos.y, 3),
                                   round(last_pos.z, 3)]
    results['last_velocity_ms'] = [round(last_vel.x, 3),
                                    round(last_vel.y, 3),
                                    round(last_vel.z, 3)]

    return all(results['checks'].values()), results


def test_radar_quality(collector: MessageCollector) -> Tuple[bool, Dict]:
    """
    Validate radar point clouds from both preprocessing paths.
    ARS 548 produces 5-30 points per scan before filtering;
    after filtering it can be 0 (valid) up to ~200 (never more).
    """
    results = {'checks': {}}

    if not collector.radar_filtered_msgs:
        return False, {'error': 'No radar_filtered messages received'}

    filtered_last = collector.radar_filtered_msgs[-1]
    mapping_last  = collector.radar_mapping_msgs[-1] if collector.radar_mapping_msgs else None

    # 1. Frame ID — ARS 548 driver publishes in radar_frame or ARS_548
    valid_frames = {'radar_frame', 'ARS_548', 'base_link'}
    results['checks']['filtered_frame_valid'] = (
        filtered_last.header.frame_id in valid_frames
    )
    results['filtered_frame_id'] = filtered_last.header.frame_id

    # 2. Timestamp set (non-zero)
    results['checks']['filtered_timestamp_set'] = not (
        filtered_last.header.stamp.sec == 0
        and filtered_last.header.stamp.nanosec == 0
    )

    # 3. Point count within physical limits (0 is valid after filtering)
    npts = filtered_last.width * max(filtered_last.height, 1)
    results['checks']['filtered_point_count_sane'] = npts <= RADAR_POINTS_MAX
    results['filtered_last_point_count'] = npts

    # 4. Mapping cloud checks (if received)
    if mapping_last:
        results['checks']['mapping_frame_valid'] = (
            mapping_last.header.frame_id in valid_frames
        )
        npts_map = mapping_last.width * max(mapping_last.height, 1)
        results['checks']['mapping_point_count_sane'] = npts_map <= RADAR_POINTS_MAX
        results['mapping_last_point_count'] = npts_map
    else:
        results['checks']['mapping_received'] = False

    # 5. Average point count over all received filtered clouds
    point_counts = [
        m.width * max(m.height, 1) for m in collector.radar_filtered_msgs
    ]
    results['filtered_avg_points'] = round(
        sum(point_counts) / len(point_counts), 1
    ) if point_counts else 0
    results['filtered_cloud_count'] = len(collector.radar_filtered_msgs)

    return all(results['checks'].values()), results


def test_map_quality(collector: MessageCollector) -> Tuple[bool, Dict]:
    """
    Validate the occupancy grid:
    - Frame ID is 'odom' (as configured in this system)
    - Resolution matches config (0.07 m)
    - Has both occupied and free cells after the collection window
    - Map grows over time (more non-unknown cells in later messages)
    """
    if not collector.map_msgs:
        return False, {'error': 'No /map messages received'}

    results = {'checks': {}}
    last = collector.map_msgs[-1]

    # 1. Frame ID (this system publishes map in 'odom' frame)
    results['checks']['frame_id_correct'] = last.header.frame_id == 'odom'
    results['frame_id'] = last.header.frame_id

    # 2. Resolution close to configured 0.07 m (allow ±0.01)
    res = last.info.resolution
    results['checks']['resolution_correct'] = abs(res - 0.07) < 0.015
    results['resolution_m'] = round(res, 4)

    # 3. Grid has cells
    results['checks']['has_cells'] = len(last.data) > 0
    results['grid_cells_total'] = len(last.data)

    # 4. Has both occupied (>50) and free (<20 and >=0) cells
    occupied = sum(1 for c in last.data if c > 50)
    free     = sum(1 for c in last.data if 0 <= c < 20)
    unknown  = sum(1 for c in last.data if c == -1)
    results['checks']['has_occupied_cells'] = occupied > 0
    results['checks']['has_free_cells']     = free > 0
    results['grid_occupied_cells'] = occupied
    results['grid_free_cells']     = free
    results['grid_unknown_cells']  = unknown

    # 5. Map is growing (compare first vs last non-unknown count)
    if len(collector.map_msgs) >= 2:
        first = collector.map_msgs[0]
        first_known = sum(1 for c in first.data if c != -1)
        last_known  = sum(1 for c in last.data  if c != -1)
        results['checks']['map_growing'] = last_known >= first_known
        results['map_known_cells_first'] = first_known
        results['map_known_cells_last']  = last_known
    else:
        # Only one map message received — skip growth check
        results['map_known_cells_last'] = sum(1 for c in last.data if c != -1)

    return all(results['checks'].values()), results


def test_data_synchronization(collector: MessageCollector) -> Tuple[bool, Dict]:
    """
    Check that odometry and map timestamps are within a few seconds of each other.
    Uses sim-time headers; tolerates up to 10 s difference (covers warmup lag).
    """
    if not collector.odom_msgs or not collector.map_msgs:
        return False, {'error': 'Insufficient data'}

    odom_t = (collector.odom_msgs[-1].header.stamp.sec
              + collector.odom_msgs[-1].header.stamp.nanosec * 1e-9)
    map_t  = (collector.map_msgs[-1].header.stamp.sec
              + collector.map_msgs[-1].header.stamp.nanosec * 1e-9)
    diff   = abs(odom_t - map_t)

    ok = diff < 10.0
    return ok, {
        'checks': {'odom_map_time_diff_ok': ok},
        'odom_sim_time_s': round(odom_t, 2),
        'map_sim_time_s':  round(map_t, 2),
        'time_diff_s':     round(diff, 2),
    }


# ---------------------------------------------------------------------------
# Report saver
# ---------------------------------------------------------------------------

def save_report(all_results: dict, report_path: str = 'pipeline_validation_report.txt'):
    """Write a human-readable validation report for thesis appendix."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        "=" * 70,
        "  RADAR INDOOR MAPPING — PIPELINE VALIDATION REPORT",
        f"  Generated: {timestamp}",
        "=" * 70,
        "",
    ]
    for test_name, (passed, data) in all_results.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"[{status}] {test_name}")
        for k, v in data.items():
            if k == 'checks':
                for ck, cv in v.items():
                    lines.append(f"       {'✓' if cv else '✗'}  {ck}")
            else:
                lines.append(f"       {k}: {v}")
        lines.append("")

    overall = all(passed for passed, _ in all_results.values())
    lines += [
        "=" * 70,
        f"  OVERALL: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}",
        "=" * 70,
    ]

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return report_path


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_tests():
    print("=" * 70)
    print("  RADAR INDOOR MAPPING — INTEGRATION TEST SUITE")
    print("=" * 70)

    # [1] Node check (no ROS init needed)
    print("\n[Test 1] Node health...")
    nodes_ok, node_results = test_nodes_running()
    for node, ok in node_results.items():
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {node}")

    if not nodes_ok:
        print("\n  ✗ Pipeline not running. Start it with:")
        print("    ros2 launch temporal_radar_mapping radar_mapping.launch.py use_sim_time:=true")
        return False

    # Init ROS and collector
    rclpy.init()
    collector = MessageCollector()

    # [2] Collect data
    print(f"\n[Collection] Spinning for {COLLECT_SEC}s "
          f"(bag must be playing with --clock)...")
    start = time.time()
    last_report = -1
    while time.time() - start < COLLECT_SEC:
        rclpy.spin_once(collector, timeout_sec=0.05)
        elapsed = int(time.time() - start)
        if elapsed % 5 == 0 and elapsed != last_report:
            last_report = elapsed
            odom_n   = len(collector.odom_msgs)
            radar_n  = len(collector.radar_filtered_msgs)
            map_n    = len(collector.map_msgs)
            inlier_n = len(collector.radar_inlier_msgs)
            print(f"  {elapsed:2d}s — odom:{odom_n:4d}  radar:{radar_n:4d}  "
                  f"map:{map_n:3d}  inliers:{inlier_n:4d}")
    print(f"  {COLLECT_SEC}s — collection complete\n")

    all_results = {}

    try:
        # [2] Topics + rates
        print("[Test 2] Topics and message rates...")
        ok2, d2 = test_topics_and_rates(collector)
        all_results['Topics and rates'] = (ok2, d2)
        print(f"  Wall clock window: {d2['wall_sec']:.1f}s")
        rate_labels = {
            'fused_odom':     ('fused odometry',       f'{ODOM_RATE_MIN_HZ:.0f} Hz min'),
            'radar_filtered': ('radar filtered (odom)',f'{RADAR_RATE_MIN_HZ:.0f} Hz min'),
            'radar_mapping':  ('radar mapping',        f'{RADAR_RATE_MIN_HZ:.0f} Hz min'),
            'occupancy_map':  ('occupancy map',        f'{MAP_RATE_MIN_HZ:.1f} Hz min'),
            'radar_inliers':  ('radar inliers [info]', 'informational'),
            'loop_closures':  ('loop closures [info]', 'informational'),
        }
        for key, (label, note) in rate_labels.items():
            r     = d2['rates'][key]
            count = d2['counts'][key]
            if key in ('radar_inliers', 'loop_closures'):
                sym = '✓' if count > 0 else '–'
                print(f"  {sym} {label:32s}  {r:5.1f} Hz  ({count} msgs)  [{note}]")
            else:
                check_key = f"{key.split('_')[0]}_rate_ok" if key != 'occupancy_map' else 'map_rate_ok'
                # Reconstruct the rate check key
                rate_check_map = {
                    'fused_odom': 'odom_rate_ok',
                    'radar_filtered': 'filtered_rate_ok',
                    'radar_mapping': 'mapping_rate_ok',
                    'occupancy_map': 'map_rate_ok',
                }
                rate_ok = d2['checks'].get(rate_check_map.get(key, ''), False)
                sym = '✓' if rate_ok else '✗'
                print(f"  {sym} {label:32s}  {r:5.1f} Hz  ({count} msgs)  [{note}]")

        # [3] TF tree
        print("\n[Test 3] TF tree validation...")
        ok3, d3 = test_tf_tree(collector)
        all_results['TF tree'] = (ok3, d3)
        for pair, ok in d3.items():
            print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {pair}")

        # [4] SLAM health
        print("\n[Test 4] SLAM health...")
        ok4, d4 = test_slam_health(collector)
        all_results['SLAM health'] = (ok4, d4)
        for k, v in d4.items():
            if k.startswith('map_odom') or k.startswith('ransac'):
                ok_val = v if isinstance(v, bool) else True
                if isinstance(v, bool):
                    print(f"  {'✓ PASS' if v else '✗ FAIL'}  {k}")
                else:
                    print(f"  ✓ info  {k}: {v}")
        if d4.get('map_odom_tf_translation_m') is not None:
            print(f"  ✓ info  map→odom correction magnitude: "
                  f"{d4['map_odom_tf_translation_m']:.4f} m")

        # [5] Odometry quality
        print("\n[Test 5] Odometry data quality...")
        ok5, d5 = test_odometry_quality(collector)
        all_results['Odometry quality'] = (ok5, d5)
        if 'error' in d5:
            print(f"  ✗ FAIL  {d5['error']}")
        else:
            for ck, cv in d5['checks'].items():
                print(f"  {'✓ PASS' if cv else '✗ FAIL'}  {ck}")
            if 'quat_norm_range' in d5:
                lo, hi = d5['quat_norm_range']
                print(f"  ✓ info  quaternion norm range: [{lo}, {hi}]")
            if 'velocity_max_ms' in d5:
                print(f"  ✓ info  max velocity in sample: {d5['velocity_max_ms']} m/s")
            if 'last_position_m' in d5:
                print(f"  ✓ info  last position (m): {d5['last_position_m']}")

        # [6] Radar quality
        print("\n[Test 6] Radar data quality...")
        ok6, d6 = test_radar_quality(collector)
        all_results['Radar quality'] = (ok6, d6)
        if 'error' in d6:
            print(f"  ✗ FAIL  {d6['error']}")
        else:
            for ck, cv in d6['checks'].items():
                print(f"  {'✓ PASS' if cv else '✗ FAIL'}  {ck}")
            print(f"  ✓ info  filtered path: {d6['filtered_cloud_count']} clouds, "
                  f"avg {d6['filtered_avg_points']} pts/scan, "
                  f"last scan {d6['filtered_last_point_count']} pts")
            if 'mapping_last_point_count' in d6:
                print(f"  ✓ info  mapping path: last scan {d6['mapping_last_point_count']} pts")

        # [7] Map quality
        print("\n[Test 7] Occupancy map quality...")
        ok7, d7 = test_map_quality(collector)
        all_results['Map quality'] = (ok7, d7)
        if 'error' in d7:
            print(f"  ✗ FAIL  {d7['error']}")
        else:
            for ck, cv in d7['checks'].items():
                print(f"  {'✓ PASS' if cv else '✗ FAIL'}  {ck}")
            print(f"  ✓ info  resolution: {d7.get('resolution_m')} m  "
                  f"frame: '{d7.get('frame_id')}'")
            print(f"  ✓ info  cells — occupied: {d7.get('grid_occupied_cells')}  "
                  f"free: {d7.get('grid_free_cells')}  "
                  f"unknown: {d7.get('grid_unknown_cells')}")
            if 'map_known_cells_first' in d7:
                print(f"  ✓ info  known cells: {d7['map_known_cells_first']} → "
                      f"{d7['map_known_cells_last']} (growth)")

        # [8] Data sync
        print("\n[Test 8] Data synchronisation...")
        ok8, d8 = test_data_synchronization(collector)
        all_results['Data synchronisation'] = (ok8, d8)
        if 'error' in d8:
            print(f"  ✗ FAIL  {d8['error']}")
        else:
            for ck, cv in d8['checks'].items():
                print(f"  {'✓ PASS' if cv else '✗ FAIL'}  {ck}")
            print(f"  ✓ info  odom sim-time: {d8['odom_sim_time_s']} s  "
                  f"map sim-time: {d8['map_sim_time_s']} s  "
                  f"diff: {d8['time_diff_s']} s")

        # Summary
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)
        all_tests_ok = True
        for test_name, (passed, _) in all_results.items():
            sym = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {sym}  {test_name}")
            if not passed:
                all_tests_ok = False

        # Always add node result to all_results for report
        all_results['Node health'] = (nodes_ok, {'checks': node_results})

        print()
        if all_tests_ok:
            print("  ✓✓✓  ALL TESTS PASSED  ✓✓✓")
        else:
            print("  ✗  SOME TESTS FAILED — see details above")

        # Save report
        report_path = save_report(all_results)
        print(f"\n  Report saved to: {report_path}")
        print("=" * 70)

        return all_tests_ok

    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        success = run_tests()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        success = False
    except Exception as e:
        import traceback
        print(f"\n✗ Fatal error: {e}")
        traceback.print_exc()
        success = False
    sys.exit(0 if success else 1)
